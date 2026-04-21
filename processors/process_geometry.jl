# This file is a part of ML-based-LAr-veto, licensed under the MIT License (MIT).
# Geometry Processor - Generate HPGe and SiPM detector geometry YAMLs for ML training

include(joinpath(@__DIR__, "..", "src", "geometry.jl"))

"""
    process_geometry(processing_config, l200, group_name; reprocess=false)

Generate HPGe and SiPM detector geometry YAML files for the specified ML group.

For each detector that is processable=true in at least one run:
- Collects status (processable, usability) across all runs in the group
- Loads geometry from legend-metadata (HPGe) or computes from constants (SiPM)
- Loads positioning from legend-pygeom extra_meta
- Computes z-positions based on string layout (HPGe)
- Fails if a detector has inconsistent geometry across runs

Output:
- generated/geometry/HPGe/{group_name}_hpge_geometry.yaml
- generated/geometry/SiPM/{group_name}_sipm_geometry.yaml
"""
function process_geometry(processing_config::PropDict, l200::LegendData, group_name::String)
    @info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    @info "Process Geometry for ML group: $group_name"
    
    # ========================================================================
    # Setup paths
    # ========================================================================
    paths = processing_config.paths
    hpge_output_dir = joinpath(paths.output.geometry, "HPGe")
    sipm_output_dir = joinpath(paths.output.geometry, "SiPM")
    hpge_output_path = joinpath(hpge_output_dir, "$(group_name)_hpge_geometry.yaml")
    sipm_output_path = joinpath(sipm_output_dir, "$(group_name)_sipm_geometry.yaml")
    
    # Load data sources
    legend_metadata = data_path(l200, "metadata")
    diodes_dir = joinpath(legend_metadata, "hardware", "detectors", "germanium", "diodes")
    pygeom_path = get(paths, :legend_pygeom, nothing)
    
    @info "Data sources" diodes_dir=diodes_dir pygeom_path=pygeom_path
    
    # ========================================================================
    # Load group definition (periods and runs)
    # ========================================================================
    groupings_path = paths.groupings
    @info "Loading group definition from $groupings_path"
    groupings = readlprops(groupings_path)
    
    if !haskey(groupings, Symbol(group_name))
        error("Group '$group_name' not found in $groupings_path")
    end
    
    group_def = groupings[Symbol(group_name)]
    periods_runs = collect_periods_runs(group_def)
    @info "Group contains $(length(periods_runs)) period-run combinations"
    

    
    # ========================================================================
    # Collect detector status across all runs (HPGe and SiPM)
    # ========================================================================
    hpge_statuses = Dict{String, DetectorStatus}()
    sipm_statuses = Dict{String, SiPMStatus}()
    periods_seen = Set{String}()
    filekeys_collected = Vector{FileKey}()
    
    for (period, run) in periods_runs
        @info "Processing" period=period run=run
        
        # Get filekey for this run
        filekey = try
            start_filekey(l200, (DataPeriod(period), DataRun(run), :phy))
        catch e
            @warn "Could not get filekey for phy, trying cal" period=period run=run
            try
                start_filekey(l200, (DataPeriod(period), DataRun(run), :cal))
            catch e2
                @warn "Skipping run - no valid filekey" period=period run=run
                continue
            end
        end
        
        push!(periods_seen, period)
        push!(filekeys_collected, filekey)
        
        # ====================================================================
        # Process HPGe detectors
        # ====================================================================
        chinfo_geds = try
            channelinfo(l200, filekey; system=:geds)
        catch e
            @warn "Could not get geds channelinfo" period=period run=run error=e
            nothing
        end
        
        if chinfo_geds !== nothing
            for i in 1:length(chinfo_geds)
                det_name = string(chinfo_geds.detector[i])
                
                if !haskey(hpge_statuses, det_name)
                    hpge_statuses[det_name] = DetectorStatus(det_name)
                end
                
                try
                    add_run_status!(
                        hpge_statuses[det_name],
                        period, run,
                        chinfo_geds.processable[i],
                        chinfo_geds.usability[i],
                        chinfo_geds.detstring[i],
                        chinfo_geds.position[i],
                        chinfo_geds.rawid[i]
                    )
                catch e
                    @error "HPGe geometry inconsistency!" detector=det_name error=e
                    rethrow(e)
                end
            end
        end
        
        # ====================================================================
        # Process SiPM detectors
        # ====================================================================
        chinfo_spms = try
            channelinfo(l200, filekey; system=:spms)
        catch e
            @warn "Could not get spms channelinfo" period=period run=run error=e
            nothing
        end
        
        if chinfo_spms !== nothing
            for i in 1:length(chinfo_spms)
                det_name = string(chinfo_spms.detector[i])
                
                # Extract location from channelinfo fields (convert to String explicitly)
                fiber_str = String(string(chinfo_spms.fiber[i]))
                barrel_str = String(fiber_str[1:2])  # "IB" or "OB" prefix
                position_str = chinfo_spms.position[i] == 1 ? "top" : "bottom"
                
                if !haskey(sipm_statuses, det_name)
                    sipm_statuses[det_name] = SiPMStatus(det_name)
                end
                
                try
                    add_sipm_run_status!(
                        sipm_statuses[det_name],
                        period, run,
                        chinfo_spms.processable[i],
                        chinfo_spms.usability[i],
                        barrel_str,
                        fiber_str,
                        position_str,
                        chinfo_spms.rawid[i]
                    )
                catch e
                    @error "SiPM location inconsistency!" detector=det_name error=e
                    rethrow(e)
                end
            end
        end
    end
    
    @info "Found detectors" hpge=length(hpge_statuses) sipm=length(sipm_statuses)
    
    # Check if we have multiple periods (potential geometry change)
    if length(periods_seen) > 1
        @warn "Group spans multiple periods - verify geometry consistency" periods=periods_seen
    end
    
    # ========================================================================
    # Load extra_meta for HPGe (rodlength, string geometry)
    # ========================================================================
    hpges, hpge_strings = (nothing, nothing)
    
    if pygeom_path !== nothing && isdir(pygeom_path) && !isempty(filekeys_collected)
        first_fk = first(filekeys_collected)
        first_timestamp = DateTime(first_fk)
        @info "Using timestamp for extra_meta validity" filekey=first_fk timestamp=first_timestamp
        hpges, hpge_strings = load_hpge_extra_meta(pygeom_path, first_timestamp)
        
        if hpges === nothing
            @warn "Could not load extra_meta - HPGe positions will be incomplete"
        end
    end
    
    # ========================================================================
    # Generate HPGe Geometry YAML
    # ========================================================================
    begin
        @info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        @info "Generating HPGe geometry"
        
        processable_hpge = filter(p -> is_ever_processable(p.second), hpge_statuses)
        @info "HPGe detectors with processable=true: $(length(processable_hpge))"
        
        # Group by string and compute z-positions
        string_detectors = Dict{Int, Vector{StringDetectorInfo}}()
        
        for (det_name, status) in processable_hpge
            status.string_id === nothing && continue
            
            diode_geom = load_diode_geometry(diodes_dir, det_name)
            if diode_geom === nothing
                @warn "Missing diode geometry" detector=det_name
                continue
            end
            
            rodlength = get_detector_rodlength(hpges, det_name)
            rodlength = rodlength === nothing ? 103.5 : rodlength
            
            string_id = status.string_id
            if !haskey(string_detectors, string_id)
                string_detectors[string_id] = StringDetectorInfo[]
            end
            
            push!(string_detectors[string_id], StringDetectorInfo(
                det_name,
                status.position_in_string,
                diode_geom.height_in_mm,
                rodlength
            ))
        end
        
        # Compute z-positions
        z_positions = Dict{String, NamedTuple}()
        for (string_id, detectors) in string_detectors
            merge!(z_positions, compute_string_z_positions(detectors))
        end
        
        # Build output
        hpge_output = Dict{String, Any}()
        
        for (det_name, status) in sort(collect(processable_hpge), by=p->p.first)
            diode_geom = load_diode_geometry(diodes_dir, det_name)
            diode_geom === nothing && continue
            
            string_radius, string_angle = get_string_geometry(hpge_strings, status.string_id)
            string_radius = string_radius === nothing ? 220.0 : string_radius
            string_angle = string_angle === nothing ? 0.0 : string_angle
            
            rodlength = get_detector_rodlength(hpges, det_name)
            rodlength = rodlength === nothing ? 103.5 : rodlength
            
            det_z = get(z_positions, det_name, (z_top=0.0, z_center=0.0, z_bottom=0.0))
            
            hpge_output[det_name] = build_detector_entry(
                det_name, status, diode_geom,
                string_radius, string_angle, rodlength, det_z
            )
        end

        # ── Post-processing: add scaled_angle, scaled_z_center ───────────
        all_angles = Float64[e["detector_position"]["cylind_coords"]["angle"]["value"] for e in values(hpge_output)]
        all_z      = Float64[e["detector_position"]["cylind_coords"]["z_center"]["value"] for e in values(hpge_output)]
        min_z, max_z = extrema(all_z)
        range_z = max_z - min_z
        @info "  HPGe scaled fields: angle /360, z_center [$min_z, $max_z] mm"
        for entry in values(hpge_output)
            pos = entry["detector_position"]
            a = pos["cylind_coords"]["angle"]["value"]
            z = pos["cylind_coords"]["z_center"]["value"]
            pos["scaled_angle"]    = round(a / 360.0; digits=6)
            pos["scaled_z_center"] = range_z ≈ 0.0 ? 0.5 : round((z - min_z) / range_z; digits=6)
        end

        mkpath(hpge_output_dir)
        write_geometry_yaml(hpge_output_path, group_name, hpge_output)
        @info "HPGe geometry complete" n_detectors=length(hpge_output)
    end
    
    # ========================================================================
    # Generate SiPM Geometry YAML
    # ========================================================================
    begin
        @info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        @info "Generating SiPM geometry"
        
        processable_sipm = filter(p -> is_sipm_ever_processable(p.second), sipm_statuses)
        @info "SiPM detectors with processable=true: $(length(processable_sipm))"
        
        # Build output
        sipm_output = Dict{String, Any}()
        
        for (det_name, status) in sort(collect(processable_sipm), by=p->p.first)
            sipm_output[det_name] = build_sipm_entry(det_name, status)
        end
        
        mkpath(sipm_output_dir)
        write_geometry_yaml(sipm_output_path, group_name, sipm_output)
        @info "SiPM geometry complete" n_detectors=length(sipm_output)
    end
    
    @info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    @info "Geometry processing complete"
    
    # ========================================================================
    # Generate Relative Geometry YAML (HPGe ↔ SiPM)
    # ========================================================================
    relative_output_dir = joinpath(paths.output.geometry, "relative")
    relative_output_path = joinpath(relative_output_dir, "$(group_name)_relative_geometry.yaml")

    begin
        @info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        @info "Generating relative geometry (HPGe ↔ SiPM)"

        if !isfile(hpge_output_path) || !isfile(sipm_output_path)
            @error "Cannot generate relative geometry — HPGe or SiPM geometry files missing" hpge=hpge_output_path sipm=sipm_output_path
        else
            relative_data, max_abs_dz = compute_relative_geometry(hpge_output_path, sipm_output_path)
            write_relative_geometry_yaml(
                relative_output_path, group_name,
                relative_data, max_abs_dz,
                hpge_output_path, sipm_output_path,
            )
            @info "Relative geometry complete" n_hpge=length(relative_data)
        end
    end

    @info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    @info "All geometry processing complete"
    
    return (hpge=hpge_output_path, sipm=sipm_output_path, relative=relative_output_path)
end

# ============================================================================
# Helper Functions
# ============================================================================

"""
    collect_periods_runs(group_def)

Convert group definition PropDict to list of (period, run) tuples.
"""
function collect_periods_runs(group_def)
    periods_runs = Tuple{String, String}[]
    
    for period in keys(group_def)
        runs = group_def[period]
        for run in runs
            push!(periods_runs, (string(period), string(run)))
        end
    end
    
    sort!(periods_runs)
    periods_runs
end
