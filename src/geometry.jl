# This file is a part of ML-based-LAr-veto, licensed under the MIT License (MIT).
# Geometry utilities for HPGe and SiPM detector positioning

using YAML
using Dates

# JSON3 is only needed for load_sipm_channelmap() — make it optional
const _HAS_JSON3 = try
    @eval using JSON3
    true
catch
    false
end

# ============================================================================
# Data Loading Functions
# ============================================================================

"""
    load_diode_geometry(diodes_dir::String, detector_name::String)

Load detector geometry (height, radius) from legend-metadata diodes directory.
Returns NamedTuple with height_in_mm and radius_in_mm, or nothing if not found.
"""
function load_diode_geometry(diodes_dir::String, detector_name::String)
    diode_file = joinpath(diodes_dir, "$(detector_name).yaml")
    !isfile(diode_file) && return nothing
    
    data = YAML.load_file(diode_file; dicttype=Dict{String, Any})
    geometry = get(data, "geometry", nothing)
    geometry === nothing && return nothing
    
    height = get(geometry, "height_in_mm", nothing)
    radius = get(geometry, "radius_in_mm", nothing)
    (height === nothing || radius === nothing) && return nothing
    
    (height_in_mm = Float64(height), radius_in_mm = Float64(radius))
end

"""
    parse_validity_timestamp(ts_str::String)

Parse a validity timestamp string like "20251108T002705Z" to DateTime.
"""
function parse_validity_timestamp(ts_str::String)
    # Handle format "YYYYMMDDTHHMMSSz" 
    Dates.DateTime(ts_str, dateformat"yyyymmddTHHMMSSZ")
end

"""
    find_extra_meta_dir(pygeom_path::String)

Find the extra_meta directory in legend-pygeom.
"""
function find_extra_meta_dir(pygeom_path::String)
    possible_paths = [
        joinpath(pygeom_path, "src", "pygeoml200", "configs", "extra_meta"),
        joinpath(pygeom_path, "src", "l200geom", "configs", "extra_meta"),
    ]
    
    for path in possible_paths
        isdir(path) && return path
    end
    
    @warn "Extra meta directory not found" tried=possible_paths
    return nothing
end

"""
    load_hpge_extra_meta(pygeom_path::String, timestamp::DateTime)

Load HPGe extra metadata (rodlength, string positions) from legend-pygeom.
Uses validity.yaml to find the correct configs for the given timestamp.
Follows LegendDataManagement validity logic with searchsortedlast.
Returns (hpges::Dict, hpge_strings::Dict) or (nothing, nothing) if not found.
"""
function load_hpge_extra_meta(pygeom_path::String, timestamp::DateTime)
    extra_meta_dir = find_extra_meta_dir(pygeom_path)
    extra_meta_dir === nothing && return (nothing, nothing)
    
    validity_file = joinpath(extra_meta_dir, "validity.yaml")
    if !isfile(validity_file)
        @warn "No validity.yaml found" path=validity_file
        return (nothing, nothing)
    end
    
    # Load validity.yaml
    validity_entries = YAML.load_file(validity_file; dicttype=Dict{Any, Any})
    
    # Parse entries and sort by valid_from timestamp
    parsed_entries = Vector{@NamedTuple{valid_from::DateTime, apply::Vector{String}}}()
    
    for entry in validity_entries
        vf_str = get(entry, "valid_from", nothing)
        apply_list = get(entry, "apply", String[])
        
        vf_str === nothing && continue
        isempty(apply_list) && continue
        
        try
            vf = parse_validity_timestamp(vf_str)
            push!(parsed_entries, (valid_from = vf, apply = apply_list))
        catch e
            @warn "Could not parse validity entry" valid_from=vf_str error=e
        end
    end
    
    if isempty(parsed_entries)
        @warn "No valid entries in validity.yaml"
        return (nothing, nothing)
    end
    
    # Sort by valid_from
    sort!(parsed_entries, by = e -> e.valid_from)
    valid_from_times = [e.valid_from for e in parsed_entries]
    
    # Use searchsortedlast like LegendDataManagement
    idx = searchsortedlast(valid_from_times, timestamp)
    
    if idx < 1
        @warn "Timestamp before first validity entry" timestamp=timestamp first_valid=valid_from_times[1]
        return (nothing, nothing)
    end
    
    config_files = parsed_entries[idx].apply
    @info "Using extra_meta configs for timestamp" timestamp=timestamp configs=config_files
    
    # Load and merge all apply configs.
    # NOTE: validity entries stack configs (e.g. p18 = [p15-config, p18-config]),
    # and a later config may *partially* override an entry — e.g. the p18 config
    # only sets `minishroud_delta_length_in_mm` for strings 1 & 7. We therefore
    # DEEP-merge per-entry: a shallow `merged[k] = v` would drop `radius_in_mm` /
    # `angle_in_deg`, silently collapsing those strings onto the (220, 0)
    # fallback in `build_hpge_geometry` (the cause of strings 1/7 overlapping).
    deepmerge!(dst::AbstractDict, src::AbstractDict) = begin
        for (k, v) in src
            if v isa AbstractDict && get(dst, k, nothing) isa AbstractDict
                deepmerge!(dst[k], v)
            else
                dst[k] = v
            end
        end
        dst
    end
    merged_hpges = Dict{Any, Any}()
    merged_hpge_strings = Dict{Any, Any}()

    for config_file in config_files
        config_path = joinpath(extra_meta_dir, config_file)
        if !isfile(config_path)
            @warn "Config file not found" config=config_file
            continue
        end

        raw = YAML.load_file(config_path; dicttype=Dict{Any, Any})

        # Merge hpges / hpge_string (later configs override earlier; partial
        # entries are merged into, not replacing, the existing entry).
        deepmerge!(merged_hpges,        get(raw, "hpges",       Dict{Any, Any}()))
        deepmerge!(merged_hpge_strings, get(raw, "hpge_string", Dict{Any, Any}()))
    end
    
    @debug "Loaded extra_meta" n_hpges=length(merged_hpges) n_strings=length(merged_hpge_strings)
    
    (merged_hpges, merged_hpge_strings)
end

# Legacy function for backward compatibility - converts period to approximate timestamp
function load_hpge_extra_meta(pygeom_path::String, period::String)
    @warn "Using legacy period-based extra_meta loading - prefer timestamp-based loading"
    # Map period to approximate start timestamp (not precise, use timestamp version when possible)
    period_timestamps = Dict(
        "p03" => DateTime(2023, 3, 11, 23, 58, 40),
        "p10" => DateTime(2024, 4, 11),
        "p11" => DateTime(2024, 4, 11),
        "p13" => DateTime(2024, 12, 2, 15, 0),
        "p14" => DateTime(2025, 4, 25, 18, 1, 15),
        "p15" => DateTime(2025, 7, 16, 16, 15, 17),
        "p16" => DateTime(2025, 7, 17),  # Between p15 and p18
        "p17" => DateTime(2025, 10, 1),  # Approximate
        "p18" => DateTime(2025, 11, 8, 0, 27, 5),
        "p19" => DateTime(2026, 1, 1),   # After p18
    )
    
    ts = get(period_timestamps, period, nothing)
    if ts === nothing
        @warn "Unknown period, using current time" period=period
        ts = now()
    end
    
    load_hpge_extra_meta(pygeom_path, ts)
end

"""
    get_detector_rodlength(hpges::Dict, detector_name::String)

Get rodlength_in_mm for a detector from extra_meta hpges dict.
"""
function get_detector_rodlength(hpges::Union{Dict, Nothing}, detector_name::String)
    hpges === nothing && return nothing
    det_meta = get(hpges, detector_name, nothing)
    det_meta === nothing && return nothing
    rodlength = get(det_meta, "rodlength_in_mm", nothing)
    rodlength === nothing ? nothing : Float64(rodlength)
end

"""
    get_string_geometry(hpge_strings::Dict, string_id::Int)

Get (radius_in_mm, angle_in_deg) for a string from extra_meta.
"""
function get_string_geometry(hpge_strings::Union{Dict, Nothing}, string_id::Int)
    hpge_strings === nothing && return (nothing, nothing)
    
    # Try both integer and string keys (YAML may parse as either)
    str_meta = get(hpge_strings, string_id, nothing)
    if str_meta === nothing
        str_meta = get(hpge_strings, string(string_id), nothing)
    end
    str_meta === nothing && return (nothing, nothing)
    
    radius = get(str_meta, "radius_in_mm", nothing)
    angle = get(str_meta, "angle_in_deg", nothing)
    
    (radius === nothing ? nothing : Float64(radius),
     angle === nothing ? nothing : Float64(angle))
end

# ============================================================================
# Position Calculation Functions
# ============================================================================

"""
    compute_cartesian_coords(radius_mm::Float64, angle_deg::Float64)

Convert cylindrical to cartesian coordinates.
Returns (x_mm, y_mm).
"""
function compute_cartesian_coords(radius_mm::Float64, angle_deg::Float64)
    angle_rad = deg2rad(mod(angle_deg, 360))
    x = radius_mm * cos(angle_rad)
    y = -radius_mm * sin(angle_rad)  # Negative because of coordinate convention
    (round(x; digits=2), round(y; digits=2))
end

"""
    compute_z_positions(z_top::Float64, height_mm::Float64)

Compute z positions for a detector.
Returns (z_top, z_center, z_bottom).
"""
function compute_z_positions(z_top::Float64, height_mm::Float64)
    z_bottom = z_top - height_mm
    z_center = (z_top + z_bottom) / 2
    (round(z_top; digits=2), round(z_center; digits=2), round(z_bottom; digits=2))
end

# ============================================================================
# Status Tracking Functions
# ============================================================================

"""
    DetectorStatus

Mutable struct to track detector status across runs.
"""
mutable struct DetectorStatus
    detector_name::String
    processable_true::Vector{String}   # List of "pXX-rYYY" strings
    processable_false::Vector{String}
    usable_on::Vector{String}
    usable_ac::Vector{String}
    usable_off::Vector{String}
    # Geometry info (should be consistent across runs)
    string_id::Union{Int, Nothing}
    position_in_string::Union{Int, Nothing}
    rawid::Union{Int, Nothing}
end

DetectorStatus(name::String) = DetectorStatus(
    name, String[], String[], String[], String[], String[], nothing, nothing, nothing
)

"""
    add_run_status!(status::DetectorStatus, period::String, run::String, 
                    processable::Bool, usability::Symbol, 
                    string_id::Int, position::Int, rawid::Int)

Add a run's status to detector tracking. Checks for geometry consistency.
"""
function add_run_status!(status::DetectorStatus, period::String, run::String,
                         processable::Bool, usability::Symbol,
                         string_id::Int, position::Int, rawid::Int)
    run_str = "$(period)-$(run)"
    
    # Track processable status
    if processable
        push!(status.processable_true, run_str)
    else
        push!(status.processable_false, run_str)
    end
    
    # Track usability status
    if usability == :on
        push!(status.usable_on, run_str)
    elseif usability == :ac
        push!(status.usable_ac, run_str)
    elseif usability == :off
        push!(status.usable_off, run_str)
    end
    
    # Check geometry consistency
    if status.string_id === nothing
        status.string_id = string_id
        status.position_in_string = position
        status.rawid = rawid
    else
        if status.string_id != string_id || status.position_in_string != position
            error("Geometry inconsistency for $(status.detector_name): " *
                  "string_id $(status.string_id) vs $string_id, " *
                  "position $(status.position_in_string) vs $position in $run_str")
        end
    end
end

"""
    is_ever_processable(status::DetectorStatus)

Check if detector was processable in at least one run.
"""
is_ever_processable(status::DetectorStatus) = !isempty(status.processable_true)

"""
    group_runs_by_period(run_strs::Vector{String})

Group run strings like ["p18-r000", "p18-r001", "p16-r002"] into 
Dict("p18" => ["r000", "r001"], "p16" => ["r002"]).
"""
function group_runs_by_period(run_strs::Vector{String})
    grouped = Dict{String, Vector{String}}()
    
    for rs in run_strs
        parts = split(rs, "-")
        length(parts) >= 2 || continue
        period = string(parts[1])
        run = string(parts[2])
        
        if !haskey(grouped, period)
            grouped[period] = String[]
        end
        push!(grouped[period], run)
    end
    
    # Sort runs within each period
    for (period, runs) in grouped
        sort!(runs)
    end
    
    grouped
end

# ============================================================================
# String Position Computation (z-coordinates)
# ============================================================================

"""
    StringDetectorInfo

Info for a detector in a string for z-position computation.
"""
struct StringDetectorInfo
    detector_name::String
    position_in_string::Int
    height_in_mm::Float64
    rodlength_in_mm::Float64
end

"""
    compute_string_z_positions(detectors::Vector{StringDetectorInfo})

Compute z positions for all detectors in a string.
Returns Dict{String, NamedTuple} with z_top, z_center, z_bottom per detector.
"""
function compute_string_z_positions(detectors::Vector{StringDetectorInfo})
    sorted = sort(detectors, by=d -> d.position_in_string)
    result = Dict{String, NamedTuple{(:z_top, :z_center, :z_bottom), Tuple{Float64, Float64, Float64}}}()
    
    current_z_top = 0.0
    for det in sorted
        z_top, z_center, z_bottom = compute_z_positions(current_z_top, det.height_in_mm)
        result[det.detector_name] = (z_top=z_top, z_center=z_center, z_bottom=z_bottom)
        current_z_top -= det.rodlength_in_mm  # Next detector starts below rod
    end
    
    result
end

# ============================================================================
# YAML Output Building
# ============================================================================

"""
    build_detector_entry(det_name::String, status::DetectorStatus, 
                         diode_geom, string_radius, string_angle, 
                         rodlength, z_positions)

Build the YAML dict entry for a single detector.
"""
function build_detector_entry(det_name::String, status::DetectorStatus,
                              diode_geom::NamedTuple, string_radius::Float64, 
                              string_angle::Float64, rodlength::Float64,
                              z_positions::NamedTuple)
    x_mm, y_mm = compute_cartesian_coords(string_radius, string_angle)
    
    Dict{String, Any}(
        "status" => Dict{String, Any}(
            "processable_true" => group_runs_by_period(status.processable_true),
            "processable_false" => group_runs_by_period(status.processable_false),
            "usable_on" => group_runs_by_period(status.usable_on),
            "usable_ac" => group_runs_by_period(status.usable_ac),
            "usable_off" => group_runs_by_period(status.usable_off),
        ),
        "detector_geometry" => Dict{String, Any}(
            "height" => Dict("value" => round(diode_geom.height_in_mm; digits=2), "unit" => "mm"),
            "radius" => Dict("value" => round(diode_geom.radius_in_mm; digits=2), "unit" => "mm"),
        ),
        "detector_position" => Dict{String, Any}(
            "string_id" => status.string_id,
            "position_in_string" => status.position_in_string,
            "cylind_coords" => Dict{String, Any}(
                "radius" => Dict("value" => round(string_radius; digits=2), "unit" => "mm"),
                "angle" => Dict("value" => round(string_angle; digits=2), "unit" => "deg"),
                "z_center" => Dict("value" => round(z_positions.z_center; digits=2), "unit" => "mm"),
            ),
            "cartes_coords" => Dict{String, Any}(
                "x" => Dict("value" => round(x_mm; digits=2), "unit" => "mm"),
                "y" => Dict("value" => round(y_mm; digits=2), "unit" => "mm"),
                "z_top" => Dict("value" => round(z_positions.z_top; digits=2), "unit" => "mm"),
                "z_center" => Dict("value" => round(z_positions.z_center; digits=2), "unit" => "mm"),
                "z_bottom" => Dict("value" => round(z_positions.z_bottom; digits=2), "unit" => "mm"),
            ),
        ),
        "additional" => Dict{String, Any}(),
    )
end

"""
    write_geometry_yaml(output_path::String, group_name::String, 
                        detectors::Dict{String, Any}, sources::Dict{String, Any})

Write the complete geometry YAML file.
Detectors are placed directly on top level (no wrapper).
"""
function write_geometry_yaml(output_path::String, group_name::String,
                             detectors::Dict{String, Any})
    # Build output with detectors on top level (no wrapper, no metadata)
    output = Dict{String, Any}()
    
    # Add all detectors directly on top level
    for (det_name, det_data) in detectors
        output[det_name] = det_data
    end
    
    mkpath(dirname(output_path))
    open(output_path, "w") do io
        YAML.write(io, output)
    end
    
    @info "Geometry YAML written" path=output_path n_detectors=length(detectors)
    output_path
end

# ============================================================================
# SiPM Geometry Functions
# ============================================================================

# SiPM geometry constants (fixed hardware dimensions)
const SIPM_OFFSET_TOP_PLATE_TO_GE_ZERO = 422.1
const SIPM_OFFSET_TOP_PLATE_TO_SIPM_TOP_OB = 29.2
const SIPM_OFFSET_IB_VERSUS_OB = 35.0
const SIPM_OB_RADIUS_MM = 290.0
const SIPM_IB_RADIUS_MM = 130.0
const SIPM_OB_FIBER_LENGTH_STRAIGHT = 1320.8
const SIPM_IB_FIBER_LENGTH = 1400.0
const SIPM_OB_BEND_RADIUS = 165.0

# Computed z-positions (rounded to 2 decimals)
const SIPM_Z_OB_TOP = round(SIPM_OFFSET_TOP_PLATE_TO_GE_ZERO - SIPM_OFFSET_TOP_PLATE_TO_SIPM_TOP_OB; digits=2)
const SIPM_Z_IB_TOP = round(SIPM_Z_OB_TOP - SIPM_OFFSET_IB_VERSUS_OB; digits=2)
const SIPM_Z_OB_BOTTOM = round(SIPM_Z_OB_TOP - SIPM_OB_FIBER_LENGTH_STRAIGHT - SIPM_OB_BEND_RADIUS; digits=2)
const SIPM_Z_IB_BOTTOM = round(SIPM_Z_IB_TOP - SIPM_IB_FIBER_LENGTH; digits=2)

"""
    SiPMStatus

Mutable struct to track SiPM status across runs.
"""
mutable struct SiPMStatus
    detector_name::String
    processable_true::Vector{String}   # List of "pXX-rYYY" strings
    processable_false::Vector{String}
    usable_on::Vector{String}
    usable_off::Vector{String}
    # Location info (should be consistent across runs)
    barrel::Union{String, Nothing}
    fiber::Union{String, Nothing}
    position::Union{String, Nothing}   # "top" or "bottom"
    rawid::Union{Int, Nothing}
end

SiPMStatus(name::String) = SiPMStatus(
    name, String[], String[], String[], String[], nothing, nothing, nothing, nothing
)

"""
    add_sipm_run_status!(status::SiPMStatus, period::String, run::String,
                         processable::Bool, usability::Symbol,
                         barrel::String, fiber::String, position::String, rawid::Int)

Add a run's status to SiPM tracking. Checks for location consistency.
"""
function add_sipm_run_status!(status::SiPMStatus, period::String, run::String,
                              processable::Bool, usability::Symbol,
                              barrel::String, fiber::String, position::String, rawid::Int)
    run_str = "$(period)-$(run)"
    
    # Track processable status
    if processable
        push!(status.processable_true, run_str)
    else
        push!(status.processable_false, run_str)
    end
    
    # Track usability status (SiPMs typically only have on/off)
    if usability == :on
        push!(status.usable_on, run_str)
    elseif usability == :off
        push!(status.usable_off, run_str)
    end
    
    # Check location consistency
    if status.barrel === nothing
        status.barrel = barrel
        status.fiber = fiber
        status.position = position
        status.rawid = rawid
    else
        if status.barrel != barrel || status.fiber != fiber || status.position != position
            error("Location inconsistency for SiPM $(status.detector_name): " *
                  "barrel $(status.barrel) vs $barrel, " *
                  "fiber $(status.fiber) vs $fiber, " *
                  "position $(status.position) vs $position in $run_str")
        end
    end
end

"""
    is_sipm_ever_processable(status::SiPMStatus)

Check if SiPM was processable in at least one run.
"""
is_sipm_ever_processable(status::SiPMStatus) = !isempty(status.processable_true)

"""
    get_sipm_module_num(fiber_name::String)

Extract module number from fiber name (e.g., "IB013014" -> 6).
"""
function get_sipm_module_num(fiber_name::String)
    div(parse(Int, fiber_name[3:5]) - 1, 2)
end

"""
    compute_sipm_angle(barrel::String, fiber::String)

Compute angle in degrees for a SiPM based on barrel and fiber.
Returns mirrored angle (coordinate convention).
"""
function compute_sipm_angle(barrel::String, fiber::String)
    module_num = get_sipm_module_num(fiber)
    
    angle_deg = if barrel == "OB"
        # OB: 20 modules, reference is OB015016
        zero_mod = get_sipm_module_num("OB015016")
        rad2deg(2 * π / 20 * (module_num - zero_mod - 0.5))
    else
        # IB: 9 modules, reference is IB013014
        zero_mod = get_sipm_module_num("IB013014")
        rad2deg(2 * π / 9 * (module_num - zero_mod - 0.5))
    end
    
    # Normalize and mirror (coordinate convention)
    angle_norm = mod(angle_deg, 360)
    mod(360 - angle_norm, 360)
end

"""
    compute_sipm_coordinates(barrel::String, fiber::String, position::String)

Compute cylindrical and cartesian coordinates for a SiPM.
Returns NamedTuple with (radius_mm, angle_deg, z_mm, x_mm, y_mm).
"""
function compute_sipm_coordinates(barrel::String, fiber::String, position::String)
    # Get radius based on barrel
    radius_mm = barrel == "IB" ? SIPM_IB_RADIUS_MM : SIPM_OB_RADIUS_MM
    
    # Get z based on barrel and position
    z_mm = if barrel == "IB"
        position == "top" ? SIPM_Z_IB_TOP : SIPM_Z_IB_BOTTOM
    else
        position == "top" ? SIPM_Z_OB_TOP : SIPM_Z_OB_BOTTOM
    end
    
    # Compute angle
    angle_deg = compute_sipm_angle(barrel, fiber)
    angle_rad = deg2rad(angle_deg)
    
    # Compute cartesian coordinates
    x_mm = radius_mm * cos(angle_rad)
    y_mm = -radius_mm * sin(angle_rad)  # Negative due to coordinate convention
    
    (
        radius_mm = round(radius_mm; digits=2),
        angle_deg = round(angle_deg; digits=2),
        z_mm = round(z_mm; digits=2),
        x_mm = round(x_mm; digits=2),
        y_mm = round(y_mm; digits=2),
    )
end

"""
    build_sipm_entry(det_name::String, status::SiPMStatus)

Build the YAML dict entry for a single SiPM.
"""
function build_sipm_entry(det_name::String, status::SiPMStatus)
    coords = compute_sipm_coordinates(status.barrel, status.fiber, status.position)
    
    Dict{String, Any}(
        "status" => Dict{String, Any}(
            "processable_true" => group_runs_by_period(status.processable_true),
            "processable_false" => group_runs_by_period(status.processable_false),
            "usable_on" => group_runs_by_period(status.usable_on),
            "usable_off" => group_runs_by_period(status.usable_off),
        ),
        "detector_position" => Dict{String, Any}(
            "barrel" => status.barrel,
            "fiber" => status.fiber,
            "position" => status.position,
            "cylind_coords" => Dict{String, Any}(
                "radius" => Dict("value" => coords.radius_mm, "unit" => "mm"),
                "angle" => Dict("value" => coords.angle_deg, "unit" => "deg"),
                "z_center" => Dict("value" => coords.z_mm, "unit" => "mm"),
            ),
            "cartes_coords" => Dict{String, Any}(
                "x" => Dict("value" => coords.x_mm, "unit" => "mm"),
                "y" => Dict("value" => coords.y_mm, "unit" => "mm"),
                "z" => Dict("value" => coords.z_mm, "unit" => "mm"),
            ),
        ),
        "additional" => Dict{String, Any}(),
    )
end

"""
    load_sipm_channelmap(pygeom_path::String)

Load SiPM channelmap from legend-pygeom dummy_geom/channelmap.json.
Returns Dict{String, NamedTuple} mapping SiPM name to (barrel, fiber, position, rawid).
"""
function load_sipm_channelmap(pygeom_path::String)
    _HAS_JSON3 || error("JSON3 is required for load_sipm_channelmap() but not available in the current environment")
    # Find channelmap.json
    possible_paths = [
        joinpath(pygeom_path, "src", "pygeoml200", "configs", "dummy_geom", "channelmap.json"),
        joinpath(pygeom_path, "src", "l200geom", "configs", "dummy_geom", "channelmap.json"),
    ]
    
    channelmap_path = nothing
    for path in possible_paths
        if isfile(path)
            channelmap_path = path
            break
        end
    end
    
    if channelmap_path === nothing
        @warn "SiPM channelmap not found" tried=possible_paths
        return nothing
    end
    
    @info "Loading SiPM channelmap from $channelmap_path"
    
    # Parse JSON
    raw = JSON3.read(read(channelmap_path, String))
    
    result = Dict{String, NamedTuple{(:barrel, :fiber, :position, :rawid), Tuple{String, String, String, Int}}}()
    
    for (name, entry) in pairs(raw)
        system = get(entry, :system, "")
        system == "spms" || continue
        
        loc = get(entry, :location, nothing)
        loc === nothing && continue
        
        barrel = get(loc, :barrel, nothing)
        fiber = get(loc, :fiber, nothing)
        position = get(loc, :position, nothing)
        
        (barrel === nothing || fiber === nothing || position === nothing) && continue
        
        daq = get(entry, :daq, nothing)
        rawid = daq === nothing ? 0 : get(daq, :rawid, 0)
        
        result[String(name)] = (
            barrel = String(barrel),
            fiber = String(fiber),
            position = String(position),
            rawid = Int(rawid),
        )
    end
    
    @info "Loaded $(length(result)) SiPM entries from channelmap"
    result
end


# ============================================================================
# Relative Geometry Computation
# ============================================================================

"""
    load_geometry_yaml_flat(path::String)

Load a generated HPGe or SiPM geometry YAML and return a flat Dict mapping
detector_name => (angle_deg, z_center_mm, ...).
"""
function load_geometry_yaml_flat(path::String)
    raw = YAML.load_file(path; dicttype=Dict{String, Any})
    return raw
end

"""
    extract_hpge_position(entry::Dict) → NamedTuple

Extract (angle_deg, z_center_mm) from a generated HPGe geometry entry.
"""
function extract_hpge_position(entry::Dict{String, Any})
    pos = entry["detector_position"]
    angle = Float64(pos["cylind_coords"]["angle"]["value"])
    z_center = Float64(pos["cylind_coords"]["z_center"]["value"])
    (angle_deg = angle, z_center_mm = z_center)
end

"""
    extract_sipm_position(entry::Dict) → NamedTuple

Extract (angle_deg, z_mm, barrel, position) from a generated SiPM geometry entry.
"""
function extract_sipm_position(entry::Dict{String, Any})
    pos = entry["detector_position"]
    angle = Float64(pos["cylind_coords"]["angle"]["value"])
    z = Float64(pos["cylind_coords"]["z_center"]["value"])
    barrel = String(pos["barrel"])
    position = String(pos["position"])
    (angle_deg = angle, z_mm = z, barrel = barrel, position = position)
end

"""
    angular_difference(a_deg::Float64, b_deg::Float64) → Float64

Compute the minimum angular difference in degrees, range [0, 180].
"""
function angular_difference(a_deg::Float64, b_deg::Float64)
    raw = mod(a_deg - b_deg, 360.0)
    min(raw, 360.0 - raw)
end

"""
    cos_proximity(angle_diff_deg::Float64) → Float64

Cosine-based proximity score:
  0°  → 1.0  (same direction)
  90° → 0.5
  180° → 0.0 (opposite side)
  270° → 0.5
  360° → 1.0

Formula: (1 + cos(θ)) / 2
"""
function cos_proximity(angle_diff_deg::Float64)
    (1.0 + cos(deg2rad(angle_diff_deg))) / 2.0
end

"""
    compute_relative_geometry(hpge_yaml_path::String, sipm_yaml_path::String)

Compute relative geometry parameters for all HPGe-SiPM pairs.

For each HPGe detector and each SiPM, computes:
- `angle_diff_deg`:  minimum angular difference [0, 180]
- `cos_proximity`:   (1 + cos(Δθ)) / 2 — proximity score [0, 1]
- `delta_z_mm`:      sipm_z - hpge_z_center (in mm)
- `scaled_delta_z`:  delta_z / max_z_extent (normalized)

Returns a Dict{String, Any} ready for YAML output, structured as:
```
HPGe_name:
  hpge_angle_deg: ...
  hpge_z_center_mm: ...
  sipms:
    SiPM_name:
      angle_diff_deg: ...
      cos_proximity: ...
      delta_z_mm: ...
      scaled_delta_z: ...
```
"""
function compute_relative_geometry(hpge_yaml_path::String, sipm_yaml_path::String)
    hpge_data = load_geometry_yaml_flat(hpge_yaml_path)
    sipm_data = load_geometry_yaml_flat(sipm_yaml_path)

    # Extract positions
    hpge_positions = Dict{String, NamedTuple}()
    for (name, entry) in hpge_data
        entry isa Dict || continue
        haskey(entry, "detector_position") || continue
        hpge_positions[name] = extract_hpge_position(entry)
    end

    sipm_positions = Dict{String, NamedTuple}()
    for (name, entry) in sipm_data
        entry isa Dict || continue
        haskey(entry, "detector_position") || continue
        sipm_positions[name] = extract_sipm_position(entry)
    end

    @info "Relative geometry: $(length(hpge_positions)) HPGe × $(length(sipm_positions)) SiPM = $(length(hpge_positions) * length(sipm_positions)) pairs"

    # First pass: compute all delta_z values to find the z extent for scaling
    all_delta_z = Float64[]
    for (_, hpge_pos) in hpge_positions
        for (_, sipm_pos) in sipm_positions
            dz = sipm_pos.z_mm - hpge_pos.z_center_mm
            push!(all_delta_z, dz)
        end
    end

    max_abs_dz = isempty(all_delta_z) ? 1.0 : maximum(abs, all_delta_z)
    if max_abs_dz ≈ 0.0
        max_abs_dz = 1.0
    end
    @info "  Z scaling: max |Δz| = $(round(max_abs_dz; digits=1)) mm"

    # Second pass: build output dict
    output = Dict{String, Any}()

    for (hpge_name, hpge_pos) in sort(collect(hpge_positions); by = p -> p.first)
        sipms_dict = Dict{String, Any}()

        for (sipm_name, sipm_pos) in sort(collect(sipm_positions); by = p -> p.first)
            adiff = angular_difference(sipm_pos.angle_deg, hpge_pos.angle_deg)
            cprox = cos_proximity(adiff)
            dz = sipm_pos.z_mm - hpge_pos.z_center_mm
            sdz = dz / max_abs_dz

            sipms_dict[sipm_name] = Dict{String, Any}(
                "angle_diff_deg"  => round(adiff; digits=2),
                "cos_proximity"   => round(cprox; digits=6),
                "delta_z_mm"      => round(dz; digits=2),
                "scaled_delta_z"  => round(sdz; digits=6),
            )
        end

        output[hpge_name] = Dict{String, Any}(
            "hpge_angle_deg"    => round(hpge_pos.angle_deg; digits=2),
            "hpge_z_center_mm"  => round(hpge_pos.z_center_mm; digits=2),
            "sipms"             => sipms_dict,
        )
    end

    return output, max_abs_dz
end

"""
    write_relative_geometry_yaml(output_path::String, group_name::String,
                                  relative_data::Dict, max_abs_dz::Float64,
                                  hpge_source::String, sipm_source::String)

Write relative geometry YAML file with metadata header.
"""
function write_relative_geometry_yaml(output_path::String, group_name::String,
                                       relative_data::Dict{String, Any},
                                       max_abs_dz::Float64,
                                       hpge_source::String, sipm_source::String)
    mkpath(dirname(output_path))

    # Count dimensions
    n_hpge = length(relative_data)
    n_sipm = n_hpge > 0 ? length(first(values(relative_data))["sipms"]) : 0

    open(output_path, "w") do io
        # Write metadata as comments (to keep YAML content clean for loading)
        println(io, "# Relative geometry for group: $group_name")
        println(io, "# Generated at: $(Dates.now())")
        println(io, "# HPGe source: $hpge_source")
        println(io, "# SiPM source: $sipm_source")
        println(io, "# HPGe detectors: $n_hpge")
        println(io, "# SiPM detectors: $n_sipm")
        println(io, "# Total pairs: $(n_hpge * n_sipm)")
        println(io, "# max_abs_delta_z_mm: $(round(max_abs_dz; digits=2))")
        println(io, "# Parameters per pair: angle_diff_deg, cos_proximity, delta_z_mm, scaled_delta_z")
        println(io, "#")
        println(io, "# cos_proximity = (1 + cos(angle_diff)) / 2")
        println(io, "#   0° → 1.0 (same direction), 90° → 0.5, 180° → 0.0 (opposite)")
        println(io, "# scaled_delta_z = delta_z_mm / max_abs_delta_z_mm")
        println(io, "")
        YAML.write(io, relative_data)
    end

    @info "Relative geometry YAML written" path=output_path n_hpge=n_hpge n_sipm=n_sipm
    output_path
end

# ============================================================================
# Orchestration helpers — used by processors/process_geometry.jl
# ============================================================================

"""
    collect_periods_runs(group_def) → Vector{Tuple{String,String}}

Convert group definition PropDict to sorted list of (period, run) tuples.
"""
function collect_periods_runs(group_def)
    periods_runs = Tuple{String, String}[]
    for period in keys(group_def)
        for run in group_def[period]
            push!(periods_runs, (string(period), string(run)))
        end
    end
    sort!(periods_runs)
end
export collect_periods_runs

"""
    collect_detector_statuses(l200, periods_runs)
        → (hpge_statuses, sipm_statuses, filekeys)

Walk all (period, run) pairs and accumulate per-detector status
(processable/usability, geometry consistency) for both HPGe and SiPM systems.
"""
function collect_detector_statuses(l200::LegendData, periods_runs::Vector{Tuple{String,String}})
    hpge_statuses = Dict{String, DetectorStatus}()
    sipm_statuses = Dict{String, SiPMStatus}()
    filekeys_collected = FileKey[]

    for (period, run) in periods_runs
        # Prefer :phy start_filekey — it resolves to the canonical phy-run start
        # timestamp, under which the detector metadata (processable / usability)
        # is valid for this run. Fall back to :cal only if :phy is unavailable.
        sel = (DataPeriod(period), DataRun(run))
        filekey, category = try
            (start_filekey(l200, (sel..., :phy)), :phy)
        catch
            try
                @warn "No :phy filekey — falling back to :cal timestamp" period=period run=run
                (start_filekey(l200, (sel..., :cal)), :cal)
            catch
                @warn "Skipping run — no :phy or :cal filekey" period=period run=run
                continue
            end
        end
        @info "Resolving detector status" period=period run=run category=category filekey=string(filekey) timestamp=DateTime(filekey)
        push!(filekeys_collected, filekey)

        chinfo_geds = try channelinfo(l200, filekey; system=:geds)
        catch e; @warn "geds channelinfo failed" period=period run=run error=e; nothing end
        if chinfo_geds !== nothing
            for i in 1:length(chinfo_geds)
                name = string(chinfo_geds.detector[i])
                get!(hpge_statuses, name, DetectorStatus(name))
                add_run_status!(hpge_statuses[name], period, run,
                    chinfo_geds.processable[i], chinfo_geds.usability[i],
                    chinfo_geds.detstring[i], chinfo_geds.position[i],
                    chinfo_geds.rawid[i])
            end
        end

        chinfo_spms = try channelinfo(l200, filekey; system=:spms)
        catch e; @warn "spms channelinfo failed" period=period run=run error=e; nothing end
        if chinfo_spms !== nothing
            for i in 1:length(chinfo_spms)
                name = string(chinfo_spms.detector[i])
                fiber_str = String(string(chinfo_spms.fiber[i]))
                barrel_str = String(fiber_str[1:2])
                position_str = chinfo_spms.position[i] == 1 ? "top" : "bottom"
                get!(sipm_statuses, name, SiPMStatus(name))
                add_sipm_run_status!(sipm_statuses[name], period, run,
                    chinfo_spms.processable[i], chinfo_spms.usability[i],
                    barrel_str, fiber_str, position_str, chinfo_spms.rawid[i])
            end
        end
    end
    (hpge_statuses, sipm_statuses, filekeys_collected)
end
export collect_detector_statuses

"""
    build_hpge_geometry(hpge_statuses, hpges, hpge_strings, diodes_dir) → Dict{String,Any}

Build the HPGe geometry output dict: loads diode geometry, groups detectors by
string, computes z positions, fills entries via `build_detector_entry`, and
applies the `scaled_angle` / `scaled_z_center` post-pass.
"""
function build_hpge_geometry(hpge_statuses::Dict{String, DetectorStatus},
                             hpges, hpge_strings, diodes_dir::String)
    processable = filter(p -> is_ever_processable(p.second), hpge_statuses)
    @info "HPGe detectors with processable=true: $(length(processable))"

    # Group by string
    string_dets = Dict{Int, Vector{StringDetectorInfo}}()
    for (name, status) in processable
        status.string_id === nothing && continue
        dg = load_diode_geometry(diodes_dir, name)
        dg === nothing && (@warn "Missing diode geometry" detector=name; continue)
        rod = get_detector_rodlength(hpges, name)
        rod = rod === nothing ? 103.5 : rod
        push!(get!(string_dets, status.string_id, StringDetectorInfo[]),
              StringDetectorInfo(name, status.position_in_string, dg.height_in_mm, rod))
    end

    z_positions = Dict{String, NamedTuple}()
    for (_, dets) in string_dets
        merge!(z_positions, compute_string_z_positions(dets))
    end

    output = Dict{String, Any}()
    for (name, status) in sort(collect(processable); by=first)
        dg = load_diode_geometry(diodes_dir, name)
        dg === nothing && continue
        sr, sa = get_string_geometry(hpge_strings, status.string_id)
        if sr === nothing || sa === nothing
            @warn "No radius/angle for string $(status.string_id) in extra_meta — " *
                  "falling back to (220 mm, 0°); detectors of this string will " *
                  "overlap others at that fallback position" detector=name
        end
        sr = sr === nothing ? 220.0 : sr
        sa = sa === nothing ? 0.0 : sa
        rod = get_detector_rodlength(hpges, name); rod = rod === nothing ? 103.5 : rod
        z = get(z_positions, name, (z_top=0.0, z_center=0.0, z_bottom=0.0))
        output[name] = build_detector_entry(name, status, dg, sr, sa, rod, z)
    end

    # Post-pass: scaled_angle + scaled_z_center
    all_z = Float64[e["detector_position"]["cylind_coords"]["z_center"]["value"] for e in values(output)]
    min_z, max_z = isempty(all_z) ? (0.0, 0.0) : extrema(all_z)
    range_z = max_z - min_z
    for e in values(output)
        pos = e["detector_position"]
        a = pos["cylind_coords"]["angle"]["value"]
        z = pos["cylind_coords"]["z_center"]["value"]
        pos["scaled_angle"]    = round(a / 360.0; digits=6)
        pos["scaled_z_center"] = range_z ≈ 0.0 ? 0.5 : round((z - min_z) / range_z; digits=6)
    end
    output
end
export build_hpge_geometry

"""
    build_sipm_geometry(sipm_statuses) → Dict{String,Any}

Build the SiPM geometry output dict from collected statuses.
"""
function build_sipm_geometry(sipm_statuses::Dict{String, SiPMStatus})
    processable = filter(p -> is_sipm_ever_processable(p.second), sipm_statuses)
    @info "SiPM detectors with processable=true: $(length(processable))"
    output = Dict{String, Any}()
    for (name, status) in sort(collect(processable); by=first)
        output[name] = build_sipm_entry(name, status)
    end
    output
end
export build_sipm_geometry

# ============================================================================
# Detector-status Markdown report (colour-coded table per (period, run))
# ============================================================================

# Colour map: emoji + HTML span background
# proc+on → green, proc+ac → yellow, proc+off → orange, !proc → red, missing → dark
const _STATUS_CELL = Dict(
    :on      => ("🟢", "#b6f5a0", "on"),
    :ac      => ("🟡", "#fff3a0", "ac"),
    :off     => ("🟠", "#ffcf8a", "off"),
    :notproc => ("🔴", "#ffa0a0", "–"),
    :missing => ("⚫", "#333333", "–"),
)

function _status_cell(run_str::String, status)
    if run_str in status.processable_false
        e, bg, lbl = _STATUS_CELL[:notproc]
        return "<span style=\"background-color:$bg\">$e $lbl</span>"
    elseif run_str in status.processable_true
        key = run_str in status.usable_on ? :on :
              run_str in status.usable_ac ? :ac :
              run_str in status.usable_off ? :off : :notproc
        e, bg, lbl = _STATUS_CELL[key]
        return "<span style=\"background-color:$bg\">$e $lbl</span>"
    else
        e, bg, lbl = _STATUS_CELL[:missing]
        return "<span style=\"background-color:$bg;color:#fff\">$e $lbl</span>"
    end
end

# SiPM variant (no :ac state)
function _status_cell_sipm(run_str::String, status)
    if run_str in status.processable_false
        e, bg, lbl = _STATUS_CELL[:notproc]
        return "<span style=\"background-color:$bg\">$e $lbl</span>"
    elseif run_str in status.processable_true
        key = run_str in status.usable_on ? :on :
              run_str in status.usable_off ? :off : :notproc
        e, bg, lbl = _STATUS_CELL[key]
        return "<span style=\"background-color:$bg\">$e $lbl</span>"
    else
        e, bg, lbl = _STATUS_CELL[:missing]
        return "<span style=\"background-color:$bg;color:#fff\">$e $lbl</span>"
    end
end

function _write_status_table(io::IO, title::String,
                             detectors::Vector{String},
                             statuses::Dict{String, <:Any},
                             periods_runs::Vector{Tuple{String,String}},
                             cell_fn)
    println(io, "## $title")
    println(io, "")
    isempty(detectors) && (println(io, "_No processable detectors._\n"); return)

    header = "| Period-Run | " * join(detectors, " | ") * " |"
    sep    = "|" * repeat("---|", length(detectors) + 1)
    println(io, header)
    println(io, sep)
    for (p, r) in periods_runs
        run_str = "$p-$r"
        row = String["`$run_str`"]
        for det in detectors
            status = get(statuses, det, nothing)
            push!(row, status === nothing ?
                "<span style=\"background-color:#333333;color:#fff\">⚫ –</span>" :
                cell_fn(run_str, status))
        end
        println(io, "| ", join(row, " | "), " |")
    end
    println(io, "")
end

"""
    generate_detector_status_report(hpge_statuses, sipm_statuses, periods_runs,
                                    report_path, group_name)

Write a colour-coded Markdown report listing all detectors that are processable
in at least one run, with one row per (period, run) showing
processable + usability state.
"""
function generate_detector_status_report(
    hpge_statuses::Dict{String, DetectorStatus},
    sipm_statuses::Dict{String, SiPMStatus},
    periods_runs::Vector{Tuple{String,String}},
    report_path::String,
    group_name::String,
)
    mkpath(dirname(report_path))
    hpge_dets = sort(collect(keys(filter(p -> is_ever_processable(p.second),     hpge_statuses))))
    sipm_dets = sort(collect(keys(filter(p -> is_sipm_ever_processable(p.second), sipm_statuses))))

    open(report_path, "w") do io
        println(io, "# Detector Status Report")
        println(io, "")
        println(io, "**Group:** `$group_name`  ")
        println(io, "**Generated:** $(Dates.now())  ")
        println(io, "**Period-Runs:** $(length(periods_runs))  ")
        println(io, "**HPGe ever processable:** $(length(hpge_dets))  ")
        println(io, "**SiPM ever processable:** $(length(sipm_dets))  ")
        println(io, "")
        println(io, "## Legend")
        println(io, "")
        println(io, "| Symbol | Meaning | Background |")
        println(io, "|---|---|---|")
        println(io, "| <span style=\"background-color:#b6f5a0\">🟢 on</span>  | processable + usability `on`  | green |")
        println(io, "| <span style=\"background-color:#fff3a0\">🟡 ac</span>  | processable + usability `ac` (HPGe only) | yellow |")
        println(io, "| <span style=\"background-color:#ffcf8a\">🟠 off</span> | processable + usability `off` | orange |")
        println(io, "| <span style=\"background-color:#ffa0a0\">🔴 –</span>   | not processable in this run   | red |")
        println(io, "| <span style=\"background-color:#333333;color:#fff\">⚫ –</span> | detector not listed for this run | dark |")
        println(io, "")
        _write_status_table(io, "HPGe detectors", hpge_dets, hpge_statuses, periods_runs, _status_cell)
        _write_status_table(io, "SiPM detectors", sipm_dets, sipm_statuses, periods_runs, _status_cell_sipm)
    end
    @info "Detector-status report saved" path=report_path
    report_path
end
export generate_detector_status_report

