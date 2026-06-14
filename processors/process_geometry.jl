# This file is a part of ML-based-LAr-veto, licensed under the MIT License (MIT).
# Geometry Processor — thin orchestrator (Juleana pattern)
#
# Steps:
#   1. Read paths + group definition.
#   2. Walk periods/runs and collect per-detector status via channelinfo.
#   3. Load HPGe extra_meta (rodlength, string positions).
#   4. Build HPGe + SiPM geometry dicts.
#   5. Write YAMLs (HPGe, SiPM, relative).
#   6. Emit a colour-coded detector-status Markdown report.

function process_geometry(processing_config::PropDict, l200::LegendData, group_name::String)
    @info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    @info "Process Geometry for ML group: $group_name"

    # ── Paths + data sources ─────────────────────────────────────────────
    paths = processing_config.paths
    hpge_out_path     = joinpath(paths.output.geometry, "HPGe", "$(group_name)_hpge_geometry.yaml")
    sipm_out_path     = joinpath(paths.output.geometry, "SiPM", "$(group_name)_sipm_geometry.yaml")
    relative_out_path = joinpath(paths.output.geometry, "relative", "$(group_name)_relative_geometry.yaml")

    diodes_dir  = joinpath(data_path(l200, "metadata"), "hardware", "detectors", "germanium", "diodes")
    pygeom_path = get(paths, :legend_pygeom, nothing)

    # ── Group definition → periods_runs ──────────────────────────────────
    groupings = readlprops(paths.groupings)
    haskey(groupings, Symbol(group_name)) || error("Group '$group_name' not found in $(paths.groupings)")
    group_def    = groupings[Symbol(group_name)]
    periods_runs = collect_periods_runs(group_def)
    @info "Group contains $(length(periods_runs)) period-run combinations"

    # ── Collect detector status (HPGe + SiPM) across all runs ────────────
    hpge_statuses, sipm_statuses, filekeys = collect_detector_statuses(l200, periods_runs)
    @info "Found detectors" hpge=length(hpge_statuses) sipm=length(sipm_statuses)

    # ── Load HPGe extra_meta (string radius/angle, rodlength) ────────────
    hpges, hpge_strings = (nothing, nothing)
    if pygeom_path !== nothing && isdir(pygeom_path) && !isempty(filekeys)
        ts = DateTime(first(filekeys))
        @info "Using timestamp for extra_meta validity" timestamp=ts
        hpges, hpge_strings = load_hpge_extra_meta(pygeom_path, ts)
        hpges === nothing && @warn "Could not load extra_meta — HPGe positions will use defaults"
    end

    # ── Build + write HPGe / SiPM geometries ─────────────────────────────
    hpge_output = build_hpge_geometry(hpge_statuses, hpges, hpge_strings, diodes_dir)
    sipm_output = build_sipm_geometry(sipm_statuses)
    write_geometry_yaml(hpge_out_path, group_name, hpge_output)
    write_geometry_yaml(sipm_out_path, group_name, sipm_output)

    # ── Relative geometry (HPGe ↔ SiPM) ──────────────────────────────────
    if isfile(hpge_out_path) && isfile(sipm_out_path)
        relative_data, max_abs_dz = compute_relative_geometry(hpge_out_path, sipm_out_path)
        write_relative_geometry_yaml(relative_out_path, group_name, relative_data,
                                     max_abs_dz, hpge_out_path, sipm_out_path)
    else
        @error "Cannot generate relative geometry — HPGe or SiPM geometry files missing"
    end

    # ── Detector status report ───────────────────────────────────────────
    status_path = joinpath(get_report_dir(processing_config, group_name), "detectorstatus.md")
    generate_detector_status_report(hpge_statuses, sipm_statuses, periods_runs, status_path, group_name)

    @info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    @info "Geometry processing complete"
    (hpge=hpge_out_path, sipm=sipm_out_path, relative=relative_out_path, status=status_path)
end
