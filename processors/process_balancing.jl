# This file is a part of ML-based-LAr-veto, licensed under the MIT License (MIT).
# Balancing Processor — read jlext → balance → write jlbal + jlbalml
#
# jlbal:   all events, detector IDs assigned to forcedtrigger (inference)
# jlbalml: PE-range filtered, down/upsampled to match counts (training)

function process_balancing(processing_config::PropDict, l200::LegendData, group_name::String)

    @info "Process Balancing for ML group: $group_name"

    bal_cfg = load_metadata_config(processing_config, :balancing)
    isnothing(bal_cfg) && (@error "No balancing config found"; return false)

    output_base = processing_config.paths.output.tier
    exclusion_rules = parse_exclusion_rules(get(bal_cfg, "ml_training_exclusion", []))

    # ── Parse detector exclusions ────────────────────────────────────────
    excl_ged_names  = String[String(x) for x in get(bal_cfg, "excluded_ged_detectors", String[])]
    excl_sipm_names = String[String(x) for x in get(bal_cfg, "excluded_sipm_detectors", String[])]
    excl_ged_ids    = resolve_excluded_ids(excl_ged_names)
    excl_sipm_ids   = resolve_excluded_ids(excl_sipm_names)
    !isempty(excl_ged_names)  && @info "Excluded HPGe:  $(join(excl_ged_names, ", "))"
    !isempty(excl_sipm_names) && @info "Excluded SiPM:  $(join(excl_sipm_names, ", "))"

    # ── Read windowed PE from jlext ──────────────────────────────────────
    datasets = Dict{String, PreparedDataset}()
    exclusion_stats = Dict{String, NamedTuple{(:n_events_dropped, :n_sipms_dropped),
                                              Tuple{Int, Int}}}()
    for name in ["sub500keV", "forcedtrigger"]
        path = get_tier_path(output_base, "jlext", group_name, name)
        isfile(path) || (@error "jlext not found — run extraction first" dataset=name; return false)
        prep = _read_wpe_from_lh5(path); prep.name = name
        n_sipms_dropped  = drop_sipm_columns!(prep, excl_sipm_ids)
        n_events_dropped = apply_ged_exclusion!(prep, excl_ged_ids)
        exclusion_stats[name] = (n_events_dropped=n_events_dropped, n_sipms_dropped=n_sipms_dropped)
        (n_events_dropped > 0 || n_sipms_dropped > 0) &&
            @info "  Exclusions applied" dataset=name events_dropped=n_events_dropped sipms_dropped=n_sipms_dropped
        datasets[name] = prep
        @info "  $name: $(prep.n_events) events"
    end

    sub = datasets["sub500keV"]; ft = datasets["forcedtrigger"]

    # ── Apply exclusion filters (training only) ──────────────────────────
    sub_valid = apply_exclusion(sub, exclusion_rules)
    ft_valid  = apply_exclusion(ft, exclusion_rules)
    @info "  Training filter: sub500keV $(length(sub_valid))/$(sub.n_events), FT $(length(ft_valid))/$(ft.n_events)"

    # ── Helper: write one tier ───────────────────────────────────────────
    function _write_tier(tbl, tier, ds_name, sipm_ids)
        path = get_tier_path(output_base, tier, group_name, ds_name)
        lh5open(path, "w") do ds
            ds[Symbol(tier)] = _fix_vov(tbl)
            ds["sipm_detector_ids"] = sipm_ids
        end
        @info "  Written $tier/$ds_name: $(length(tbl)) events"
    end

    # ══════════════════════════════════════════════════════════════════════
    # jlbalml — training: balanced counts, PE filtered
    # ══════════════════════════════════════════════════════════════════════
    n_sub_ml = length(sub_valid); n_ft_ml = length(ft_valid)
    n_target = n_sub_ml  # match to sub500keV count

    if n_ft_ml >= n_target
        ft_ml_idxs = ft_valid[randperm(n_ft_ml)[1:n_target]]  # downsample
    else
        extra = [ft_valid[rand(1:n_ft_ml)] for _ in 1:(n_target - n_ft_ml)]
        ft_ml_idxs = vcat(ft_valid, extra)  # upsample
    end

    sub_ml_det = sub.ged_detector_id[sub_valid]
    ft_ml_det  = sample_detector_ids(sub, length(ft_ml_idxs))
    @info "  jlbalml: sub500keV=$n_target, FT=$(length(ft_ml_idxs)) (was $n_ft_ml)"

    _write_tier(build_bal_table(sub, sub_valid, sub_ml_det), "jlbalml", "sub500keV", sub.sipm_detector_ids)
    _write_tier(build_bal_table(ft, ft_ml_idxs, ft_ml_det), "jlbalml", "forcedtrigger", ft.sipm_detector_ids)

    # ══════════════════════════════════════════════════════════════════════
    # jlbal — inference: all events
    # ══════════════════════════════════════════════════════════════════════
    sub_all = collect(1:sub.n_events)
    ft_all  = collect(1:ft.n_events)
    ft_inf_det = sample_detector_ids(sub, ft.n_events)

    _write_tier(build_bal_table(sub, sub_all, sub.ged_detector_id), "jlbal", "sub500keV", sub.sipm_detector_ids)
    _write_tier(build_bal_table(ft, ft_all, ft_inf_det), "jlbal", "forcedtrigger", ft.sipm_detector_ids)

    # ── Pass-through (e.g. physics) → jlbal only ────────────────────────
    for ds_name in String.(get(bal_cfg, "pass_through", String[]))
        path = get_tier_path(output_base, "jlext", group_name, ds_name)
        isfile(path) || (@warn "jlext not found for $ds_name — skipping"; continue)
        prep = _read_wpe_from_lh5(path); prep.name = ds_name
        n_sipms_dropped  = drop_sipm_columns!(prep, excl_sipm_ids)
        n_events_dropped = apply_ged_exclusion!(prep, excl_ged_ids)
        exclusion_stats[ds_name] = (n_events_dropped=n_events_dropped, n_sipms_dropped=n_sipms_dropped)
        (n_events_dropped > 0 || n_sipms_dropped > 0) &&
            @info "  Exclusions applied" dataset=ds_name events_dropped=n_events_dropped sipms_dropped=n_sipms_dropped
        datasets[ds_name] = prep
        tbl = build_bal_table(prep, collect(1:prep.n_events), prep.ged_detector_id)
        _write_tier(tbl, "jlbal", ds_name, prep.sipm_detector_ids)
    end

    # ── Report ───────────────────────────────────────────────────────────
    report_dir = get_report_dir(processing_config, group_name)
    generate_balancing_report(datasets, report_dir, group_name, exclusion_rules,
                              n_sub_ml, length(ft_ml_idxs), n_ft_ml;
                              excluded_ged_names=excl_ged_names,
                              excluded_sipm_names=excl_sipm_names,
                              exclusion_stats=exclusion_stats)

    @info "Balancing complete"
    return true
end
