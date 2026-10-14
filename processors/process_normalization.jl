# Processor: Normalization — read jlbal/jlbalml → scale + geometry → write jlnorm/jlnormml
#
# Input:  generated/tier/jlbal(ml)/<group>/{sub500keV,forcedtrigger}.lh5
# Output: generated/tier/jlnorm(ml)/<group>/{train,val,test|all}.lh5

import Random
using Random: shuffle!
using Statistics: median

function process_normalization(processing_config::PropDict, l200::LegendData, group_name::String)

    @info "Process Normalization for ML group: $group_name"

    norm_cfg = load_metadata_config(processing_config, :normalization)
    isnothing(norm_cfg) && (@error "No normalization config found"; return false)

    output_base   = String(processing_config.paths.output.tier)
    geometry_base = String(processing_config.paths.output.geometry)
    plot_dir      = get_plot_dir(processing_config, group_name, :normalization)
    report_dir    = get_report_dir(processing_config, group_name)

    scaling_cfg  = get(norm_cfg, "scaling", Dict())
    geom_cfg     = get(norm_cfg, "geometry_features", Dict())
    split_cfg    = get(norm_cfg, "split", Dict("train" => 0.7, "test" => 0.2, "val" => 0.1))

    # ── Load geometry YAMLs ──────────────────────────────────────────────
    rel_geom  = load_relative_geometry(group_name, geometry_base)
    hpge_geom = load_hpge_geometry(group_name, geometry_base)
    sipm_geom = load_sipm_geometry(group_name, geometry_base)
    @info "  Geometry loaded: $(length(hpge_geom)) HPGe, $(length(sipm_geom)) SiPM, $(length(rel_geom)) relative"

    rel_features  = String.(get(geom_cfg, "relative", String[]))
    hpge_features = String.(get(geom_cfg, "hpge", String[]))
    sipm_features = String.(get(geom_cfg, "sipm", String[]))

    # ── Process training + inference tiers ────────────────────────────────
    report_data = Dict{String, Any}()
    sipm_ids = UInt32[]  # populated on first tier read
    trig_cfg = get(norm_cfg, "trigger_scaling", nothing)
    trig_pe_params = Dict{String,Any}()

    for tier in ["jlbalml", "jlbal"]
        @info "━━ Normalization: $tier"
        tier_sym = Symbol(tier)
        out_tier = tier == "jlbalml" ? "jlnormml" : "jlnorm"

        # ── Read datasets ────────────────────────────────────────────────
        datasets = Dict{String, Any}()
        for ds_name in ["sub500keV", "forcedtrigger"]
            path = get_tier_path(output_base, tier, group_name, ds_name)
            isfile(path) || (@error "Missing $tier/$ds_name"; return false)
            tbl  = lh5open(path, "r") do f; f[string(tier_sym)][:]; end
            sids = lh5open(path, "r") do f; f["sipm_detector_ids"][:]; end
            datasets[ds_name] = (table=tbl, sipm_ids=Vector{UInt32}(sids))
            @info "  $ds_name: $(length(tbl)) events"
        end

        sipm_ids = datasets["sub500keV"].sipm_ids

        # ── Geometry augmentation ────────────────────────────────────────
        geom_matrices = Dict{String, Dict{String, Any}}()
        for ds_name in ["sub500keV", "forcedtrigger"]
            tbl = datasets[ds_name].table
            assigned = tbl.assigned_ged_detector
            fm = Dict{String, Any}()
            for feat in rel_features
                fm[feat] = build_feature_matrix(assigned, sipm_ids, rel_geom, feat)
            end
            for feat in hpge_features
                fm[feat] = build_hpge_feature_vector(assigned, hpge_geom, feat)
            end
            geom_matrices[ds_name] = fm
        end
        @info "  Geometry features: relative=$(rel_features), hpge=$(hpge_features)"

        # ── Activity plots (jlbalml, unscaled PE) ───────────────────────
        if tier == "jlbalml"
            sub_mat_full = pe_sums_to_matrix(datasets["sub500keV"].table, :sipm_pe_sums)
            ft_mat_full  = pe_sums_to_matrix(datasets["forcedtrigger"].table, :sipm_pe_sums)
            preliminary = get(get(processing_config, :plots, PropDict()), :preliminary, true)
            save_activity_plots(sub_mat_full, ft_mat_full, sipm_ids, hpge_geom, sipm_geom,
                                datasets["sub500keV"].table.assigned_ged_detector,
                                datasets["forcedtrigger"].table.assigned_ged_detector,
                                plot_dir, group_name; preliminary)
        end

        # ── Scaling pipeline per column ──────────────────────────────────
        scaled = Dict{String, Dict{String, Matrix{Float64}}}()
        before_matrices = Dict{String, Dict{String, Matrix{Float64}}}()
        scaling_params = Dict{String, Dict{String, Any}}()

        for (col, steps) in scaling_cfg
            col = String(col)
            col_sym = Symbol(col)
            sub_M = pe_sums_to_matrix(datasets["sub500keV"].table, col_sym)
            ft_M  = pe_sums_to_matrix(datasets["forcedtrigger"].table, col_sym)

            # Store unscaled copies for plots
            for (dn, M) in [("sub500keV", sub_M), ("forcedtrigger", ft_M)]
                get!(before_matrices, dn, Dict{String, Matrix{Float64}}())[col] = copy(M)
            end

            # Compute global params across both datasets, then apply
            gparams = compute_global_params([sub_M, ft_M], steps)
            sub_M_scaled = copy(sub_M); ft_M_scaled = copy(ft_M)
            apply_scaling_pipeline!(sub_M_scaled, steps, gparams)
            apply_scaling_pipeline!(ft_M_scaled, steps, gparams)

            for (dn, M) in [("sub500keV", sub_M_scaled), ("forcedtrigger", ft_M_scaled)]
                get!(scaled, dn, Dict{String, Matrix{Float64}}())[col] = M
            end
            scaling_params[col] = gparams
            @info "  Scaled $col: global_min=$(get(gparams, "global_min", "N/A")), global_max=$(get(gparams, "global_max", "N/A"))"
        end

        # ── PE distribution plots (jlbalml, before/after) ───────────────
        if tier == "jlbalml"
            save_pe_distribution_plots(before_matrices, scaled, sipm_ids, plot_dir, group_name; preliminary)
        end

        # ── Trigger-level feature scaling ────────────────────────────────
        trig_data = Dict{String, Dict{Symbol, Any}}()

        if trig_cfg !== nothing
            pe_steps   = get(trig_cfg, "pe", Vector())
            time_range = get(trig_cfg, "time", Dict())
            time_range = get(time_range, "range", [-10, 10])

            # Read trigger VoV columns
            sub_tbl = datasets["sub500keV"].table
            ft_tbl  = datasets["forcedtrigger"].table

            sub_pe_vov  = sub_tbl.trigger_pe_vals
            ft_pe_vov   = ft_tbl.trigger_pe_vals
            sub_t_vov   = sub_tbl.trigger_times_us
            ft_t_vov    = ft_tbl.trigger_times_us
            sub_did_vov = sub_tbl.trigger_det_ids
            ft_did_vov  = ft_tbl.trigger_det_ids

            # Compute global trigger PE params across both datasets
            trig_pe_params = compute_trigger_pe_global_params([sub_pe_vov, ft_pe_vov], pe_steps)
            @info "  Trigger PE global: min=$(trig_pe_params["global_min"]), max=$(trig_pe_params["global_max"])"

            for (ds_name, pe_vov, t_vov, did_vov) in [
                    ("sub500keV",      sub_pe_vov,  sub_t_vov,  sub_did_vov),
                    ("forcedtrigger",  ft_pe_vov,   ft_t_vov,   ft_did_vov)]
                td = Dict{Symbol, Any}()
                td[:trig_pe_scaled]   = scale_trigger_pe(pe_vov, pe_steps, trig_pe_params)
                td[:trig_time_scaled] = scale_trigger_times(t_vov, time_range)
                td[:trig_det_ids]     = VectorOfVectors(did_vov)
                td[:trig_count]       = Int32[length(pe_vov[i]) for i in eachindex(pe_vov)]
                trig_data[ds_name] = td
                @info "  Trigger features ($ds_name): $(length(pe_vov)) events, median triggers=$(length(pe_vov) > 0 ? Int(median(td[:trig_count])) : 0)"
            end
            scaling_params["trigger_pe"] = trig_pe_params

            # Trigger distribution plots (jlbalml only)
            if tier == "jlbalml"
                save_trigger_distribution_plots(datasets, trig_data, plot_dir, group_name; preliminary)
            end
        end

        # ── Build output columns per dataset ─────────────────────────────
        ds_columns = Dict{String, Dict{Symbol, Any}}()
        for ds_name in ["sub500keV", "forcedtrigger"]
            tbl = datasets[ds_name].table
            cols = Dict{Symbol, Any}()
            # Scaled PE columns → VoV
            for (col, M) in get(scaled, ds_name, Dict())
                cols[Symbol(col, "_scaled")] = VectorOfVectors(collect(eachrow(Float32.(M))))
            end
            # Geometry features
            gm = geom_matrices[ds_name]
            for feat in rel_features
                cols[Symbol(feat)] = VectorOfVectors(collect(eachrow(Float32.(gm[feat]))))
            end
            for feat in hpge_features
                cols[Symbol(feat)] = Float32.(gm[feat])
            end
            # Pass-through scalar columns
            cols[:assigned_ged_detector] = UInt32.(tbl.assigned_ged_detector)
            cols[:event_sum_pe]       = Float32.(tbl.event_sum_pe)
            cols[:event_multiplicity] = Int32.(tbl.event_multiplicity)
            for extra in (:event_sum_pe_prompt, :event_multiplicity_prompt,
                          :event_sum_pe_delayed, :event_multiplicity_delayed, :ged_energy_keV)
                hasproperty(tbl, extra) && (cols[extra] = extra in (:event_multiplicity_prompt, :event_multiplicity_delayed) ?
                    Int32.(getproperty(tbl, extra)) : Float32.(getproperty(tbl, extra)))
            end
            # Trigger-level features (VoV + scalar count)
            if haskey(trig_data, ds_name)
                for (k, v) in trig_data[ds_name]
                    cols[k] = v
                end
            end
            ds_columns[ds_name] = cols
        end

        # ── Label + combine + shuffle ────────────────────────────────────
        n_sub = length(datasets["sub500keV"].table)
        n_ft  = length(datasets["forcedtrigger"].table)
        n_total = n_sub + n_ft

        all_keys = union(keys(ds_columns["sub500keV"]), keys(ds_columns["forcedtrigger"]))
        combined = Dict{Symbol, Any}()
        for k in all_keys
            sub_col = get(ds_columns["sub500keV"], k, nothing)
            ft_col  = get(ds_columns["forcedtrigger"], k, nothing)
            (sub_col === nothing || ft_col === nothing) && continue
            combined[k] = sub_col isa VectorOfVectors ?
                VectorOfVectors(vcat(collect(sub_col), collect(ft_col))) :
                vcat(sub_col, ft_col)
        end
        combined[:label] = vcat(ones(Int8, n_sub), zeros(Int8, n_ft))

        perm = collect(1:n_total); shuffle!(perm)
        for (k, v) in combined
            combined[k] = v isa VectorOfVectors ?
                VectorOfVectors([v[i] for i in perm]) : v[perm]
        end
        @info "  Combined: $n_total events (signal=$n_sub, bg=$n_ft), shuffled"

        # ── Write output ─────────────────────────────────────────────────
        if tier == "jlbalml"
            r_train = Float64(get(split_cfg, "train", 0.7))
            r_test  = Float64(get(split_cfg, "test",  0.2))
            n_train = round(Int, n_total * r_train)
            n_test  = round(Int, n_total * r_test)
            n_val   = n_total - n_train - n_test
            splits = [("train", 1:n_train), ("test", (n_train+1):(n_train+n_test)),
                      ("val", (n_train+n_test+1):n_total)]
            @info "  Split: train=$n_train, test=$n_test, val=$n_val"
            for (split_name, range) in splits
                _write_norm_file(output_base, out_tier, group_name, split_name,
                                 combined, range, sipm_ids)
            end
        else
            _write_norm_file(output_base, out_tier, group_name, "all",
                             combined, 1:n_total, sipm_ids)
        end

        report_data[tier] = (scaling_params=scaling_params, n_sub=n_sub, n_ft=n_ft,
                             n_total=n_total, before=before_matrices, after=scaled,
                             geom_features=vcat(rel_features, hpge_features),
                             trig_data=trig_data)
    end

    # ── Pass-through datasets (physics etc.) → jlnorm ────────────────────
    known = Set(["sub500keV", "forcedtrigger"])
    jlbal_dir = joinpath(output_base, "jlbal", group_name)
    if isdir(jlbal_dir)
        for f in readdir(jlbal_dir)
            m = match(r"l200-\w+-(\w+)-tier_jlbal\.lh5", f)
            m === nothing && continue
            ds_name = String(m.captures[1])
            ds_name in known && continue
            _normalize_passthrough(ds_name, group_name, output_base,
                                   scaling_cfg, rel_geom, hpge_geom, sipm_geom,
                                   rel_features, hpge_features, sipm_ids,
                                   trig_cfg, trig_pe_params)
        end
    end

    # ── Report ───────────────────────────────────────────────────────────
    generate_normalization_report(report_data, report_dir, group_name; output_base)

    @info "Normalization complete"
    return true
end

# ============================================================================
# Helpers
# ============================================================================

"""Write a single norm output file (train/val/test/all) from combined columns."""
function _write_norm_file(output_base, tier, group_name, split_name,
                          combined::Dict{Symbol,Any}, range, sipm_ids)
    path = get_tier_path(output_base, tier, group_name, split_name)
    tbl_cols = Dict{Symbol, Any}()
    for (k, v) in combined
        tbl_cols[k] = v isa VectorOfVectors ?
            VectorOfVectors([v[i] for i in range]) : v[range]
    end
    ks = Tuple(sort(collect(keys(tbl_cols))))
    tbl = Table(NamedTuple{ks}(Tuple(tbl_cols[k] for k in ks)))
    lh5open(path, "w") do ds
        ds[tier] = _fix_vov(tbl)
        ds["sipm_detector_ids"] = sipm_ids
    end
    n_sig = sum(tbl_cols[:label] .== 1)
    @info "  Written $tier/$split_name: $(length(range)) events (signal=$n_sig, bg=$(length(range)-n_sig))"
end

"""Normalize a pass-through dataset (e.g. physics) → jlnorm, no label, independent scaling."""
function _normalize_passthrough(ds_name, group_name, output_base,
                                scaling_cfg, rel_geom, hpge_geom, sipm_geom,
                                rel_features, hpge_features, sipm_ids,
                                trig_cfg, trig_pe_params)
    @info "  Pass-through: $ds_name"
    path = get_tier_path(output_base, "jlbal", group_name, ds_name)
    isfile(path) || return
    tbl  = lh5open(path, "r") do f; f["jlbal"][:]; end
    sids = lh5open(path, "r") do f; f["sipm_detector_ids"][:]; end

    cols = Dict{Symbol, Any}()
    assigned = tbl.assigned_ged_detector

    for (col, steps) in scaling_cfg
        col = String(col)
        M = pe_sums_to_matrix(tbl, Symbol(col))
        M_scaled = copy(M)
        apply_scaling_pipeline!(M_scaled, steps)
        cols[Symbol(col, "_scaled")] = VectorOfVectors(collect(eachrow(Float32.(M_scaled))))
    end

    for feat in rel_features
        cols[Symbol(feat)] = VectorOfVectors(collect(eachrow(Float32.(
            build_feature_matrix(assigned, Vector{UInt32}(sids), rel_geom, feat)))))
    end
    for feat in hpge_features
        cols[Symbol(feat)] = Float32.(build_hpge_feature_vector(assigned, hpge_geom, feat))
    end

    cols[:assigned_ged_detector] = UInt32.(assigned)
    cols[:event_sum_pe]       = Float32.(tbl.event_sum_pe)
    cols[:event_multiplicity] = Int32.(tbl.event_multiplicity)
    for extra in (:event_sum_pe_prompt, :event_multiplicity_prompt,
                  :event_sum_pe_delayed, :event_multiplicity_delayed, :ged_energy_keV)
        hasproperty(tbl, extra) && (cols[extra] = extra in (:event_multiplicity_prompt, :event_multiplicity_delayed) ?
            Int32.(getproperty(tbl, extra)) : Float32.(getproperty(tbl, extra)))
    end

    # Trigger-level features (if configured)
    if trig_cfg !== nothing && hasproperty(tbl, :trigger_pe_vals)
        pe_steps   = get(trig_cfg, "pe", Vector())
        time_range = get(get(trig_cfg, "time", Dict()), "range", [-10, 10])
        cols[:trig_pe_scaled]   = scale_trigger_pe(tbl.trigger_pe_vals, pe_steps, trig_pe_params)
        cols[:trig_time_scaled] = scale_trigger_times(tbl.trigger_times_us, time_range)
        cols[:trig_det_ids]     = VectorOfVectors(tbl.trigger_det_ids)
        cols[:trig_count]       = Int32[length(tbl.trigger_pe_vals[i]) for i in eachindex(tbl.trigger_pe_vals)]
    end

    ks = Tuple(sort(collect(keys(cols))))
    out_tbl = Table(NamedTuple{ks}(Tuple(cols[k] for k in ks)))
    out_path = get_tier_path(output_base, "jlnorm", group_name, ds_name)
    lh5open(out_path, "w") do ds
        ds["jlnorm"] = _fix_vov(out_tbl)
        ds["sipm_detector_ids"] = Vector{UInt32}(sids)
    end
    @info "  Written jlnorm/$ds_name: $(length(tbl)) events (no label)"
end
