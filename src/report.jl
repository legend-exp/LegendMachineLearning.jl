# This file is a part of ML-based-LAr-veto, licensed under the MIT License (MIT).
# Report generation helpers — Markdown reports for processors

using Printf, Statistics
using Tables: Tables

# ============================================================================
# Detector Name Lookup
# ============================================================================

"""
    build_rawid_to_name(l200, group_name) → Dict{UInt32, String}

Build a mapping from detector raw IDs to human-readable names
by scanning channelinfo for all runs in the given ML group.
"""
function build_rawid_to_name(l200::LegendData, group_name::String)
    rawid_to_name = Dict{UInt32, String}()
    groupings_path = joinpath(dirname(@__DIR__), "config", "ml_groupings.yaml")
    group_def = get_ml_grouping(groupings_path, group_name)

    for (period_str, runs_def) in group_def
        period_str = String(period_str)
        period, runs = parse_period_run(period_str, runs_def)
        for run in runs
            filekey = try
                start_filekey(l200, (period, run, :phy))
            catch
                try start_filekey(l200, (period, run, :cal)) catch; continue end
            end
            for sys in (:spms, :geds)
                try
                    chinfo = channelinfo(l200, filekey; system=sys)
                    for i in 1:length(chinfo)
                        rawid_to_name[UInt32(chinfo.detector[i])] = string(chinfo.detector[i])
                    end
                catch; end
            end
        end
    end
    rawid_to_name
end
export build_rawid_to_name

# ============================================================================
# Extraction Report
# ============================================================================

"""
    generate_extraction_report(results, report_dir, group_name, l200) → String

Write a Markdown extraction report summarizing all datasets.
Returns the path to the written report file.
"""
function generate_extraction_report(
    results::Dict{String, PreparedDataset},
    report_dir::String,
    group_name::String,
    l200::LegendData,
)
    mkpath(report_dir)
    report_path = joinpath(report_dir, "extraction.md")

    rawid_to_name = build_rawid_to_name(l200, group_name)
    _det_name(rawid) = get(rawid_to_name, UInt32(rawid), string(rawid))

    open(report_path, "w") do io
        println(io, "# Extraction Report")
        println(io, "")
        println(io, "**Group:** `$group_name`  ")
        println(io, "**Generated:** $(Dates.now())  ")
        println(io, "")

        for (ds_name, ds) in sort(collect(results); by=first)
            s = ds.stats
            println(io, "---")
            println(io, "## Dataset: `$ds_name`")
            println(io, "")
            println(io, "| Metric | Value |")
            println(io, "|--------|-------|")
            @printf(io, "| Events read | %d |\n", get(s, "events_read_total", 0))
            @printf(io, "| Events after filter | %d |\n", get(s, "events_after_filter", 0))
            @printf(io, "| Events removed (NaN/Inf) | %d |\n", get(s, "events_nan_inf_removed", 0))
            @printf(io, "| **Events final** | **%d** |\n", get(s, "events_final", ds.n_events))
            n_zero = count(==(0.0), ds.event_sum_pe)
            @printf(io, "| Events zero PE | %d (%.1f%%) |\n", n_zero, 100.0 * n_zero / max(ds.n_events, 1))
            @printf(io, "| SiPMs | %d |\n", ds.n_sipms)
            @printf(io, "| Mean multiplicity | %.2f |\n", isempty(ds.event_multiplicity) ? 0.0 : mean(ds.event_multiplicity))
            @printf(io, "| Mean PE sum | %.2f |\n", isempty(ds.event_sum_pe) ? 0.0 : mean(ds.event_sum_pe))
            println(io, "")

            # SiPM detectors (collapsible)
            println(io, "<details><summary>$(ds.n_sipms) SiPM detectors</summary>\n")
            println(io, "| # | Detector | RawID | Non-DC Triggers |")
            println(io, "|---|----------|-------|-----------------|")
            for (i, did) in enumerate(ds.sipm_detector_ids)
                n_trigs = length(get(ds.per_det_raw_trig_pe, did, Float64[]))
                @printf(io, "| %d | %s | %d | %d |\n", i, _det_name(did), did, n_trigs)
            end
            println(io, "</details>\n")

            # HPGe detectors (collapsible, only for signal datasets)
            if any(!=(UInt32(0)), ds.ged_detector_id)
                ged_counts = Dict{UInt32, Int}()
                for gid in ds.ged_detector_id
                    gid == UInt32(0) && continue
                    ged_counts[gid] = get(ged_counts, gid, 0) + 1
                end
                sorted_geds = sort(collect(ged_counts); by=last, rev=true)
                println(io, "<details><summary>$(length(sorted_geds)) HPGe detectors</summary>\n")
                println(io, "| # | Detector | RawID | Events |")
                println(io, "|---|----------|-------|--------|")
                for (i, (gid, cnt)) in enumerate(sorted_geds)
                    @printf(io, "| %d | %s | %d | %d |\n", i, _det_name(gid), gid, cnt)
                end
                println(io, "</details>\n")
            end
        end
    end

    @info "Report saved" path=report_path
    report_path
end
export generate_extraction_report

# ============================================================================
# Balancing Report
# ============================================================================

"""
    generate_balancing_report(datasets, report_dir, group_name,
                              exclusion_rules, n_sub_ml, n_ft_ml, n_ft_valid)

Write a Markdown balancing report.
"""
function generate_balancing_report(
    datasets::Dict{String, PreparedDataset},
    report_dir::String,
    group_name::String,
    exclusion_rules::Vector,
    n_sub_ml::Int, n_ft_ml::Int, n_ft_valid::Int,
)
    mkpath(report_dir)
    path = joinpath(report_dir, "balancing.md")

    open(path, "w") do io
        println(io, "# Balancing Report")
        println(io, "")
        println(io, "**Group:** `$group_name`  ")
        println(io, "**Generated:** $(Dates.now())  ")
        if !isempty(exclusion_rules)
            println(io, "**Training exclusion rules:**  ")
            for rule in exclusion_rules
                println(io, "- `$(rule.key)` $(rule.op) $(rule.val)  ")
            end
        end
        println(io, "")

        # Summary table
        println(io, "## Training (jlbalml)")
        println(io, "")
        println(io, "| Dataset | Total | After exclusion | After balancing |")
        println(io, "|---------|-------|-----------------|-----------------|")

        sub = get(datasets, "sub500keV", nothing)
        ft  = get(datasets, "forcedtrigger", nothing)
        sub_n = isnothing(sub) ? 0 : sub.n_events
        ft_n  = isnothing(ft) ? 0 : ft.n_events

        @printf(io, "| sub500keV | %d | %d | %d |\n", sub_n, n_sub_ml, n_sub_ml)
        action = n_ft_valid >= n_sub_ml ? "downsampled" : "upsampled"
        @printf(io, "| forcedtrigger | %d | %d | %d (%s) |\n", ft_n, n_ft_valid, n_ft_ml, action)
        println(io, "")

        println(io, "## Inference (jlbal)")
        println(io, "")
        println(io, "| Dataset | Events |")
        println(io, "|---------|--------|")
        for (name, ds) in sort(collect(datasets); by=first)
            @printf(io, "| %s | %d |\n", name, ds.n_events)
        end
        println(io, "")

        # Per-dataset stats
        for (name, ds) in sort(collect(datasets); by=first)
            println(io, "---")
            println(io, "## $name")
            println(io, "")
            pe = ds.event_sum_pe
            if !isempty(pe)
                m = Statistics.mean(pe)
                n_zero = count(==(0.0), pe)
                @printf(io, "- Events: %d\n", ds.n_events)
                @printf(io, "- Mean PE sum (full): %.2f\n", m)
                @printf(io, "- Zero PE events: %d (%.1f%%)\n", n_zero, 100.0 * n_zero / max(ds.n_events, 1))
                @printf(io, "- Mean multiplicity: %.2f\n", Statistics.mean(ds.event_multiplicity))
                @printf(io, "- SiPMs: %d\n", ds.n_sipms)
                n_trigs = sum(length(v) for v in ds.trigger_pe_vals; init=0)
                @printf(io, "- Raw triggers (±10μs): %d\n", n_trigs)
            end
            println(io, "")
        end
    end

    @info "Balancing report saved" path=path
    path
end
export generate_balancing_report

# ============================================================================
# Normalization Report
# ============================================================================

"""
    generate_normalization_report(report_data, report_dir, group_name)

Write a Markdown normalization report with before/after statistics per column,
scaling parameters, geometry features, and split distribution.
"""
function generate_normalization_report(report_data::Dict{String,Any},
                                       report_dir::String, group_name::String;
                                       output_base::String="")
    mkpath(report_dir)
    path = joinpath(report_dir, "normalization.md")

    _stats(M) = begin
        pos = filter(>(0), vec(M))
        isempty(pos) && return (min=0.0, max=0.0, mean=0.0, std=0.0, n_pos=0, n_zero=count(==(0.0), M))
        (min=minimum(pos), max=maximum(pos), mean=Statistics.mean(pos),
         std=Statistics.std(pos), n_pos=length(pos), n_zero=count(==(0.0), M))
    end

    open(path, "w") do io
        println(io, "# Normalization Report")
        println(io, "")
        println(io, "**Group:** `$group_name`  ")
        println(io, "**Generated:** $(Dates.now())  ")
        println(io, "")

        for tier in ["jlbalml", "jlbal"]
            td = get(report_data, tier, nothing)
            td === nothing && continue
            out_tier = tier == "jlbalml" ? "jlnormml" : "jlnorm"

            println(io, "---")
            println(io, "## $tier → $out_tier")
            println(io, "")
            @printf(io, "- sub500keV: %d events\n", td.n_sub)
            @printf(io, "- forcedtrigger: %d events\n", td.n_ft)
            @printf(io, "- Combined: %d events\n", td.n_total)
            println(io, "- Geometry features: $(join(td.geom_features, ", "))")
            println(io, "")

            # Scaling parameters
            println(io, "### Scaling Parameters")
            println(io, "")
            println(io, "| Column | global_min | global_max |")
            println(io, "|--------|-----------|-----------|")
            for (col, params) in sort(collect(td.scaling_params); by=first)
                gmin = round(get(params, "global_min", NaN); digits=6)
                gmax = round(get(params, "global_max", NaN); digits=6)
                println(io, "| $col | $gmin | $gmax |")
            end
            println(io, "")

            # Before/after statistics per column per dataset
            for ds_name in ["sub500keV", "forcedtrigger"]
                println(io, "### $ds_name — Before / After")
                println(io, "")
                println(io, "| Column | Stage | Min | Max | Mean | σ | N>0 | N=0 |")
                println(io, "|--------|-------|-----|-----|------|---|-----|-----|")
                before_ds = get(td.before, ds_name, Dict())
                after_ds  = get(td.after, ds_name, Dict())
                for col in sort(collect(union(keys(before_ds), keys(after_ds))))
                    for (stage, data) in [("before", before_ds), ("after", after_ds)]
                        M = get(data, col, nothing)
                        M === nothing && continue
                        s = _stats(M)
                        @printf(io, "| %s | %s | %.4f | %.4f | %.4f | %.4f | %d | %d |\n",
                                col, stage, s.min, s.max, s.mean, s.std, s.n_pos, s.n_zero)
                    end
                end
                println(io, "")
            end

            # Trigger-level statistics
            trig = hasproperty(td, :trig_data) ? td.trig_data : nothing
            if trig !== nothing && !isempty(trig)
                println(io, "### Trigger-Level Features")
                println(io, "")
                println(io, "| Dataset | Events | Total Triggers | Median/Event | Mean PE | Mean |t| [μs] |")
                println(io, "|---------|--------|---------------|--------------|---------|--------------|")
                for ds_name in ["sub500keV", "forcedtrigger"]
                    haskey(trig, ds_name) || continue
                    tdds = trig[ds_name]
                    n_evts = length(tdds[:trig_count])
                    n_total_trigs = sum(tdds[:trig_count])
                    med_trigs = n_evts > 0 ? Int(Statistics.median(tdds[:trig_count])) : 0
                    flat_pe = reduce(vcat, tdds[:trig_pe_scaled])
                    flat_t  = reduce(vcat, tdds[:trig_time_scaled])
                    mean_pe = isempty(flat_pe) ? 0.0 : Statistics.mean(Float64.(flat_pe))
                    mean_at = isempty(flat_t) ? 0.0 : Statistics.mean(abs.(Float64.(flat_t)))
                    @printf(io, "| %s | %d | %d | %d | %.4f | %.4f |\n",
                            ds_name, n_evts, n_total_trigs, med_trigs, mean_pe, mean_at)
                end
                println(io, "")
            end
        end

        # ── jlnormml output structure diagram ────────────────────────────
        # Read one split file to inspect the actual key structure and types
        println(io, "---")
        println(io, "## Output Structure: jlnormml")
        println(io, "")
        println(io, "Each split file (`train.lh5`, `val.lh5`, `test.lh5`) has the following structure:")
        println(io, "")
        struct_path = get_tier_path(output_base, "jlnormml", group_name, "train")
        if isfile(struct_path)
            tbl = lh5open(struct_path, "r") do f; f["jlnormml"][:]; end
            sids = lh5open(struct_path, "r") do f; f["sipm_detector_ids"][:]; end
            n_events = length(tbl)
            n_sipms = length(sids)
            println(io, "```")
            println(io, "$(basename(struct_path))")
            println(io, "├── jlnormml/                ($n_events events × columns below)")
            col_names = sort(collect(Tables.columnnames(tbl)))
            for (ci, col) in enumerate(col_names)
                prefix = ci == length(col_names) ? "│   └── " : "│   ├── "
                cv = Tables.getcolumn(tbl, col)
                if cv isa VectorOfVectors
                    inner_T = eltype(eltype(cv))
                    # Check if fixed-length (matrix-like) or variable-length
                    lens = unique(length.(cv[1:min(100, length(cv))]))
                    if length(lens) == 1
                        println(io, prefix, col, "  VoV{$inner_T}  [n_events × $(lens[1])]")
                    else
                        med_len = Int(Statistics.median(length.(cv)))
                        println(io, prefix, col, "  VoV{$inner_T}  [n_events × variable, median=$(med_len)]")
                    end
                else
                    println(io, prefix, col, "  Vector{$(eltype(cv))}  [n_events]")
                end
            end
            println(io, "└── sipm_detector_ids        Vector{UInt32}  [$n_sipms]")
            println(io, "```")
            println(io, "")
            println(io, "**Legend:**")
            println(io, "- `VoV{T} [n × k]` — VectorOfVectors, fixed inner length `k` (one value per SiPM)")
            println(io, "- `VoV{T} [n × variable]` — VectorOfVectors, variable inner length (one value per trigger)")
            println(io, "- `Vector{T} [n]` — scalar column (one value per event)")
        else
            println(io, "_Train split file not found — structure not available._")
        end
        println(io, "")
    end

    @info "Normalization report saved" path=path
    path
end
export generate_normalization_report

# ============================================================================
# Training Report
# ============================================================================

"""
    generate_training_report(metadata, cfg, arch_name, report_dir, group_name, model_path, log_path) → String

Append a training-run section to `<report_dir>/training.md`.
All values are read from `cfg` and `metadata` — nothing hardcoded.
Returns the report file path.
"""
function generate_training_report(metadata::Dict, cfg::Dict, arch_name::String,
                                  report_dir::String, group_name::String,
                                  model_path::String, log_path::String)
    ts  = get(metadata, "timestamp_utc", "unknown")
    model_name = "$(group_name)_$(arch_name)_$(ts)"
    report_path = joinpath(report_dir, arch_name, "$(model_name).md")
    mkpath(dirname(report_path))

    fm  = get(metadata, "final_metrics", Dict())
    ac  = get(cfg, "architecture", Dict())
    tc  = get(cfg, "training", Dict())
    ic  = get(cfg, "input", Dict())
    sc  = get(ic, "sipm", Dict())
    hc  = get(ic, "hpge", Dict())
    dims = get(metadata, "input_dims", Dict())

    open(report_path, "w") do io
        println(io, "# Training Report — $(model_name)")
        println(io, "")
        println(io, "**Timestamp:** $(ts)  ")
        println(io, "**Architecture:** `$(arch_name)`  ")
        println(io, "**Model file:** `$(basename(model_path))`  ")
        println(io, "")

        # Input
        println(io, "### Input")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        println(io, "| SiPM features | $(join(get(sc, "features", []), ", ")) |")
        println(io, "| SiPM layout | $(get(sc, "layout", "-")) |")
        println(io, "| SiPM channels | $(get(dims, "sipm_channels", "-")) |")
        println(io, "| HPGe enabled | $(get(hc, "enabled", false)) |")
        if Bool(get(hc, "enabled", false))
            println(io, "| HPGe features | $(join(get(hc, "features", []), ", ")) |")
            println(io, "| HPGe output_dim | $(get(hc, "output_dim", "-")) |")
            println(io, "| HPGe hidden_width | $(get(hc, "hidden_width", "-")) |")
        end
        println(io, "")

        # Architecture
        println(io, "### Architecture")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        for k in ("hidden_widths", "activation", "dropout", "batch_norm")
            haskey(ac, k) && println(io, "| $(k) | $(ac[k]) |")
        end
        println(io, "| SiPM feature dim | $(get(dims, "sipm_feature_dim", "-")) |")
        println(io, "| Det feature dim | $(get(dims, "det_feature_dim", "-")) |")
        println(io, "")

        # Training hyperparameters
        println(io, "### Training")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        for k in ("epochs", "batch_size", "learning_rate", "weight_decay",
                   "optimizer", "seed", "early_stopping_patience",
                   "lr_scheduler", "lr_min")
            haskey(tc, k) && @printf(io, "| %s | %s |\n", k, tc[k])
        end
        println(io, "")

        # Results
        println(io, "### Results")
        println(io, "| Metric | Value |")
        println(io, "|--------|-------|")
        @printf(io, "| Best epoch | %d |\n", get(metadata, "best_epoch", 0))
        @printf(io, "| Best val loss | %.6f |\n", get(metadata, "best_val_loss", NaN))
        for (split, prefix) in [("train", "train"), ("val", "val"), ("test", "test")]
            lk, ak = "$(prefix)_loss", "$(prefix)_acc"
            if haskey(fm, lk)
                @printf(io, "| %s loss | %.6f |\n", split, fm[lk])
                @printf(io, "| %s acc  | %.2f%% |\n", split, fm[ak] * 100)
            end
        end
        println(io, "")
        println(io, "**Log:** `$(basename(log_path))`  ")
        println(io, "")
    end

    @info "  Training report updated: $report_path"
    return report_path
end

# ============================================================================
# Prediction Report
# ============================================================================

"""
    generate_prediction_report(metadata, cfg, arch_name, results, report_path, group_name) → String

Write a Markdown prediction report per model run.
`results` is a Dict with keys like :threshold_ft, :threshold_k40, :threshold_method,
:datasets (Vector of NamedTuples with :name, :n, :n_zero, :n_high, :n_model, :has_label),
:sf_vals (optional K40/K42 survival fractions from plot_k40_k42_survival).
"""
function generate_prediction_report(metadata::Dict, cfg::Dict, arch_name::String,
                                    results::Dict, report_path::String, group_name::String)
    mkpath(dirname(report_path))

    ts  = get(metadata, "timestamp_utc", "unknown")
    fm  = get(metadata, "final_metrics", Dict())
    dims = get(metadata, "input_dims", Dict())
    ic  = get(cfg, "input", Dict())
    sc  = get(ic, "sipm", Dict())
    hc  = get(ic, "hpge", Dict())
    ac  = get(cfg, "architecture", Dict())

    open(report_path, "w") do io
        println(io, "# Prediction Report")
        println(io, "")
        println(io, "**Group:** `$group_name`  ")
        println(io, "**Architecture:** `$arch_name`  ")
        println(io, "**Timestamp:** $ts  ")
        model_file = get(metadata, "model_file", "")
        !isempty(model_file) && println(io, "**Model:** `$model_file`  ")
        println(io, "**Generated:** $(Dates.now())  ")
        println(io, "")

        # Training metrics (from JLD2 metadata)
        if !isempty(fm)
            println(io, "## Training Metrics (from model)")
            println(io, "| Split | Loss | Accuracy |")
            println(io, "|-------|------|----------|")
            for split in ("train", "val", "test")
                lk, ak = "$(split)_loss", "$(split)_acc"
                haskey(fm, lk) && @printf(io, "| %s | %.6f | %.2f%% |\n", split, fm[lk], fm[ak] * 100)
            end
            println(io, "")
        end

        # Model config summary
        println(io, "## Model Configuration")
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        println(io, "| SiPM features | $(join(get(sc, "features", []), ", ")) |")
        println(io, "| SiPM channels | $(get(dims, "sipm_channels", "-")) |")
        println(io, "| HPGe enabled | $(get(hc, "enabled", false)) |")
        println(io, "| Hidden widths | $(get(ac, "hidden_widths", "-")) |")
        println(io, "| Dropout | $(get(ac, "dropout", "-")) |")
        println(io, "| BatchNorm | $(get(ac, "batch_norm", "-")) |")
        println(io, "")

        # Prediction results per dataset
        datasets = get(results, :datasets, NamedTuple[])
        if !isempty(datasets)
            println(io, "## Prediction Results")
            println(io, "| Dataset | Events | PE=0 → 0.0 | PE≥threshold → 1.0 | Model inference |")
            println(io, "|---------|--------|------------|-------------------|-----------------|")
            for ds in datasets
                @printf(io, "| %s | %d | %d | %d | %d |\n",
                        ds.name, ds.n, ds.n_zero, ds.n_high, ds.n_model)
            end
            println(io, "")
        end

        # Thresholds
        t_method = get(results, :threshold_method, "ft_matched_4x4_survival")
        t_ft = get(results, :threshold_ft, NaN)
        t_k40 = get(results, :threshold_k40, NaN)
        active_val = if t_method == "k40_matched_4x4_survival" && !isnan(t_k40)
            t_k40
        elseif !isnan(t_ft)
            t_ft
        else
            NaN
        end
        pe_hard = get(results, :pe_hard_veto, 25.0)

        println(io, "## Thresholds")
        println(io, "")
        println(io, "**Method:** `$t_method`  ")
        isnan(active_val) || @printf(io, "**Active threshold:** %.6f  \n", active_val)
        println(io, "")
        println(io, "| Threshold | Value |")
        println(io, "|-----------|-------|")
        isnan(t_ft) || @printf(io, "| FT-matched (4×4 FT survival) | %.6f |\n", t_ft)
        isnan(t_k40) || @printf(io, "| K40-matched (4×4 K40 survival) | %.6f |\n", t_k40)
        @printf(io, "| Hard veto (PE ≥) | %.0f |\n", pe_hard)
        println(io, "")

        # K40/K42 survival fractions
        sf_vals = get(results, :sf_vals, nothing)
        if sf_vals !== nothing
            println(io, "## K40/K42 Survival Fractions")
            println(io, "| Peak | Cut | SF (%) | σ (%) |")
            println(io, "|------|-----|--------|-------|")
            for (pi, pname) in enumerate(["⁴⁰K", "⁴²K"])
                for cut in ["4×4", "NN"]
                    sf, σ = get(sf_vals, (pi, cut), (NaN, NaN))
                    isnan(sf) || @printf(io, "| %s | %s | %.1f | %.1f |\n", pname, cut, sf*100, σ*100)
                end
            end
            println(io, "")
        end
    end

    @info "  Prediction report saved: $report_path"
    return report_path
end
