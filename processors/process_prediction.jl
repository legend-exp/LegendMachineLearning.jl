# ==============================================================================
# Processor: Prediction — Load model + data, compute predictions, write jlpred/jlpredml
# ==============================================================================
#
# Thin orchestrator that delegates to src/ modules:
#   - src/ml/architectures/registry.jl → build_model, unwrap_config
#   - src/ml/model_io.jl              → load_model, find_model
#   - src/features.jl                 → load_prediction_tier, assemble_features, resolve_input_features
#   - src/prediction.jl               → predict_with_rules, compute_pred_4x4, write_pred_lh5
#   - src/plotting.jl                 → plot_prediction_histogram, plot_survival_cdf,
#                          plot_physics_energy_spectrum, plot_k40_k42_survival,
#                          compute_k40_threshold
#   - src/report.jl     → generate_prediction_report
#
# Input:  generated/tier/jlnorm/<group>/   (single "all" file)
#         generated/tier/jlnormml/<group>/  (train / val / test splits)
#         generated/model/<group>/<arch>/<latest>.jld2
# Output: generated/tier/jlpred/<group>/l200-<group>-{all,<ds>}-tier_jlpred.lh5
#         generated/tier/jlpredml/<group>/l200-<group>-{train,val,test}-tier_jlpredml.lh5
#         generated/plots/<group>/prediction/<arch>/<group>_<arch>_<ts>_*.png
#         generated/reports/<group>/prediction/<arch>/<group>_<arch>_<ts>.md
# ==============================================================================

using CairoMakie

# ── Load shared ML modules (models, model I/O) ──────────────────────────────
include(joinpath(@__DIR__, "..", "src", "ml_setup.jl"))


# ══════════════════════════════════════════════════════════════════════════════
# Main processor entry point
# ══════════════════════════════════════════════════════════════════════════════

function process_prediction(processing_config::PropDict, l200::LegendData, group_name::String;
                            model::String="", selection::Int=1)

    @info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    @info "Process Prediction for group: $group_name"

    # ── Find and load model ──────────────────────────────────────────────────
    model_dir = if !isempty(model)
        joinpath(processing_config.paths.output.model, group_name, model)
    else
        joinpath(processing_config.paths.output.model, group_name)
    end
    isdir(model_dir) || error("Model directory not found: $model_dir")
    model_path = find_model(model_dir; selection=selection)
    @info "  Loading model: $(basename(model_path))"

    ps, st, metadata = load_model(model_path)

    # ── Restore config from JLD2 metadata (schema v3 only) ───────────────────
    haskey(metadata, "config") && haskey(metadata, "architecture_name") ||
        error("Model must be schema v3 (has 'config' + 'architecture_name' in metadata)")
    arch_name = metadata["architecture_name"]
    cfg       = metadata["config"]
    @info "  Architecture: $arch_name"

    n_sipm = Int(metadata["input_dims"]["sipm_channels"])
    sipm_feats, det_feats = resolve_input_features(cfg)
    @info "  Features: SiPM=$(length(sipm_feats))  Det=$(length(det_feats))  n_sipm=$n_sipm"

    nn_model, layout = build_model(arch_name, cfg, n_sipm)
    @info "  Model reconstructed: $arch_name  layout=$layout"

    # ── SiPM ordering permutation (from model metadata) ──────────────────────
    sipm_ord  = get(metadata, "sipm_ordering", Dict())
    sipm_perm = get(sipm_ord, "enabled", false) ? Int.(sipm_ord["permutation"]) : nothing
    @info "  SiPM ordering: $(sipm_perm !== nothing ? "enabled ($(length(sipm_perm)) ch)" : "disabled")"

    if haskey(metadata, "final_metrics")
        fm = metadata["final_metrics"]
        @info @sprintf("  Training metrics — train=%.1f%%  val=%.1f%%  test=%.1f%%",
                       fm["train_acc"] * 100, fm["val_acc"] * 100, fm["test_acc"] * 100)
    end

    # ── Paths & versioned suffix ─────────────────────────────────────────────
    tier_base = processing_config.paths.output.tier
    ts = get(metadata, "timestamp_utc", Dates.format(Dates.now(Dates.UTC), "yyyymmdd_HHMMSS"))
    suffix = "$(group_name)_$(arch_name)_$(ts)"

    plot_base   = joinpath(processing_config.paths.output.plots, group_name, "prediction", arch_name)
    report_base = joinpath(processing_config.paths.output.reports, group_name, "prediction", arch_name)

    # ── Prediction config (thresholds, hard classification) ──────────────────
    pred_cfg = load_metadata_config(processing_config, :prediction)
    threshold_method = if pred_cfg !== nothing && haskey(pred_cfg, "threshold_method")
        String(pred_cfg["threshold_method"])
    else
        "ft_matched_4x4_survival"
    end
    hard_cfg = pred_cfg !== nothing ? get(pred_cfg, "hard_classification", nothing) : nothing
    pe_hard_veto = parse_hard_classification(hard_cfg)
    @info "  Threshold method: $threshold_method"
    @info @sprintf("  Hard classification: pe_sum=0 → 0.0, pe_sum≥%.0f → 1.0", pe_hard_veto)

    # ── Collectors ───────────────────────────────────────────────────────────
    jlpred_preds  = Vector{Float32}()
    jlpred_labels = Vector{Float32}()
    jlpred_4x4    = Vector{Int8}()
    threshold_ft  = NaN
    threshold_k40 = NaN
    report_datasets = NamedTuple[]

    # ── Helper: run inference on one tier file ───────────────────────────────
    function _predict_tier(path::AbstractString, tier_key::AbstractString,
                           out_tier::AbstractString, ds_name::AbstractString)
        data = load_prediction_tier(path, tier_key, cfg)
        @info @sprintf("  Loaded %d events", data.n)

        xs, xd = assemble_features(data, cfg, sipm_perm)
        pred_ml  = predict_with_rules(nn_model, ps, st, xs, xd, data.event_sum_pe; pe_hard_veto)
        pred_4x4 = compute_pred_4x4(data.event_sum_pe, data.event_multiplicity)

        n_zero  = count(pred_ml .== 0f0)
        n_high  = count(pred_ml .== 1f0)
        n_model = data.n - n_zero - n_high

        out_path = joinpath(tier_base, out_tier, group_name,
                            "l200-$(group_name)-$(ds_name)-tier_$(out_tier).lh5")
        write_pred_lh5(out_path, out_tier, data, pred_ml, pred_4x4, data.sids)

        push!(report_datasets, (name=ds_name, n=data.n, n_zero=n_zero, n_high=n_high,
                                n_model=n_model, has_label=data.has_label))
        return data, pred_ml, pred_4x4
    end

    # ── Process jlnormml (train/val/test → jlpredml) ────────────────────────
    normml_dir = joinpath(tier_base, "jlnormml", group_name)
    for split in ("train", "val", "test")
        split_file = joinpath(normml_dir, "l200-$(group_name)-$(split)-tier_jlnormml.lh5")
        isfile(split_file) || (@warn "  Missing jlnormml/$split — skipping"; continue)
        @info "  ─── jlnormml/$split → jlpredml/$split ───"
        _predict_tier(split_file, "jlnormml", "jlpredml", split)
    end

    # ── Discover and process jlnorm datasets (all + physics etc.) ────────────
    norm_dir = joinpath(tier_base, "jlnorm", group_name)
    if isdir(norm_dir)
        for f in sort(readdir(norm_dir))
            endswith(f, ".lh5") || continue
            m = match(r"l200-\w+-(\w+)-tier_jlnorm\.lh5", f)
            m === nothing && continue
            ds_name = m.captures[1]

            extra_file = joinpath(norm_dir, f)
            @info "  ─── jlnorm/$ds_name → jlpred/$ds_name ───"
            data, pred_ml, pred_4x4 = _predict_tier(extra_file, "jlnorm", "jlpred", ds_name)

            # Collect "all" dataset for histogram + survival CDF plots
            if ds_name == "all" && data.has_label
                append!(jlpred_preds, pred_ml)
                append!(jlpred_labels, data.y)
                append!(jlpred_4x4, pred_4x4)

                # Compute FT-matched threshold NOW so it's available for physics datasets
                threshold_ft = plot_survival_cdf(jlpred_preds, jlpred_labels, jlpred_4x4,
                    joinpath(plot_base, "$(suffix)_survival_cdf.png");
                    title="Cumulative Survival Probability (jlpred)")
                @info @sprintf("  threshold_ft (FT-matched): %.4f", threshold_ft)
            end

            # K40 threshold + veto only for unlabelled physics data
            ged_e = get(data, :ged_energy_keV, nothing)
            if ged_e !== nothing && !data.has_label && isnan(threshold_k40)
                threshold_k40, _, _ = compute_k40_threshold(ged_e, pred_ml, pred_4x4)
                @info @sprintf("  threshold_k40 (K40-matched): %.4f", threshold_k40)
            end

            # Apply binary veto for physics output
            if ged_e !== nothing && !data.has_label
                active_threshold = if threshold_method == "k40_matched_4x4_survival" && !isnan(threshold_k40)
                    threshold_k40
                elseif !isnan(threshold_ft)
                    threshold_ft
                else
                    0.5
                end
                veto_ml_vec = Int8.(pred_ml .>= Float32(active_threshold))
                @info @sprintf("  Veto (threshold=%.4f): %d/%d vetoed (%.1f%%)",
                               active_threshold, count(==(Int8(1)), veto_ml_vec), data.n,
                               100.0 * count(==(Int8(1)), veto_ml_vec) / max(data.n, 1))

                # Rewrite with veto_ml column
                data_with_veto = merge(data, (veto_ml = veto_ml_vec,))
                out_path = joinpath(tier_base, "jlpred", group_name,
                                    "l200-$(group_name)-$(ds_name)-tier_jlpred.lh5")
                write_pred_lh5(out_path, "jlpred", data_with_veto, pred_ml, pred_4x4, data.sids)

                # Physics plots
                plot_physics_energy_spectrum(ged_e, pred_4x4, veto_ml_vec,
                    joinpath(plot_base, "$(suffix)_$(ds_name)_energy_spectrum.png"))
                plot_k40_k42_survival(ged_e, pred_4x4, veto_ml_vec,
                    joinpath(plot_base, "$(suffix)_$(ds_name)_k40_k42_spectrum.png"))
            end
        end
    end

    # ── Prediction histogram from jlpred ("all") ─────────────────────────────
    if !isempty(jlpred_preds)
        plot_prediction_histogram(jlpred_preds, jlpred_labels,
            joinpath(plot_base, "$(suffix)_prediction_histogram.png");
            title="Prediction Distribution (jlpred)")
    end

    # ── Prediction report ────────────────────────────────────────────────────
    report_path = joinpath(report_base, "$(suffix).md")
    report_results = Dict{Symbol,Any}(
        :threshold_ft     => threshold_ft,
        :threshold_k40    => threshold_k40,
        :threshold_method => threshold_method,
        :pe_hard_veto     => pe_hard_veto,
        :datasets         => report_datasets,
    )
    metadata_report = copy(metadata)
    metadata_report["model_file"] = basename(model_path)
    generate_prediction_report(metadata_report, cfg, arch_name, report_results,
                               report_path, group_name)

    @info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    @info "Prediction complete"
    return true
end
