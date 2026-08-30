# ==============================================================================
# Processor: HPO — Hyperband search around train_model
# ==============================================================================
#
# Reads the per-group search space from `metadata/hpo/<group>.yaml`, runs
# Hyperband (via Hyperopt.jl) over the chosen architecture, retraining a fresh
# model each trial. Data is loaded once before the loop. Hyperband's
# `resources` parameter is mapped to the per-trial epoch budget so weak trials
# get pruned after a few epochs while promising ones get the full schedule.
#
# Output (under `generated/hpo/<group>/<arch>/`):
#   study.csv          one row per trial (resources, params, val_loss, wall_s)
#   best_config.yaml   the architecture+training block of the winning trial,
#                      ready to paste into metadata/training/<group>.yaml
#   hpo_report.md      Markdown summary: best params, top-K trials, settings
#
# This processor does NOT modify metadata/training/<group>.yaml. The user
# decides whether/which knobs to copy over.

include(joinpath(@__DIR__, "..", "src", "ml_setup.jl"))

using Hyperopt
using YAML
using Printf, Dates, Statistics
using Random

function process_hpo(processing_config::PropDict, l200::LegendData, group_name::String;
                     model::String = "mlp_detector_concat")

    @info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    @info "Process HPO for group: $group_name  (model=$model)"

    # ── Load HPO search-space config ─────────────────────────────────────────
    hpo_cfg = load_metadata_config(processing_config, :hpo)
    isnothing(hpo_cfg) && error("No HPO config found for group $group_name")
    haskey(hpo_cfg, model) || error("HPO YAML has no `$model:` block for $group_name " *
                                     "(found: $(collect(keys(hpo_cfg))))")
    hpo = hpo_cfg[model]

    hb_cfg = get(hpo, "hyperband", Dict())
    R   = Int(get(hb_cfg, "R",   80))
    eta = Int(get(hb_cfg, "eta", 3))
    inner_kind = String(get(hb_cfg, "inner_sampler", "random"))
    trial_extras = get(hpo, "trial", Dict{String,Any}())
    objective = get(hpo, "objective", Dict{String,Any}("metric" => "val_loss"))
    metric_name = String(get(objective, "metric", "val_loss"))
    extra_ds    = haskey(objective, "physics_dataset") ? String(objective["physics_dataset"]) : nothing
    metric_name in ("val_loss", "k42_sf", "composite") ||
        error("HPO objective.metric must be `val_loss`, `k42_sf` or `composite`, got `$metric_name`")

    # Effective weight on the k42_sf component — used to decide whether we need
    # to load the physics dataset and whether to load the hard-classification rules.
    k42_weight = if metric_name == "composite"
        Float64(get(get(objective, "weights", Dict{String,Any}()), "k42_sf", 0.0))
    elseif metric_name == "k42_sf"
        1.0
    else
        0.0
    end
    if metric_name == "composite"
        w_raw = get(objective, "weights", Dict{String,Any}())
        w_vl  = Float64(get(w_raw, "val_loss",    0.0))
        w_og  = Float64(get(w_raw, "overfit_gap", 0.0))
        sum_w = k42_weight + w_vl + w_og
        sum_w > 0 || error("HPO objective.weights sum to 0 — at least one of " *
                            "[k42_sf, val_loss, overfit_gap] must be > 0")
        @info @sprintf("  Composite weights: k42_sf=%.2f  val_loss=%.2f  overfit_gap=%.2f",
                        k42_weight, w_vl, w_og)
    end
    k42_weight > 0 && extra_ds === nothing &&
        error("HPO needs `objective.physics_dataset:` set when k42_sf weight > 0")

    # ── Hard-classification rules from the prediction config ────────────────
    # When the k42_sf component has any weight, the HPO trial must reproduce
    # what `process_prediction` does at inference time: pe_sum=0 → pred=0,
    # pe_sum ≥ pe_hard_veto → pred=1. Otherwise the HPO objective is biased
    # relative to the SF the prediction processor will report on the same model.
    pe_hard_veto = Float32(15.0)
    if k42_weight > 0
        pred_cfg = load_metadata_config(processing_config, :prediction)
        if pred_cfg !== nothing
            pe_hard_veto = parse_hard_classification(get(pred_cfg, "hard_classification", nothing))
            @info @sprintf("  Hard-classification (from prediction yaml): pe_sum=0 → 0.0,  pe_sum ≥ %.0f → 1.0",
                            pe_hard_veto)
        else
            @warn "k42_sf weight > 0 but no metadata/prediction/$group_name.yaml found — " *
                  "using default pe_hard_veto=$pe_hard_veto. Trials will not match prediction-time SF."
        end
    end
    ss = get(hpo, "search_space", nothing)
    isnothing(ss) && error("HPO YAML for $group_name/$model has no `search_space:` block")
    space = parse_search_space(ss)
    param_names = String[String(p.first) for p in space]
    candidates  = collect.(getfield.(space, :second))

    # Estimated trial count — sum of all bracket sizes Hyperband will try
    smax = floor(Int, log(eta, R))
    est_trials = sum(ceil(Int, (smax + 1) * R / R * (eta^s) / (s + 1)) for s in smax:-1:0)
    @info "  Hyperband: R=$R η=$eta  ($(smax + 1) brackets, ~$est_trials trials)"
    @info "  Inner sampler: $inner_kind"
    @info "  Objective: $metric_name" * (extra_ds === nothing ? "" : "  (physics_dataset=$extra_ds)")
    @info "  Search-space dims: $(length(space))"
    for (n, vals) in zip(param_names, candidates)
        @info "    $n → $(length(vals)) values  [$(first(vals)) … $(last(vals))]"
    end

    # ── Load training cfg (data + base hyperparams) ──────────────────────────
    raw_train_cfg = load_metadata_config(processing_config, :training)
    isnothing(raw_train_cfg) && error("HPO needs metadata/training/$group_name.yaml as base")
    haskey(raw_train_cfg, model) && raw_train_cfg[model] isa Dict ||
        error("training YAML for $group_name has no `$model:` block to use as base")
    raw_train_cfg["selected_architecture"] = model
    arch_name, base_cfg = unwrap_config(raw_train_cfg)
    @assert arch_name == model

    # ── Load data ONCE (skip per-trial reloading) ────────────────────────────
    # Skip the physics-tier load when the k42_sf component has zero weight.
    prepared = prepare_hpo_data(processing_config, group_name, base_cfg;
                                 extra_dataset = (k42_weight > 0 ? extra_ds : nothing))

    # ── Output paths ─────────────────────────────────────────────────────────
    # Reports + best-config land under the group's report dir (same convention
    # as extraction.md / normalization.md / training/<arch>/<ts>/), grouped
    # under `hpo/<arch>/` so multiple architectures coexist cleanly.
    out_dir = joinpath(processing_config.paths.output.reports, group_name, "hpo", arch_name)
    mkpath(out_dir)
    study_csv = joinpath(out_dir, "study.csv")
    best_yml  = joinpath(out_dir, "best_config.yaml")
    report_md = joinpath(out_dir, "hpo_report.md")
    log_dir   = joinpath(processing_config.paths.output.logs, group_name, "hpo", arch_name)
    mkpath(log_dir)

    # CSV header — last data column is named after the active metric
    csv_io = open(study_csv, "w")
    println(csv_io, "trial,resources," * join(param_names, ",") * ",$metric_name,wall_s")
    flush(csv_io)
    trial_counter = Ref(0)

    # ── Hyperband objective closure ──────────────────────────────────────────
    # Hyperopt.hyperband(f, candidates) calls `f(resources, params_vector)` for
    # the first bracket (Vector with one entry per param), and `f(resources,
    # state)` for subsequent brackets where `state` is the same Vector
    # carried over.
    #
    # BOHB's KDE samplers extrapolate beyond the YAML-listed candidates in
    # later brackets — they use Gaussians that have non-zero density outside
    # `[lo, hi]`. For *categorical* dims (numeric `choice` like width_mult,
    # string `choice` like activation, bool like batch_norm) we therefore snap
    # the raw sample to the nearest valid candidate via `_resolve_sample`, so
    # the trial actually trains a config that matches the YAML's intent.
    # Continuous (range/log_range) values are left raw here — they get
    # clamped inside `apply_search_overrides!` to safe physical ranges.
    #
    # Two return paths:
    #   - `params` (resolved values) → trial run + CSV log
    #   - `params_vec` (raw values)  → BOHB state, so the KDE keeps modelling
    #                                   the continuous dims accurately
    function _resolve_sample(v, vals)
        # Bool first — `Bool <: Real` in Julia, so without this guard a numeric
        # extrapolation like v=2 would be snapped to "1 ≈ true" rather than
        # treated as the second categorical index (false).
        if eltype(vals) === Bool
            v isa Bool && return v
            return vals[clamp(round(Int, Float64(v)), 1, length(vals))]
        elseif length(vals) > 5 && eltype(vals) <: Real
            return v                                          # continuous range
        elseif eltype(vals) <: Real
            v_num = Float64(v)
            return vals[argmin(abs.(Float64.(vals) .- v_num))]   # numeric choice → snap
        else
            v in vals && return v                             # already valid
            idx = clamp(round(Int, Float64(v)), 1, length(vals))
            return vals[idx]                                  # string choice → index lookup
        end
    end

    function hpo_objective(resources, params_vec)
        trial_counter[] += 1
        ti = trial_counter[]
        resolved_vec = [_resolve_sample(v, candidates[i]) for (i, v) in enumerate(params_vec)]
        params = Dict{String,Any}(name => v for (name, v) in zip(param_names, resolved_vec))
        trial_log = joinpath(log_dir, @sprintf("trial_%04d.csv", ti))

        val_loss = Inf
        t0 = time()
        try
            val_loss = run_hpo_trial(arch_name, base_cfg, params,
                                      Int(round(resources)), prepared, trial_extras;
                                      objective = objective, pe_hard_veto = pe_hard_veto,
                                      log_path = trial_log)
        catch e
            @warn "HPO trial $ti failed" exception=e
            val_loss = Inf
        end
        wall = round(time() - t0; digits=1)
        @info @sprintf("  HPO trial %3d  resources=%4d  %s=%.4f  (%.1fs)",
                       ti, Int(round(resources)), metric_name, val_loss, wall)

        # CSV: log RESOLVED params (the ones the model actually trained on),
        # not the raw BOHB samples — much easier to read and reproduce.
        pvals = join([string(get(params, n, "")) for n in param_names], ",")
        println(csv_io, "$ti,$(Int(round(resources))),$pvals,$val_loss,$wall")
        flush(csv_io)

        # Hyperband state must stay raw so BOHB's continuous KDEs keep working
        return val_loss, params_vec
    end

    # ── Run Hyperband ───────────────────────────────────────────────────────
    @info "  Starting Hyperband search (this may take hours)…"
    flush(stderr); flush(stdout)
    t0_total = time()
    inner = make_inner_sampler(inner_kind, space)
    # Workaround for Hyperopt.jl v0.5.6 bug: when `inner = BOHB(...)` and the
    # candidates are passed as Vector{Vector}, `update_KDEs` calls
    # `MultiKDE.KDEMulti(dims, records, bw, candidates)` and the only matching
    # method requires `candidates::Tuple`. Passing a Tuple here works around it
    # without affecting the random sampler path.
    ho = Hyperopt.hyperband(hpo_objective, Tuple(candidates); R = R, η = eta, inner = inner)
    elapsed_s = round(time() - t0_total; digits=1)
    close(csv_io)

    # ── Best result + write outputs ──────────────────────────────────────────
    # ho.minimum is normally (loss, resources_at_best, best_state_tuple). If the
    # final bracket fails (all-Inf, KDE corner-case, etc.), Hyperopt collapses
    # the value to a bare Float64. Recover the best from the per-trial history
    # in that case so we still write best_config.yaml + report.
    best_loss, best_res, best_state = if ho.minimum isa Tuple && length(ho.minimum) == 3
        ho.minimum
    else
        @warn "ho.minimum was not a (loss, resources, state) tuple — recovering best from history" got=typeof(ho.minimum)
        finite_ix = findall(isfinite, ho.results)
        isempty(finite_ix) && error("HPO finished with no successful trials — no best to report")
        bi = finite_ix[argmin(ho.results[finite_ix])]
        (Float64(ho.results[bi]), NaN, ho.history[bi])
    end
    best_pairs = Dict{String,Any}(name => v for (name, v) in zip(param_names, best_state))
    res_str = isfinite(best_res) ? string(Int(round(best_res))) : "?"
    @info @sprintf("HPO complete: %d trials in %.1fs  best %s=%.4f at resources=%s",
                   trial_counter[], elapsed_s, metric_name, best_loss, res_str)
    @info "  Best params: $best_pairs"

    # Build the best-config YAML (architecture+training block only)
    best_cfg = deepcopy(Dict{String,Any}(base_cfg))
    apply_search_overrides!(best_cfg, best_pairs)
    YAML.write_file(best_yml, Dict{String,Any}(arch_name => best_cfg))
    @info "  Best-config YAML: $best_yml"

    # Markdown report
    generate_hpo_report(report_md, group_name, arch_name, ho, param_names, candidates,
                        best_pairs, best_loss, best_res, R, eta,
                        trial_counter[], elapsed_s; metric_name = metric_name)
    @info "  HPO report: $report_md"

    return true
end
