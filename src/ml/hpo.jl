# Hyperparameter Optimization helpers
#
# Wraps Hyperopt.jl's Hyperband sampler around the existing training loop so
# the HPO processor can reuse `train_model` without copying it. Loads data
# exactly once (outside the trial loop) — only the model + parameter init
# differ between trials, so we save the ~minute of data loading per trial.
#
# Public API:
#   parse_search_space(ssdict)            → Vector of (name, sampler-iterable)
#   apply_search_overrides!(cfg, params)  → mutate `cfg` in place with new HPs
#   prepare_hpo_data(processing_config, group_name, cfg)
#                                         → (nt_tr, nt_va, perm, train_sids, dev_fn, using_gpu)
#   run_hpo_trial(arch_name, base_cfg, overrides, max_epochs, prepared)
#                                         → val_loss::Float64
#
# Conventions for the search-space YAML (see metadata/hpo/<group>.yaml):
#   <param_name>:
#     type: range       lo: 0.1   hi: 0.4   [n: 30]      → LinRange
#     type: log_range   lo: 1e-4  hi: 5e-3  [n: 100]     → log10 LinRange
#     type: choice      values: [a, b, c]                → finite list
#
# Special parameter names that are NOT direct YAML keys, mapped in apply_search_overrides!:
#   width_mult        → multiplies architecture.hidden_widths element-wise (rounded to int)
#   activation        → architecture.activation
#   dropout           → architecture.dropout
#   learning_rate     → training.learning_rate
#   weight_decay      → training.weight_decay
#   batch_size        → training.batch_size

using Hyperopt
using Random
using Lux
using Statistics: mean
using Printf
using MLUtils: DataLoader

# ── Search-space parsing ────────────────────────────────────────────────────

"""
    parse_search_space(ss::Dict) → Vector{Pair{Symbol, AbstractVector}}

Convert the `search_space:` YAML block into a list of (name, iterable) pairs
that can be spliced into Hyperopt.jl's `@hyperopt for` macro.
"""
function parse_search_space(ss::AbstractDict)
    out = Pair{Symbol, AbstractVector}[]
    for (k, spec) in ss
        name = Symbol(k)
        t = String(get(spec, "type", "range"))
        if t == "range"
            lo = Float64(spec["lo"]); hi = Float64(spec["hi"])
            n  = Int(get(spec, "n", 30))
            push!(out, name => collect(LinRange(lo, hi, n)))
        elseif t == "log_range"
            lo = Float64(spec["lo"]); hi = Float64(spec["hi"])
            n  = Int(get(spec, "n", 50))
            push!(out, name => collect(exp10.(LinRange(log10(lo), log10(hi), n))))
        elseif t == "choice"
            push!(out, name => collect(spec["values"]))
        else
            error("Unknown search-space sampler type '$t' for parameter '$k' " *
                  "(supported: range / log_range / choice)")
        end
    end
    out
end

# ── Inner-sampler selection (RandomSampler vs BOHB) ────────────────────────

"""
    _bohb_dim_for_values(vals) → Hyperopt.DimensionType

Map one search-space value vector to the BOHB dimension type it expects.
Heuristic: numeric ranges with > 5 distinct levels (typical of `range` /
`log_range`) → `Continuous`; small numeric sets such as a `choice: [0.5, 1.0,
1.5, 2.0]` → `Categorical`; non-numeric (strings, symbols) → `UnorderedCategorical`.

The 5-level cutoff is a tradeoff: a `choice` with up to 5 numeric levels
gets discrete-KDE treatment (correct semantics), while a denser `range`
keeps continuous-KDE smoothing.
"""
function _bohb_dim_for_values(vals)
    if eltype(vals) <: Real
        length(unique(vals)) > 5 ? Hyperopt.Continuous() :
                                   Hyperopt.Categorical(length(vals))
    else
        Hyperopt.UnorderedCategorical(length(vals))
    end
end

"""
    make_inner_sampler(kind::String, space) → Hyperopt.Sampler

Construct the Hyperband inner sampler from a YAML string. Supports:
  - `"random"` (default) → `RandomSampler` — fast, no model overhead
  - `"bohb"`            → `BOHB` with auto-derived dim types (Bayesian +
    pruning, BOHB paper). Needs ≳ 4 observations per budget bracket before
    its KDE model kicks in; falls back to random sampling until then.
"""
function make_inner_sampler(kind::String, space)
    k = lowercase(strip(kind))
    if k == "random"
        return Hyperopt.RandomSampler()
    elseif k == "bohb"
        dims = Hyperopt.DimensionType[_bohb_dim_for_values(p.second) for p in space]
        return Hyperopt.BOHB(dims = dims)
    else
        error("Unknown HPO inner_sampler '$kind' (expected: random | bohb)")
    end
end

# ── Apply sampled hyperparams onto a cfg dict ───────────────────────────────

"""
    apply_search_overrides!(cfg::Dict, params::Dict)

Mutate `cfg` (one architecture block of the training YAML) in place so that
the sampled hyperparameter values from one HPO trial become the active
training configuration. Knows about a small set of "logical" parameter names
(see file header) and maps them to the right nested keys.
"""
function apply_search_overrides!(cfg::AbstractDict, params::AbstractDict)
    arch = cfg["architecture"]
    train = cfg["training"]
    base_widths = collect(Int, arch["hidden_widths"])

    for (k, v) in params
        ks = String(k)
        if ks == "width_mult"
            wm = max(0.1, Float64(v))   # BOHB-KDE may extrapolate; keep widths positive
            arch["hidden_widths"] = [max(1, round(Int, w * wm)) for w in base_widths]
        elseif ks == "hidden_widths"
            arch["hidden_widths"] = collect(Int, v)
        elseif ks == "dropout"
            arch["dropout"] = clamp(Float64(v), 0.0, 0.95)   # BOHB-KDE can extrapolate
        elseif ks == "activation"
            arch["activation"] = String(v)
        elseif ks == "batch_norm"
            # Accept Bool directly; if a numeric extrapolation slipped through,
            # clamp to {0, 1} so `Bool(2)` etc. doesn't crash the trial.
            arch["batch_norm"] = v isa Bool ? v : Bool(clamp(round(Int, Float64(v)), 0, 1))
        elseif ks == "learning_rate"
            train["learning_rate"] = max(Float64(v), 1.0e-8)   # negative / zero lr is invalid
        elseif ks == "weight_decay"
            train["weight_decay"] = max(Float64(v), 0.0)       # negative wd is invalid
        elseif ks == "batch_size"
            train["batch_size"] = Int(v)
        else
            error("apply_search_overrides!: unknown HPO parameter '$ks'")
        end
    end
    cfg
end

# ── One-time data loading + everything that does NOT depend on hyperparams ──

"""
    prepare_hpo_data(processing_config, group_name, cfg) → NamedTuple

Run the data-loading + ordering + feature-assembly pipeline that
`process_training` does up to the model-build step. Returns a NamedTuple with
`nt_tr`, `nt_va`, `perm`, `train_sids`, `dev_fn`, `using_gpu`, `actual_n_sipm`
so each HPO trial can build its model + DataLoader and train without redoing
this expensive step.

Requires the bindings normally exported by process_training.jl (already in
scope when running inside the same Julia session, since the HPO processor
loads it as a peer).
"""
function prepare_hpo_data(processing_config::PropDict, group_name::String, cfg::AbstractDict;
                          extra_dataset::Union{String, Nothing} = nothing)
    sipm_feats, det_feats = resolve_input_features(cfg)
    @info "  HPO: SiPM features: $(sipm_feats)"
    @info "  HPO: Detector features: $(det_feats)"

    tier_base = processing_config.paths.output.tier
    norm_dir  = joinpath(tier_base, "jlnormml", group_name)
    split_path(s) = joinpath(norm_dir, "l200-$(group_name)-$(s)-tier_jlnormml.lh5")
    for s in ("train", "val")
        isfile(split_path(s)) || error("HPO: missing split file: $(split_path(s))")
    end

    @info "  HPO: loading jlnormml splits…"
    d_train = load_training_split(split_path("train"), cfg)
    d_val   = load_training_split(split_path("val"),   cfg)
    @info @sprintf("  HPO: samples train=%d val=%d", d_train.n, d_val.n)

    train_sids    = d_train.sids
    actual_n_sipm = length(train_sids)

    sipm_ord_cfg = get(get(cfg["input"], "sipm", Dict()), "ordering", Dict())
    perm = nothing
    if Bool(get(sipm_ord_cfg, "enabled", false))
        gbase = processing_config.paths.output.geometry
        perm, _ = build_sipm_ordering(train_sids, gbase, group_name)
    end

    function _make_loader_inputs(data, c, p; with_label::Bool = true)
        nt = assemble_features(data, c, p)
        kept = (; (k => v for (k, v) in pairs(nt) if v !== nothing)...)
        return with_label ? merge(kept, (label = reshape(data.y, 1, :),)) : kept
    end

    nt_tr = _make_loader_inputs(d_train, cfg, perm)
    nt_va = _make_loader_inputs(d_val,   cfg, perm)
    d_train = nothing; d_val = nothing
    GC.gc()

    # ── Optional: extra physics-tier dataset for objective metrics like k42_sf ──
    extra = nothing
    if extra_dataset !== nothing
        phys_path = joinpath(tier_base, "jlnorm", group_name,
                              "l200-$(group_name)-$(extra_dataset)-tier_jlnorm.lh5")
        isfile(phys_path) || error("HPO: extra physics file missing: $phys_path")
        @info "  HPO: loading extra physics dataset '$extra_dataset' from jlnorm…"
        d_phys = load_prediction_tier(phys_path, "jlnorm", cfg)
        d_phys.ged_energy_keV === nothing &&
            error("HPO extra dataset $extra_dataset has no ged_energy_keV column")
        nt_phys = _make_loader_inputs(d_phys, cfg, perm; with_label = false)
        @info @sprintf("  HPO: extra dataset events=%d (energy %.0f-%.0f keV)",
                       d_phys.n,
                       Float64(minimum(d_phys.ged_energy_keV)),
                       Float64(maximum(d_phys.ged_energy_keV)))
        extra = (nt = nt_phys,
                 ged_e = d_phys.ged_energy_keV,
                 event_sum_pe = d_phys.event_sum_pe,
                 event_multiplicity = d_phys.event_multiplicity,
                 n = d_phys.n)
        d_phys = nothing; GC.gc()
    end

    dev_fn, using_gpu = select_device()

    return (; nt_tr, nt_va, perm, train_sids, actual_n_sipm, dev_fn, using_gpu, extra)
end

# ── One Hyperband trial ─────────────────────────────────────────────────────

"""
    _hpo_forward_pass(model, ps, st, nt, dev_fn; batchsize=1024) → Vector{Float32}

Run the model in test-mode over a NamedTuple of input tensors and return
the sigmoid'd predictions as a flat CPU vector. Used by HPO objectives that
need predictions on a held-out physics dataset (e.g. `k42_sf`).
"""
function _hpo_forward_pass(model, ps, st, nt::NamedTuple, dev_fn; batchsize::Int = 1024)
    # `train_model` snapshots best_ps/best_st via `_snap` (Array → CPU). For a
    # GPU forward pass we need them back on the same device as `data`, otherwise
    # `Lux.apply` raises "Objects are on devices with different types".
    ps      = fmap(dev_fn, ps)
    eval_st = fmap(dev_fn, Lux.testmode(st))
    dl = DataLoader(nt; batchsize=batchsize, shuffle=false, partial=true)
    out = Float32[]
    for raw in dl
        data = fmap(dev_fn, raw)
        ŷ, _ = Lux.apply(model, data, ps, eval_st)
        # ŷ shape: (1, B) — collapse to vector, σ to probabilities
        append!(out, vec(Array(ŷ)))
    end
    return Float32.(1 ./ (1 .+ exp.(-out)))   # σ(logits)
end

"""
    run_hpo_trial(arch_name, base_cfg, overrides, max_epochs, prepared, trial_extras;
                  objective, seed, log_path) → metric::Float64

Build a fresh model from `base_cfg` with `overrides` applied, train for at
most `max_epochs` (Hyperband's resource budget for this bracket) using the
preloaded data in `prepared`, and return the configured objective metric
(lower is better — Hyperopt minimises).

`objective` is the `objective:` block from the HPO YAML:
  - `metric: val_loss` (default) → returns the best validation BCE loss
  - `metric: k42_sf`             → forwards the trained model on the
    `extra_dataset` (must be loaded via `prepare_hpo_data`'s
    `extra_dataset = ...` arg), computes the K40-matched threshold against
    a 4×4 baseline, and returns the K42 survival fraction at that threshold.

`trial_extras` is the `trial:` block from the HPO YAML — used to override
per-trial training settings (early_stopping_patience, batch_size, etc.) that
are common to every trial of this study.
"""
function run_hpo_trial(arch_name::String, base_cfg::AbstractDict,
                       overrides::AbstractDict, max_epochs::Int,
                       prepared::NamedTuple, trial_extras::AbstractDict;
                       objective::AbstractDict = Dict{String,Any}("metric" => "val_loss"),
                       pe_hard_veto::Float32 = Float32(15.0),
                       seed::Int = 1234, log_path::Union{String,Nothing} = nothing)

    # Deep-copy of the cfg so we can mutate without touching the caller's dict.
    cfg = deepcopy(Dict{String,Any}(base_cfg))

    # First the static per-trial overrides (patience, batch_size, …).
    apply_search_overrides!(cfg, Dict{String,Any}(
        k => v for (k, v) in trial_extras if k in
        ("dropout", "batch_norm", "activation", "learning_rate",
         "weight_decay", "batch_size", "width_mult")
    ))
    # Then the searched hyperparams (override the static ones if both exist).
    apply_search_overrides!(cfg, overrides)

    # Per-trial training-section knobs that are not in the architecture.
    tc = cfg["training"]
    tc["epochs"] = max_epochs
    if haskey(trial_extras, "early_stopping_patience")
        tc["early_stopping_patience"] = Int(trial_extras["early_stopping_patience"])
    end
    if haskey(trial_extras, "optimizer")
        tc["optimizer"] = String(trial_extras["optimizer"])
    end
    if haskey(trial_extras, "seed")
        seed = Int(trial_extras["seed"])
    end

    Random.seed!(seed)
    model, _ = build_model(arch_name, cfg, prepared.actual_n_sipm)

    rng = Random.MersenneTwister(seed + 1)
    ps, st = Lux.setup(rng, model)
    if prepared.using_gpu
        ps = prepared.dev_fn(ps)
        st = prepared.dev_fn(st)
    end

    bs = Int(tc["batch_size"])
    train_dl = DataLoader(prepared.nt_tr; batchsize=bs, shuffle=true,  partial=true)
    val_dl   = DataLoader(prepared.nt_va; batchsize=bs, shuffle=false, partial=true)

    best_ps, best_st, _, best_vl, best_tl = train_model(model, ps, st, train_dl, val_dl, cfg,
                                                         prepared.dev_fn; log_path=log_path)

    metric = String(get(objective, "metric", "val_loss"))

    # Backwards-compat shortcuts: `metric: val_loss` / `metric: k42_sf` map onto
    # the composite path with weight=1 on that single component (and 0 elsewhere).
    weights = if metric == "composite"
        w_raw = get(objective, "weights", Dict{String,Any}())
        Dict{String,Float64}(
            "k42_sf"      => Float64(get(w_raw, "k42_sf",      0.0)),
            "val_loss"    => Float64(get(w_raw, "val_loss",    0.0)),
            "overfit_gap" => Float64(get(w_raw, "overfit_gap", 0.0)),
        )
    elseif metric == "val_loss"
        Dict{String,Float64}("k42_sf"=>0.0, "val_loss"=>1.0, "overfit_gap"=>0.0)
    elseif metric == "k42_sf"
        Dict{String,Float64}("k42_sf"=>1.0, "val_loss"=>0.0, "overfit_gap"=>0.0)
    else
        error("Unknown HPO objective metric '$metric' (supported: val_loss, k42_sf, composite)")
    end

    sum_w = sum(values(weights))
    sum_w > 0 || error("HPO objective.weights sum to 0 — at least one component must be > 0")

    # ── k42_sf component (only computed if its weight > 0) ────────────────
    k42 = 0.0
    if weights["k42_sf"] > 0
        prepared.extra === nothing &&
            error("HPO weights.k42_sf > 0 requires `objective.physics_dataset:` to be set " *
                  "so prepare_hpo_data loads the extra dataset")
        pred_ml = _hpo_forward_pass(model, best_ps, best_st, prepared.extra.nt,
                                     prepared.dev_fn; batchsize=bs)
        # Reproduce `predict_with_rules` overrides so the HPO objective matches
        # the SF the prediction processor will report on the same model.
        ev_pe = Float32.(prepared.extra.event_sum_pe)
        @inbounds for i in eachindex(pred_ml)
            if ev_pe[i] == 0f0
                pred_ml[i] = 0f0
            elseif ev_pe[i] >= pe_hard_veto
                pred_ml[i] = 1f0
            end
        end
        pred_4x4 = Int8.((ev_pe                                   .>= 4f0) .|
                         (Int.(prepared.extra.event_multiplicity) .>= 4))
        ged_e    = Float32.(prepared.extra.ged_e)
        threshold_k40, _, _ = compute_k40_threshold(ged_e, pred_ml, pred_4x4)
        isnan(threshold_k40) && return Inf   # trial failed
        k42_raw = compute_k42_sf_at_k40_threshold(ged_e, pred_ml, threshold_k40)
        isnan(k42_raw) && return Inf
        k42 = Float64(k42_raw)
    end

    # ── overfit_gap = max(0, val_loss - train_loss) (only positive gap counts) ─
    # Negative gap (train > val, very rare) is not punished — we only care about
    # the model fitting the train set tighter than the val set.
    gap = max(0.0, Float64(best_vl) - Float64(best_tl))

    composite = weights["k42_sf"]      * k42 +
                weights["val_loss"]    * Float64(best_vl) +
                weights["overfit_gap"] * gap

    @info @sprintf("    components: k42_sf=%.4f val_loss=%.4f train_loss=%.4f gap=%.4f → composite=%.4f",
                    k42, best_vl, best_tl, gap, composite)

    return composite
end
