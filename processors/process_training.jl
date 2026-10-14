# ==============================================================================
# Processor: Training — Metadata-driven model training (Lux + CUDA/CPU, Zygote AD)
# ==============================================================================
#
# Thin orchestrator that delegates to src/ modules:
#   - src/ml/architectures/registry.jl → build_model, unwrap_config
#   - src/ml/model_io.jl              → save_model
#   - src/ml/training_loop.jl         → train_model
#   - src/ml/device.jl                → select_device, gpu_sync_gc
#   - src/ml/evaluation.jl            → eval_loop
#   - src/features.jl                 → load_training_split, assemble_features, build_sipm_ordering
#
# The model architecture, hyperparameters, and input features are fully
# determined by metadata/training/<group>.yaml — no hardcoded feature lists.
#
# YAML schema (v3): top-level key = architecture name
#   mlp_detector_concat:
#     input: { sipm: {features, layout, ordering}, hpge: {...} }
#     architecture: { hidden_widths, activation, dropout, batch_norm }
#     training: { epochs, batch_size, learning_rate, ... }
#
# Input:  generated/tier/jlnormml/<group>/{train,val,test}.lh5
# Output: generated/model/<group>/<group>_<arch>_<timestamp>.jld2

# ── Load shared ML modules (models, training loop, model I/O) ────────────────
include(joinpath(@__DIR__, "..", "src", "ml_setup.jl"))

function process_training(processing_config::PropDict, l200::LegendData, group_name::String;
                          model::String="")

    @info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    @info "Process Training for group: $group_name"

    # ── Load + unwrap training config ────────────────────────────────────────
    raw_cfg = load_metadata_config(processing_config, :training)
    isnothing(raw_cfg) && error("No training config found for group $group_name")

    # The processing_config `model:` kwarg, if set, overrides the YAML's
    # `selected_architecture:` line. This lets the same per-group YAML carry
    # multiple architecture blocks, with the active one chosen at the
    # processing-config level (no YAML edits to switch).
    if !isempty(model)
        if haskey(raw_cfg, model) && raw_cfg[model] isa Dict
            raw_cfg["selected_architecture"] = model
        else
            avail = filter(k -> k != "selected_architecture", collect(keys(raw_cfg)))
            error("kwarg model='$model' but no top-level block by that name in training YAML. " *
                  "Available blocks: $avail")
        end
    end
    arch_name, cfg = unwrap_config(raw_cfg)

    tc = cfg["training"]
    seed = Int(get(tc, "seed", 1234))
    Random.seed!(seed)

    @info "  Architecture: $arch_name"

    # ── Resolve input features from config ───────────────────────────────────
    sipm_feats, det_feats = resolve_input_features(cfg)
    @info "  SiPM features: $(sipm_feats)"
    @info "  Detector features: $(det_feats)"

    # ── Locate jlnormml split files ──────────────────────────────────────────
    tier_base = processing_config.paths.output.tier
    norm_dir  = joinpath(tier_base, "jlnormml", group_name)
    split_path(s) = joinpath(norm_dir, "l200-$(group_name)-$(s)-tier_jlnormml.lh5")
    for s in ("train", "val", "test")
        isfile(split_path(s)) || error("Missing split file: $(split_path(s))")
    end

    # ── Load data (config-driven columns) ────────────────────────────────────
    @info "  Loading jlnormml splits..."
    d_train = load_training_split(split_path("train"), cfg)
    d_val   = load_training_split(split_path("val"), cfg)
    d_test  = load_training_split(split_path("test"), cfg)

    @info @sprintf("  Samples: train=%d  val=%d  test=%d",
                   d_train.n, d_val.n, d_test.n)
    @info @sprintf("  Train labels: signal=%d  bg=%d",
                   count(>=(0.5f0), d_train.y), count(<(0.5f0), d_train.y))

    train_sids   = d_train.sids
    actual_n_sipm = length(train_sids)

    # ── SiPM ordering ────────────────────────────────────────────────────────
    sipm_ord_cfg = get(get(cfg["input"], "sipm", Dict()), "ordering", Dict())
    order_detectors = Bool(get(sipm_ord_cfg, "enabled", false))
    perm          = nothing
    ordered_names = nothing
    if order_detectors
        @info "  Building SiPM channel ordering..."
        gbase = processing_config.paths.output.geometry
        perm, ordered_names = build_sipm_ordering(train_sids, gbase, group_name)
        @info "  Ordering applied: $(length(perm)) channels reordered"
    end

    @info "  n_sipm=$(actual_n_sipm)  features_per_channel=$(length(sipm_feats))"

    # ── Build model via registry ─────────────────────────────────────────────
    model, layout = build_model(arch_name, cfg, actual_n_sipm)

    # ── Assemble feature NamedTuples (skip nothing entries) ──────────────────
    function _make_loader_inputs(data, cfg, perm)
        nt = assemble_features(data, cfg, perm)
        y  = reshape(data.y, 1, :)
        # Drop any nothing entries (e.g. trig/mask/geom for MLP architecture).
        kept = (; (k => v for (k,v) in pairs(nt) if v !== nothing)...)
        return merge(kept, (label = y,))
    end

    nt_tr = _make_loader_inputs(d_train, cfg, perm)
    nt_va = _make_loader_inputs(d_val,   cfg, perm)
    nt_te = _make_loader_inputs(d_test,  cfg, perm)

    # Free raw data
    d_train_n = d_train.n; d_val_n = d_val.n; d_test_n = d_test.n
    d_train = nothing; d_val = nothing; d_test = nothing
    gpu_sync_gc()

    # Logging dims
    sipm_dim = haskey(nt_tr, :sipm) ? size(nt_tr.sipm, 1) : 0
    det_dim  = haskey(nt_tr, :det)  ? size(nt_tr.det,  1) : 0
    @info @sprintf("  Feature dims: SiPM=%d  Det=%d", sipm_dim, det_dim)
    if haskey(nt_tr, :trig)
        sz = size(nt_tr.trig)
        @info @sprintf("  Trigger tensor: F=%d  max_K=%d  n_sipm=%d  N=%d", sz...)
    end
    if haskey(nt_tr, :geom)
        sz = size(nt_tr.geom)
        @info @sprintf("  Geometry tensor: G=%d  n_sipm=%d  N=%d", sz...)
    end

    # ── Initialise model + device ────────────────────────────────────────────
    dev_fn, using_gpu = select_device()
    rng = Random.MersenneTwister(seed + 1)
    ps, st = Lux.setup(rng, model)

    if using_gpu
        @info "  Moving params to GPU..."
        ps = dev_fn(ps)
        st = dev_fn(st)
    end

    # ── DataLoaders ──────────────────────────────────────────────────────────
    bs = Int(tc["batch_size"])
    train_dl = DataLoader(nt_tr; batchsize=bs, shuffle=true,  partial=true)
    val_dl   = DataLoader(nt_va; batchsize=bs, shuffle=false, partial=true)
    test_dl  = DataLoader(nt_te; batchsize=bs, shuffle=false, partial=true)

    # ── Train ─────────────────────────────────────────────────────────────────
    @info @sprintf("  Training: %d epochs, batch=%d, lr=%.2e, optimizer=%s, AD=Zygote",
                   Int(tc["epochs"]), bs, tc["learning_rate"], tc["optimizer"])

    log_dir = joinpath(processing_config.paths.output.logs, group_name)
    mkpath(log_dir)
    train_log_path = joinpath(log_dir, "$(group_name)_training_$(Dates.format(Dates.now(Dates.UTC), "yyyymmdd_HHMMSS")).csv")

    flush(stderr); flush(stdout)
    best_ps, best_st, best_ep, best_vl, _ = train_model(
        model, ps, st, train_dl, val_dl, cfg, dev_fn; log_path=train_log_path)

    # ── Final evaluation on all splits (CPU, testmode) ────────────────────────
    best_st_eval = Lux.testmode(best_st)
    trl, tra = eval_loop(model, best_ps, best_st_eval,
                          DataLoader(nt_tr; batchsize=bs), identity)
    vll, vla = eval_loop(model, best_ps, best_st_eval,
                          DataLoader(nt_va; batchsize=bs), identity)
    tel, tea = eval_loop(model, best_ps, best_st_eval,
                          DataLoader(nt_te; batchsize=bs), identity)

    @info @sprintf("  Final metrics — train: loss=%.4f acc=%.1f%%  val: loss=%.4f acc=%.1f%%  test: loss=%.4f acc=%.1f%%",
                   trl, tra * 100, vll, vla * 100, tel, tea * 100)

    # ── Save model + metadata (schema v3) ────────────────────────────────────
    model_dir = joinpath(processing_config.paths.output.model, group_name, arch_name)
    ts = Dates.format(Dates.now(Dates.UTC), "yyyymmdd_HHMMSS")
    save_path = joinpath(model_dir, "$(group_name)_$(arch_name)_$(ts).jld2")

    sipm_names_default = [string(DetectorId(id)) for id in train_sids]
    metadata = Dict{String,Any}(
        "schema_version"      => 3,
        "group_name"          => group_name,
        "timestamp_utc"       => ts,
        "ad_backend"          => "Zygote",

        "architecture_name"   => arch_name,
        "config"              => cfg,

        "sipm_ordering" => Dict{String,Any}(
            "enabled"       => order_detectors,
            "permutation"   => perm !== nothing ? perm : collect(1:actual_n_sipm),
            "ordered_names" => ordered_names !== nothing ? ordered_names : sipm_names_default,
            "groups"        => get(sipm_ord_cfg, "groups", String[]),
        ),

        "input_dims" => Dict(
            "sipm_feature_dim"  => sipm_dim,
            "det_feature_dim"   => det_dim,
            "sipm_channels"     => actual_n_sipm,
        ),

        "best_epoch"   => best_ep,
        "best_val_loss" => Float64(best_vl),
        "final_metrics" => Dict(
            "train_loss" => Float64(trl), "train_acc" => Float64(tra),
            "val_loss"   => Float64(vll), "val_acc"   => Float64(vla),
            "test_loss"  => Float64(tel), "test_acc"  => Float64(tea),
        ),

        "sipm_detector_ids" => train_sids,
    )

    save_model(save_path, best_ps, best_st, metadata)

    # ── Training report ──────────────────────────────────────────────────────
    # Per-run subfolder (`<arch>/<timestamp>/`) so multiple runs don't mix.
    report_dir = joinpath(processing_config.paths.output.reports, group_name, "training", arch_name, ts)
    generate_training_report(metadata, cfg, arch_name, report_dir, group_name,
                             save_path, train_log_path)

    # ── Loss plot ────────────────────────────────────────────────────────────
    plot_dir = joinpath(processing_config.paths.output.plots, group_name, "training", arch_name, ts)
    plot_path = joinpath(plot_dir, "$(group_name)_$(arch_name)_$(ts)_loss.png")
    preliminary = get(Dict(processing_config.plots), :preliminary, false)
    plot_training_loss(train_log_path, plot_path, metadata; preliminary=preliminary)

    @info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    @info "Training complete"
    return true
end
