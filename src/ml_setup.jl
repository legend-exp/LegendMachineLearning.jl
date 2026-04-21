# One-shot include for all ML-dependent modules.
# Safe to include from multiple processors — only runs once.
#
# Usage (from processors/):
#   include(joinpath(@__DIR__, "..", "src", "ml_setup.jl"))

if !@isdefined(_ML_LOADED)
    const _ml_dir = joinpath(@__DIR__, "ml")

    # ── Core: imports, device, loss, optimizer ───────────────────────────
    include(joinpath(_ml_dir, "imports.jl"))
    include(joinpath(_ml_dir, "device.jl"))
    include(joinpath(_ml_dir, "losses.jl"))
    include(joinpath(_ml_dir, "optimizers.jl"))

    # ── Layers: building blocks ──────────────────────────────────────────
    include(joinpath(_ml_dir, "layers", "blocks.jl"))
    include(joinpath(_ml_dir, "layers", "pooling.jl"))
    include(joinpath(_ml_dir, "layers", "attention.jl"))
    include(joinpath(_ml_dir, "layers", "set_ops.jl"))

    # ── Architectures: registry + model definitions ──────────────────────
    include(joinpath(_ml_dir, "architectures", "registry.jl"))
    include(joinpath(_ml_dir, "architectures", "det_concat_mlp.jl"))
    include(joinpath(_ml_dir, "architectures", "hierarchical_set.jl"))
    include(joinpath(_ml_dir, "architectures", "set_transformer.jl"))

    # ── Training, evaluation, model I/O ──────────────────────────────────
    include(joinpath(_ml_dir, "evaluation.jl"))
    include(joinpath(_ml_dir, "training_loop.jl"))
    include(joinpath(_ml_dir, "model_io.jl"))

    const _ML_LOADED = true
end
