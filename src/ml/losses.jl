# Loss functions
#
# Provides:
#   _bce(ŷ, y)       — numerically stable binary cross-entropy on raw logits
#   _accuracy(ŷ, y)  — classification accuracy from logits
#   LOSS_REGISTRY     — Dict{String, Function} for config-driven loss selection
#   build_loss(cfg)   — look up loss function from training config

"""Numerically stable BCE on raw logits (no sigmoid applied)."""
_bce(ŷ, y) = mean(max.(ŷ, 0f0) .- ŷ .* y .+ log1p.(exp.(-abs.(ŷ))))

"""Classification accuracy from raw logits."""
_accuracy(ŷ, y) = Float32(mean((σ.(ŷ) .>= 0.5f0) .== (y .>= 0.5f0)))

# ── Loss Registry ────────────────────────────────────────────────────────────

const LOSS_REGISTRY = Dict{String, Function}(
    "bce" => _bce,
)

"""
    build_loss(cfg::Dict) → Function

Return a loss function based on `cfg["training"]["loss"]` (default: "bce").
"""
function build_loss(cfg::Dict)
    tc = cfg["training"]
    name = lowercase(get(tc, "loss", "bce"))
    haskey(LOSS_REGISTRY, name) || error(
        "Unknown loss: '$name'. Registered: $(collect(keys(LOSS_REGISTRY)))")
    return LOSS_REGISTRY[name]
end
