# Loss functions
#
# Provides:
#   _bce(ŷ, y)       — numerically stable binary cross-entropy on raw logits
#   _accuracy(ŷ, y)  — classification accuracy from logits
#   LOSS_REGISTRY     — Dict{String, Function} for config-driven loss selection
#   build_loss(cfg)   — look up loss function from training config

"""Numerically stable BCE on raw logits (no sigmoid applied)."""
_bce(ŷ, y) = mean(max.(ŷ, 0f0) .- ŷ .* y .+ log1p.(exp.(-abs.(ŷ))))

"""
    _bce_label_smooth(ŷ, y; α_pos, β_neg)

Asymmetric label-smoothed BCE. Targets are softened toward 0.5 by `α_pos` for
positive labels (noisy: sub500keV) and by `β_neg` for negative labels (clean:
ForcedTrigger):

    y_smooth = y · (1 - α_pos) · 1 + (1 - y) · β_neg
                                                = (1 - α_pos)·y + β_neg·(1 - y)

For α_pos=β_neg=0 this reduces exactly to `_bce`.
"""
function _bce_label_smooth(ŷ, y; α_pos::Float32, β_neg::Float32)
    y_s = (1f0 - α_pos) .* y .+ β_neg .* (1f0 .- y)
    return mean(max.(ŷ, 0f0) .- ŷ .* y_s .+ log1p.(exp.(-abs.(ŷ))))
end

"""Classification accuracy from raw logits."""
_accuracy(ŷ, y) = Float32(mean((σ.(ŷ) .>= 0.5f0) .== (y .>= 0.5f0)))

# ── Loss Registry ────────────────────────────────────────────────────────────

const LOSS_REGISTRY = Dict{String, Function}(
    "bce" => (cfg) -> _bce,
    "bce_label_smooth" => function (cfg::Dict)
        ls = get(cfg["training"], "label_smoothing", Dict())
        α  = Float32(get(ls, "alpha_pos", 0.0))
        β  = Float32(get(ls, "beta_neg",  0.0))
        return (ŷ, y) -> _bce_label_smooth(ŷ, y; α_pos=α, β_neg=β)
    end,
)

"""
    build_loss(cfg::Dict) → Function

Return a loss function based on `cfg["training"]["loss"]` (default: "bce").
Each registry entry is a *factory* `(cfg) → loss_fn` so it can pull
hyperparameters (e.g. label smoothing α/β) from the config.
"""
function build_loss(cfg::Dict)
    tc = cfg["training"]
    name = lowercase(get(tc, "loss", "bce"))
    haskey(LOSS_REGISTRY, name) || error(
        "Unknown loss: '$name'. Registered: $(collect(keys(LOSS_REGISTRY)))")
    return LOSS_REGISTRY[name](cfg)
end
