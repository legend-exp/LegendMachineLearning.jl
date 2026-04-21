# Reusable MLP building blocks
#
# Provides:
#   ACT_FNS               — Dict mapping activation name → function
#   resolve_activation(s) — look up activation by name string
#   dense_block(in, out, act; dp, bn) → Vector of Lux layers
#   mlp_block(dims, act; dp, bn)      → Lux.Chain
#   det_embedding(cfg)                → Lux.Chain or identity

# ── Activation lookup ────────────────────────────────────────────────────────

const ACT_FNS = Dict{String, Function}(
    "relu"  => relu,
    "gelu"  => gelu,
    "swish" => swish,
    "tanh"  => tanh,
)

"""Look up an activation function by name (default: relu)."""
resolve_activation(name::String) = get(ACT_FNS, lowercase(name), relu)

# ── Dense block: Dense → [BN] → activation → [Dropout] ──────────────────────

"""
    dense_block(in_d, out_d, act; dp=0f0, bn=false) → Vector{<:Lux.AbstractLuxLayer}

One Dense layer followed by optional BatchNorm, activation, and Dropout.
Uses fused `Dense(in => out, act)` when there is no BatchNorm (better for AD).
When BatchNorm is enabled, activation is applied after BN via `Dense(in => out)`
+ `BatchNorm(out, act)`.
"""
function dense_block(in_d::Int, out_d::Int, act::Function;
                     dp::Float32=0f0, bn::Bool=false)
    layers = if bn
        # Dense (linear) → BN with fused activation
        Any[Lux.Dense(in_d => out_d), Lux.BatchNorm(out_d, act)]
    else
        # Dense with fused activation (single Lux layer — simpler AD graph)
        Any[Lux.Dense(in_d => out_d, act)]
    end
    dp > 0 && push!(layers, Lux.Dropout(dp))
    return layers
end

# ── MLP block: sequence of dense blocks ──────────────────────────────────────

"""
    mlp_block(dims::Vector{Int}, act; dp=0f0, bn=false) → Lux.Chain

Build an MLP from a list of layer dimensions.
Example: `mlp_block([128, 256, 256], relu; dp=0.2f0, bn=true)` creates
  Dense(128→256) + BN + relu + Dropout → Dense(256→256) + BN + relu + Dropout
"""
function mlp_block(dims::Vector{Int}, act::Function;
                   dp::Float32=0f0, bn::Bool=false)
    length(dims) >= 2 || error("mlp_block needs at least [in, out], got $dims")
    layers = Any[]
    for i in 1:(length(dims)-1)
        append!(layers, dense_block(dims[i], dims[i+1], act; dp, bn))
    end
    return Lux.Chain(layers...)
end

# ── Detector (HPGe) embedding — shared across all architectures ──────────────

"""
    det_embedding(hpge_cfg::Dict, act; bn=false) → (chain, output_dim)

Build the HPGe detector embedding MLP from the `input.hpge` config block.
Returns `(Lux.WrappedFunction(identity), input_dim)` if embedding is disabled.

Config keys: `enabled`, `features`, `hidden_width`, `output_dim`.
"""
function det_embedding(hpge_cfg::Dict, act::Function; bn::Bool=false)
    enabled = Bool(get(hpge_cfg, "enabled", false))
    din = enabled ? length(hpge_cfg["features"]) : 2

    if !enabled
        return Lux.WrappedFunction(identity), din
    end

    hw   = Int(hpge_cfg["hidden_width"])
    dout = Int(hpge_cfg["output_dim"])

    # Build: input → hidden (with BN+act) → linear output
    layers = Any[]
    append!(layers, dense_block(din, hw, act; dp=0f0, bn=bn))
    push!(layers, Lux.Dense(hw => dout))

    return Lux.Chain(layers...), dout
end
