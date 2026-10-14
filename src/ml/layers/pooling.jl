# Set pooling — masked reductions over a chosen axis.
#
# Used by HierarchicalSetModel (pool over K triggers per SiPM, then over n_sipm)
# and FlatSetTransformer (PMA, future). Pooling ops are pure functions
# (no trainable parameters) so they wrap into Lux via `Lux.WrappedFunction`.
#
# Convention: input layout is (features × … × items × batch). The pool axis
# (``dims``) collapses to size 1 then is dropped via reshape.
#
# Provides:
#   masked_pool(x, mask; op, dims)   — single op (:sum, :max, :mean)
#   masked_multi_pool(x, mask; ops, dims) — concat of multiple ops along feature dim
#   resolve_pool_op(name)            — string → Symbol
#
# `mask` must broadcast against `x` along ``dims``. Padded slots have mask=false.

const POOL_OPS = (:sum, :max, :mean)

"""Resolve a pool-op name from YAML to an internal Symbol."""
function resolve_pool_op(name)::Symbol
    s = Symbol(lowercase(string(name)))
    s in POOL_OPS || error("Unknown pool op: $name. Allowed: $(POOL_OPS)")
    s
end

# Replace padded slots with neutral element for the op so the reduction is
# unaffected by them. For :max we use a very negative number; for :sum / :mean
# we use 0 (and divide by the true count).
@inline _neutral(::Val{:sum},  ::Type{T}) where {T} = zero(T)
@inline _neutral(::Val{:mean}, ::Type{T}) where {T} = zero(T)
@inline _neutral(::Val{:max},  ::Type{T}) where {T} = T(-1f30)

"""
    masked_pool(x, mask; op, dims) → Array

Reduce `x` along `dims` using `op` (∈ POOL_OPS), respecting `mask` (true =
real, false = padded). Output drops the reduced dimension.

`mask` is broadcasted against `x`; for the typical hierarchical layout
`x::(F × K × N × B)` and `mask::(K × N × B)` (no feature dim) you should pass
the mask as `reshape(mask, 1, size(mask)...)` so it broadcasts correctly.
"""
function masked_pool(x::AbstractArray{T}, mask::AbstractArray{Bool};
                     op::Symbol, dims::Int) where {T}
    op in POOL_OPS || error("Unknown pool op: $op")
    nT = _neutral(Val(op), T)
    x_masked = ifelse.(mask, x, nT)
    # Per-slot true count (post-reduction shape)
    c = dropdims(sum(T.(mask); dims=dims); dims=dims)
    has_any = c .>= 1
    if op === :sum
        return dropdims(sum(x_masked; dims=dims); dims=dims)
    elseif op === :max
        s = dropdims(maximum(x_masked; dims=dims); dims=dims)
        # Zero out slots with no real entries (otherwise the very-negative
        # neutral leaks downstream and blows up BatchNorm).
        return ifelse.(has_any, s, zero(T))
    else  # :mean — divide by per-slot true count
        s = dropdims(sum(x_masked; dims=dims); dims=dims)
        return s ./ max.(c, T(1))
    end
end

"""
    masked_multi_pool(x, mask; ops::Vector{Symbol}, dims::Int) → Array

Apply each op in `ops` and concatenate along the feature dimension (dim 1).
Output feature dim = size(x, 1) * length(ops).
"""
function masked_multi_pool(x::AbstractArray, mask::AbstractArray{Bool};
                           ops::Vector{Symbol}, dims::Int)
    isempty(ops) && error("masked_multi_pool needs at least one op")
    parts = [masked_pool(x, mask; op=o, dims=dims) for o in ops]
    return cat(parts...; dims=1)
end

"""
    masked_pool_layer(ops, dims) → Lux.WrappedFunction

Wrap a multi-op masked pool into a Lux layer. The forward call expects the
input to be a `(x, mask)` tuple so the wrapper knows the mask.
"""
function masked_pool_layer(ops::Vector{Symbol}, dims::Int)
    Lux.WrappedFunction(t::Tuple{<:AbstractArray, <:AbstractArray{Bool}} ->
                         masked_multi_pool(t[1], t[2]; ops=ops, dims=dims))
end
