# Attention primitives for the Flat Set Transformer (Lee et al. 2019).
#
# Provides:
#   ISAB(embed_dim, n_inducing, n_heads; ffn_hidden, attn_dropout) — Induced Set Attention Block
#   PMA(embed_dim, n_seeds, n_heads; attn_dropout)                — Pooling by Multihead Attention
#   make_attention_mask(bool_mask)                                — bool→additive bias helper
#   _mha_with_bias(mha, ps, st, q, k, v, bias)                    — MHA forward routed through `bias=`
#
# Both layers wrap `Lux.MultiHeadAttention` for the dot-product attention math
# and add the learnable inducing points (ISAB) / seed vectors (PMA) on top.
#
# Why a custom `_mha_with_bias` helper:
# Lux.MultiHeadAttention's `mask=` API path goes through LuxLib's
# `apply_attention_mask_bias(logits, mask, ::Nothing)` which uses
# `typemin(Float32) = -Inf32`. Zygote's broadcast-pullback for `ifelse` on
# `(Bool, Float32, -Inf32)` infers a `Union{}` element type and aborts on
# GPU. The `bias=` path simply does `logits .+ bias` (no Inf), so we route
# our additive mask through there instead.

import LuxLib

# ── Simple LayerNorm that works for any input rank ─────────────────────────
#
# Normalizes along dim 1 (feature dim). `Lux.LayerNorm` initialises affine
# params as `(D, 1)` (trailing-1 for 4D image input) which trips up the
# `expand_layernorm_dims` dispatch with default `dims=:` on 3D inputs of
# shape `(D, set_len, batch)` — the shape we use throughout the Set
# Transformer. This is a thin replacement that broadcasts cleanly.

struct ChannelLayerNorm <: Lux.AbstractLuxLayer
    dim::Int
    eps::Float32
end

ChannelLayerNorm(dim::Int; eps::Float32 = 1f-5) = ChannelLayerNorm(dim, eps)

Lux.initialparameters(::AbstractRNG, m::ChannelLayerNorm) = (
    scale = ones(Float32, m.dim),
    bias  = zeros(Float32, m.dim),
)
Lux.initialstates(::AbstractRNG, ::ChannelLayerNorm) = NamedTuple()
Lux.parameterlength(m::ChannelLayerNorm) = 2 * m.dim
Lux.statelength(::ChannelLayerNorm) = 0

function Lux.apply(m::ChannelLayerNorm, x::AbstractArray, ps, st)
    μ  = mean(x; dims=1)
    σ² = var(x; dims=1, mean=μ, corrected=false)
    x_norm = (x .- μ) ./ sqrt.(σ² .+ m.eps)
    sz = ntuple(i -> i == 1 ? m.dim : 1, ndims(x))
    γ  = reshape(ps.scale, sz)
    β  = reshape(ps.bias,  sz)
    return γ .* x_norm .+ β, st
end

# ── Mask helper ──────────────────────────────────────────────────────────────

"""
    make_attention_mask(real_mask::AbstractArray{Bool, 2}) → AbstractArray{Float32, 4}

Convert a per-token bool padding mask of shape `(kv_len, B)` into an additive
attention mask of shape `(kv_len, 1, 1, B)`. Padded positions get `MASK_NEG`
(a very negative finite Float32), real positions get `0`. The result
broadcasts against the attention scores `(kv_len, q_len, n_heads, B)`.

We use a finite `-1f9` (not `-Inf32`) because Zygote's broadcast adjoint
infers a `Union{}` element type when `-Inf` appears, breaking GPU
compatibility. `softmax(-1f9 + ...) ≈ 0` to far below Float32 precision, so
this is functionally equivalent to `-Inf`.
"""
const MASK_NEG = -1f9

function make_attention_mask(real_mask::AbstractArray{Bool, 2})
    kv_len, B = size(real_mask)
    a = ifelse.(real_mask, 0f0, MASK_NEG)
    return reshape(a, kv_len, 1, 1, B)
end

# ── MHA forward via LuxLib's `bias=` path (AD-clean on GPU) ─────────────────

"""
    _mha_with_bias(mha, ps, st, q, k, v, bias) → (y, new_st)

Mirror of `Lux.apply_multiheadattention` but routes the additive `bias`
through `LuxLib.scaled_dot_product_attention` directly (so the masked path
stays out of `apply_attention_mask_bias`'s `typemin = -Inf` branch).

Reuses the projection sub-layers of `mha::Lux.MultiHeadAttention`. The
expected input shape per tensor is `(D, N_*, B)` and `bias` must be
broadcastable to the attention scores `(N_kv, N_q, n_heads, B)`.
"""
function _mha_with_bias(mha::Lux.MultiHeadAttention, ps, st,
                        q::AbstractArray, k::AbstractArray, v::AbstractArray,
                        bias)
    q, q_st = Lux.apply(mha.q_proj, q, ps.q_proj, st.q_proj)
    k, k_st = Lux.apply(mha.k_proj, k, ps.k_proj, st.k_proj)
    v, v_st = Lux.apply(mha.v_proj, v, ps.v_proj, st.v_proj)

    dropout = Lux.StatefulLuxLayer(
        mha.attention_dropout, ps.attention_dropout, st.attention_dropout
    )

    x, _ = LuxLib.scaled_dot_product_attention(
        reshape(q, size(q, 1) ÷ mha.nheads, mha.nheads, size(q)[2:end]...),
        reshape(k, size(k, 1) ÷ mha.nheads, mha.nheads, size(k)[2:end]...),
        reshape(v, size(v, 1) ÷ mha.nheads, mha.nheads, size(v)[2:end]...);
        head_dim = 1,
        token_dim = 3,
        fdrop = dropout,
        mask = nothing,
        bias = bias,
    )
    x = reshape(x, size(x, 1) * mha.nheads, size(x)[3:end]...)

    y, out_st = Lux.apply(mha.out_proj, x, ps.out_proj, st.out_proj)
    return y, (q_proj=q_st, k_proj=k_st, v_proj=v_st,
               attention_dropout=dropout.st, out_proj=out_st)
end

# ── ISAB — Induced Set Attention Block ───────────────────────────────────────

"""
    ISAB(embed_dim, n_inducing, n_heads; ffn_hidden=4*embed_dim,
         attn_dropout=0.0f0)

Set transformer block with O(N·M) complexity via M learnable inducing points.

Forward pass: `(X, mask)` where `X::(D, N, B)` and `mask::(N, B)::Bool`.
Returns `Y::(D, N, B)`.
"""
struct ISAB{MHA1, MHA2, FF, LN1, LN2} <: Lux.AbstractLuxLayer
    embed_dim::Int
    n_inducing::Int
    mha1::MHA1
    mha2::MHA2
    ffn::FF
    ln1::LN1
    ln2::LN2
end

function ISAB(embed_dim::Int, n_inducing::Int, n_heads::Int;
              ffn_hidden::Int = 4 * embed_dim,
              attn_dropout::Float32 = 0f0)
    mha1 = Lux.MultiHeadAttention(embed_dim; nheads=n_heads,
                                   attention_dropout_probability=attn_dropout)
    mha2 = Lux.MultiHeadAttention(embed_dim; nheads=n_heads,
                                   attention_dropout_probability=attn_dropout)
    ffn  = Lux.Chain(Lux.Dense(embed_dim => ffn_hidden, gelu),
                     Lux.Dense(ffn_hidden => embed_dim))
    ln1  = ChannelLayerNorm(embed_dim)
    ln2  = ChannelLayerNorm(embed_dim)
    return ISAB(embed_dim, n_inducing, mha1, mha2, ffn, ln1, ln2)
end

function Lux.initialparameters(rng::AbstractRNG, m::ISAB)
    scale = Float32(1 / sqrt(m.embed_dim))
    return (
        inducing = randn(rng, Float32, m.embed_dim, m.n_inducing) .* scale,
        mha1 = Lux.initialparameters(rng, m.mha1),
        mha2 = Lux.initialparameters(rng, m.mha2),
        ffn  = Lux.initialparameters(rng, m.ffn),
        ln1  = Lux.initialparameters(rng, m.ln1),
        ln2  = Lux.initialparameters(rng, m.ln2),
    )
end

function Lux.initialstates(rng::AbstractRNG, m::ISAB)
    return (
        mha1 = Lux.initialstates(rng, m.mha1),
        mha2 = Lux.initialstates(rng, m.mha2),
        ffn  = Lux.initialstates(rng, m.ffn),
        ln1  = Lux.initialstates(rng, m.ln1),
        ln2  = Lux.initialstates(rng, m.ln2),
    )
end

Lux.parameterlength(m::ISAB) = m.embed_dim * m.n_inducing +
    Lux.parameterlength(m.mha1) + Lux.parameterlength(m.mha2) +
    Lux.parameterlength(m.ffn)  + Lux.parameterlength(m.ln1)  +
    Lux.parameterlength(m.ln2)
Lux.statelength(m::ISAB) = Lux.statelength(m.mha1) + Lux.statelength(m.mha2) +
    Lux.statelength(m.ffn) + Lux.statelength(m.ln1) + Lux.statelength(m.ln2)

function Lux.apply(m::ISAB, (X, mask)::Tuple, ps, st)
    D, _, B = size(X)

    # Broadcast inducing points (D, M) → (D, M, B).
    I_b = reshape(ps.inducing, D, m.n_inducing, 1) .+ similar(X, D, m.n_inducing, B) .* 0f0
    # Cleaner equivalent: `repeat(reshape(ps.inducing, D, M, 1), 1, 1, B)` —
    # the form above avoids a copy by adding a zero-tensor with the right
    # device & shape (works for both Array and CuArray).

    # Step 1: I queries X, with X-side additive padding bias
    attn_bias = make_attention_mask(mask)                                # (N,1,1,B) → broadcasts
    H, st_mha1 = _mha_with_bias(m.mha1, ps.mha1, st.mha1, I_b, X, X, attn_bias)
    # H: (D, M, B) — no padding (M inducing slots are all real)
    H, st_ln1 = Lux.apply(m.ln1, H, ps.ln1, st.ln1)

    # Step 2: X queries H (no bias — H has no padding)
    X2, st_mha2 = _mha_with_bias(m.mha2, ps.mha2, st.mha2, X, H, H, nothing)
    X2 = X .+ X2                                                          # residual
    X2, st_ln2 = Lux.apply(m.ln2, X2, ps.ln2, st.ln2)

    # FFN with residual
    Yf, st_ffn = Lux.apply(m.ffn, X2, ps.ffn, st.ffn)
    Y = X2 .+ Yf

    new_st = (mha1=st_mha1, mha2=st_mha2, ffn=st_ffn, ln1=st_ln1, ln2=st_ln2)
    return Y, new_st
end

# ── PMA — Pooling by Multihead Attention ─────────────────────────────────────

"""
    PMA(embed_dim, n_seeds, n_heads; attn_dropout=0.0f0)

Variable-length set → fixed-size summary via attention from `n_seeds`
learnable seed vectors. For binary classification use `n_seeds=1`.

Forward pass: `(X, mask)` where `X::(D, N, B)` and `mask::(N, B)::Bool`.
Returns `Y::(D, n_seeds, B)`. Squeeze `dim=2` downstream when `n_seeds=1`.
"""
struct PMA{MHA} <: Lux.AbstractLuxLayer
    embed_dim::Int
    n_seeds::Int
    mha::MHA
end

function PMA(embed_dim::Int, n_seeds::Int, n_heads::Int;
             attn_dropout::Float32 = 0f0)
    mha = Lux.MultiHeadAttention(embed_dim; nheads=n_heads,
                                  attention_dropout_probability=attn_dropout)
    return PMA(embed_dim, n_seeds, mha)
end

function Lux.initialparameters(rng::AbstractRNG, m::PMA)
    scale = Float32(1 / sqrt(m.embed_dim))
    return (
        seed = randn(rng, Float32, m.embed_dim, m.n_seeds) .* scale,
        mha  = Lux.initialparameters(rng, m.mha),
    )
end
Lux.initialstates(rng::AbstractRNG, m::PMA) = (mha = Lux.initialstates(rng, m.mha),)

Lux.parameterlength(m::PMA) = m.embed_dim * m.n_seeds + Lux.parameterlength(m.mha)
Lux.statelength(m::PMA) = Lux.statelength(m.mha)

function Lux.apply(m::PMA, (X, mask)::Tuple, ps, st)
    D, _, B = size(X)
    S_b = reshape(ps.seed, D, m.n_seeds, 1) .+ similar(X, D, m.n_seeds, B) .* 0f0
    attn_bias = make_attention_mask(mask)
    Y, st_mha = _mha_with_bias(m.mha, ps.mha, st.mha, S_b, X, X, attn_bias)
    return Y, (mha = st_mha,)
end
