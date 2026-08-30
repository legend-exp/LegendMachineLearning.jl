# FlatSetTransformer — all triggers as one set, ISAB × L → PMA → tail
#
# Architecture:
#   Inputs (NamedTuple): trig (F, K, N, B), mask (K, N, B),
#                        geom (G, N, B), det (D_det, B)
#
#   1) Broadcast geom along K and concat with trig features:
#         (F, K, N, B) ⊕ (G, K, N, B) → (F+G, K, N, B)
#      Flatten dims (K, N) into one set axis:
#         (F+G, K·N, B)         + flat mask (K·N, B)
#
#   2) Embed Linear(F+G → D) + LayerNorm
#         → (D, K·N, B)
#
#   3) Stack of ISAB layers × L (each refines the set with cross-trigger
#      attention via M inducing points; complexity O(K·N · M))
#
#   4) PMA with one seed → (D, B)
#
#   5) Concat with HPGe embedding, then tail MLP → logit
#
# Reuses pieces from: layers/blocks.jl (det_embedding, dense_block, mlp_block),
# layers/attention.jl (ISAB, PMA).

# ── Lux container ────────────────────────────────────────────────────────────

struct FlatSetTransformer{E, IS, PM, D, T} <: Lux.AbstractLuxContainerLayer{
        (:embed, :isabs, :pma, :det, :tail)}
    embed::E         # Chain: Dense(F+G → D) → LayerNorm
    isabs::IS        # Chain of ISAB layers (each consumes (X, mask))
    pma::PM          # PMA layer with n_seeds=1
    det::D           # HPGe embedding chain
    tail::T          # Dense → activation → … → Dense(→1)
end

function Lux.apply(m::FlatSetTransformer, inputs::NamedTuple, ps, st)
    trig = inputs.trig         # (F, K, N, B)
    mask = inputs.mask         # (K, N, B) Bool
    geom = inputs.geom         # (G, N, B)
    det  = inputs.det          # (D_det, B)

    F, K, N_sipm, B = size(trig)
    G = size(geom, 1)

    # 1) Broadcast geom over K, concat with trig, flatten K·N.
    geom_K = reshape(geom, G, 1, N_sipm, B) .+
             similar(trig, G, K, N_sipm, B) .* 0f0      # (G, K, N, B)
    x = cat(trig, geom_K; dims=1)                       # (F+G, K, N, B)
    x_flat = reshape(x, F + G, K * N_sipm, B)           # (F+G, K·N, B)
    mask_flat = reshape(mask, K * N_sipm, B)            # (K·N, B) Bool

    # 2) Embed (Dense + LayerNorm) — broadcast over the set axis K·N.
    h, st_e = Lux.apply(m.embed, x_flat, ps.embed, st.embed)   # (D, K·N, B)

    # 3) ISAB stack — each layer takes (X, mask). We accumulate updated
    #    states into a plain Tuple (immutable, Zygote-AD-clean) and convert
    #    to a NamedTuple at the end. `Base.setindex(::NamedTuple, ...)` would
    #    work in forward but has no AD adjoint.
    ks = keys(st.isabs)
    sts_tuple = ()
    for k in ks
        layer = getfield(m.isabs.layers, k)
        ps_i  = getfield(ps.isabs, k)
        st_i  = getfield(st.isabs, k)
        h, st_i_new = Lux.apply(layer, (h, mask_flat), ps_i, st_i)
        sts_tuple = (sts_tuple..., st_i_new)
    end
    st_is = NamedTuple{ks}(sts_tuple)

    # 4) PMA → (D, n_seeds=1, B), then squeeze the seed axis.
    pooled, st_pm = Lux.apply(m.pma, (h, mask_flat), ps.pma, st.pma)
    e = dropdims(pooled; dims=2)                                # (D, B)

    # 5) HPGe + tail.
    hd, st_d = Lux.apply(m.det, det, ps.det, st.det)
    out, st_t = Lux.apply(m.tail, vcat(e, hd), ps.tail, st.tail)

    new_st = (embed=st_e, isabs=st_is, pma=st_pm, det=st_d, tail=st_t)
    return out, new_st
end

# ── A thin wrapper to give the ISAB stack a stable layer-name structure ─────

# We store the ISAB layers as a `Lux.Chain` so Lux's container machinery
# auto-handles their parameters/states. The forward pass loops over
# `m.isabs.layers` manually because each layer needs the (X, mask) tuple.

# ── Builder ──────────────────────────────────────────────────────────────────

function _build_set_transformer(cfg::Dict, n_sipm::Int)
    ic = cfg["input"]
    ac = cfg["architecture"]
    hc = get(ic, "hpge", Dict())

    trig_cfg = ic["triggers"]
    F = length(trig_cfg["features"])
    G = length(get(ic["sipm"], "geometry", String[]))

    # ── Embed: Linear(F+G → D) + LayerNorm ────────────────────────────
    em = ac["embed"]
    D = Int(em["out_dim"])
    em_act = resolve_activation(get(em, "activation", "gelu"))
    em_ln  = Bool(get(em, "layer_norm", true))
    embed_layers = Any[Lux.Dense(F + G => D, em_act)]
    em_ln && push!(embed_layers, ChannelLayerNorm(D))
    embed = Lux.Chain(embed_layers...)

    # ── ISAB stack ────────────────────────────────────────────────────
    is = ac["isab"]
    L = Int(is["n_layers"])
    n_heads_isab = Int(is["n_heads"])
    n_inducing  = Int(is["n_inducing"])
    ffn_hidden  = Int(get(is, "ffn_hidden", 4 * D))
    attn_dp     = Float32(get(is, "attention_dropout", 0))
    isab_stack = Lux.Chain([
        ISAB(D, n_inducing, n_heads_isab; ffn_hidden=ffn_hidden, attn_dropout=attn_dp)
        for _ in 1:L
    ]...)

    # ── PMA ───────────────────────────────────────────────────────────
    pm = ac["pma"]
    n_seeds = Int(get(pm, "n_seeds", 1))
    n_heads_pma = Int(pm["n_heads"])
    pma = PMA(D, n_seeds, n_heads_pma; attn_dropout=attn_dp)

    # ── HPGe embedding ────────────────────────────────────────────────
    tail_cfg = ac["tail"]
    tail_act = resolve_activation(get(tail_cfg, "activation", "tanh"))
    tail_dp  = Float32(get(tail_cfg, "dropout", 0))
    tail_bn  = Bool(get(tail_cfg, "batch_norm", false))
    det_chain, dout = det_embedding(hc, tail_act; bn=tail_bn)

    # ── Tail (event vec ⊕ HPGe → logit) ───────────────────────────────
    tail_widths = Int.(tail_cfg["hidden_widths"])
    tail_dims = vcat(D + dout, tail_widths)
    tail_layers = Any[]
    for i in 1:(length(tail_dims) - 1)
        append!(tail_layers, dense_block(tail_dims[i], tail_dims[i+1], tail_act;
                                          dp=tail_dp, bn=tail_bn))
    end
    push!(tail_layers, Lux.Dense(tail_dims[end] => 1))
    tail = Lux.Chain(tail_layers...)

    model = FlatSetTransformer(embed, isab_stack, pma, det_chain, tail)

    layout = :flat_set
    hpge_feats = get(hc, "features", String[])
    @info "  Model: triggers F=$F + geom G=$G → embed(D=$D) → ISAB×$L (M=$n_inducing, h=$n_heads_isab)"
    @info "         → PMA(seeds=$n_seeds, h=$n_heads_pma) → (D=$D)"
    @info "         + HPGe$hpge_feats → embed=$dout → tail$tail_dims → 1"
    return model, layout
end

register_architecture!("set_transformer", _build_set_transformer)
