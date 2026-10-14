# HierarchicalSetModel — Two-level set model: Triggers → SiPM → Event
#
# Architecture (per event):
#   Triggers (F × K × N × B)
#       └─ φ-MLP (shared weights, applied per-trigger via PerElementModel)
#                 → (E × K × N × B)
#       └─ masked pool over K (sum/max/mean — concat of selected ops)
#                 → (E·|ops_K| × N × B)
#       └─ vcat with per-SiPM geometry (G × N × B)
#                 → ((E·|ops_K| + G) × N × B)
#       └─ SiPM-MLP (shared weights, per-SiPM via PerElementModel)
#                 → (S × N × B)
#       └─ pool over N (mean/max — concat) — no mask, all SiPMs are real
#                 → (S·|ops_N| × B)
#       └─ vcat with HPGe embedding (D × B)
#       └─ tail MLP → Dense(→1) → logit
#
# Inputs (NamedTuple):
#   trig :: Float32 (F × max_K × n_sipm × B)
#   mask :: Bool    (max_K × n_sipm × B)
#   geom :: Float32 (G × n_sipm × B)
#   det  :: Float32 (D × B)

# ── Lux layer type ───────────────────────────────────────────────────────────

struct HierarchicalSetModel{P, S, D, T} <: Lux.AbstractLuxContainerLayer{
        (:phi, :sipm_mlp, :det, :tail)}
    phi::P                 # PerElementModel wrapping the φ-MLP
    pool_K_ops::Vector{Symbol}    # which masked-pool ops to apply over K
    sipm_mlp::S            # PerElementModel wrapping per-SiPM MLP
    pool_N_ops::Vector{Symbol}    # which (unmasked) pool ops to apply over N
    det::D                 # HPGe embedding chain
    tail::T                # tail MLP → logit
end

function Lux.apply(m::HierarchicalSetModel, inputs::NamedTuple, ps, st)
    trig = inputs.trig                                    # (F, K, N, B)
    mask = inputs.mask                                    # (K, N, B)
    geom = inputs.geom                                    # (G, N, B)
    det  = inputs.det                                     # (D, B)

    # 1) φ-MLP per trigger (broadcast over K, N, B)
    h, st_phi = Lux.apply(m.phi, trig, ps.phi, st.phi)    # (E, K, N, B)

    # 2) Pool over K with mask. Mask reshape so it broadcasts against (E,K,N,B).
    mask_b = reshape(mask, 1, size(mask)...)              # (1, K, N, B)
    pooled_K = masked_multi_pool(h, mask_b; ops=m.pool_K_ops, dims=2)  # (Eκ,N,B)

    # 3) Concat per-SiPM geometry along feature dim
    v = cat(pooled_K, geom; dims=1)                       # (Eκ+G, N, B)

    # 4) SiPM-level MLP (per-SiPM shared weights, broadcast over N, B)
    u, st_sm = Lux.apply(m.sipm_mlp, v, ps.sipm_mlp, st.sipm_mlp)  # (S, N, B)

    # 5) Pool over N (the n_sipm axis). All slots real → no mask needed.
    # Use a plain Bool array on the same device as `u` (BitArray would not
    # land on the GPU when `u` is a CuArray).
    m_all_N = similar(u, Bool, 1, size(u, 2), size(u, 3))
    fill!(m_all_N, true)
    pooled_N = masked_multi_pool(u, m_all_N; ops=m.pool_N_ops, dims=2)  # (Sν,B)

    # 6) HPGe embedding + concat + tail
    hd, st_d = Lux.apply(m.det,  det, ps.det,  st.det)
    e = vcat(pooled_N, hd)
    out, st_t = Lux.apply(m.tail, e, ps.tail, st.tail)

    new_st = (phi=st_phi, sipm_mlp=st_sm, det=st_d, tail=st_t)
    return out, new_st
end

# ── Builder ──────────────────────────────────────────────────────────────────

function _build_hierarchical_set(cfg::Dict, n_sipm::Int)
    ic = cfg["input"]
    ac = cfg["architecture"]
    hc = get(ic, "hpge", Dict())

    trig_cfg = ic["triggers"]
    F = length(trig_cfg["features"])                     # trigger feature dim
    G = length(get(ic["sipm"], "geometry", String[]))    # geometry feature dim

    # ── φ-MLP (per trigger) ────────────────────────────────────────────
    phi_cfg = ac["phi"]
    phi_act = resolve_activation(get(phi_cfg, "activation", "gelu"))
    phi_dp  = Float32(get(phi_cfg, "dropout",   0))
    phi_bn  = Bool(get(phi_cfg, "batch_norm", false))
    phi_widths = Int.(phi_cfg["hidden_widths"])
    phi_dims = vcat(F, phi_widths)
    phi_inner = mlp_block(phi_dims, phi_act; dp=phi_dp, bn=phi_bn)
    E = phi_dims[end]
    phi = PerElementModel(phi_inner)

    # ── Pool ops over K (triggers per SiPM) — applied as plain function call
    pool_K_cfg = ac["pooling"]
    pool_K_ops = Symbol[resolve_pool_op(o) for o in pool_K_cfg["ops"]]
    Eκ = E * length(pool_K_ops)

    # ── SiPM-MLP (per SiPM, after concat with geometry) ────────────────
    sm_cfg = ac["sipm_mlp"]
    sm_act = resolve_activation(get(sm_cfg, "activation", "gelu"))
    sm_dp  = Float32(get(sm_cfg, "dropout",   0))
    sm_bn  = Bool(get(sm_cfg, "batch_norm", false))
    sm_widths = Int.(sm_cfg["hidden_widths"])
    sm_dims = vcat(Eκ + G, sm_widths)
    sipm_inner = mlp_block(sm_dims, sm_act; dp=sm_dp, bn=sm_bn)
    S = sm_dims[end]
    sipm_mlp = PerElementModel(sipm_inner)

    # ── Pool over N (n_sipm) — no mask (all SiPMs are real)
    pool_N_cfg = ac["event_pooling"]
    pool_N_ops = Symbol[resolve_pool_op(o) for o in pool_N_cfg["ops"]]
    Sν = S * length(pool_N_ops)

    # ── HPGe embedding ─────────────────────────────────────────────────
    tail_cfg = ac["tail"]
    tail_act = resolve_activation(get(tail_cfg, "activation", "tanh"))
    tail_dp  = Float32(get(tail_cfg, "dropout",   0))
    tail_bn  = Bool(get(tail_cfg, "batch_norm", false))
    det_chain, dout = det_embedding(hc, tail_act; bn=tail_bn)

    # ── Tail (event vec ⊕ HPGe → logit) ────────────────────────────────
    tail_widths = Int.(tail_cfg["hidden_widths"])
    tail_dims = vcat(Sν + dout, tail_widths)
    tail_layers = Any[]
    for i in 1:(length(tail_dims)-1)
        append!(tail_layers, dense_block(tail_dims[i], tail_dims[i+1], tail_act;
                                          dp=tail_dp, bn=tail_bn))
    end
    push!(tail_layers, Lux.Dense(tail_dims[end] => 1))
    tail = Lux.Chain(tail_layers...)

    model = HierarchicalSetModel(phi, pool_K_ops, sipm_mlp, pool_N_ops,
                                  det_chain, tail)

    layout = :hierarchical
    @info "  Model: triggers F=$F → φ$phi_dims → poolK($(pool_K_ops))=$Eκ"
    @info "         + geom G=$G → SiPM-MLP$sm_dims → poolN($(pool_N_ops))=$Sν"
    @info "         + HPGe$(get(hc,"features",[])) → embed=$dout → tail$tail_dims → 1"
    return model, layout
end

register_architecture!("hierarchical_set", _build_hierarchical_set)
