# DetConcatMLP — Two-branch MLP: SiPM prefix + HPGe embedding → concat → tail
#
# Architecture:
#   SiPM features ─→ [prefix MLP (optional)] ──┐
#                                                ├─ vcat ─→ [tail MLP] → Dense(→1) → logit
#   HPGe features ─→ [embedding MLP]  ──────────┘
#
# Uses shared building blocks from ml/layers/blocks.jl.

# ── Lux layer type ───────────────────────────────────────────────────────────

struct DetConcatMLP{P,D,C,T} <: Lux.AbstractLuxContainerLayer{(:prefix, :det, :cat, :tail)}
    prefix::P       # SiPM prefix branch (identity or Chain)
    det::D          # HPGe embedding branch (identity or Chain)
    cat::C          # VCat fusion layer
    tail::T         # shared tail MLP → logit
end

function Lux.apply(m::DetConcatMLP, (xs, xd)::Tuple, ps, st)
    hs, ss = Lux.apply(m.prefix, xs, ps.prefix, st.prefix)
    hd, sd = Lux.apply(m.det,    xd, ps.det,    st.det)
    hc, sc = Lux.apply(m.cat, (hs, hd), ps.cat, st.cat)
    out, s = Lux.apply(m.tail,   hc, ps.tail,   st.tail)
    return out, (prefix=ss, det=sd, cat=sc, tail=s)
end

# ── Builder ──────────────────────────────────────────────────────────────────

function _build_mlp_detector_concat(cfg::Dict, n_sipm::Int)
    ac = cfg["architecture"]
    ic = cfg["input"]
    sc = ic["sipm"]
    hc = get(ic, "hpge", Dict())

    nf  = length(sc["features"])
    act = resolve_activation(get(ac, "activation", "relu"))

    wr = ac["hidden_widths"]
    ws = wr isa Vector ? Int.(wr) : fill(Int(wr), 1)

    dp = Float32(get(ac, "dropout", 0))
    bn = Bool(get(ac, "batch_norm", false))

    pre_n  = clamp(Int(get(hc, "concat_after_layers", 0)), 0, length(ws))
    post_n = length(ws) - pre_n

    # ── SiPM prefix branch ───────────────────────────────────────────────
    pw = n_sipm * nf
    if pre_n > 0
        pre_dims = vcat(pw, ws[1:pre_n])
        prefix = mlp_block(pre_dims, act; dp, bn)
        pw = ws[pre_n]
    else
        prefix = Lux.WrappedFunction(identity)
    end

    # ── HPGe detector embedding ──────────────────────────────────────────
    det_branch, dout = det_embedding(hc, act; bn)

    # ── Tail: concat → hidden layers → logit(1) ─────────────────────────
    tail_dims = vcat(pw + dout, ws[pre_n+1:end])
    tail_layers = Any[]
    for i in 1:(length(tail_dims)-1)
        append!(tail_layers, dense_block(tail_dims[i], tail_dims[i+1], act; dp, bn))
    end
    push!(tail_layers, Lux.Dense(tail_dims[end] => 1))
    tail = Lux.Chain(tail_layers...)

    model = DetConcatMLP(
        prefix, det_branch,
        Lux.WrappedFunction(xs -> vcat(xs...)),
        tail)

    layout = Symbol(get(sc, "layout", "by_channel"))

    din = Bool(get(hc, "enabled", false)) ? length(hc["features"]) : 2
    @info "  Model: SiPM=$(n_sipm*nf)→prefix($pre_n layers)→$pw  |  Det=$(din)→embed→$dout  |  Concat=$(pw+dout)→tail($post_n layers)→1"
    @info "  Hidden widths: $(ws)  dropout=$(dp)  BN=$(bn)"
    return model, layout
end

register_architecture!("mlp_detector_concat", _build_mlp_detector_concat)
