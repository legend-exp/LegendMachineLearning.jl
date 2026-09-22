# Variable-length set helpers — apply per-element MLPs over a non-batch axis,
# pad ragged sequences into dense tensors, and propagate masks.
#
# Used by HierarchicalSetModel (φ-MLP over K triggers per SiPM) and
# FlatSetTransformer (future).
#
# Provides:
#   pad_ragged(vov, max_K, n_sipm; pad_value)       — VoV → dense tensor + mask
#   apply_per_element(model, x; element_dims)       — broadcast a model over inner dims
#   PerElementChain(inner; element_dims)            — Lux-compatible wrapper
#
# Convention for trigger tensors: (F × K × N × B) where
#   F = trigger features, K = max triggers, N = n_sipm, B = batch.
# A φ-MLP that maps F → E features must run on every (k,n,b) slot,
# i.e. broadcast over (K,N,B). We do it by reshaping to a 2D matrix
# (F × K·N·B), running the MLP once, then reshaping back.

# ── Per-element model wrapper ────────────────────────────────────────────────

"""
    PerElementModel(inner)

Lux layer wrapper that flattens dims 2:end into the batch dim, calls `inner`,
and reshapes back. `inner` should be a normal Lux model that consumes
`(F × N_total)` and returns `(E × N_total)`.

Caveat: avoid putting `Lux.BatchNorm` inside `inner`. The cuDNN BN inference
kernel fails (`CUDNN_STATUS_NOT_SUPPORTED`) when `N_total = ∏(dims 2:end)` is
large (≳ 200k). This happens because each per-SiPM call sees `n_sipm ·
batch_size` "samples". Use no normalization or `LayerNorm` (which has no
running stats and avoids the cuDNN path) inside `inner`.
"""
struct PerElementModel{M} <: Lux.AbstractLuxWrapperLayer{:inner}
    inner::M
end

function Lux.apply(m::PerElementModel, x::AbstractArray, ps, st)
    sz   = size(x)              # (F, d2, d3, ..., dN)
    F    = sz[1]
    rest = sz[2:end]
    N_total = prod(rest)
    x2d  = reshape(x, F, N_total)
    y2d, new_st = Lux.apply(m.inner, x2d, ps, st)
    E    = size(y2d, 1)
    return reshape(y2d, E, rest...), new_st
end

# ── Ragged → dense padding ───────────────────────────────────────────────────

"""
    pad_per_sipm_triggers(trig_feats, trig_det_ids, max_K, n_sipm;
                          pad_value=0f0) → (Array{Float32,4}, Array{Bool,3})

Build a dense per-SiPM trigger tensor from per-event ragged vectors.

Inputs (all length n_events, each event has K_e ragged triggers):
  trig_feats     :: Vector of NamedTuple, where each field is
                    Vector{Vector{Float32}} of length n_events. Each inner
                    Vector{Float32} has length K_e (one entry per trigger).
                    Equivalently a Vector{Tuple{Vector{Float32}, ...}} per event.
  trig_det_ids   :: Vector{Vector{Int}} (or UInt32) — slot index in 1:n_sipm
                    per trigger (already mapped from DetectorId via sipm_perm).

Output:
  X :: Float32 (F × max_K × n_sipm × n_events) — F = number of features
  M :: Bool    (max_K × n_sipm × n_events)     — true where a real trigger sits

Triggers are inserted in their input order; the (max_K+1)-th and beyond are
truncated. Multiple triggers of the same SiPM in one event get adjacent slots
along the K-axis.
"""
function pad_per_sipm_triggers(
        feature_vectors::NTuple{F, <:AbstractVector{<:AbstractVector{<:Real}}},
        trig_det_slots::AbstractVector{<:AbstractVector{<:Integer}},
        max_K::Int, n_sipm::Int; pad_value::Real = 0f0,
    ) where {F}

    n_events = length(trig_det_slots)
    @assert all(length.(feature_vectors) .== n_events) "feature vectors must all have $n_events events"

    X = fill(Float32(pad_value), F, max_K, n_sipm, n_events)
    # Plain Bool array (not BitArray) — `dev_fn`/`fmap` only descends into
    # AbstractArray{T} where T is a primitive bits type that has a CuArray
    # equivalent. BitArrays stay on CPU and break GPU broadcasts.
    M = fill(false, max_K, n_sipm, n_events)

    # next-slot pointer per (sipm, event) — reused across the inner loop below
    next_slot = Vector{Int}(undef, n_sipm)

    @inbounds for e in 1:n_events
        slots = trig_det_slots[e]
        K_e   = length(slots)
        K_e == 0 && continue

        # reset slot pointer per event
        fill!(next_slot, 1)

        for t in 1:K_e
            s = Int(slots[t])
            (s < 1 || s > n_sipm) && continue
            k = next_slot[s]
            k > max_K && continue              # truncate excess triggers
            for f in 1:F
                X[f, k, s, e] = Float32(feature_vectors[f][e][t])
            end
            M[k, s, e] = true
            next_slot[s] = k + 1
        end
    end

    return X, M
end

"""
    detid_to_slot(trig_det_ids, sids, sipm_perm) → Vector{Vector{Int}}

Map raw SiPM detector IDs in `trig_det_ids` to slot indices 1..length(sids).
If `sipm_perm` is given, the slot is permuted to match the model's reordered
SiPM axis. Triggers whose detector ID is not in `sids` get slot 0 (skipped at
padding time).
"""
function detid_to_slot(trig_det_ids::AbstractVector{<:AbstractVector{<:Integer}},
                       sids::AbstractVector,
                       sipm_perm::Union{Vector{Int}, Nothing})
    # Build inverse permutation: original_slot → new_slot
    # If sipm_perm == nothing we use identity.
    n_sipm = length(sids)
    inv = if sipm_perm === nothing
        collect(1:n_sipm)
    else
        ip = Vector{Int}(undef, n_sipm)
        @inbounds for (new_pos, old_pos) in enumerate(sipm_perm)
            ip[old_pos] = new_pos
        end
        ip
    end
    # detid → original_slot (1-based)
    id_to_orig = Dict{UInt32, Int}()
    @inbounds for (i, d) in enumerate(sids)
        id_to_orig[UInt32(d)] = i
    end

    out = Vector{Vector{Int}}(undef, length(trig_det_ids))
    @inbounds for (e, ids) in enumerate(trig_det_ids)
        slots = Vector{Int}(undef, length(ids))
        for t in eachindex(ids)
            orig = get(id_to_orig, UInt32(ids[t]), 0)
            slots[t] = orig == 0 ? 0 : inv[orig]
        end
        out[e] = slots
    end
    return out
end
