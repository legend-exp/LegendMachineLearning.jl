# Integrated Gradients attribution for SiPM inputs of any model whose
# `Lux.apply(model, ::NamedTuple, ps, st)` exposes `inputs.sipm` and `inputs.det`.
#
# Provides:
#   integrated_gradients(model, ps, st, x_sipm, x_det; ...) → Matrix{Float32}
#   aggregate_sipm_ig(ig, n_sipm; mode, n_feat) → reshape per-SiPM
#   group_events_by_hpge(x_det, hpge_yaml_path; tol) → per-HPGe event indices
#   attribution_heatmap_for_subset(model, ps, st, inputs, subset, n_sipm, hpge_yaml; ...)
#
# Design notes:
#   - Logit attribution (no σ) — matches BCE-trained classifier convention.
#   - HPGe input is held fixed, only SiPM input is interpolated from baseline.
#   - Midpoint rule: α_k = (k − 0.5)/n_steps. More accurate than left-Riemann
#     at low n_steps; matches what TF/Captum default to.
#   - CPU only. The val split is small enough; avoids cuDNN-disabled GPU path.

using YAML
using Statistics: mean

"""
    integrated_gradients(model, ps, st, x_sipm, x_det;
                         baseline_sipm = zeros(Float32, size(x_sipm)),
                         n_steps = 32, batch_size = 256) → Matrix{Float32}

Per-feature, per-event Integrated Gradients of the model logit w.r.t. the SiPM
input. Returns a `(n_features × n_events)` matrix with the same shape as
`x_sipm`. The HPGe input `x_det` is held fixed throughout the integration —
this conditions the explanation on the per-event HPGe identity.
"""
function integrated_gradients(model, ps, st,
                              x_sipm::AbstractMatrix{Float32},
                              x_det::AbstractMatrix{Float32};
                              baseline_sipm::AbstractMatrix{Float32} = fill!(similar(x_sipm), 0f0),
                              n_steps::Int = 32,
                              batch_size::Int = 256)
    size(x_sipm, 2) == size(x_det, 2) ||
        error("integrated_gradients: x_sipm has $(size(x_sipm,2)) cols, x_det has $(size(x_det,2))")
    size(baseline_sipm) == size(x_sipm) ||
        error("integrated_gradients: baseline_sipm shape $(size(baseline_sipm)) ≠ x_sipm $(size(x_sipm))")
    n_steps >= 1 || error("integrated_gradients: n_steps must be ≥ 1")

    st_eval = Lux.testmode(st)
    nF, B = size(x_sipm)
    ig = similar(x_sipm, Float32, nF, B)        # device-matched (CPU Array or CuArray)

    αs = Float32.([(k - 0.5) / n_steps for k in 1:n_steps])

    for i in 1:batch_size:B
        j = min(i + batch_size - 1, B)
        idxs = i:j

        xs = x_sipm[:, idxs]
        bs = baseline_sipm[:, idxs]
        xd = x_det[:, idxs]
        diff = xs .- bs

        grad_sum = fill!(similar(xs), 0f0)       # device-matched
        for α in αs
            xα = bs .+ α .* diff
            loss(z) = sum(first(Lux.apply(model, (sipm = z, det = xd), ps, st_eval)))
            grad_sum .+= Zygote.gradient(loss, xα)[1]
        end
        ig[:, idxs] .= diff .* grad_sum ./ Float32(n_steps)
    end
    return ig
end

"""
    aggregate_sipm_ig(ig, n_sipm; mode = :sum_abs, n_feat = 3)

Reshape a `(n_sipm * n_feat × B)` IG matrix laid out `by_channel` into a
per-SiPM aggregate. Layout: row `(s−1)*n_feat + f` corresponds to SiPM `s`,
feature `f` (matches `assemble_sipm_matrix` with `layout = :by_channel`).

Modes:
  - `:sum_abs`  → `Matrix{Float32}(n_sipm × B)`,         cell = Σ_f |IG|
  - `:per_feat` → `Array{Float32,3}(n_sipm × n_feat × B)`, cell = |IG|
                  (per-feature absolute attribution; positive in both modes)
"""
function aggregate_sipm_ig(ig::AbstractMatrix{Float32}, n_sipm::Int;
                           mode::Symbol = :sum_abs, n_feat::Int = 3)
    nF, B = size(ig)
    nF == n_sipm * n_feat ||
        error("aggregate_sipm_ig: ig has $nF rows but expected $(n_sipm*n_feat) " *
              "for n_sipm=$n_sipm, n_feat=$n_feat")

    # by_channel layout: row (s−1)*n_feat + f → reshape (nF, B) → (n_feat, n_sipm, B).
    # Vectorized so this works for both Array and CuArray.
    ig_3d = reshape(ig, n_feat, n_sipm, B)

    if mode === :sum_abs
        return dropdims(sum(abs.(ig_3d); dims = 1); dims = 1)         # (n_sipm, B)
    elseif mode === :per_feat
        return permutedims(abs.(ig_3d), (2, 1, 3))                    # (n_sipm, n_feat, B)
    else
        error("aggregate_sipm_ig: unknown mode $mode (use :sum_abs or :per_feat)")
    end
end

"""
    group_events_by_hpge(x_det, hpge_yaml_path; tol = 1f-5) → NamedTuple

Group event indices by HPGe identity. Two events sharing the same
`(scaled_angle, scaled_z_center)` tuple (within `tol`) come from the same
HPGe detector. The YAML provides the detector name, string id, and
position-in-string for each (scaled_angle, scaled_z_center) pair.

Returned NamedTuple is sorted by `(string_id, position_in_string)`, with
unmatched groups (no entry in the YAML within `tol`) appended at the end:
  - keys::Vector{Tuple{Float32,Float32}}
  - names::Vector{String}             — `"unknown(sa, sz)"` for no-match
  - idxs_for_key::Vector{Vector{Int}}
  - string_ids::Vector{Int}           — 0 for unknowns
  - positions::Vector{Int}            — 0 for unknowns
"""
function group_events_by_hpge(x_det::AbstractMatrix{Float32}, hpge_yaml_path::String;
                              tol::Float32 = 1f-5)
    size(x_det, 1) == 2 ||
        error("group_events_by_hpge: x_det must have 2 rows (scaled_angle, scaled_z_center), got $(size(x_det,1))")
    isfile(hpge_yaml_path) || error("group_events_by_hpge: file not found: $hpge_yaml_path")

    raw = YAML.load_file(hpge_yaml_path; dicttype=Dict{String,Any})

    # YAML catalog: (sa, sz, name, string_id, position_in_string)
    catalog = Tuple{Float32, Float32, String, Int, Int}[]
    for (name, entry) in raw
        entry isa Dict || continue
        haskey(entry, "detector_position") || continue
        dp = entry["detector_position"]
        haskey(dp, "scaled_angle") && haskey(dp, "scaled_z_center") || continue
        sa = Float32(dp["scaled_angle"])
        sz = Float32(dp["scaled_z_center"])
        sid = Int(get(dp, "string_id", 0))
        pos = Int(get(dp, "position_in_string", 0))
        push!(catalog, (sa, sz, String(name), sid, pos))
    end

    # Bucket events by quantized (sa, sz)
    quant(x) = round(Int, x / tol)
    buckets = Dict{Tuple{Int,Int}, Vector{Int}}()
    for e in axes(x_det, 2)
        key = (quant(x_det[1, e]), quant(x_det[2, e]))
        push!(get!(buckets, key, Int[]), e)
    end

    # Resolve each bucket against the catalog
    keys_out = Tuple{Float32,Float32}[]
    names_out = String[]
    idxs_out = Vector{Int}[]
    sids_out = Int[]
    pos_out = Int[]

    for (_, ev_idxs) in buckets
        e0 = ev_idxs[1]
        sa = x_det[1, e0]; sz = x_det[2, e0]
        match_name = ""
        match_sid = 0
        match_pos = 0
        for (csa, csz, cname, csid, cpos) in catalog
            if abs(csa - sa) < tol && abs(csz - sz) < tol
                match_name = cname
                match_sid = csid
                match_pos = cpos
                break
            end
        end
        if isempty(match_name)
            match_name = "unknown($(round(sa; digits=4)),$(round(sz; digits=4)))"
        end
        push!(keys_out, (sa, sz))
        push!(names_out, match_name)
        push!(idxs_out, ev_idxs)
        push!(sids_out, match_sid)
        push!(pos_out, match_pos)
    end

    # Sort by (string_id, position_in_string); unknowns (sid=0) at the end.
    sort_key(i) = sids_out[i] == 0 ? (typemax(Int), pos_out[i]) : (sids_out[i], pos_out[i])
    perm = sortperm(collect(eachindex(keys_out)); by = sort_key)

    return (
        keys = keys_out[perm],
        names = names_out[perm],
        idxs_for_key = idxs_out[perm],
        string_ids = sids_out[perm],
        positions = pos_out[perm],
    )
end

"""
    attribution_heatmap_for_subset(model, ps, st, inputs, subset_idxs, n_sipm, hpge_yaml_path;
                                   n_steps=32, batch_size=256, per_feat=false) → NamedTuple

End-to-end: slice `inputs` to a subset (e.g., one event class), compute
Integrated Gradients on every event in the subset, group events by HPGe,
and average per-SiPM `|IG|` to produce an `n_hpge × n_sipm` heatmap matrix.

`inputs` must have `.sipm::Matrix{Float32}(n_features × n_total)` and
`.det::Matrix{Float32}(n_det_features × n_total)` (the output of
`assemble_features`).

Returned NamedTuple:
  - matrix          :: Matrix{Float32}(n_hpge × n_sipm)
  - matrix_per_feat :: Array{Float32,3}(n_hpge × n_sipm × n_feat) | nothing
  - hpge_keys, hpge_names, hpge_strings, hpge_pos, n_per_hpge

Rows are ordered as in `group_events_by_hpge`.
"""
function attribution_heatmap_for_subset(model, ps, st, inputs::NamedTuple,
                                        subset_idxs::AbstractVector{<:Integer},
                                        n_sipm::Int, hpge_yaml_path::String;
                                        n_steps::Int = 32, batch_size::Int = 256,
                                        per_feat::Bool = false,
                                        n_feat::Int = 3)
    haskey(inputs, :sipm) && haskey(inputs, :det) ||
        error("attribution_heatmap_for_subset: inputs must contain :sipm and :det")
    isempty(subset_idxs) && error("attribution_heatmap_for_subset: empty subset")

    x_sipm_sub = inputs.sipm[:, subset_idxs]
    x_det_sub  = inputs.det[:,  subset_idxs]

    @info @sprintf("  IG: %d events × %d steps × %d features", length(subset_idxs), n_steps, size(x_sipm_sub, 1))
    ig = integrated_gradients(model, ps, st, x_sipm_sub, x_det_sub;
                              n_steps = n_steps, batch_size = batch_size)

    # Move IG + det back to host: grouping uses scalar indexing, and the per-
    # row mean assignment into the host matrix `M` needs CPU values.
    ig_h    = Array(ig)
    x_det_h = Array(x_det_sub)

    abs_per_sipm = aggregate_sipm_ig(ig_h, n_sipm; mode = :sum_abs, n_feat = n_feat)
    pf = per_feat ? aggregate_sipm_ig(ig_h, n_sipm; mode = :per_feat, n_feat = n_feat) : nothing

    grp = group_events_by_hpge(x_det_h, hpge_yaml_path)
    H = length(grp.keys)
    M = zeros(Float32, H, n_sipm)
    n_per = zeros(Int, H)
    M_pf = per_feat ? zeros(Float32, H, n_sipm, n_feat) : nothing

    for (h, ev) in enumerate(grp.idxs_for_key)
        isempty(ev) && continue
        n_per[h] = length(ev)
        M[h, :] = vec(mean(view(abs_per_sipm, :, ev); dims = 2))
        if per_feat
            for f in 1:n_feat
                M_pf[h, :, f] = vec(mean(view(pf, :, f, ev); dims = 2))
            end
        end
    end

    return (
        matrix = M,
        matrix_per_feat = M_pf,
        hpge_keys = grp.keys,
        hpge_names = grp.names,
        hpge_strings = grp.string_ids,
        hpge_pos = grp.positions,
        n_per_hpge = n_per,
        idxs_for_key = grp.idxs_for_key,  # 1-based into the subset (NOT the full dataset)
    )
end

"""
    ig_completeness_residual(model, ps, st, x_sipm, x_det; baseline_sipm, n_steps, batch_size)

Diagnostic: compute the relative residual of the IG completeness axiom
`Σ_i IG_i ≈ F(x) − F(baseline)`. Returns `(mean_rel_resid::Float32, n::Int)`.
For a correct implementation with sufficient `n_steps` this should be ≪ 1%.
"""
function ig_completeness_residual(model, ps, st,
                                  x_sipm::AbstractMatrix{Float32},
                                  x_det::AbstractMatrix{Float32};
                                  baseline_sipm::AbstractMatrix{Float32} = fill!(similar(x_sipm), 0f0),
                                  n_steps::Int = 32,
                                  batch_size::Int = 256)
    st_eval = Lux.testmode(st)
    f_x  = Array(vec(first(Lux.apply(model, (sipm = x_sipm,        det = x_det), ps, st_eval))))
    f_b  = Array(vec(first(Lux.apply(model, (sipm = baseline_sipm, det = x_det), ps, st_eval))))
    Δ    = f_x .- f_b
    ig   = integrated_gradients(model, ps, st, x_sipm, x_det;
                                baseline_sipm = baseline_sipm,
                                n_steps = n_steps, batch_size = batch_size)
    sum_ig = Array(vec(sum(ig; dims = 1)))
    denom  = mean(abs.(Δ))
    denom <= 0 && return (0f0, length(Δ))
    return (Float32(mean(abs.(sum_ig .- Δ)) / denom), length(Δ))
end
