# Feature loading + assembly — config-driven, shared between training + prediction
#
# Provides:
#   - load_training_split(path, cfg) — load LH5 split, columns determined by config
#   - load_prediction_tier(path, tier_key, cfg) — load LH5 tier for prediction
#   - assemble_features(data, cfg, sipm_perm) → (x_sipm, x_det, y)
#   - build_sipm_ordering(sids, geometry_base, group_name) → (perm, ordered_names)
#   - resolve_input_features(cfg) — extract input_features from config
#
# Config layout (unwrapped — no top-level architecture key):
#   input:
#     sipm:
#       features: [sipm_pe_sums_prompt_scaled, ...]
#       layout: by_channel
#       ordering: {enabled: true, groups: [...]}
#     hpge:
#       enabled: true
#       features: [ged_angle_norm, ged_z_center_norm]

# ══════════════════════════════════════════════════════════════════════════════
# Input feature resolution
# ══════════════════════════════════════════════════════════════════════════════

# Feature names in config must match LH5 column names exactly — no mapping needed.

"""
    resolve_input_features(cfg::Dict) → (sipm_features, det_features)

Extract input feature lists from unwrapped config (no top-level architecture key).
Reads `cfg["input"]["sipm"]["features"]` and `cfg["input"]["hpge"]["features"]`.
"""
function resolve_input_features(cfg::Dict)
    ic = cfg["input"]
    sc = ic["sipm"]
    hc = get(ic, "hpge", Dict())

    sipm_feats = String.(sc["features"])
    isempty(sipm_feats) && error("input.sipm.features must not be empty")

    det_feats = if Bool(get(hc, "enabled", false)) && haskey(hc, "features")
        String.(hc["features"])
    else
        String[]
    end

    return sipm_feats, det_feats
end

# ══════════════════════════════════════════════════════════════════════════════
# VoV → Matrix helper
# ══════════════════════════════════════════════════════════════════════════════

"""Convert a VoV column from LH5 table into a Float32 matrix (n_events × n_channels)."""
function _vov_to_matrix(col, n::Int, ns::Int)
    M = Matrix{Float32}(undef, n, ns)
    @inbounds for i in 1:n, j in 1:ns
        M[i, j] = Float32(col[i][j])
    end
    return M
end

# ══════════════════════════════════════════════════════════════════════════════
# Data loading — training splits
# ══════════════════════════════════════════════════════════════════════════════

"""
    load_training_split(path::String, cfg::Dict) → NamedTuple

Load a single jlnormml LH5 split. Columns loaded are determined by the
`input_features` config block (or `input_mode` fallback).

Returns a NamedTuple with:
  - sipm :: Dict{String, Matrix{Float32}}  — sipm feature matrices (n × n_channels)
  - det  :: Dict{String, Vector{Float32}}  — detector feature vectors (n,)
  - y    :: Vector{Float32}                — labels
  - sids :: Vector{UInt32}                 — SiPM detector IDs
  - n    :: Int                            — number of events
"""
function load_training_split(path::String, cfg::Dict)
    sipm_feats, det_feats = resolve_input_features(cfg)

    tbl  = lh5open(path, "r") do f; f["jlnormml"][:]; end
    sids = lh5open(path, "r") do f; f["sipm_detector_ids"][:]; end
    n  = length(tbl)
    ns = length(first(getproperty(tbl, Symbol(first(sipm_feats)))))

    # Load SiPM features (VoV → Matrix)
    sipm = Dict{String, Matrix{Float32}}()
    for feat in sipm_feats
        col = Symbol(feat)
        hasproperty(tbl, col) || error("Column :$col not found in $path (feature: $feat)")
        sipm[feat] = _vov_to_matrix(getproperty(tbl, col), n, ns)
    end

    # Load detector features (scalar columns)
    det = Dict{String, Vector{Float32}}()
    for feat in det_feats
        col = Symbol(feat)
        hasproperty(tbl, col) || error("Column :$col not found in $path (feature: $feat)")
        det[feat] = Float32.(getproperty(tbl, col))
    end

    return (sipm = sipm, det = det, y = Float32.(tbl.label),
            sids = Vector{UInt32}(sids), n = n)
end

# ══════════════════════════════════════════════════════════════════════════════
# Data loading — prediction tier (jlnorm / jlnormml)
# ══════════════════════════════════════════════════════════════════════════════

"""
    load_prediction_tier(path::String, tier_key::String, cfg::Dict) → NamedTuple

Load an LH5 file for prediction. Loads input features from config plus
prediction-specific columns (event_sum_pe, event_multiplicity, energy, etc.).
"""
function load_prediction_tier(path::String, tier_key::String, cfg::Dict)
    sipm_feats, det_feats = resolve_input_features(cfg)

    tbl  = lh5open(path, "r") do f; f[tier_key][:]; end
    sids = lh5open(path, "r") do f; f["sipm_detector_ids"][:]; end
    n  = length(tbl)
    ns = length(first(getproperty(tbl, Symbol(first(sipm_feats)))))

    sipm = Dict{String, Matrix{Float32}}()
    for feat in sipm_feats
        col = Symbol(feat)
        hasproperty(tbl, col) || error("Column :$col not found in $path (feature: $feat)")
        sipm[feat] = _vov_to_matrix(getproperty(tbl, col), n, ns)
    end

    det = Dict{String, Vector{Float32}}()
    for feat in det_feats
        col = Symbol(feat)
        hasproperty(tbl, col) || error("Column :$col not found in $path (feature: $feat)")
        det[feat] = Float32.(getproperty(tbl, col))
    end

    has_label = hasproperty(tbl, :label)
    y = has_label ? Float32.(tbl.label) : fill(NaN32, n)

    has_energy = hasproperty(tbl, :ged_energy_keV)
    energy = has_energy ? Float32.(tbl.ged_energy_keV) : nothing

    # Also load pe/cos unconditionally for write_pred_lh5 (full-window columns)
    pe_full = if hasproperty(tbl, :sipm_pe_sums_scaled)
        _vov_to_matrix(tbl.sipm_pe_sums_scaled, n, ns)
    else
        get(sipm, "sipm_pe_sums_scaled", nothing)
    end
    cos_full = if hasproperty(tbl, :cos_proximity)
        _vov_to_matrix(tbl.cos_proximity, n, ns)
    else
        get(sipm, "cos_proximity", nothing)
    end

    # Prompt/delayed for write-back
    pe_prompt  = get(sipm, "sipm_pe_sums_prompt_scaled", nothing)
    pe_delayed = get(sipm, "sipm_pe_sums_delayed_scaled", nothing)

    return (sipm = sipm, det = det, y = y, sids = Vector{UInt32}(sids), n = n,
            has_label = has_label, ged_energy_keV = energy,
            event_sum_pe       = Float32.(tbl.event_sum_pe),
            event_multiplicity = Int32.(tbl.event_multiplicity),
            pe = pe_full, cos = cos_full,
            ga = collect(values(det))[1],  # first det feature for backwards compat
            gr = collect(values(det))[length(det)],  # second det feature
            pe_prompt = pe_prompt, pe_delayed = pe_delayed,
            event_sum_pe_prompt       = hasproperty(tbl, :event_sum_pe_prompt)       ? Float32.(tbl.event_sum_pe_prompt)       : nothing,
            event_multiplicity_prompt = hasproperty(tbl, :event_multiplicity_prompt) ? Int32.(tbl.event_multiplicity_prompt) : nothing,
            event_sum_pe_delayed       = hasproperty(tbl, :event_sum_pe_delayed)       ? Float32.(tbl.event_sum_pe_delayed)       : nothing,
            event_multiplicity_delayed = hasproperty(tbl, :event_multiplicity_delayed) ? Int32.(tbl.event_multiplicity_delayed) : nothing,
    )
end

# ══════════════════════════════════════════════════════════════════════════════
# Feature assembly
# ══════════════════════════════════════════════════════════════════════════════

"""
    assemble_sipm_matrix(data, cfg, sipm_perm) → Matrix{Float32} (features × samples)

Build the SiPM input matrix from loaded data + config.
Columns are interleaved per-channel (by_channel layout) or stacked (by_feature).
"""
function assemble_sipm_matrix(data::NamedTuple, cfg::Dict,
                              sipm_perm::Union{Vector{Int}, Nothing})
    sipm_feats, _ = resolve_input_features(cfg)
    layout = Symbol(get(get(cfg, "input", Dict())["sipm"], "layout", "by_channel"))

    matrices = Matrix{Float32}[]
    for feat in sipm_feats
        M = data.sipm[feat]
        if sipm_perm !== nothing
            M = M[:, sipm_perm]
        end
        push!(matrices, M)
    end

    n  = size(matrices[1], 1)
    ns = size(matrices[1], 2)
    nf = length(matrices)

    out = Matrix{Float32}(undef, n, nf * ns)
    if layout == :by_channel
        @inbounds for j in 1:ns, (fi, M) in enumerate(matrices)
            out[:, nf*(j-1) + fi] = @view M[:, j]
        end
    else  # by_feature
        @inbounds for (fi, M) in enumerate(matrices)
            out[:, (fi-1)*ns+1 : fi*ns] = M
        end
    end

    return permutedims(out)  # → (features × samples) for Lux
end

"""
    assemble_det_matrix(data, cfg) → Matrix{Float32} (n_det_features × samples)

Build the detector feature matrix from loaded data + config.
"""
function assemble_det_matrix(data::NamedTuple, cfg::Dict)
    _, det_feats = resolve_input_features(cfg)
    n = data.n
    nd = length(det_feats)
    out = Matrix{Float32}(undef, nd, n)
    for (i, feat) in enumerate(det_feats)
        out[i, :] = data.det[feat]
    end
    return out
end

"""
    assemble_features(data, cfg, sipm_perm) → (x_sipm, x_det)

Convenience wrapper: build both SiPM and detector feature matrices.
"""
function assemble_features(data::NamedTuple, cfg::Dict,
                           sipm_perm::Union{Vector{Int}, Nothing})
    return assemble_sipm_matrix(data, cfg, sipm_perm), assemble_det_matrix(data, cfg)
end

# ══════════════════════════════════════════════════════════════════════════════
# SiPM ordering
# ══════════════════════════════════════════════════════════════════════════════

"""
    build_sipm_ordering(sids, geometry_base, group_name) → (perm, ordered_names)

Build SiPM column permutation: IB_top → IB_bottom → OB_top → OB_bottom, ascending angle.
"""
function build_sipm_ordering(sids::Vector{UInt32}, geometry_base::String, group_name::String)
    path = joinpath(geometry_base, "SiPM", "$(group_name)_sipm_geometry.yaml")
    isfile(path) || error("SiPM geometry not found: $path")
    raw = YAML.load_file(path; dicttype=Dict{String,Any})

    GROUP_RANK = Dict("IB_top"=>1, "IB_bottom"=>2, "OB_top"=>3, "OB_bottom"=>4)
    names = [string(DetectorId(id)) for id in sids]

    sort_keys = map(names) do nm
        e = get(raw, nm, nothing)
        (e === nothing || !haskey(e, "detector_position")) && return (99, 0.0)
        p = e["detector_position"]
        grp = "$(p["barrel"])_$(p["position"])"
        (get(GROUP_RANK, grp, 99), Float64(p["cylind_coords"]["angle"]["value"]))
    end

    perm = sortperm(sort_keys)

    for (rank, gname_str) in enumerate(["IB_top", "IB_bottom", "OB_top", "OB_bottom"])
        cnt = count(k -> k[1] == rank, sort_keys)
        @info "    $gname_str: $cnt SiPMs"
    end
    return perm, names[perm]
end
