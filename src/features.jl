# Feature loading + assembly — config-driven, shared between training + prediction
#
# Provides:
#   - load_training_split(path, cfg) — load LH5 split, columns determined by config
#   - load_prediction_tier(path, tier_key, cfg) — load LH5 tier for prediction
#   - assemble_features(data, cfg, sipm_perm) → NamedTuple of input arrays
#   - build_sipm_ordering(sids, geometry_base, group_name) → (perm, ordered_names)
#   - resolve_input_features(cfg) — extract input_features from config
#   - resolve_geometry_features(cfg) / resolve_trigger_config(cfg)
#
# Config layout (unwrapped — no top-level architecture key):
#   input:
#     sipm:
#       features: [sipm_pe_sums_prompt_scaled, ...]   # MLP path (per-SiPM scalars)
#       geometry: [cos_proximity, scaled_delta_z]     # set-model path (per-SiPM geometry)
#       layout: by_channel
#       ordering: {enabled: true, groups: [...]}
#     triggers:                                        # set-model path
#       features: [trig_time_scaled, trig_pe_scaled]
#       max_per_sipm: 16
#       pad_value: 0.0
#     hpge:
#       enabled: true
#       features: [scaled_angle, scaled_z_center]

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

    sipm_feats = String.(get(sc, "features", String[]))

    det_feats = if Bool(get(hc, "enabled", false)) && haskey(hc, "features")
        String.(hc["features"])
    else
        String[]
    end

    return sipm_feats, det_feats
end

"""
    resolve_geometry_features(cfg::Dict) → Vector{String}

Per-SiPM geometry features (e.g. `cos_proximity`, `scaled_delta_z`) consumed
by set-style models. Empty if not configured.
"""
function resolve_geometry_features(cfg::Dict)
    sc = cfg["input"]["sipm"]
    String.(get(sc, "geometry", String[]))
end

"""
    resolve_trigger_config(cfg::Dict) → NamedTuple or nothing

Trigger-sequence config block. Returns `nothing` if not present.
"""
function resolve_trigger_config(cfg::Dict)
    haskey(cfg["input"], "triggers") || return nothing
    tc = cfg["input"]["triggers"]
    feats = String.(get(tc, "features", ["trig_time_scaled", "trig_pe_scaled"]))
    return (
        features    = feats,
        max_per_sipm = Int(get(tc, "max_per_sipm", 16)),
        pad_value   = Float32(get(tc, "pad_value", 0.0)),
    )
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
    geom_feats = resolve_geometry_features(cfg)
    trig_cfg   = resolve_trigger_config(cfg)

    tbl  = lh5open(path, "r") do f; f["jlnormml"][:]; end
    sids = lh5open(path, "r") do f; f["sipm_detector_ids"][:]; end
    n  = length(tbl)
    # Determine per-SiPM count from any per-SiPM column (sipm_feats or geom_feats)
    probe_feat = !isempty(sipm_feats) ? first(sipm_feats) :
                 !isempty(geom_feats) ? first(geom_feats) :
                 error("config has neither input.sipm.features nor input.sipm.geometry")
    ns = length(first(getproperty(tbl, Symbol(probe_feat))))

    # Load SiPM scalar features (per-event vectors of length n_sipm → Matrix)
    sipm = Dict{String, Matrix{Float32}}()
    for feat in sipm_feats
        col = Symbol(feat)
        hasproperty(tbl, col) || error("Column :$col not found in $path (feature: $feat)")
        sipm[feat] = _vov_to_matrix(getproperty(tbl, col), n, ns)
    end

    # Load per-SiPM geometry features (set-model branch)
    geom = Dict{String, Matrix{Float32}}()
    for feat in geom_feats
        col = Symbol(feat)
        hasproperty(tbl, col) || error("Column :$col not found in $path (geometry: $feat)")
        geom[feat] = _vov_to_matrix(getproperty(tbl, col), n, ns)
    end

    # Load detector features (per-event scalar columns)
    det = Dict{String, Vector{Float32}}()
    for feat in det_feats
        col = Symbol(feat)
        hasproperty(tbl, col) || error("Column :$col not found in $path (feature: $feat)")
        det[feat] = Float32.(getproperty(tbl, col))
    end

    # Load per-event trigger VoVs (set-model branch). VoVs come back as
    # AbstractVector{<:AbstractVector{T}} from LegendHDF5IO; we keep them
    # ragged here and pad later in `assemble_trigger_tensor`.
    trig = nothing
    if trig_cfg !== nothing
        col_ids = :trig_det_ids
        hasproperty(tbl, col_ids) || error("Column :trig_det_ids not in $path (required for triggers)")
        trig_data = Dict{String, AbstractVector}()
        for feat in trig_cfg.features
            col = Symbol(feat)
            hasproperty(tbl, col) || error("Column :$col not found in $path (trigger feature: $feat)")
            trig_data[feat] = getproperty(tbl, col)
        end
        trig = (
            features = trig_cfg.features,
            data     = trig_data,                       # Dict{name → VoV{Float32}}
            det_ids  = getproperty(tbl, col_ids),       # VoV{UInt32}
        )
    end

    return (sipm = sipm, geom = geom, det = det, trig = trig,
            y = Float32.(tbl.label),
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
    geom_feats = resolve_geometry_features(cfg)
    trig_cfg   = resolve_trigger_config(cfg)

    tbl  = lh5open(path, "r") do f; f[tier_key][:]; end
    sids = lh5open(path, "r") do f; f["sipm_detector_ids"][:]; end
    n  = length(tbl)
    probe_feat = !isempty(sipm_feats) ? first(sipm_feats) :
                 !isempty(geom_feats) ? first(geom_feats) :
                 error("config has neither input.sipm.features nor input.sipm.geometry")
    ns = length(first(getproperty(tbl, Symbol(probe_feat))))

    sipm = Dict{String, Matrix{Float32}}()
    for feat in sipm_feats
        col = Symbol(feat)
        hasproperty(tbl, col) || error("Column :$col not found in $path (feature: $feat)")
        sipm[feat] = _vov_to_matrix(getproperty(tbl, col), n, ns)
    end

    geom = Dict{String, Matrix{Float32}}()
    for feat in geom_feats
        col = Symbol(feat)
        hasproperty(tbl, col) || error("Column :$col not found in $path (geometry: $feat)")
        geom[feat] = _vov_to_matrix(getproperty(tbl, col), n, ns)
    end

    det = Dict{String, Vector{Float32}}()
    for feat in det_feats
        col = Symbol(feat)
        hasproperty(tbl, col) || error("Column :$col not found in $path (feature: $feat)")
        det[feat] = Float32.(getproperty(tbl, col))
    end

    trig = nothing
    if trig_cfg !== nothing && hasproperty(tbl, :trig_det_ids)
        trig_data = Dict{String, AbstractVector}()
        for feat in trig_cfg.features
            col = Symbol(feat)
            hasproperty(tbl, col) || error("Column :$col not found in $path (trigger feature: $feat)")
            trig_data[feat] = getproperty(tbl, col)
        end
        trig = (
            features = trig_cfg.features,
            data     = trig_data,
            det_ids  = getproperty(tbl, :trig_det_ids),
        )
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

    return (sipm = sipm, geom = geom, det = det, trig = trig, y = y,
            sids = Vector{UInt32}(sids), n = n,
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
    assemble_geometry_tensor(data, cfg, sipm_perm) → Array{Float32,3} | nothing

Per-SiPM geometry features stacked into a `(G × n_sipm × n_events)` tensor.
Returns `nothing` if no geometry features are configured.
"""
function assemble_geometry_tensor(data::NamedTuple, cfg::Dict,
                                  sipm_perm::Union{Vector{Int}, Nothing})
    geom_feats = resolve_geometry_features(cfg)
    isempty(geom_feats) && return nothing
    n  = data.n
    ns = size(first(values(data.geom)), 2)
    G  = length(geom_feats)
    out = Array{Float32, 3}(undef, G, ns, n)
    for (gi, feat) in enumerate(geom_feats)
        M = data.geom[feat]                            # (n × ns)
        if sipm_perm !== nothing
            M = M[:, sipm_perm]
        end
        # M is (n × ns); we want (G × ns × n)
        @inbounds for e in 1:n, s in 1:ns
            out[gi, s, e] = M[e, s]
        end
    end
    return out
end

"""
    assemble_trigger_tensor(data, cfg, sipm_perm) → (X, mask) | nothing

Build the dense trigger tensor `X::(F × max_K × n_sipm × n_events)` and
`mask::(max_K × n_sipm × n_events)` from ragged per-event trigger VoVs.

Returns `nothing` if `cfg["input"]["triggers"]` is not configured.
"""
function assemble_trigger_tensor(data::NamedTuple, cfg::Dict,
                                 sipm_perm::Union{Vector{Int}, Nothing})
    trig_cfg = resolve_trigger_config(cfg)
    trig_cfg === nothing && return nothing
    data.trig === nothing && error("trigger data not loaded — re-load split with trigger config")

    n_sipm = length(data.sids)

    # Map raw detector IDs in trig_det_ids to slot indices in 1..n_sipm
    slot_lists = detid_to_slot(data.trig.det_ids, data.sids, sipm_perm)

    # Pack feature VoVs into a tuple so pad_per_sipm_triggers can iterate them.
    fv = Tuple(data.trig.data[feat] for feat in trig_cfg.features)
    X, M = pad_per_sipm_triggers(fv, slot_lists,
                                 trig_cfg.max_per_sipm, n_sipm;
                                 pad_value = trig_cfg.pad_value)
    return (X, M)
end

"""
    assemble_features(data, cfg, sipm_perm) → NamedTuple

Build all input arrays needed by any registered architecture. Architectures
pull from this NamedTuple via `Lux.apply(model, inputs, ps, st)`. Keys present
in the output:
  - sipm  :: Matrix{Float32} (features × samples) — MLP-style SiPM features
  - det   :: Matrix{Float32} (n_det × samples)    — HPGe scalars
  - geom  :: Array{Float32, 3} (G × n_sipm × samples) | nothing
  - trig  :: Array{Float32, 4} (F × max_K × n_sipm × samples) | nothing
  - mask  :: Array{Bool, 3} (max_K × n_sipm × samples) | nothing

Empty matrices are still returned for `sipm`/`det` so old code paths keep
working — set-only models simply ignore those.
"""
function assemble_features(data::NamedTuple, cfg::Dict,
                           sipm_perm::Union{Vector{Int}, Nothing})
    sipm_feats = resolve_input_features(cfg)[1]
    x_sipm = isempty(sipm_feats) ? Matrix{Float32}(undef, 0, data.n) :
                                    assemble_sipm_matrix(data, cfg, sipm_perm)
    x_det  = assemble_det_matrix(data, cfg)
    x_geom = assemble_geometry_tensor(data, cfg, sipm_perm)
    trig_pair = assemble_trigger_tensor(data, cfg, sipm_perm)
    x_trig = trig_pair === nothing ? nothing : trig_pair[1]
    x_mask = trig_pair === nothing ? nothing : trig_pair[2]
    return (sipm = x_sipm, det = x_det, geom = x_geom, trig = x_trig, mask = x_mask)
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
