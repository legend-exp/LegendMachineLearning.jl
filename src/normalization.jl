# This file is a part of ML-based-LAr-veto, licensed under the MIT License (MIT).
# Normalization helpers — pipeline scaling, geometry feature lookup

import YAML

# ============================================================================
# VoV → Matrix conversion
# ============================================================================

"""
    pe_sums_to_matrix(tbl, col::Symbol) → Matrix{Float64}

Convert a VectorOfVectors column from a jlbal/jlbalml table to Matrix{Float64}
(n_events × n_sipms). Returns zeros matrix if column is absent.
"""
function pe_sums_to_matrix(tbl, col::Symbol)
    n = length(tbl)
    n == 0 && return Matrix{Float64}(undef, 0, 0)
    if !hasproperty(tbl, col)
        n_sipms = length(tbl.sipm_pe_sums[1])
        return zeros(Float64, n, n_sipms)
    end
    vov = getproperty(tbl, col)
    n_sipms = length(vov[1])
    M = Matrix{Float64}(undef, n, n_sipms)
    @inbounds for i in 1:n
        row = vov[i]
        for j in 1:n_sipms; M[i, j] = Float64(row[j]); end
    end
    M
end
export pe_sums_to_matrix

# ============================================================================
# Scaling pipeline — apply config steps in order
# ============================================================================

"""
    apply_scaling_pipeline!(M, steps, global_params) → Dict{String,Any}

Apply a list of scaling steps (from YAML config) to matrix M in-place.
`global_params` is a Dict that may contain pre-computed "global_min"/"global_max"
(shared across datasets). Returns scaling parameters used.

Supported steps:
  - `clip: [lo, hi]`                                       → clamp values to [lo, hi]
  - `transform: log1p`                                     → log(1+x), zeros stay zero
  - `normalize: {method: minmax, range: [0,1], exclude_zero: true/false}`
                                                            → min-max scale to [lo,hi]
  - `normalize: minmax`                                    → shorthand for above with defaults
"""
function apply_scaling_pipeline!(M::Matrix{Float64}, steps::Vector,
                                 global_params::Dict{String,Any}=Dict{String,Any}())
    params = Dict{String,Any}()
    for step in steps
        step isa Dict || continue
        for (op, arg) in step
            op = String(op)
            if op == "clip"
                lo, hi = Float64(arg[1]), Float64(arg[2])
                _clip_matrix!(M, lo, hi)
                params["clip"] = [lo, hi]
            elseif op == "transform"
                method = String(arg)
                if method == "log1p"
                    _log1p_nonneg!(M)
                else
                    @warn "Unknown transform: $method"
                end
                params["transform"] = method
            elseif op == "normalize"
                method, out_range, exclude_zero = _parse_normalize_arg(arg)
                if method == "minmax"
                    gmin = get(global_params, "global_min", NaN)
                    gmax = get(global_params, "global_max", NaN)
                    if isnan(gmin) || isnan(gmax)
                        gmin, gmax = exclude_zero ? _compute_global_minmax_positive(M) : _compute_global_minmax(M)
                    end
                    _minmax_scale!(M, gmin, gmax, out_range, exclude_zero)
                    params["global_min"] = gmin
                    params["global_max"] = gmax
                    params["out_range"] = out_range
                    params["exclude_zero"] = exclude_zero
                else
                    @warn "Unknown normalize method: $method"
                end
                params["normalize"] = method
            end
        end
    end
    params
end
export apply_scaling_pipeline!

"""
    compute_global_params(matrices, steps) → Dict{String,Any}

Pre-compute global parameters (e.g. min/max after clip+transform) across
multiple matrices. Used to share parameters between datasets.
"""
function compute_global_params(matrices::Vector{Matrix{Float64}}, steps::Vector)
    params = Dict{String,Any}()
    # Apply clip + transform to copies, then compute global min/max
    works = [copy(M) for M in matrices]
    for step in steps
        step isa Dict || continue
        for (op, arg) in step
            op = String(op)
            if op == "clip"
                lo, hi = Float64(arg[1]), Float64(arg[2])
                for W in works; _clip_matrix!(W, lo, hi); end
            elseif op == "transform" && String(arg) == "log1p"
                for W in works; _log1p_nonneg!(W); end
            elseif op == "normalize"
                method, _, exclude_zero = _parse_normalize_arg(arg)
                if method == "minmax"
                    gmin, gmax = exclude_zero ? compute_global_minmax_positive(works) : compute_global_minmax(works)
                    params["global_min"] = gmin
                    params["global_max"] = gmax
                end
            end
        end
    end
    params
end
export compute_global_params

# ============================================================================
# Normalize argument parser
# ============================================================================

"""Parse normalize config: string shorthand or Dict with method/range/exclude_zero."""
function _parse_normalize_arg(arg)
    if arg isa Dict
        method = String(get(arg, "method", "minmax"))
        r = get(arg, "range", [0, 1])
        out_range = (Float64(r[1]), Float64(r[2]))
        exclude_zero = Bool(get(arg, "exclude_zero", false))
    else
        method = String(arg)
        out_range = (0.0, 1.0)
        exclude_zero = true  # backwards compat: plain "minmax" = exclude zeros
    end
    (method, out_range, exclude_zero)
end

# ============================================================================
# Internal helpers
# ============================================================================

function _clip_matrix!(M::Matrix{Float64}, lo::Float64, hi::Float64)
    @inbounds for i in eachindex(M)
        v = M[i]
        M[i] = v < lo ? lo : (v > hi ? hi : v)
    end
    M
end

function _log1p_nonneg!(M::Matrix{Float64})
    @inbounds for i in eachindex(M)
        v = M[i]; M[i] = v <= 0.0 ? 0.0 : log1p(v)
    end
    M
end

function _compute_global_minmax_positive(M::Matrix{Float64})
    gmin, gmax = Inf, -Inf
    @inbounds for v in M
        v > 0.0 && (v < gmin && (gmin = v); v > gmax && (gmax = v))
    end
    isfinite(gmin) && isfinite(gmax) ? (gmin, gmax) : (0.0, 1.0)
end

function _compute_global_minmax(M::Matrix{Float64})
    gmin, gmax = Inf, -Inf
    @inbounds for v in M
        v < gmin && (gmin = v); v > gmax && (gmax = v)
    end
    isfinite(gmin) && isfinite(gmax) ? (gmin, gmax) : (0.0, 1.0)
end

function compute_global_minmax_positive(matrices::Vector{Matrix{Float64}})
    gmin, gmax = Inf, -Inf
    for M in matrices
        lmin, lmax = _compute_global_minmax_positive(M)
        lmin < gmin && (gmin = lmin); lmax > gmax && (gmax = lmax)
    end
    isfinite(gmin) && isfinite(gmax) ? (gmin, gmax) : (0.0, 1.0)
end
export compute_global_minmax_positive

function compute_global_minmax(matrices::Vector{Matrix{Float64}})
    gmin, gmax = Inf, -Inf
    for M in matrices
        lmin, lmax = _compute_global_minmax(M)
        lmin < gmin && (gmin = lmin); lmax > gmax && (gmax = lmax)
    end
    isfinite(gmin) && isfinite(gmax) ? (gmin, gmax) : (0.0, 1.0)
end
export compute_global_minmax

function _minmax_scale!(M::Matrix{Float64}, gmin::Float64, gmax::Float64,
                        out_range::Tuple{Float64,Float64}=(0.0, 1.0),
                        exclude_zero::Bool=true)
    denom = gmax - gmin
    out_lo, out_hi = out_range
    out_span = out_hi - out_lo
    if denom ≈ 0.0
        @inbounds for i in eachindex(M)
            (!exclude_zero || M[i] > 0.0) && (M[i] = out_lo)
        end
        return M
    end
    @inbounds for i in eachindex(M)
        v = M[i]
        if exclude_zero && v <= 0.0
            continue  # leave zeros as-is
        end
        M[i] = out_lo + (v - gmin) / denom * out_span
    end
    M
end

# ============================================================================
# Geometry Feature Loaders
# ============================================================================

"""
    load_relative_geometry(group_name, geometry_base) → Dict{hpge → Dict{sipm → Dict}}

Load relative geometry YAML.  Returns nested dict for feature lookup:
  `rel_geom[hpge_name][sipm_name][feature_name]` → Float64
"""
function load_relative_geometry(group_name::String, geometry_base::String)
    path = joinpath(geometry_base, "relative", "$(group_name)_relative_geometry.yaml")
    isfile(path) || error("Relative geometry not found: $path")
    raw = YAML.load_file(path; dicttype=Dict{String,Any})
    geom = Dict{String, Dict{String, Dict{String,Any}}}()
    for (hpge_name, entry) in raw
        entry isa Dict && haskey(entry, "sipms") || continue
        geom[hpge_name] = entry["sipms"]
    end
    geom
end
export load_relative_geometry

"""
    load_hpge_geometry(group_name, geometry_base) → Dict{name → Dict{feature → Float64}}

Load HPGe geometry YAML.  Returns dict with pre-computed scaled features:
  `hpge_geom[det_name]["scaled_angle"]` → Float64
"""
function load_hpge_geometry(group_name::String, geometry_base::String)
    path = joinpath(geometry_base, "HPGe", "$(group_name)_hpge_geometry.yaml")
    isfile(path) || error("HPGe geometry not found: $path")
    raw = YAML.load_file(path; dicttype=Dict{String,Any})
    result = Dict{String, Dict{String,Float64}}()
    for (name, entry) in raw
        entry isa Dict && haskey(entry, "detector_position") || continue
        pos = entry["detector_position"]
        d = Dict{String,Float64}()
        for k in ("scaled_angle", "scaled_z_center")
            haskey(pos, k) && (d[k] = Float64(pos[k]))
        end
        # Also store raw values for plots
        if haskey(pos, "cylind_coords")
            cc = pos["cylind_coords"]
            d["angle_deg"]    = Float64(cc["angle"]["value"])
            d["z_center_mm"]  = Float64(cc["z_center"]["value"])
        end
        # Store string info if available
        haskey(pos, "string_id")          && (d["string_id"] = Float64(pos["string_id"]))
        haskey(pos, "position_in_string") && (d["position_in_string"] = Float64(pos["position_in_string"]))
        result[name] = d
    end
    result
end
export load_hpge_geometry

"""
    load_sipm_geometry(group_name, geometry_base) → Dict{name → Dict{feature → Float64}}

Load SiPM geometry YAML.  Returns dict with position features per SiPM.
"""
function load_sipm_geometry(group_name::String, geometry_base::String)
    path = joinpath(geometry_base, "SiPM", "$(group_name)_sipm_geometry.yaml")
    isfile(path) || error("SiPM geometry not found: $path")
    raw = YAML.load_file(path; dicttype=Dict{String,Any})
    result = Dict{String, Dict{String,Float64}}()
    for (name, entry) in raw
        entry isa Dict && haskey(entry, "detector_position") || continue
        pos = entry["detector_position"]
        d = Dict{String,Float64}()
        if haskey(pos, "cylind_coords")
            cc = pos["cylind_coords"]
            haskey(cc, "angle") && (d["angle_deg"] = Float64(cc["angle"]["value"]))
            haskey(cc, "z")     && (d["z_mm"]      = Float64(cc["z"]["value"]))
        end
        result[name] = d
    end
    result
end
export load_sipm_geometry

# ============================================================================
# Geometry Feature Builders
# ============================================================================

"""
    build_feature_matrix(assigned_ged, sipm_ids, rel_geom, feature) → Matrix{Float32}

Build (n_events × n_sipms) matrix of a relative geometry feature (e.g. cos_proximity).
Looks up `rel_geom[hpge_name][sipm_name][feature]` per event×SiPM pair.
Missing entries filled with 0.5.
"""
function build_feature_matrix(assigned_ged::AbstractVector{UInt32},
                              sipm_ids::Vector{UInt32},
                              rel_geom::Dict{String,Dict{String,Dict{String,Any}}},
                              feature::String)
    n_events, n_sipms = length(assigned_ged), length(sipm_ids)
    sipm_names = [string(DetectorId(s)) for s in sipm_ids]
    M = Matrix{Float32}(undef, n_events, n_sipms)
    @inbounds for i in 1:n_events
        hpge_name = string(DetectorId(assigned_ged[i]))
        hpge_sipms = get(rel_geom, hpge_name, nothing)
        for j in 1:n_sipms
            if hpge_sipms === nothing
                M[i, j] = 0.5f0
            else
                sipm_entry = get(hpge_sipms, sipm_names[j], nothing)
                M[i, j] = sipm_entry === nothing ? 0.5f0 : Float32(get(sipm_entry, feature, 0.5))
            end
        end
    end
    M
end
export build_feature_matrix

"""
    build_hpge_feature_vector(assigned_ged, hpge_geom, feature) → Vector{Float32}

Build per-event vector of an HPGe geometry feature (e.g. scaled_angle).
Looks up `hpge_geom[det_name][feature]` per event. Missing → 0.5.
"""
function build_hpge_feature_vector(assigned_ged::AbstractVector{UInt32},
                                   hpge_geom::Dict{String,Dict{String,Float64}},
                                   feature::String)
    n = length(assigned_ged)
    v = Vector{Float32}(undef, n)
    @inbounds for i in 1:n
        name = string(DetectorId(assigned_ged[i]))
        info = get(hpge_geom, name, nothing)
        v[i] = info === nothing ? 0.5f0 : Float32(get(info, feature, 0.5))
    end
    v
end
export build_hpge_feature_vector

# ============================================================================
# Trigger-Level Feature Scaling
# ============================================================================

"""
    scale_trigger_pe(pe_vov, steps, global_params) → VoV{Float32}

Apply the PE scaling pipeline to per-trigger PE values (VoV).
Uses the same clip → transform → normalize steps as PE sums.
"""
function scale_trigger_pe(pe_vov::AbstractVector{<:AbstractVector},
                          steps::Vector, global_params::Dict{String,Any})
    out = Vector{Vector{Float32}}(undef, length(pe_vov))
    gmin = get(global_params, "global_min", NaN)
    gmax = get(global_params, "global_max", NaN)
    denom = (isnan(gmin) || isnan(gmax) || gmax ≈ gmin) ? 1.0 : gmax - gmin
    # Parse normalize config once
    norm_range = (0.0, 1.0)
    norm_exclude_zero = false
    for step in steps
        step isa Dict || continue
        for (op, arg) in step
            if String(op) == "normalize"
                _, norm_range, norm_exclude_zero = _parse_normalize_arg(arg)
            end
        end
    end
    out_lo, out_span = norm_range[1], norm_range[2] - norm_range[1]
    for i in eachindex(pe_vov)
        raw = pe_vov[i]
        scaled = Vector{Float32}(undef, length(raw))
        for k in eachindex(raw)
            v = Float64(raw[k])
            for step in steps
                step isa Dict || continue
                for (op, arg) in step
                    op = String(op)
                    if op == "clip"
                        lo, hi = Float64(arg[1]), Float64(arg[2])
                        v = clamp(v, lo, hi)
                    elseif op == "transform" && String(arg) == "log1p"
                        v = v > 0.0 ? log1p(v) : 0.0
                    elseif op == "normalize"
                        if norm_exclude_zero && v <= 0.0
                            # leave zero as-is
                        else
                            v = out_lo + (v - gmin) / denom * out_span
                        end
                    end
                end
            end
            scaled[k] = Float32(v)
        end
        out[i] = scaled
    end
    VectorOfVectors(out)
end
export scale_trigger_pe

"""
    compute_trigger_pe_global_params(pe_vovs, steps) → Dict{String,Any}

Compute global min/max for trigger PE values after clip+transform,
across multiple VoV datasets (e.g. sub500keV + forcedtrigger).
"""
function compute_trigger_pe_global_params(pe_vovs::Vector, steps::Vector)
    # Parse exclude_zero from normalize config
    exclude_zero = false
    for step in steps
        step isa Dict || continue
        for (op, arg) in step
            if String(op) == "normalize"
                result = _parse_normalize_arg(arg)
                exclude_zero = result[3]
            end
        end
    end
    gmin, gmax = Inf, -Inf
    for vov in pe_vovs
        for evts in vov
            for v_raw in evts
                v = Float64(v_raw)
                for step in steps
                    step isa Dict || continue
                    for (op, arg) in step
                        op = String(op)
                        if op == "clip"
                            v = clamp(v, Float64(arg[1]), Float64(arg[2]))
                        elseif op == "transform" && String(arg) == "log1p"
                            v = v > 0.0 ? log1p(v) : 0.0
                        elseif op == "normalize"
                            break
                        end
                    end
                end
                if exclude_zero
                    v > 0.0 && (v < gmin && (gmin = v); v > gmax && (gmax = v))
                else
                    v < gmin && (gmin = v); v > gmax && (gmax = v)
                end
            end
        end
    end
    params = Dict{String,Any}()
    params["global_min"] = isfinite(gmin) ? gmin : 0.0
    params["global_max"] = isfinite(gmax) ? gmax : 1.0
    params
end
export compute_trigger_pe_global_params

"""
    scale_trigger_times(time_vov, time_range) → VoV{Float32}

Linearly scale trigger times from `time_range` (e.g. [-10, 10] μs) to [-1, 1].
"""
function scale_trigger_times(time_vov::AbstractVector{<:AbstractVector},
                             time_range::Vector)
    lo, hi = Float64(time_range[1]), Float64(time_range[2])
    half_range = (hi - lo) / 2.0
    center = (hi + lo) / 2.0
    out = Vector{Vector{Float32}}(undef, length(time_vov))
    for i in eachindex(time_vov)
        raw = time_vov[i]
        scaled = Vector{Float32}(undef, length(raw))
        for k in eachindex(raw)
            scaled[k] = Float32(clamp((Float64(raw[k]) - center) / half_range, -1.0, 1.0))
        end
        out[i] = scaled
    end
    VectorOfVectors(out)
end
export scale_trigger_times
