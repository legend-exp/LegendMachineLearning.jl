# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

"Layout version of the JLD2 model file; loading a different version is an error."
const MODEL_FORMAT_VERSION = 1
export MODEL_FORMAT_VERSION

# Writes go to a temporary file in the target directory and are renamed into place, so an interrupted write never leaves a truncated file at the final path.
function _atomic_write(writer, path::AbstractString)
    mkpath(dirname(path))
    tmp = tempname(dirname(path); cleanup = false)
    try
        writer(tmp)
        mv(tmp, path; force = true)
    finally
        isfile(tmp) && rm(tmp)
    end
    return path
end

"SHA-256 of the file at `path`, as a lowercase hex string."
file_sha256(path::AbstractString) = open(io -> bytes2hex(sha256(io)), path)
export file_sha256

"""
    save_model(path, m::LegendMLModel) -> sha256::String

Write the model as JLD2: the machine in its serializable form, the feature
extractor, the category table, the metadata, and `MODEL_FORMAT_VERSION`.
Returns the SHA-256 of the written file.
"""
function save_model(path::AbstractString, m::LegendMLModel)
    _atomic_write(
        tmp -> jldsave(tmp; format_version = MODEL_FORMAT_VERSION, machine = serializable(m.machine),
            features = m.features, categories = m.categories, metadata = m.metadata), path)
    return file_sha256(path)
end
export save_model

"""
    load_model(path; expected_sha256=nothing) -> LegendMLModel

Read a model written by `save_model`. When `expected_sha256` is given the
file digest must match it.
"""
function load_model(path::AbstractString; expected_sha256::Union{Nothing,AbstractString} = nothing)
    isfile(path) || throw(ArgumentError("model file $path does not exist"))
    if expected_sha256 !== nothing
        actual = file_sha256(path)
        actual == expected_sha256 ||
            throw(
                ArgumentError(
                    "SHA-256 of $path is $actual, expected $expected_sha256: the model file does not belong to the parameter entry"
                )
            )
    end
    jldopen(path, "r") do f
        v = f["format_version"]
        v == MODEL_FORMAT_VERSION ||
            throw(ArgumentError("model file format version $v, this package reads version $MODEL_FORMAT_VERSION"))
        mach = f["machine"]
        restore!(mach)
        return LegendMLModel(mach, f["features"], f["categories"], f["metadata"])
    end
end
export load_model

"""
    write_sample(path, s::WaveformSample, c::ClusterResult; stats=nothing)

LH5 layout: table `sample` and arrays `clustering/assignment`, `clustering/exemplars`,
and, when `stats` is given, the table `stats` with one row per sample row (e.g.
the rule inputs `build_sample` returns). Identifiers and stratum names are stored
as strings; hyperparameters and convergence go to the parameter database (see
`cluster_result_to_props`).
"""
function write_sample(path::AbstractString, s::WaveformSample, c::ClusterResult; stats = nothing)
    length(c.assignment) == length(s) ||
        throw(DimensionMismatch("$(length(c.assignment)) cluster assignments for $(length(s)) sample rows"))
    isnothing(stats) || length(stats) == length(s) ||
        throw(DimensionMismatch("$(length(stats)) stats rows for $(length(s)) sample rows"))
    _atomic_write(path) do tmp
        lh5open(tmp, "w") do f
            f["sample"] = Table(waveform = s.waveform, filekey = string.(s.filekey), eventnumber = s.eventnumber,
                channel = string.(s.channel), detector = string.(s.detector), det_type = String.(s.det_type),
                daqenergy = s.daqenergy, baseline = s.baseline, timestamp = s.timestamp,
                stratum = String.(s.stratum))
            f["clustering/assignment"] = c.assignment
            f["clustering/exemplars"] = c.exemplars
            return isnothing(stats) || (f["stats"] = stats)
        end
    end
    return nothing
end
export write_sample

"""
    read_sample(path) -> (sample, assignment, exemplars, stats)

`stats` is the table written with the sample, or `nothing` for a file without one.
"""
function read_sample(path::AbstractString)
    isfile(path) || throw(ArgumentError("sample file $path does not exist"))
    lh5open(path, "r") do f
        t = f["sample"][:]
        s = WaveformSample(t.waveform, FileKey.(t.filekey), Int.(t.eventnumber), ChannelId.(t.channel),
            DetectorId.(t.detector), Symbol.(t.det_type), Float64.(t.daqenergy), Float64.(t.baseline),
            collect(typeof(1.0u"s"), t.timestamp), Symbol.(t.stratum))
        stats = haskey(f, "stats") ? f["stats"][:] : nothing
        return s, Int.(f["clustering/assignment"][:]), Int.(f["clustering/exemplars"][:]), stats
    end
end
export read_sample

"""
    cluster_result_to_props(c::ClusterResult) -> PropDict

Hyperparameters, `converged` and `iterations` as a parameter-database entry.
"""
function cluster_result_to_props(c::ClusterResult)
    pd = PropDict()
    for (k, v) in pairs(c.hyperparameters)
        pd[k] = v
    end
    pd.converged = c.converged
    pd.iterations = c.iterations
    return pd
end
export cluster_result_to_props

"""
    cluster_result(props::PropDict, assignment, exemplars) -> ClusterResult

Inverse of `cluster_result_to_props` combined with the arrays from `read_sample`.
"""
function cluster_result(props::PropDict, assignment::AbstractVector{<:Integer}, exemplars::AbstractVector{<:Integer})
    hp = (damp = Float64(props.damp), preference = Float64(props.preference),
        preference_quantile = Float64(props.preference_quantile), maxiter = Int(props.maxiter),
        tol = Float64(props.tol), metric = String(props.metric))
    return ClusterResult(
        hp,
        collect(Int, assignment),
        collect(Int, exemplars),
        Bool(props.converged),
        Int(props.iterations)
    )
end
export cluster_result
