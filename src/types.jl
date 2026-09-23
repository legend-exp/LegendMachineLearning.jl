# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

"""
    abstract type AbstractFeatureExtractor

Maps an array of waveforms to a feature matrix with one row per waveform.
Subtypes implement `features(fe, wvfs)`.
"""
abstract type AbstractFeatureExtractor end
export AbstractFeatureExtractor

"""
    WaveformSample

Waveforms selected for training together with everything needed to trace
each one back to its raw event. `stratum` names the selection rule each row
was drawn by. All vectors share the same axes.
"""
struct WaveformSample
    waveform::ArrayOfRDWaveforms
    filekey::Vector{FileKey}
    eventnumber::Vector{Int}
    channel::Vector{ChannelId}
    detector::Vector{DetectorId}
    det_type::Vector{Symbol}
    daqenergy::Vector{Float64}
    baseline::Vector{Float64}
    timestamp::Vector{typeof(1.0u"s")}
    stratum::Vector{Symbol}
    function WaveformSample(waveform, filekey, eventnumber, channel, detector, det_type,
            daqenergy, baseline, timestamp, stratum)
        ax = axes(waveform, 1)
        for (name, v) in ((:filekey, filekey), (:eventnumber, eventnumber), (:channel, channel),
                          (:detector, detector), (:det_type, det_type), (:daqenergy, daqenergy),
                          (:baseline, baseline), (:timestamp, timestamp), (:stratum, stratum))
            axes(v, 1) == ax || throw(DimensionMismatch("field $name has axes $(axes(v, 1)), waveform has $ax"))
        end
        new(waveform, filekey, eventnumber, channel, detector, det_type, daqenergy, baseline, timestamp, stratum)
    end
end
export WaveformSample

Base.length(s::WaveformSample) = length(s.waveform)

Base.getindex(s::WaveformSample, idxs) = WaveformSample(
    s.waveform[idxs], s.filekey[idxs], s.eventnumber[idxs], s.channel[idxs],
    s.detector[idxs], s.det_type[idxs], s.daqenergy[idxs], s.baseline[idxs],
    s.timestamp[idxs], s.stratum[idxs])

function Base.vcat(samples::WaveformSample...)
    WaveformSample(
        ArrayOfRDWaveforms(reduce(vcat, (collect(s.waveform) for s in samples))),
        reduce(vcat, (s.filekey for s in samples)),
        reduce(vcat, (s.eventnumber for s in samples)),
        reduce(vcat, (s.channel for s in samples)),
        reduce(vcat, (s.detector for s in samples)),
        reduce(vcat, (s.det_type for s in samples)),
        reduce(vcat, (s.daqenergy for s in samples)),
        reduce(vcat, (s.baseline for s in samples)),
        reduce(vcat, (s.timestamp for s in samples)),
        reduce(vcat, (s.stratum for s in samples)))
end

"""
    ClusterResult

Affinity-propagation result on a `WaveformSample`: `assignment[i]` is the
cluster index of sample row `i`, `exemplars[k]` is the sample row of the
exemplar of cluster `k`.
"""
struct ClusterResult
    hyperparameters::NamedTuple
    assignment::Vector{Int}
    exemplars::Vector{Int}
    converged::Bool
    iterations::Int
end
export ClusterResult

"""
    LegendMLModel

A fitted MLJ machine together with the feature extractor it was trained
on, the label-to-category table, and provenance metadata. Calling the model
on waveforms returns integer labels.
"""
struct LegendMLModel{M, F<:AbstractFeatureExtractor}
    machine::M
    features::F
    categories::Dict{Int,Symbol}
    metadata::PropDict
end
export LegendMLModel
