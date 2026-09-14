# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

"""
    abstract type AbstractFeatureExtractor

Maps an array of waveforms to a feature matrix with one row per waveform.
Subtypes implement `features(fe, wvfs)`.
"""
abstract type AbstractFeatureExtractor end
export AbstractFeatureExtractor

"""
    HaarFeatures(; n_samples=256, normalize=true, bl_window=0.0u"µs"..25.0u"µs")

Feature extractor: baseline subtraction (mean over `bl_window`), repeated
pairwise Haar averaging down to `n_samples` samples, and, if `normalize`,
division of each waveform by the larger absolute value of its extrema.
The number of Haar passes is derived from the input length, so one extractor
handles presummed and full-resolution waveforms of the same detector.
"""
struct HaarFeatures{Q<:Unitful.Time} <: AbstractFeatureExtractor
    n_samples::Int
    normalize::Bool
    bl_window::ClosedInterval{Q}
end
export HaarFeatures

function HaarFeatures(; n_samples::Int=256, normalize::Bool=true,
        bl_window::ClosedInterval=0.0u"µs"..25.0u"µs")
    n_samples > 0 || throw(ArgumentError("n_samples must be positive, got $n_samples"))
    HaarFeatures(n_samples, normalize, bl_window)
end

"""
    haar_passes(fe::HaarFeatures, n_input::Integer) -> Int

Number of pairwise Haar averaging passes that bring `n_input` samples down to
`fe.n_samples`. Throws when the ratio is not a power of two.
"""
function haar_passes(fe::HaarFeatures, n_input::Integer)
    ratio, r = divrem(n_input, fe.n_samples)
    (r == 0 && ratio ≥ 1 && ispow2(ratio)) ||
        throw(ArgumentError("waveform length $n_input is not a power-of-two multiple of n_samples = $(fe.n_samples)"))
    Int(log2(ratio))
end

"""
    features(fe::HaarFeatures, wvfs::AbstractVector{<:RDWaveform}) -> Matrix{Float32}

Feature matrix of size `(length(wvfs), fe.n_samples)`.
"""
function features(fe::HaarFeatures, wvfs::AbstractVector{<:RDWaveform})
    isempty(wvfs) && throw(ArgumentError("cannot extract features from an empty waveform array"))
    Base.require_one_based_indexing(wvfs)
    w = wvfs isa ArrayOfRDWaveforms ? wvfs : convert(ArrayOfRDWaveforms, collect(wvfs))
    lengths = length.(w.signal)
    n_input = first(lengths)
    all(==(n_input), lengths) ||
        throw(ArgumentError("all waveforms must have the same length; found lengths $(sort!(unique(lengths)))"))
    n_passes = haar_passes(fe, n_input)
    bl = signalstats.(w, leftendpoint(fe.bl_window), rightendpoint(fe.bl_window))
    w = shift_waveform.(w, -bl.mean)
    flt = HaarAveragingFilter(2)
    for _ in 1:n_passes
        w = flt.(w)
    end
    if fe.normalize
        # extrema returns (min, max); the larger magnitude sets the scale, flat waveforms stay unscaled
        nf = map(e -> max(abs(first(e)), abs(last(e))), extrema.(w.signal))
        replace!(nf, zero(eltype(nf)) => one(eltype(nf)))
        w = multiply_waveform.(w, inv.(nf))
    end
    Matrix{Float32}(permutedims(flatview(VectorOfSimilarArrays(w.signal))))
end
export features
