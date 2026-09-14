# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

using Test
using LegendMachineLearning
using LegendMachineLearning: HaarFeatures, features, haar_passes
using RadiationDetectorSignals
using Unitful
using IntervalSets

# 1024-sample step waveform with a baseline offset of 100 ADC and a step of +400 at sample 513
step_wf(offset, amp; n=1024) = RDWaveform(range(0u"µs", step=0.016u"µs", length=n),
    vcat(fill(Float64(offset), n ÷ 2), fill(Float64(offset + amp), n ÷ 2)))

@testset "HaarFeatures" begin
    fe = HaarFeatures(n_samples=256, normalize=true, bl_window=0.0u"µs"..2.0u"µs")

    @testset "haar_passes" begin
        @test haar_passes(fe, 1024) == 2
        @test haar_passes(fe, 256) == 0
        @test_throws "not a power-of-two multiple" haar_passes(fe, 768)
        @test_throws "not a power-of-two multiple" haar_passes(fe, 128)
    end

    @testset "shape, baseline removal and normalization" begin
        wvfs = ArrayOfRDWaveforms([step_wf(100, 400), step_wf(-50, 800), step_wf(0, 0)])
        X = features(fe, wvfs)
        @test X isa Matrix{Float32}
        @test size(X) == (3, 256)
        # baseline removed: first feature samples are ≈ 0
        @test all(abs.(X[:, 1]) .< 1e-4)
        # normalized: the step waveforms end at exactly 1
        @test X[1, end] ≈ 1 atol=1e-5
        @test X[2, end] ≈ 1 atol=1e-5
        # a flat waveform has zero extrema and must not produce NaN
        @test all(isfinite, X[3, :])
    end

    @testset "normalize=false keeps amplitude" begin
        fe0 = HaarFeatures(n_samples=256, normalize=false, bl_window=0.0u"µs"..2.0u"µs")
        X = features(fe0, ArrayOfRDWaveforms([step_wf(100, 400)]))
        # two Haar passes scale a constant by sqrt(2)^2 = 2
        @test X[1, end] ≈ 800 rtol=1e-6
    end

    @testset "views and plain vectors of waveforms" begin
        wvfs = ArrayOfRDWaveforms([step_wf(100, 400), step_wf(-50, 800), step_wf(0, 0)])
        X = features(fe, wvfs)
        @test features(fe, view(wvfs, 2:3)) == X[2:3, :]
        @test features(fe, collect(wvfs)) == X
    end

    @testset "errors" begin
        wvfs = ArrayOfRDWaveforms([step_wf(100, 400)])
        @test_throws "empty" features(fe, wvfs[1:0])
    end
end
