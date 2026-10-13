# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

using Test
using LegendMachineLearning
using LegendMachineLearning: waveform_stats
using RadiationDetectorSignals: RDWaveform, ArrayOfRDWaveforms
using StableRNGs, Unitful, IntervalSets

@testset "waveform_stats" begin
    rng = StableRNG(1)
    # 781 samples at 128 ns, the layout of the presummed LEGEND-200 trace.
    t = range(0.0u"ns", step = 128.0u"ns", length = 781)
    x = ustrip.(u"µs", t)
    trace(t0, a; tau = 400.0, bl = 1000.0, slope = 0.0, noise = 2.0) =
        [bl + slope * xi + noise * randn(rng) + (xi >= t0 ? a * exp(-(xi - t0) / tau) : 0.0) for xi in x]
    sigs = [
        trace(48.0, 5000.0),                                                # ordinary pulse
        trace(48.0, 0.0; slope = 20.0),                                       # sloped baseline
        trace(48.0, 5000.0) .+ trace(70.0, 5000.0; bl = 0.0, noise = 0.0),     # two pulses
        trace(48.0, 0.0),                                                   # noise only
        fill(1000.0, 781)                                                  # flat, noise-free
    ]
    wvfs = ArrayOfRDWaveforms([RDWaveform(t, s) for s in sigs])
    windows = (bl_window = 0.0u"µs"..39.0u"µs", tail_window = 70.0u"µs"..95.0u"µs")
    st = waveform_stats(wvfs; windows...)
    @test length(st) == 5

    @test isapprox(st.blmean[1], 1000; atol = 1)
    @test isapprox(st.wf_max[1], 5000; rtol = 0.01)
    @test st.amplitude == st.wf_max
    @test isapprox(st.tail_tau[1], 400u"µs"; rtol = 0.05)
    @test isapprox(st.t50[1], 48u"µs"; atol = 0.2u"µs")
    @test st.t10[1] <= st.t50[1] <= st.t90[1]
    @test st.risetime[1] == st.t90[1] - st.t10[1] && st.risetime[1] < 300u"ns"
    @test st.n_triggers[1] == 1
    @test st.blsigma[1] < 4 * st.bl_slope_sigma[1]

    # A sloped baseline spreads the plain sigma but not the sigma about the fitted line.
    @test isapprox(st.blslope[2], 20u"µs^-1"; rtol = 0.05)
    @test st.blsigma[2] > 4 * st.bl_slope_sigma[2]

    @test st.n_triggers[3] == 2
    @test st.n_triggers[4] == 0
    @test st.wf_max[4] < 10 * st.blsigma[4]

    # Nothing crosses on a flat trace: the crossing times are NaN and so is the decay time.
    @test isnan(st.t50[5]) && isnan(st.risetime[5]) && isnan(st.tail_tau[5])
    @test st.n_triggers[5] == 0

    # Arrays larger than one processing block give one row per waveform in order.
    many = ArrayOfRDWaveforms([RDWaveform(t, trace(48.0, 100.0 * (i % 3))) for i in 1:10_050])
    st_many = waveform_stats(many; windows...)
    @test length(st_many) == 10_050
    @test isapprox(st_many.wf_max[10_050], 0; atol = 20) && isapprox(st_many.wf_max[10_049], 200; atol = 20)

    # A Float32 time axis, as in the phy raw files, gives the same Float64 columns.
    t32 = range(0.0f0u"ns", step = 128.0f0u"ns", length = 781)
    st32 = waveform_stats(ArrayOfRDWaveforms([RDWaveform(t32, sigs[1]), RDWaveform(t32, sigs[5])]); windows...)
    @test eltype(st32.t50) == eltype(st.t50) && eltype(st32.risetime) == eltype(st.risetime)
    @test isconcretetype(eltype(st32.tail_tau)) && isconcretetype(eltype(st32.blslope))
    @test isapprox(st32.t50[1], st.t50[1]; atol = 1u"ns") && isnan(st32.t50[2])

    @test_throws "no waveforms" waveform_stats(wvfs[Int[]]; windows...)
end
