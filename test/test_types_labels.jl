# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

using Test
using LegendMachineLearning
using LegendMachineLearning: WaveformSample, ClusterResult, propagate_labels, UNLABELED,
    qc_categories, class_fractions
using RadiationDetectorSignals, Unitful
using LegendDataManagement: FileKey, ChannelId, DetectorId
using PropDicts

flat_wf(n) = RDWaveform(range(0u"µs", step=0.016u"µs", length=n), zeros(n))

function make_sample(n)
    WaveformSample(
        ArrayOfRDWaveforms([flat_wf(8) for _ in 1:n]),
        fill(FileKey("l200-p03-r001-cal-20230318T012144Z"), n),
        collect(1:n),
        fill(ChannelId(1121600), n),
        fill(DetectorId(:V00048B), n),
        fill(:icpc, n),
        collect(Float64, 1:n))
end

@testset "WaveformSample" begin
    s = make_sample(4)
    @test length(s) == 4
    s2 = s[2:3]
    @test length(s2) == 2
    @test s2.eventnumber == [2, 3]
    @test s2.daqenergy == [2.0, 3.0]
    s3 = vcat(s, s2)
    @test length(s3) == 6
    @test s3.eventnumber == [1, 2, 3, 4, 2, 3]
    @test_throws DimensionMismatch WaveformSample(s.waveform, s.filekey, s.eventnumber[1:3],
        s.channel, s.detector, s.det_type, s.daqenergy)
end

@testset "propagate_labels" begin
    cluster = ClusterResult((damp=0.5, preference=-1.0), [1, 2, 2, 3, 1], [1, 2, 4], true, 10)
    @test propagate_labels(cluster, [0, 5, 7]) == [0, 5, 5, 7, 0]
    @test_throws "unlabeled" propagate_labels(cluster, [0, UNLABELED, 7])
    @test_throws DimensionMismatch propagate_labels(cluster, [0, 5])
end

@testset "categories and fractions" begin
    cfg = PropDict(:categories => PropDict(0 => "normal", 1 => "discharge"))
    cats = qc_categories(cfg)
    @test cats == Dict(0 => :normal, 1 => :discharge)
    f = class_fractions([0, 0, 1, 0], cats)
    @test f[0] == 0.75 && f[1] == 0.25
    @test_throws "not a configured category" class_fractions([0, 2], cats)
    @test_throws "empty label set" class_fractions(Int[], cats)
end
