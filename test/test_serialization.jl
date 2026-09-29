# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

using Test
using LegendMachineLearning
using LegendMachineLearning: save_model, load_model, file_sha256, write_sample, read_sample,
    cluster_result_to_props, cluster_result, train_svm, predict_labels, HaarFeatures, MODEL_FORMAT_VERSION
using PropDicts, StableRNGs, Unitful, IntervalSets
using RadiationDetectorSignals
using LegendDataManagement: FileKey, ChannelId, DetectorId
using LegendHDF5IO
using TypedTables: Table

svm_cfg() = PropDict(:cost => PropDict(:min => 0.1, :max => 10.0, :n => 2),
    :gamma => PropDict(:min => 0.01, :max => 1.0, :n => 2), :n_folds => 3, :probability => false)

@testset "model round trip" begin
    rng = StableRNG(3)
    X = vcat(-1 .+ 0.3 .* randn(rng, 30, 4), 1 .+ 0.3 .* randn(rng, 30, 4))
    y = vcat(fill(0, 30), fill(3, 30))
    mach, _ = train_svm(X, y, svm_cfg(); rng)
    meta = PropDict(:period => "p03", :training_filekey => "l200-p03-r001-cal-20230318T012144Z", :n_sample => 60)
    m = LegendMLModel(mach, HaarFeatures(n_samples=4, bl_window=0.0u"µs"..1.0u"µs"), Dict(0 => :normal, 3 => :other), meta)
    mktempdir() do dir
        path = joinpath(dir, "model.jld2")
        digest = save_model(path, m)
        @test digest == file_sha256(path)
        @test length(digest) == 64
        m2 = load_model(path; expected_sha256=digest)
        @test m2.features == m.features
        @test m2.categories == m.categories
        @test m2.metadata.period == "p03"
        @test m2.metadata.n_sample == 60
        @test predict_labels(m2, X) == predict_labels(m, X)
        @test_throws "SHA-256" load_model(path; expected_sha256="0"^64)
    end
end

@testset "sample round trip" begin
    wf(i) = RDWaveform(range(0u"µs", step=0.016u"µs", length=8), Float64.(i:i+7))
    s = WaveformSample(ArrayOfRDWaveforms([wf(1), wf(2), wf(3)]),
        FileKey.(["l200-p03-r001-cal-20230318T012144Z", "l200-p03-r001-cal-20230318T012144Z", "l200-p03-r001-cal-20230318T022144Z"]),
        [10, 20, 30], ChannelId.([1121600, 1121600, 1084803]), DetectorId.([:V00048B, :V00048B, :B00035A]),
        [:icpc, :icpc, :bege], [100.0, 0.0, 5000.0], [15000.0, 14980.0, 15020.0],
        [1.0e9, 1.0e9 + 0.5, 1.0e9 + 1.25]u"s", [:uniform, :zero_energy, :high_energy])
    c = ClusterResult((damp=0.7, preference=-12.5, preference_quantile=0.05, maxiter=100, tol=1e-6, metric="Cityblock"),
        [1, 1, 2], [1, 3], true, 42)
    mktempdir() do dir
        path = joinpath(dir, "sample.lh5")
        stats = Table(tail_tau=[400.0, NaN, -12.5]u"µs", t50=[48.0, 47.5, 60.0]u"µs", n_triggers=[1, 0, 2],
            baseline_dev=[0.3, Inf, 7.5])
        write_sample(path, s, c; stats)
        s2, assignment, exemplars, stats2 = read_sample(path)
        @test isequal(collect(stats2.tail_tau), collect(stats.tail_tau))
        @test stats2.t50 == stats.t50 && stats2.n_triggers == stats.n_triggers
        @test stats2.baseline_dev == stats.baseline_dev
        @test_throws DimensionMismatch write_sample(joinpath(dir, "bad.lh5"), s, c; stats=stats[1:2])
        write_sample(joinpath(dir, "plain.lh5"), s, c)
        @test isnothing(last(read_sample(joinpath(dir, "plain.lh5"))))
        @test length(s2) == 3
        @test s2.waveform isa ArrayOfRDWaveforms
        @test all(s2.waveform[i].signal == wf(i).signal for i in 1:3)
        @test all(s2.waveform[i].time == wf(i).time for i in 1:3)
        @test s2.filekey == s.filekey
        @test s2.eventnumber == s.eventnumber
        @test s2.channel == s.channel
        @test s2.detector == s.detector
        @test s2.det_type == s.det_type
        @test s2.daqenergy == s.daqenergy
        @test s2.baseline == s.baseline
        @test s2.timestamp == s.timestamp
        @test s2.stratum == s.stratum
        @test assignment == c.assignment && exemplars == c.exemplars
        props = cluster_result_to_props(c)
        @test props.converged == true && props.iterations == 42 && props.damp == 0.7
        c2 = cluster_result(props, assignment, exemplars)
        @test c2.hyperparameters == c.hyperparameters
        @test c2.converged == c.converged && c2.iterations == c.iterations
    end
end
