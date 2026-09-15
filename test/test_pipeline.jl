# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

using Test
using LegendMachineLearning
using LegendMachineLearning: features_config, run_qc_sampling, run_qc_training, UNLABELED, load_model,
    file_sha256, qc_model_filename, write_sample, read_sample
using PropDicts, StableRNGs, Unitful, IntervalSets
using RadiationDetectorSignals
using LegendDataManagement: FileKey, ChannelId, DetectorId

@testset "features_config" begin
    fe = features_config(PropDict(:features => PropDict(:n_samples => 128, :normalize => false, :bl_window => [0.0, 10.0])))
    @test fe.n_samples == 128 && fe.normalize == false
    @test fe.bl_window == 0.0u"µs"..10.0u"µs"
end

# The sampling/training entry points touch LegendData file paths that the test
# metadata does not define (tier jlqml). Exercise the training half on a
# synthetic sample file through the path-based internals instead.
@testset "training from a sample file" begin
    rng = StableRNG(9)
    wf(a) = RDWaveform(range(0u"µs", step=0.016u"µs", length=16), a .+ 0.01 .* randn(rng, 16))
    sig = vcat([fill(-1.0, 16) for _ in 1:40], [vcat(fill(-1.0, 8), fill(1.0, 8)) for _ in 1:40])
    n = length(sig)
    s = WaveformSample(ArrayOfRDWaveforms(wf.(sig)),
        fill(FileKey("l200-p03-r001-cal-20230318T012144Z"), n), collect(1:n),
        fill(ChannelId(1121600), n), fill(DetectorId(:V00048B), n), fill(:icpc, n), fill(1000.0, n))
    c = ClusterResult((damp=0.5, preference=-1.0, preference_quantile=0.05, maxiter=10, tol=1e-6, metric="Cityblock"),
        vcat(fill(1, 40), fill(2, 40)), [1, 41], true, 5)
    training_fk = FileKey("l200-p03-r002-cal-20230501T000000Z")
    cfg = PropDict(
        :features => PropDict(:n_samples => 8, :normalize => false, :bl_window => [0.0, 0.05]),
        :svm => PropDict(:cost => PropDict(:min => 0.1, :max => 10.0, :n => 2),
            :gamma => PropDict(:min => 0.01, :max => 1.0, :n => 2), :n_folds => 3, :probability => false),
        :categories => PropDict(0 => "normal", 4 => "step"))
    mktempdir() do dir
        sample_path = joinpath(dir, "l200-p03-r001-cal-20230318T012144Z-tier_jlqml.lh5")
        write_sample(sample_path, s, c)
        pars = PropDict(:sample => PropDict(:sample_sha256 => file_sha256(sample_path)),
            :affinity_propagation => LegendMachineLearning.cluster_result_to_props(c),
            :exemplar_labels => [0, 4])
        model, entry, rep = LegendMachineLearning._train_from_sample_file(sample_path, training_fk, cfg, pars; rng)
        @test isfile(qc_model_filename(sample_path))
        @test entry.model_sha256 == file_sha256(qc_model_filename(sample_path))
        @test model(s.waveform[[1, 41]]) == [0, 4]
        @test rep.classes == [0, 4]
        # the model is named after the passed-in training filekey, not the
        # filekey of the sample rows (s.filekey is a different run)
        @test entry.training_filekey == string(training_fk)
        @test model.metadata.training_filekey == string(training_fk)
        # unlabeled exemplar is refused
        pars_bad = merge(pars, PropDict(:exemplar_labels => [0, UNLABELED]))
        @test_throws "unlabeled" LegendMachineLearning._train_from_sample_file(sample_path, training_fk, cfg, pars_bad; rng)
        # stale sample is refused
        pars_stale = merge(pars, PropDict(:sample => PropDict(:sample_sha256 => "0"^64)))
        @test_throws "sample file changed" LegendMachineLearning._train_from_sample_file(sample_path, training_fk, cfg, pars_stale; rng)
    end
end
