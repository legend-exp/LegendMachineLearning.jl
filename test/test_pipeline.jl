# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

using Test
using LegendMachineLearning
using LegendMachineLearning: features_config, run_qc_sampling, run_qc_training, UNLABELED, load_model,
    file_sha256, qc_model_filename, write_sample, read_sample, train_svm, classify_filekey
using PropDicts, StableRNGs, Unitful, IntervalSets
using RadiationDetectorSignals
using LegendDataManagement: LegendData, FileKey, ChannelId, DetectorId, DataTier, DataPeriod, DataRun,
    writelprops, readlprops, search_disk, read_ldata
using LegendTestData: legend_test_data_path

@testset "features_config" begin
    fe = features_config(
        PropDict(:features => PropDict(:n_samples => 128, :normalize => false, :bl_window => [0.0, 10.0]))
    )
    @test fe.n_samples == 128 && fe.normalize == false
    @test fe.bl_window == 0.0u"µs"..10.0u"µs"
end

@testset "scan grid is YAML-serializable" begin
    # The same shape `run_qc_sampling` writes to `pars.affinity_propagation.scan`.
    scan = (damp = [0.5, 0.9], preference_quantile = [0.05, 0.5], preference = [-1.0, -2.0],
        n_clusters = [3 4; 5 6], converged = [true false; true true], iterations = [10 20; 30 40])
    pd = PropDict(:damp => scan.damp, :preference_quantile => scan.preference_quantile,
        :preference => scan.preference, :n_clusters => LegendMachineLearning._grid_rows(scan.n_clusters),
        :converged => LegendMachineLearning._grid_rows(scan.converged),
        :iterations => LegendMachineLearning._grid_rows(scan.iterations))
    mktempdir() do dir
        path = joinpath(dir, "scan.yaml")
        writelprops(path, pd)
        back = readlprops(path)
        @test [collect(r) for r in eachrow(scan.n_clusters)] == back.n_clusters
        @test [collect(r) for r in eachrow(scan.converged)] == back.converged
        @test [collect(r) for r in eachrow(scan.iterations)] == back.iterations
    end
end

# The sampling/training entry points touch LegendData file paths that the test
# metadata does not define (tier jlqml). Exercise the training half on a
# synthetic sample file through the path-based internals instead.
@testset "training from a sample file" begin
    rng = StableRNG(9)
    wf(a) = RDWaveform(range(0u"µs", step = 0.016u"µs", length = 16), a .+ 0.01 .* randn(rng, 16))
    sig = vcat([fill(-1.0, 16) for _ in 1:40], [vcat(fill(-1.0, 8), fill(1.0, 8)) for _ in 1:40])
    n = length(sig)
    s = WaveformSample(ArrayOfRDWaveforms(wf.(sig)),
        fill(FileKey("l200-p03-r001-cal-20230318T012144Z"), n), collect(1:n),
        fill(ChannelId(1121600), n), fill(DetectorId(:V00048B), n), fill(:icpc, n), fill(1000.0, n),
        fill(15000.0, n), collect(Float64, 1:n) .* u"s", fill(:uniform, n))
    c = ClusterResult(
        (damp = 0.5, preference = -1.0, preference_quantile = 0.05, maxiter = 10, tol = 1e-6, metric = "Cityblock"),
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
        @test_throws "unlabeled" LegendMachineLearning._train_from_sample_file(
            sample_path,
            training_fk,
            cfg,
            pars_bad;
            rng
        )
        # stale sample is refused
        pars_stale = merge(pars, PropDict(:sample => PropDict(:sample_sha256 => "0"^64)))
        @test_throws "sample file changed" LegendMachineLearning._train_from_sample_file(
            sample_path,
            training_fk,
            cfg,
            pars_stale;
            rng
        )
    end
end

@testset "classify_filekey on LegendTestData" begin
    # A probabilistic two-class model on Haar features of the 781-sample presummed
    # waveforms; the classes are synthetic, the test checks the plumbing.
    rng = StableRNG(4)
    X = Float32.(vcat(-0.5 .+ 0.1 .* randn(rng, 30, 196), 0.5 .+ 0.1 .* randn(rng, 30, 196)))
    y = vcat(fill(0, 30), fill(3, 30))
    svm_cfg = PropDict(:cost => PropDict(:min => 1.0, :max => 1.0, :n => 1),
        :gamma => PropDict(:min => 0.01, :max => 0.01, :n => 1), :n_folds => 3, :probability => true)
    mach, _ = train_svm(X, y, svm_cfg; rng)
    model = LegendMLModel(mach, HaarFeatures(n_samples=196, bl_window=0.0u"µs"..39.0u"µs"),
        Dict(0 => :normal, 3 => :other), PropDict(:labels => y))
    withenv("LEGEND_DATA_CONFIG" => joinpath(legend_test_data_path(), "data", "lh5", "prod-ref-l200", "config.json")) do
        l200 = LegendData(:l200)
        fk = first(search_disk(FileKey, l200.tier[DataTier(:raw), :cal, DataPeriod(18), DataRun(0)]))
        det = DetectorId(:V97061A)
        raw = read_ldata((:eventnumber, :timestamp, :daqenergy, :waveform_presummed), l200, DataTier(:raw), fk, det)

        t = classify_filekey(l200, fk, model; detectors=["V97061A"])
        @test propertynames(t) == (:detector, :eventnumber, :timestamp, :daqenergy, :waveform, :label, :margin)
        @test t.eventnumber == raw.eventnumber && t.timestamp == raw.timestamp && t.daqenergy == raw.daqenergy
        @test all(==(det), t.detector)
        @test t.waveform isa ArrayOfRDWaveforms
        @test t.waveform.signal == raw.waveform_presummed.signal
        @test (t.label, t.margin) == predict_with_margin(model, raw.waveform_presummed)
        @test all(in((0, 3)), t.label) && all(m -> 0.5 ≤ m ≤ 1, t.margin)

        # one block per detector entry; the cap draws a sorted subset of the events of the file
        t2 = classify_filekey(l200, fk, model; detectors=[det, det], n_max_per_detector=4, rng=StableRNG(1))
        @test length(t2) == 8
        first4 = t2.eventnumber[1:4]
        @test issorted(first4) && first4 ⊆ raw.eventnumber
        @test t2.waveform.signal[1:4] == raw.waveform_presummed.signal[findall(in(first4), raw.eventnumber)]

        @test_throws "not an aux trigger" classify_filekey(l200, fk, model; detectors=[det], exclude=[:pulser])
        @test_throws "detectors is empty" classify_filekey(l200, fk, model; detectors=String[])
    end
end
