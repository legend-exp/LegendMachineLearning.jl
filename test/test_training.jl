# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

using Test
using LegendMachineLearning
using LegendMachineLearning: train_svm, svm_grid, class_weights, predict_labels, predict_with_margin,
    HaarFeatures
using PropDicts, StableRNGs, Unitful, IntervalSets
using RadiationDetectorSignals

function svm_cfg(; probability = false)
    return PropDict(
        :cost => PropDict(:min => 0.1, :max => 10.0, :n => 3),
        :gamma => PropDict(:min => 0.01, :max => 1.0, :n => 3),
        :n_folds => 3, :probability => probability)
end

# two classes: label 0 clusters around -1, label 3 around +1, in 4 dimensions
function two_classes(rng, n)
    X = vcat(-1 .+ 0.3 .* randn(rng, n, 4), 1 .+ 0.3 .* randn(rng, n, 4))
    y = vcat(fill(0, n), fill(3, n))
    return X, y
end

@testset "training" begin
    rng = StableRNG(7)
    X, y = two_classes(rng, 40)

    @testset "grid and weights" begin
        costs, gammas = svm_grid(svm_cfg())
        @test costs ≈ [0.1, 1.0, 10.0]
        @test length(gammas) == 3 && gammas[1] ≈ 0.01 && gammas[end] ≈ 1.0
        w = class_weights([0, 0, 0, 3])
        @test w[3] == 3 * w[0]
        @test (w[0] + w[3]) / 2 ≈ 1
    end

    @testset "train_svm deterministic" begin
        mach, rep = train_svm(X, y, svm_cfg(); groups = repeat([:a, :b], 40), rng = StableRNG(1))
        @test size(rep.cv_score) == (3, 3)
        @test rep.best.cv_score == maximum(rep.cv_score)
        @test rep.best.cost in rep.cost && rep.best.gamma in rep.gamma
        @test rep.classes == [0, 3]
        @test size(rep.confusion) == (2, 2)
        @test sum(rep.confusion) == round(Int, rep.holdout_fraction * 80)
        @test all(>=(0.9), skipmissing(rep.class_accuracy))
        @test rep.groups == [:a, :b]
        @test all(rep.group_accuracy .>= 0.9)
        m = LegendMLModel(mach, HaarFeatures(n_samples = 4, bl_window = 0.0u"µs"..0.001u"µs"),
            Dict(0 => :normal, 3 => :other), PropDict())
        @test predict_labels(m, X[1:3, :]) == [0, 0, 0]
        @test predict_labels(m, X[(end - 2):end, :]) == [3, 3, 3]
        @test_throws "probabilistic" predict_with_margin(m, X[1:2, :])
    end

    @testset "class_accuracy for a rare class" begin
        # Two balanced classes plus one rare class of a single row: under
        # `rng=StableRNG(1)` the stratified holdout split deterministically
        # gives that class no holdout row (verified directly), so its
        # class_accuracy must be missing rather than a fabricated 0.0.
        Xr = vcat(-1 .+ 0.3 .* randn(StableRNG(20), 38, 4), 1 .+ 0.3 .* randn(StableRNG(21), 39, 4),
            5 .+ 0.3 .* randn(StableRNG(22), 1, 4))
        yr = vcat(fill(0, 38), fill(3, 39), fill(7, 1))
        _, rep_r = train_svm(Xr, yr, svm_cfg(); rng = StableRNG(1))
        i_rare = only(findall(==(7), rep_r.classes))
        @test ismissing(rep_r.class_accuracy[i_rare])
        @test eltype(rep_r.class_accuracy) == Union{Missing,Float64}
    end

    @testset "probabilistic model with margins" begin
        mach, _ = train_svm(X, y, svm_cfg(probability = true); rng = StableRNG(1))
        m = LegendMLModel(mach, HaarFeatures(n_samples = 4, bl_window = 0.0u"µs"..0.001u"µs"),
            Dict(0 => :normal, 3 => :other), PropDict())
        labels, margin = predict_with_margin(m, X[[1, end], :])
        @test labels == [0, 3]
        @test all(0.5 .<= margin .<= 1.0)
    end

    @testset "callable on waveforms" begin
        # 4-sample features from 8-sample waveforms (one Haar pass); flat vs step shapes
        fe = HaarFeatures(n_samples = 4, normalize = false, bl_window = 0.0u"µs"..0.02u"µs")
        rng2 = StableRNG(4)
        wf(v) = RDWaveform(range(0u"µs", step = 0.016u"µs", length = 8), repeat(v, inner = 2) .+ 0.01 .* randn(rng2, 8))
        wvfs = ArrayOfRDWaveforms(vcat([wf(zeros(4)) for _ in 1:20], [wf([0.0, 0.0, 1.0, 1.0]) for _ in 1:20]))
        yw = vcat(fill(0, 20), fill(3, 20))
        mach, _ = train_svm(LegendMachineLearning.features(fe, wvfs), yw, svm_cfg(); rng = StableRNG(1))
        m = LegendMLModel(mach, fe, Dict(0 => :normal, 3 => :other), PropDict())
        @test m(wvfs[[1, end]]) == [0, 3]
    end

    @testset "errors" begin
        @test_throws DimensionMismatch train_svm(X, y[1:5], svm_cfg())
        @test_throws "at least two classes" train_svm(X, fill(0, 80), svm_cfg())
        rare_groups = vcat(fill(:a, 79), [:rare])
        @test_throws "no rows in the holdout split" train_svm(X, y, svm_cfg(); groups = rare_groups, rng = StableRNG(1))
    end
end
