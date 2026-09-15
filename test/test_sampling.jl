# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

using Test
using LegendMachineLearning
using LegendMachineLearning: select_indices, training_filekey, build_sample
using StableRNGs, PropDicts
using LegendTestData
using LegendDataManagement

@testset "select_indices" begin
    rng = StableRNG(11)
    e = vcat(zeros(30), rand(rng, 100.0:5000.0, 970))
    idx = select_indices(e, 100; zero_energy_fraction=0.1, rng)
    @test length(idx) == 100
    @test issorted(idx) && allunique(idx)
    @test count(iszero, e[idx]) == 10
    # fewer zero-energy events than the quota: take all of them
    e2 = vcat(zeros(3), rand(rng, 100.0:5000.0, 97))
    idx2 = select_indices(e2, 50; zero_energy_fraction=0.2, rng)
    @test count(iszero, e2[idx2]) == 3 && length(idx2) == 50
    # request more than available
    @test select_indices(e2, 500; zero_energy_fraction=0.2, rng) == 1:100
    @test_throws "between 0 and 1" select_indices(e2, 5; zero_energy_fraction=1.5, rng)
end

@testset "build_sample on LegendTestData" begin
    activate_legend_test_data_config()
    l200 = LegendData(:l200)
    period, category = DataPeriod(3), DataCategory(:cal)
    fk = training_filekey(l200, period, category)
    @test fk.period == period && fk.category == category
    cfg = PropDict(
        :detectors => ["V01240A"],   # channel ch1084804 in the p03 channel map (see Step 3)
        :n_max => 40, :n_files_max => 1, :waveform_column => :waveform,
        :strata => PropDict(:zero_energy_fraction => 0.1))
    s = build_sample(l200, fk, cfg; rng=StableRNG(5))
    @test length(s) == 40
    @test all(==(fk.period), getproperty.(s.filekey, :period))
    @test allunique(zip(s.filekey, s.eventnumber))
    @test all(s.det_type .== s.det_type[1])
    @test_throws "not found" build_sample(l200, fk, merge(cfg, PropDict(:detectors => ["V00000X"])); rng=StableRNG(5))
end
