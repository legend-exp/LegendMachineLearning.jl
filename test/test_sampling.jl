# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

using Test
using LegendMachineLearning
using LegendMachineLearning: select_indices, training_filekey, build_sample
using StableRNGs, PropDicts
using LegendTestData
using LegendDataManagement
using TypedTables
using RadiationDetectorSignals: ArrayOfRDWaveforms

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
    # Point straight at the raw-tier test data, as LegendDataManagement's own
    # test suite does; `activate_legend_test_data_config()` targets a config
    # file without the `setups` wrapper `LegendDataConfig` requires, and the
    # p03 raw files use the old `chXXXX/raw` group layout that current
    # `read_ldata` cannot read. The p18 raw files have detector-named groups
    # but their detectors are absent from the test-data channel maps, so
    # `chinfo` is supplied by hand rather than via `channelinfo`.
    ENV["LEGEND_DATA_CONFIG"] = joinpath(legend_test_data_path(), "data", "lh5", "prod-ref-l200", "config.json")
    l200 = LegendData(:l200)
    raw_filekeys = search_disk(FileKey, l200.tier[DataTier(:raw), :cal, DataPeriod(18), DataRun(0)])
    fk = first(raw_filekeys)
    chinfo = Table(detector=[DetectorId(:V97061A)], channel=[ChannelId(1104000)], det_type=[:icpc])
    cfg = PropDict(
        :detectors => ["V97061A"],
        :n_max => 12, :n_files_max => 2, :waveform_column => :waveform_presummed,
        :strata => PropDict(:zero_energy_fraction => 0.1))

    s = build_sample(l200, fk, cfg; rng=StableRNG(5), chinfo)
    @test length(s) == 12
    @test allunique(zip(s.filekey, s.eventnumber))
    @test all(in(raw_filekeys), s.filekey)
    @test all(==(:icpc), s.det_type)
    @test all(==(DetectorId(:V97061A)), s.detector)
    @test s.waveform isa ArrayOfRDWaveforms

    cfg1 = merge(cfg, PropDict(:n_files_max => 1))
    s1 = build_sample(l200, fk, cfg1; rng=StableRNG(5), chinfo)
    @test length(s1) <= 10
    @test length(unique(s1.filekey)) == 1

    # "V00000A" parses as a syntactically valid detector id (unlike "V00000X",
    # whose suffix letter the HPGe id regex rejects before `build_sample`'s
    # own "not found" check ever runs) but names no detector in `chinfo`.
    @test_throws "not found" build_sample(l200, fk, merge(cfg, PropDict(:detectors => ["V00000A"])); rng=StableRNG(5), chinfo)
end
