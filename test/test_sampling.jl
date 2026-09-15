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
    # fewer non-zero events than the remainder: the zero stratum tops up
    e3 = vcat(zeros(30), rand(rng, 100.0:5000.0, 5))
    idx3 = select_indices(e3, 20; zero_energy_fraction=0.1, rng)
    @test length(idx3) == 20
    @test count(iszero, e3[idx3]) == 15
end

@testset "build_sample on LegendTestData" begin
    # Point straight at the raw-tier test data, as LegendDataManagement's own
    # test suite does; `activate_legend_test_data_config()` targets a config
    # file without the `setups` wrapper `LegendDataConfig` requires, and the
    # p03 raw files use the old `chXXXX/raw` group layout that current
    # `read_ldata` cannot read. The p18 raw files have detector-named groups
    # but their detectors are absent from the test-data channel maps, so
    # `chinfo` is supplied by hand rather than via `channelinfo`.
    withenv("LEGEND_DATA_CONFIG" => joinpath(legend_test_data_path(), "data", "lh5", "prod-ref-l200", "config.json")) do
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

        # Every sampled row traces back to the exact raw event it was read from.
        for i in 1:length(s)
            raw = read_ldata((:daqenergy, :eventnumber, :waveform_presummed), l200, DataTier(:raw), s.filekey[i], DetectorId(:V97061A))
            j = only(findall(==(s.eventnumber[i]), raw.eventnumber))
            @test raw.daqenergy[j] == s.daqenergy[i]
            @test raw.waveform_presummed[j].signal == s.waveform[i].signal
        end

        # with both raw files available, the run-wide quota of 12 is still met
        cfg2 = merge(cfg, PropDict(:n_files_max => 2))
        s2 = build_sample(l200, fk, cfg2; rng=StableRNG(5), chinfo)
        @test length(s2) == 12

        # restricted to the first raw file (10 rows), the candidate pool falls
        # short of the quota of 12: all 10 rows are returned, with a warning
        cfg1 = merge(cfg, PropDict(:n_files_max => 1))
        s1 = @test_logs (:warn, r"only 10 rows") match_mode=:any build_sample(l200, fk, cfg1; rng=StableRNG(5), chinfo)
        @test length(s1) == 10
        @test length(unique(s1.filekey)) == 1

        # "V00000A" parses as a syntactically valid detector id (unlike "V00000X",
        # whose suffix letter the HPGe id regex rejects before `build_sample`'s
        # own "not found" check ever runs) but names no detector in `chinfo`.
        @test_throws "not found" build_sample(l200, fk, merge(cfg, PropDict(:detectors => ["V00000A"])); rng=StableRNG(5), chinfo)
        @test_throws "is empty" build_sample(l200, fk, merge(cfg, PropDict(:detectors => String[])); rng=StableRNG(5), chinfo)
        @test_throws "no raw files" build_sample(l200, FileKey("l200-p18-r099-cal-20251107T191821Z"), cfg; rng=StableRNG(5), chinfo)
    end
end
