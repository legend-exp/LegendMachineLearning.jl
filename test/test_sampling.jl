# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

using Test
using LegendMachineLearning
using LegendMachineLearning: select_indices, flag_coincident, scan_candidates,
    training_filekey, build_sample
using Unitful
using StableRNGs, PropDicts
using LegendTestData
using LegendDataManagement
using Statistics: median, quantile
using TypedTables
using RadiationDetectorSignals: ArrayOfRDWaveforms

@testset "select_indices" begin
    rng = StableRNG(11)
    n_rows = 1000
    daqenergy = vcat(zeros(200), rand(rng, 100.0:5000.0, 750), fill(9000.0, 50))
    # Rows 201:250 carry far-off baselines at an energy well below the high-energy
    # threshold, so they can only be drawn as baseline outliers.
    daqenergy[201:250] .= 1000.0
    baseline = 15000 .+ 20 .* randn(rng, n_rows)
    baseline[201:250] .= 15400.0
    pulser = falses(n_rows)
    pulser[1:40] .= true
    forced_trigger = falses(n_rows)
    forced_trigger[1:80] .= true
    cand = Table(; daqenergy, baseline, pulser, forced_trigger)

    n = 200
    idx, stratum = select_indices(cand, n; exclude=[:pulser, :forced_trigger], zero_energy=0.2,
        high_energy=0.15, high_energy_quantile=0.9, baseline_outlier=0.15, baseline_mad=5.0, rng)
    @test length(idx) == n && length(stratum) == n
    @test issorted(idx) && allunique(idx)
    @test !any(pulser[idx]) && !any(forced_trigger[idx])
    @test count(==(:zero_energy), stratum) == 40
    @test count(==(:high_energy), stratum) == 30
    @test count(==(:baseline_outlier), stratum) == 30
    @test count(==(:uniform), stratum) == 100

    pool = findall(i -> !pulser[i] && !forced_trigger[i], eachindex(daqenergy))
    med = median(baseline[pool])
    dev = median(abs.(baseline[pool] .- med))
    threshold = quantile([daqenergy[i] for i in pool if daqenergy[i] > 0], 0.9)
    drawn_as(name) = [i for (i, s) in zip(idx, stratum) if s == name]
    @test all(iszero, daqenergy[drawn_as(:zero_energy)])
    @test all(≥(threshold), daqenergy[drawn_as(:high_energy)])
    @test all(b -> abs(b - med) > 5.0 * dev, baseline[drawn_as(:baseline_outlier)])
    @test all(>(0), daqenergy[drawn_as(:uniform)])

    # The whole pool, labeled by the first rule each row satisfies.
    idx_all, stratum_all = select_indices(cand, 10_000; exclude=[:pulser, :forced_trigger],
        zero_energy=0.2, high_energy=0.15, baseline_outlier=0.15, baseline_mad=5.0, rng)
    @test idx_all == pool
    @test length(stratum_all) == length(pool)
    @test all(iszero, daqenergy[[i for (i, s) in zip(idx_all, stratum_all) if s == :zero_energy]])
    @test all(>(0), daqenergy[[i for (i, s) in zip(idx_all, stratum_all) if s == :uniform]])

    # Too few triggered rows for the remainder of `n`: zero-energy rows top it up.
    scarce = Table(daqenergy=vcat(zeros(60), fill(500.0, 5)), baseline=fill(15000.0, 65))
    idx_s, stratum_s = select_indices(scarce, 20; zero_energy=0.1, rng)
    @test length(idx_s) == 20 && issorted(idx_s) && allunique(idx_s)
    @test count(==(:zero_energy), stratum_s) == 2
    @test count(==(:uniform), stratum_s) == 5
    @test count(==(:top_up), stratum_s) == 13

    # The baseline sits about 2000 ADC apart in the two groups. Against the pool-wide
    # median that offset swamps the spread and no row stands out; against the median of
    # its own group only the two planted rows do.
    m = 100
    bl_a = 15000 .+ rand(rng, -30.0:1.0:30.0, m)
    bl_b = 17000 .+ rand(rng, -30.0:1.0:30.0, m)
    bl_a[3] = 15600.0
    bl_b[7] = 17600.0
    grouped = Table(daqenergy=fill(1000.0, 2m), baseline=vcat(bl_a, bl_b),
        file=vcat(fill(:a, m), fill(:b, m)))
    idx_g, stratum_g = select_indices(grouped, 40; baseline_outlier=0.05, baseline_mad=5.0,
        baseline_group=:file, rng)
    @test [i for (i, s) in zip(idx_g, stratum_g) if s == :baseline_outlier] == [3, m + 7]
    _, stratum_p = select_indices(grouped, 40; baseline_outlier=0.05, baseline_mad=5.0, rng)
    @test count(==(:baseline_outlier), stratum_p) == 0
    @test_throws "baseline_group names" select_indices(grouped, 10; baseline_group=:run, rng)

    @test_throws "zero_energy must be between 0 and 1" select_indices(cand, 10; zero_energy=1.5, rng)
    @test_throws "sum to at most 1" select_indices(cand, 10; zero_energy=0.6, high_energy=0.6, rng)
    @test_throws "high_energy_quantile must be between 0 and 1" select_indices(cand, 10; high_energy_quantile=1.5, rng)
    @test_throws "n must be non-negative" select_indices(cand, -1; rng)
    @test_throws "not a column of the candidate table" select_indices(cand, 10; exclude=[:muon_veto], rng)
end

@testset "flag_coincident" begin
    window = 125.0u"µs"
    ts = [1.0, 2.0, 3.0, 4.0]u"s"
    @test flag_coincident(ts, [2.0, 4.0]u"s", window) == [false, true, false, true]
    # an aux trigger counts from either side of the row, and `ts` need not be sorted
    @test flag_coincident([3.0, 1.0]u"s", [1.0 + 100e-6, 2.0 - 100e-6]u"s", window) == [false, true]
    @test flag_coincident([2.0]u"s", [2.0 - 100e-6]u"s", window) == [true]
    @test flag_coincident([1.0, 2.0]u"s", [1.0 + 200e-6]u"s", window) == [false, false]
    @test flag_coincident(ts, typeof(1.0u"s")[], window) == fill(false, 4)
    @test flag_coincident(typeof(1.0u"s")[], [1.0]u"s", window) == Bool[]
    @test_throws "must be sorted" flag_coincident(ts, [4.0, 2.0]u"s", window)
end

@testset "build_sample on LegendTestData" begin
    # Point straight at the raw-tier test data, as LegendDataManagement's own
    # test suite does; `activate_legend_test_data_config()` targets a config
    # file without the `setups` wrapper `LegendDataConfig` requires, and the
    # p03 raw files use the old `chXXXX/raw` group layout that current
    # `read_ldata` cannot read. The p18 raw files have detector-named groups
    # but their detectors are absent from the test-data channel maps, so
    # `chinfo` is supplied by hand rather than via `channelinfo`. The test
    # config defines no metadata path either, so `runinfo` is out of reach and
    # the candidate files are listed with `search_disk` instead of
    # `raw_filekeys`. The files hold a single germanium group and no aux
    # trigger channels, hence the empty `aux_triggers` and `exclude`.
    withenv("LEGEND_DATA_CONFIG" => joinpath(legend_test_data_path(), "data", "lh5", "prod-ref-l200", "config.json")) do
        l200 = LegendData(:l200)
        fks = search_disk(FileKey, l200.tier[DataTier(:raw), :cal, DataPeriod(18), DataRun(0)])
        fk = first(fks)
        det = DetectorId(:V97061A)
        chinfo = Table(detector=[det], channel=[ChannelId(1104000)], det_type=[:icpc])
        cfg = PropDict(
            :detectors => ["V97061A"],
            :n_max => 12, :waveform_column => :waveform_presummed,
            :strata => PropDict(:aux_triggers => PropDict(), :exclude => String[],
                :zero_energy => 0.2, :high_energy => 0.15, :high_energy_quantile => 0.9,
                :baseline_outlier => 0.15, :baseline_mad => 5.0))

        cand = scan_candidates(l200, fks, det, ())
        @test TypedTables.columnnames(cand) == (:filekey, :eventnumber, :timestamp, :daqenergy, :baseline)
        @test length(cand) == 20

        s = build_sample(l200, fk, cfg; rng=StableRNG(5), chinfo, filekeys=fks)
        @test length(s) == 12
        @test allunique(zip(s.filekey, s.eventnumber))
        @test all(in(fks), s.filekey)
        @test all(==(:icpc), s.det_type)
        @test all(==(det), s.detector)
        @test s.waveform isa ArrayOfRDWaveforms
        @test all(in((:zero_energy, :high_energy, :baseline_outlier, :uniform, :top_up)), s.stratum)
        @test length(s.baseline) == 12 && length(s.timestamp) == 12

        # Every sampled row traces back to the exact raw event it was read from.
        for i in 1:length(s)
            raw = read_ldata((:daqenergy, :baseline, :timestamp, :eventnumber, :waveform_presummed),
                l200, DataTier(:raw), s.filekey[i], det)
            j = only(findall(==(s.eventnumber[i]), raw.eventnumber))
            @test raw.daqenergy[j] == s.daqenergy[i]
            @test raw.baseline[j] == s.baseline[i]
            @test raw.timestamp[j] == s.timestamp[i]
            @test raw.waveform_presummed[j].signal == s.waveform[i].signal
        end

        # Restricted to the first raw file (10 rows), the candidate pool falls
        # short of the quota of 12: all 10 rows are returned, with a warning.
        s1 = @test_logs (:warn, r"only 10 rows") match_mode=:any build_sample(l200, fk, cfg;
            rng=StableRNG(5), chinfo, filekeys=fks[1:1])
        @test length(s1) == 10
        @test length(unique(s1.filekey)) == 1

        # "V00000A" parses as a syntactically valid detector id (unlike "V00000X",
        # whose suffix letter the HPGe id regex rejects before `build_sample`'s
        # own "not found" check ever runs) but names no detector in `chinfo`.
        @test_throws "not found" build_sample(l200, fk, merge(cfg, PropDict(:detectors => ["V00000A"]));
            rng=StableRNG(5), chinfo, filekeys=fks)
        @test_throws "is empty" build_sample(l200, fk, merge(cfg, PropDict(:detectors => String[]));
            rng=StableRNG(5), chinfo, filekeys=fks)
        @test_throws "no raw files" build_sample(l200, fk, cfg; rng=StableRNG(5), chinfo, filekeys=FileKey[])
    end
end
