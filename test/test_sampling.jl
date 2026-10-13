# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

using Test
using LegendMachineLearning
using LegendMachineLearning:
    select_indices, flag_coincident, scan_candidates, build_sample,
    add_pool_columns, rule_predicate, scan_rows
using Unitful
using StableRNGs, PropDicts
using LegendTestData
using LegendDataManagement
using TypedTables
using RadiationDetectorSignals: ArrayOfRDWaveforms


@testset "select_indices" begin
    rng = StableRNG(11)
    n_rows = 1000
    daqenergy = vcat(zeros(200), rand(rng, 100.0:5000.0, 750), fill(9000.0, 50))
    saturated = vcat(falses(950), trues(50))
    tail_tau = fill(400.0u"µs", n_rows)
    tail_tau[301:340] .= 50.0u"µs"
    cand = Table(; daqenergy, saturated, tail_tau)
    rules = [(:zero_energy, t -> t.daqenergy .== 0, 0.2),
        (:saturated, t -> t.saturated, 0.1),
        (:bad_tau, t -> t.tail_tau .< 150u"µs", 0.1)]
    uniform = t -> t.daqenergy .> 0

    idx, stratum = select_indices(cand, 200; rules, uniform, rng)
    @test length(idx) == 200 && length(stratum) == 200
    @test issorted(idx) && allunique(idx)
    @test [count(==(s), stratum) for s in (:zero_energy, :saturated, :bad_tau, :uniform)] == [40, 20, 20, 120]
    drawn_as(name) = [i for (i, s) in zip(idx, stratum) if s == name]
    @test all(iszero, daqenergy[drawn_as(:zero_energy)])
    @test all(saturated[drawn_as(:saturated)])
    @test all(<(150u"µs"), tail_tau[drawn_as(:bad_tau)])
    @test all(>(0), daqenergy[drawn_as(:uniform)])

    # A rule with fewer rows than its quota gives what it has; the later strata absorb the rest.
    short = [(:saturated, t -> t.saturated, 0.3), (:bad_tau, t -> t.tail_tau .< 150u"µs", 0.1)]
    _, stratum_short = select_indices(cand, 200; rules = short, uniform, rng)
    @test [count(==(s), stratum_short) for s in (:saturated, :bad_tau, :uniform)] == [50, 20, 130]

    # The whole table, labeled by the first rule each row matches.
    idx_all, stratum_all = select_indices(cand, 10_000; rules, uniform, rng)
    @test idx_all == 1:n_rows
    @test stratum_all[[1, 310, 960, 500]] == [:zero_energy, :bad_tau, :saturated, :uniform]

    # Too few rows for the remainder of `n` match `uniform`: the rest tops it up.
    scarce = Table(daqenergy = vcat(zeros(60), fill(500.0, 5)))
    _, stratum_s = select_indices(scarce, 20; rules = [(:zero_energy, t -> t.daqenergy .== 0, 0.1)], uniform, rng)
    @test [count(==(s), stratum_s) for s in (:zero_energy, :uniform, :top_up)] == [2, 5, 13]

    # `missing` and `NaN` in a predicate result count as not selected.
    odd = Table(daqenergy = fill(1.0, 4), x = Any[missing, NaN, true, false])
    @test last(select_indices(odd, 4; rules = [(:x, t -> t.x, 1.0)], uniform, rng)) ==
          [:uniform, :uniform, :x, :uniform]
    @test_throws "expected a Bool" select_indices(odd, 2; rules = [(:x, t -> fill(1, 4), 0.5)], uniform, rng)
    @test_throws DimensionMismatch select_indices(odd, 2; rules = [(:x, t -> [true], 0.5)], uniform, rng)

    @test_throws "fraction of rule zero_energy must be between 0 and 1" select_indices(cand, 10;
        rules = [(:zero_energy, t -> t.daqenergy .== 0, 1.5)], uniform, rng)
    @test_throws "sum to at most 1" select_indices(cand, 10;
        rules = [(:a, t -> t.saturated, 0.6), (:b, t -> t.saturated, 0.6)], uniform, rng)
    @test_throws "must be unique" select_indices(cand, 10;
        rules = [(:a, t -> t.saturated, 0.1), (:a, t -> t.saturated, 0.1)], uniform, rng)
    @test_throws "reserved" select_indices(cand, 10; rules = [(:uniform, t -> t.saturated, 0.1)], uniform, rng)
    @test_throws "n must be non-negative" select_indices(cand, -1; rules, uniform, rng)
end

@testset "rule_predicate" begin
    t = Table(daqenergy = [0.0, 10.0, 10.0, 10.0], tail_tau = [400.0, 400.0, 100.0, NaN]u"µs",
        t50 = [41.0, 43.0, 43.0, NaN]u"µs")
    @test rule_predicate("daqenergy > 0 && !(tail_tau > 150us && tail_tau < 2000us)")(t) == [false, false, true, true]
    @test rule_predicate("t50 > 42000ns")(t) == [false, true, true, false]
    @test_throws "not allowed" rule_predicate("run(`ls`)")
end

@testset "scan_rows" begin
    rng = StableRNG(3)
    e = vcat(zeros(50), fill(10.0, 20), zeros(30))
    few = scan_rows(e, 15, rng)
    @test length(few) == 15 && issorted(few) && allunique(few) && all(>(0), e[few])
    many = scan_rows(e, 40, rng)
    @test length(many) == 40 && issorted(many) && count(>(0), e[many]) == 20
    @test scan_rows(e, 1000, rng) == 1:100
end

@testset "add_pool_columns" begin
    # The baseline sits about 2000 ADC apart in the two files: each row is compared with
    # the median of its own file, so only the two planted rows stand out.
    rng = StableRNG(5)
    m = 100
    bl_a = 15000 .+ rand(rng, -30.0:1.0:30.0, m)
    bl_b = 17000 .+ rand(rng, -30.0:1.0:30.0, m)
    bl_a[3] = 15600.0
    bl_b[7] = 17600.0
    fk_a, fk_b = FileKey("l200-p03-r001-cal-20230318T012144Z"), FileKey("l200-p03-r001-cal-20230318T022144Z")
    pool = Table(filekey = vcat(fill(fk_a, m), fill(fk_b, m)), baseline = vcat(bl_a, bl_b),
        daqenergy = vcat(zeros(10), collect(1.0:190.0)))
    p = add_pool_columns(pool)
    @test findall(>(5), p.baseline_dev) == [3, m + 7]
    @test all(iszero, p.daqenergy_rank[1:10])
    @test p.daqenergy_rank[11] ≈ 1 / 190 && p.daqenergy_rank[end] == 1
    @test issorted(p.daqenergy_rank[11:end])
    @test p.baseline == pool.baseline
    @test_throws "pool is empty" add_pool_columns(pool[Int[]])
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
    # `raw_filekeys`. The rows are drawn without aux-trigger flags; those are
    # covered by the `flag_coincident` tests.
    withenv("LEGEND_DATA_CONFIG" => joinpath(legend_test_data_path(), "data", "lh5", "prod-ref-l200", "config.json")) do
        l200 = LegendData(:l200)
        fks = search_disk(FileKey, l200.tier[DataTier(:raw), :cal, DataPeriod(18), DataRun(0)])
        fk = first(fks)
        det = DetectorId(:V97061A)
        chinfo = Table(detector = [det], channel = [ChannelId(1104000)], det_type = [:icpc])
        rules = [PropDict(:name => "zero_energy", :select => "daqenergy == 0", :fraction => 0.2),
            PropDict(:name => "late", :select => "t50 > 50000ns", :fraction => 0.3),
            PropDict(:name => "high_energy", :select => "daqenergy_rank > 0.8", :fraction => 0.2)]
        cfg = PropDict(
            :detectors => ["V97061A"],
            :n_max => 12, :waveform_column => :waveform_presummed,
            :strata => PropDict(:aux_triggers => PropDict(), :exclude => String[], :n_scan => 15,
                :bl_window => [0.0, 39.0], :tail_window => [70.0, 95.0], :rules => rules,
                :uniform => "daqenergy > 0"))

        cand = scan_candidates(l200, fks, det, ())
        @test TypedTables.columnnames(cand) ==
              (:filekey, :eventnumber, :timestamp, :daqenergy, :baseline, :t_sat_lo, :t_sat_hi)
        @test length(cand) == 20

        drawn = build_sample(l200, fk, cfg; rng = StableRNG(5), chinfo, filekeys = fks)
        s = drawn.sample
        @test length(s) == 12 && length(drawn.stats) == 12
        @test allunique(zip(s.filekey, s.eventnumber))
        @test all(in(fks), s.filekey)
        @test all(==(:icpc), s.det_type)
        @test all(==(det), s.detector)
        @test s.waveform isa ArrayOfRDWaveforms
        @test all(in((:zero_energy, :late, :high_energy, :uniform, :top_up)), s.stratum)
        @test all(>(0.8), drawn.stats.daqenergy_rank[s.stratum .== :high_energy])
        @test all(>(50000u"ns"), drawn.stats.t50[s.stratum .== :late])
        @test (:t_sat_lo, :baseline_dev, :daqenergy_rank, :tail_tau, :n_triggers) ⊆ propertynames(drawn.stats)
        @test drawn.n_scan == PropDict(:V97061A => 15)
        @test sort(collect(keys(drawn.n_matched.V97061A))) == [:high_energy, :late, :zero_energy]
        @test drawn.n_matched.V97061A.zero_energy == 0

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

        # Restricted to the first raw file (10 rows), the pool falls short of the
        # quota of 12: all 10 rows are returned, with a warning.
        drawn1 = @test_logs (:warn, r"only 10 rows") match_mode=:any build_sample(l200, fk, cfg;
            rng = StableRNG(5), chinfo, filekeys = fks[1:1])
        @test length(drawn1.sample) == 10
        @test length(unique(drawn1.sample.filekey)) == 1

        # "V00000A" parses as a syntactically valid detector id (unlike "V00000X",
        # whose suffix letter the HPGe id regex rejects before `build_sample`'s
        # own "not found" check ever runs) but names no detector in `chinfo`.
        @test_throws "not found" build_sample(l200, fk, merge(cfg, PropDict(:detectors => ["V00000A"]));
            rng = StableRNG(5), chinfo, filekeys = fks)
        @test_throws "is empty" build_sample(l200, fk, merge(cfg, PropDict(:detectors => String[]));
            rng = StableRNG(5), chinfo, filekeys = fks)
        @test_throws "no raw files" build_sample(l200, fk, cfg; rng = StableRNG(5), chinfo, filekeys = FileKey[])
        with_strata(f) = PropDict((k => (k == :strata ? f(v) : v) for (k, v) in cfg)...)
        small_scan = with_strata(st -> PropDict((k => (k == :n_scan ? 5 : v) for (k, v) in st)...))
        @test_throws "below the per-detector share" build_sample(l200, fk, small_scan;
            rng = StableRNG(5), chinfo, filekeys = fks)
        no_window = with_strata(st -> PropDict((k => v for (k, v) in st if k != :tail_window)...))
        @test_throws "tail_window is required" build_sample(l200, fk, no_window;
            rng = StableRNG(5), chinfo, filekeys = fks)
    end
end
