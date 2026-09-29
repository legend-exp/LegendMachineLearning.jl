# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

"""
    training_filekey(data, period, category) -> FileKey

Start key of the first analysis run of `period` for `category`.
"""
function training_filekey(data::LegendData, period::DataPeriod, category::DataCategory)
    rinfo = runinfo(data, period)
    cat = Symbol(string(category))
    runs = filter(r -> getproperty(r, cat).is_analysis_run, rinfo)
    isempty(runs) && throw(ArgumentError("period $period has no analysis run in category $category"))
    getproperty(first(runs), cat).startkey
end
export training_filekey

"""
    raw_filekeys(data, period, category) -> Vector{FileKey}

Raw filekeys of every analysis run of `period` in `category`, in run order.
A run that `runinfo` lists but whose raw directory is absent or empty is
skipped with a warning; an empty result is an error.
"""
function raw_filekeys(data::LegendData, period::DataPeriod, category::DataCategory)
    cat = Symbol(string(category))
    filekeys = FileKey[]
    for r in runinfo(data, period)
        getproperty(r, cat).is_analysis_run || continue
        dir = data.tier[DataTier(:raw), category, period, r.run]
        run_filekeys = isdir(dir) ? search_disk(FileKey, dir) : FileKey[]
        if isempty(run_filekeys)
            @warn "no raw files for $period $(r.run) $category"
            continue
        end
        append!(filekeys, run_filekeys)
    end
    isempty(filekeys) && throw(ArgumentError("period $period has no raw files in category $category"))
    filekeys
end
export raw_filekeys

"""
    flag_coincident(ts, aux_ts, window) -> Vector{Bool}

True at each entry of `ts` that lies within `window` of some entry of
`aux_ts`. `aux_ts` must be sorted, `ts` need not be.
"""
function flag_coincident(ts, aux_ts, window)
    issorted(aux_ts) || throw(ArgumentError("aux_ts must be sorted"))
    map(ts) do t
        i = searchsortedfirst(aux_ts, t - window)
        i ≤ lastindex(aux_ts) && aux_ts[i] ≤ t + window
    end
end

"""
    scan_candidates(data, filekeys, det, aux_triggers; ts_window) -> TypedTables.Table

Row scalars of every raw event of `det` in `filekeys`: `filekey`,
`eventnumber`, `timestamp`, `daqenergy`, `baseline`, and the DAQ's counts of
samples saturated at the low and high end of the ADC range, `t_sat_lo` and
`t_sat_hi`, plus one `Bool` column per entry of `aux_triggers` (`name =>
DetectorId` pairs or a dictionary of them), e.g. `:pulser =>
DetectorId(:PULS01)`. A flag is true
where that aux channel has a triggered event (`daqenergy > 0`) within
`ts_window` of the row's timestamp, the same coincidence the hit tier uses
to mark pulser events.

Matching by time rather than by row works in both categories: a cal file
stores only the channels that fired, so an aux channel there holds its own
triggers and its own row count, while a phy file writes every DAQ event to
every channel.
"""
function scan_candidates(data::LegendData, filekeys::AbstractVector{FileKey}, det::DetectorId, aux_triggers;
        ts_window::Unitful.Time=125u"µs")
    aux = collect(aux_triggers)
    flag_names = Symbol[Symbol(first(p)) for p in aux]
    allunique(flag_names) || throw(ArgumentError("aux trigger names must be unique, got $flag_names"))
    aux_dets = DetectorId[DetectorId(last(p)) for p in aux]
    filekey = FileKey[]
    eventnumber = Int[]
    timestamp = typeof(1.0u"s")[]
    daqenergy = Float64[]
    baseline = Float64[]
    t_sat_lo = Int[]
    t_sat_hi = Int[]
    flags = [Bool[] for _ in flag_names]
    for fk in filekeys
        t = read_ldata((:daqenergy, :baseline, :timestamp, :eventnumber, :t_sat_lo, :t_sat_hi),
            data, DataTier(:raw), fk, det)
        for (flag, a) in zip(flags, aux_dets)
            at = read_ldata((:daqenergy, :timestamp), data, DataTier(:raw), fk, a)
            append!(flag, flag_coincident(t.timestamp, sort(at.timestamp[at.daqenergy .> 0]), ts_window))
        end
        append!(filekey, Iterators.repeated(fk, length(t.eventnumber)))
        append!(eventnumber, t.eventnumber)
        append!(timestamp, t.timestamp)
        append!(daqenergy, t.daqenergy)
        append!(baseline, t.baseline)
        append!(t_sat_lo, t.t_sat_lo)
        append!(t_sat_hi, t.t_sat_hi)
    end
    Table(; filekey, eventnumber, timestamp, daqenergy, baseline, t_sat_lo, t_sat_hi,
        NamedTuple{Tuple(flag_names)}(Tuple(flags))...)
end
export scan_candidates


"""
    add_pool_columns(pool) -> TypedTables.Table

`pool` (a table with `filekey`, `baseline` and `daqenergy` columns) with two
columns that place each row within the pool:

- `baseline_dev`: `|baseline - m|` in units of the median of these absolute
  deviations over the pool, where `m` is the median baseline of the pool rows
  of the same `filekey`. The FPGA baseline drifts from file to file, so each
  row is compared with its own file. When more than half the pool sits exactly
  on its file median the unit is zero, and the column holds `NaN` and `Inf`.
- `daqenergy_rank`: empirical quantile of `daqenergy` among the pool rows with
  `daqenergy > 0`, in (0, 1]; 0 for the other rows.
"""
function add_pool_columns(pool)
    isempty(pool) && throw(ArgumentError("the pool is empty"))
    rows_of = Dict{eltype(pool.filekey),Vector{Int}}()
    for (i, fk) in pairs(pool.filekey)
        push!(get!(rows_of, fk, Int[]), i)
    end
    deviation = similar(pool.baseline, Float64)
    for rows in values(rows_of)
        deviation[rows] .= abs.(pool.baseline[rows] .- median(view(pool.baseline, rows)))
    end
    triggered = sort(filter(>(0), pool.daqenergy))
    rank = map(e -> e > 0 ? searchsortedlast(triggered, e) / length(triggered) : 0.0, pool.daqenergy)
    Table(merge(columns(pool), (baseline_dev = deviation ./ median(deviation), daqenergy_rank = rank)))
end

"""
    rule_predicate(expr::AbstractString) -> Function

Predicate of a selection rule written as a LEGEND Julia expression over the
column names of a candidate table, e.g. `"daqenergy > 0 && tail_tau < 2000us"`;
unit literals such as `us` and `ns` are understood (see
`LegendDataManagement.ljl_propfunc`). The returned function maps a table to one
result per row.
"""
function rule_predicate(expr::AbstractString)
    pf = ljl_propfunc(expr)
    t -> pf.(t)
end

# One Bool per row of `cand`; a `missing` or `NaN` result counts as not selected.
function _rule_mask(select, cand)
    v = select(cand)
    length(v) == length(cand) ||
        throw(DimensionMismatch("rule predicate returned $(length(v)) values for $(length(cand)) rows"))
    map(v) do x
        x isa Bool && return x
        (ismissing(x) || (x isa AbstractFloat && isnan(x))) && return false
        throw(ArgumentError("rule predicate returned $(repr(x)), expected a Bool, missing or NaN"))
    end
end

"""
    select_indices(cand, n; rules, uniform, rng) -> (idx, stratum)

Sorted row indices of the candidate table `cand` together with the name of
the stratum each one was drawn by, `length(idx) == min(n, length(cand))`.

`rules` is a sequence of `(name::Symbol, select, fraction)` in which
`select(cand)` gives one `Bool` per row of `cand`; a `missing` or `NaN` result
counts as `false`. The rules are drawn in order, without replacement and each
only from rows the earlier ones left, up to `round(Int, fraction * n)` rows
each; a rule with fewer rows than its quota gives what it has and leaves the
rest of its share to the later strata. `:uniform` then fills the rest of `n`
from the rows for which `uniform(cand)` is true, and `:top_up` from any rows
still left.

`n` at or above `length(cand)` returns every row, labeled by the first rule
it matches, else `:uniform` or `:top_up`.
"""
function select_indices(cand, n::Integer; rules, uniform, rng::AbstractRNG=Random.default_rng())
    n ≥ 0 || throw(ArgumentError("n must be non-negative, got $n"))
    names = Symbol[name for (name, _, _) in rules]
    allunique(names) || throw(ArgumentError("rule names must be unique, got $names"))
    reserved = intersect(names, (:uniform, :top_up))
    isempty(reserved) || throw(ArgumentError("rule names $reserved are reserved"))
    fractions = [fraction for (_, _, fraction) in rules]
    for (name, f) in zip(names, fractions)
        0 ≤ f ≤ 1 || throw(ArgumentError("fraction of rule $name must be between 0 and 1, got $f"))
    end
    sum(fractions; init=0.0) ≤ 1 ||
        throw(ArgumentError("the rule fractions must sum to at most 1, got $(sum(fractions))"))

    matching_uniform = _rule_mask(uniform, cand)
    strata = Tuple{Symbol,Vector{Bool},Int}[(name, _rule_mask(select, cand), round(Int, fraction * n))
        for (name, select, fraction) in rules]
    push!(strata, (:uniform, matching_uniform, n), (:top_up, fill(true, axes(matching_uniform)), n))

    rows = eachindex(matching_uniform)
    if n ≥ length(rows)
        return collect(rows), [strata[findfirst(s -> s[2][i], strata)][1] for i in rows]
    end
    idx = Int[]
    stratum = Symbol[]
    drawn = fill(false, axes(matching_uniform))
    for (name, mask, quota) in strata
        available = [i for i in rows if mask[i] && !drawn[i]]
        take = min(quota, length(available), n - length(idx))
        for i in shuffle(rng, available)[1:take]
            drawn[i] = true
            push!(idx, i)
            push!(stratum, name)
        end
    end
    perm = sortperm(idx)
    idx[perm], stratum[perm]
end
export select_indices

"""
    scan_rows(daqenergy, n, rng) -> Vector{Int}

Sorted indices of `min(n, length(daqenergy))` rows: rows with `daqenergy > 0`
drawn uniformly first, then, if those run out, rows with `daqenergy <= 0`. In
a phy pool almost every row is an untriggered read with `daqenergy == 0`, and
a uniform draw would leave the waveform rules few triggered events to act on.
"""
function scan_rows(daqenergy, n::Integer, rng::AbstractRNG)
    order = vcat(shuffle(rng, findall(>(0), daqenergy)), shuffle(rng, findall(<=(0), daqenergy)))
    sort!(order[1:min(n, length(order))])
end

"""
    _detector_sample(data, filekeys, row, n_det, strata, waveform_column, rng)
        -> (; sample, stats, n_scan, n_matched)

`n_det` rows for the detector of `row`, drawn in two passes. The scalar scan
of `filekeys` gives the candidate rows; rows flagged by an aux trigger named in
`strata.exclude` leave the pool, and `add_pool_columns` places the rest within
it. `scan_rows` picks `strata.n_scan` pool rows, triggered ones first; their
waveforms are read back from each file by timestamp and summarized by
`waveform_stats`. From these scan
rows `select_indices` draws `n_det` by `strata.rules`, and the sample takes the
waveforms already read.

`stats` holds, for the sample rows, the rule inputs the sample does not carry:
the saturation counts, the pool columns and the waveform statistics.
`n_scan` is the number of scan rows and `n_matched` the number of them each
rule selects.
"""
function _detector_sample(data::LegendData, filekeys::AbstractVector{FileKey}, row,
        n_det::Integer, strata::NamedTuple, waveform_column::Symbol, rng)
    det, ch, det_type = row.detector, row.channel, row.det_type
    # `read_ldata` selects raw-tier channel groups by `DetectorId`, never `ChannelId`;
    # `ch` and `det_type` only ever fill the sample's columns below.
    cand = scan_candidates(data, filekeys, det, strata.aux_triggers; strata.ts_window)
    isempty(cand) && throw(ArgumentError("no raw events for $det in $(length(filekeys)) raw files"))
    for name in strata.exclude
        name in propertynames(cand) ||
            throw(ArgumentError("exclude names $name, which is not an aux trigger of the candidate table"))
    end
    excluded = fill(false, length(cand))
    for name in strata.exclude
        excluded .|= getproperty(cand, name)
    end
    any(!, excluded) || throw(ArgumentError("no candidate rows left for $det after excluding $(strata.exclude)"))
    pool = add_pool_columns(cand[findall(!, excluded)])
    length(pool) < n_det &&
        @warn "detector $det: only $(length(pool)) rows in $(length(filekeys)) raw files, quota is $n_det"
    scan = pool[scan_rows(pool.daqenergy, strata.n_scan, rng)]

    # The scan rows are regrouped by file, so the waveforms read file by file line up with them.
    files = unique(scan.filekey)
    rows_of = Dict(fk => Int[] for fk in files)
    for (i, fk) in pairs(scan.filekey)
        push!(rows_of[fk], i)
    end
    wvf_parts = map(files) do fk
        timestamp = scan.timestamp[rows_of[fk]]
        wf = read_ldata((waveform_column, :timestamp), data, (DataTier(:raw), fk, timestamp, det))
        wf.timestamp == timestamp ||
            throw(ErrorException("the waveforms read from $fk for $det do not carry the timestamps they were selected by"))
        collect(getproperty(wf, waveform_column))
    end
    scan = scan[reduce(vcat, [rows_of[fk] for fk in files])]
    wvfs = ArrayOfRDWaveforms(reduce(vcat, wvf_parts))

    stats = waveform_stats(wvfs; strata.bl_window, strata.tail_window)
    full = Table(merge(columns(scan), columns(stats)))
    n_matched = PropDict((name => count(_rule_mask(select, full)) for (name, select, _) in strata.rules)...)
    idx, stratum = select_indices(full, n_det; strata.rules, strata.uniform, rng)
    n = length(idx)
    sample = WaveformSample(wvfs[idx], full.filekey[idx], full.eventnumber[idx], fill(ch, n), fill(det, n),
        fill(det_type, n), full.daqenergy[idx], full.baseline[idx], full.timestamp[idx], stratum)
    stat_names = (:t_sat_lo, :t_sat_hi, :baseline_dev, :daqenergy_rank, propertynames(stats)...)
    sample_stats = Table(NamedTuple{stat_names}(map(c -> getproperty(full, c)[idx], stat_names)))
    (; sample, stats=sample_stats, n_scan=length(full), n_matched)
end

"""
    build_sample(data, filekey, config; rng, chinfo, filekeys) -> (; sample, stats, n_scan, n_matched)

Sample for the period and category of `filekey`: `config.n_max` waveforms
split equally over `config.detectors`, each drawn by `_detector_sample` from
the raw files `filekeys` and read from `config.waveform_column`. A detector
whose pool falls short of its share of `config.n_max` contributes all of it
and `@warn`s about the shortfall instead of silently returning fewer rows than
the caller might expect.

`config.strata` holds the selection parameters:

- `aux_triggers` maps flag names to the detector ids of the aux trigger
  channels, `ts_window` is the coincidence window in µs by which their triggers
  are matched to the detector's events (125 when absent), and `exclude` lists
  the flags whose events stay out of the pool;
- `n_scan` is the number of pool rows per detector whose waveforms are read and
  summarized, rows with `daqenergy > 0` first (see `scan_rows`; 200 000 when
  absent, at least the per-detector share of `n_max`);
- `bl_window` and `tail_window` (`[start, stop]` in µs, required) are the
  windows of `waveform_stats`;
- `rules` is a list of `{name, select, fraction}` with `select` a LEGEND Julia
  expression over the columns of the scan table (see `rule_predicate`), and
  `uniform` the expression of the rows the remainder is drawn from
  (`"daqenergy > 0"` when absent). See `select_indices`.

The scan table has the columns of `scan_candidates` and `add_pool_columns`
and those of `waveform_stats`.

Returns the sample, the rule inputs of its rows (`stats`, one row per sample
row), and per detector the number of scan rows (`n_scan`) and the number of
them each rule selects (`n_matched`), both as `PropDict`s keyed by detector.

`filekeys` defaults to the raw files of every analysis run of the period, so
rows may come from any run of it; `filekey` itself only names the period and
the category. `chinfo` supplies the `detector`, `channel` and `det_type`
columns used to look up each entry of `config.detectors`; it defaults to the
full channel map of `filekey`. Pass any other table with those three columns
to run without a channel map, e.g. a hand-built `TypedTables.Table` against
raw files that predate or lack channel-map coverage.
"""
function build_sample(data::LegendData, filekey::FileKey, config::PropDict;
        rng::AbstractRNG=Random.default_rng(), chinfo=channelinfo(data, filekey; system=:geds),
        filekeys::AbstractVector{FileKey}=raw_filekeys(data, filekey.period, filekey.category))
    detectors = DetectorId.(config.detectors)
    isempty(detectors) && throw(ArgumentError("config.detectors is empty"))
    isempty(filekeys) && throw(ArgumentError("no raw files for $(filekey.period) $(filekey.category)"))
    n_det = cld(Int(config.n_max), length(detectors))
    sc = config.strata
    for key in (:bl_window, :tail_window)
        haskey(sc, key) || throw(ArgumentError("config.strata.$key is required"))
    end
    window(v) = Float64(v[1]) * u"µs" .. Float64(v[2]) * u"µs"
    strata = (aux_triggers=[Symbol(k) => DetectorId(string(v)) for (k, v) in get(sc, :aux_triggers, PropDict())],
        exclude=Symbol.(get(sc, :exclude, Symbol[])),
        ts_window=Float64(get(sc, :ts_window, 125)) * u"µs",
        n_scan=Int(get(sc, :n_scan, 200_000)),
        bl_window=window(sc.bl_window), tail_window=window(sc.tail_window),
        rules=[(Symbol(r.name), rule_predicate(String(r.select)), Float64(r.fraction)) for r in get(sc, :rules, [])],
        uniform=rule_predicate(String(get(sc, :uniform, "daqenergy > 0"))))
    strata.n_scan ≥ n_det ||
        throw(ArgumentError("config.strata.n_scan = $(strata.n_scan) is below the per-detector share $n_det of n_max"))
    rows = map(detectors) do det
        i = findfirst(==(det), chinfo.detector)
        isnothing(i) && throw(ArgumentError("detector $det not found in the channel map of $filekey"))
        chinfo[i]
    end
    parts = [_detector_sample(data, filekeys, row, n_det, strata, Symbol(config.waveform_column), rng)
             for row in rows]
    by_detector(f) = PropDict((Symbol(string(row.detector)) => f(p) for (row, p) in zip(rows, parts))...)
    (sample=vcat((p.sample for p in parts)...), stats=reduce(vcat, [p.stats for p in parts]),
        n_scan=by_detector(p -> p.n_scan), n_matched=by_detector(p -> p.n_matched))
end
export build_sample
