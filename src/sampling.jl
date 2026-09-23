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
`eventnumber`, `timestamp`, `daqenergy` and `baseline`, plus one `Bool`
column per entry of `aux_triggers` (`name => DetectorId` pairs or a
dictionary of them), e.g. `:pulser => DetectorId(:PULS01)`. A flag is true
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
    flags = [Bool[] for _ in flag_names]
    for fk in filekeys
        t = read_ldata((:daqenergy, :baseline, :timestamp, :eventnumber), data, DataTier(:raw), fk, det)
        for (flag, a) in zip(flags, aux_dets)
            at = read_ldata((:daqenergy, :timestamp), data, DataTier(:raw), fk, a)
            append!(flag, flag_coincident(t.timestamp, sort(at.timestamp[at.daqenergy .> 0]), ts_window))
        end
        append!(filekey, Iterators.repeated(fk, length(t.eventnumber)))
        append!(eventnumber, t.eventnumber)
        append!(timestamp, t.timestamp)
        append!(daqenergy, t.daqenergy)
        append!(baseline, t.baseline)
    end
    Table(; filekey, eventnumber, timestamp, daqenergy, baseline,
        NamedTuple{Tuple(flag_names)}(Tuple(flags))...)
end
export scan_candidates

"""
    select_indices(cand, n; exclude, zero_energy, high_energy, high_energy_quantile,
                   baseline_outlier, baseline_mad, baseline_group, rng) -> (idx, stratum)

Sorted row indices of the candidate table `cand` together with the name of
the stratum each one was drawn by, `length(idx) == min(n, pool size)`. The
pool holds the rows in which no flag column named by `exclude` is true;
`cand` needs a `daqenergy` and a `baseline` column besides those.

The strata are drawn in turn, without replacement and each one only from
rows the earlier ones left: `:zero_energy` (`daqenergy == 0`),
`:high_energy` (`daqenergy` at or above the `high_energy_quantile` quantile
of the pool's triggered energies), `:baseline_outlier` (`baseline` further
than `baseline_mad` times the pool's median absolute deviation from its
reference baseline), and `:uniform` for the rest of `n` from the
triggered rows. Each of the first three is capped at `round(Int, fraction *
n)` rows, and at what the pool holds, so a stratum that runs short leaves
its share to the later ones. Zero-energy rows labeled `:top_up` make up the
count when the triggered rows run out.

The reference baseline is the pool's median. When `baseline_group` names a
column of `cand`, each row is instead referred to the median of the pool
rows sharing its value in that column, which keeps a baseline that shifts
between groups from turning a whole group into outliers.

`n` at or above the pool size returns the whole pool, each row labeled by
the first stratum rule it satisfies.
"""
function select_indices(cand, n::Integer;
        exclude=Symbol[], zero_energy::Real=0, high_energy::Real=0,
        high_energy_quantile::Real=0.9, baseline_outlier::Real=0, baseline_mad::Real=5,
        baseline_group=nothing, rng::AbstractRNG=Random.default_rng())
    n ≥ 0 || throw(ArgumentError("n must be non-negative, got $n"))
    0 ≤ high_energy_quantile ≤ 1 ||
        throw(ArgumentError("high_energy_quantile must be between 0 and 1, got $high_energy_quantile"))
    fractions = (; zero_energy, high_energy, baseline_outlier)
    for (name, f) in pairs(fractions)
        0 ≤ f ≤ 1 || throw(ArgumentError("$name must be between 0 and 1, got $f"))
    end
    sum(fractions) ≤ 1 ||
        throw(ArgumentError("the stratum fractions must sum to at most 1, got $(sum(fractions))"))
    for name in exclude
        name in propertynames(cand) ||
            throw(ArgumentError("exclude names $name, which is not a column of the candidate table"))
    end

    isnothing(baseline_group) || baseline_group in propertynames(cand) ||
        throw(ArgumentError("baseline_group names $baseline_group, which is not a column of the candidate table"))

    e, bl = cand.daqenergy, cand.baseline
    excluded = fill(false, axes(e))
    for name in exclude
        excluded .|= getproperty(cand, name)
    end
    pool = findall(!, excluded)
    # Only pool rows are ever asked for their deviation; the rest keep the zero they start at.
    deviation = zeros(float(eltype(bl)), axes(bl))
    if isnothing(baseline_group)
        isempty(pool) || (deviation[pool] .= bl[pool] .- median(bl[pool]))
    else
        group = getproperty(cand, baseline_group)
        for g in unique(group[pool])
            rows = [i for i in pool if isequal(group[i], g)]
            deviation[rows] .= bl[rows] .- median(bl[rows])
        end
    end
    spread = isempty(pool) ? zero(eltype(deviation)) : median(abs.(deviation[pool]))
    triggered = [e[i] for i in pool if e[i] > 0]
    # With no triggered row in the pool the high-energy stratum stays empty instead of
    # matching every row.
    threshold = isempty(triggered) ? Inf : quantile(triggered, high_energy_quantile)
    rules = ((:zero_energy, i -> iszero(e[i]), round(Int, zero_energy * n)),
        (:high_energy, i -> e[i] ≥ threshold, round(Int, high_energy * n)),
        (:baseline_outlier, i -> abs(deviation[i]) > baseline_mad * spread, round(Int, baseline_outlier * n)),
        (:uniform, i -> e[i] > 0, n),
        (:top_up, i -> iszero(e[i]), n))

    if n ≥ length(pool)
        return pool, map(pool) do i
            k = findfirst(r -> r[2](i), rules)
            isnothing(k) ? :uniform : rules[k][1]
        end
    end
    idx = Int[]
    stratum = Symbol[]
    drawn = Set{Int}()
    for (name, rule, quota) in rules
        available = [i for i in pool if !(i in drawn) && rule(i)]
        take = min(quota, length(available), n - length(idx))
        for i in shuffle(rng, available)[1:take]
            push!(drawn, i)
            push!(idx, i)
            push!(stratum, name)
        end
    end
    perm = sortperm(idx)
    idx[perm], stratum[perm]
end
export select_indices

"""
    _detector_sample(data, filekeys, row, n_det, strata, waveform_column, rng) -> WaveformSample

`n_det` rows for the detector of `row`, drawn by `select_indices` from the
candidate rows of `filekeys`. Warns and returns all candidate rows when
there are fewer of them than `n_det`. The waveforms are read back from each
file by the timestamps of the drawn rows, which is what ties them to those
rows.
"""
function _detector_sample(data::LegendData, filekeys::AbstractVector{FileKey}, row,
        n_det::Integer, strata::NamedTuple, waveform_column::Symbol, rng)
    det, ch, det_type = row.detector, row.channel, row.det_type
    # `read_ldata` selects raw-tier channel groups by `DetectorId`, never `ChannelId`;
    # `ch` and `det_type` only ever fill the sample's columns below.
    cand = scan_candidates(data, filekeys, det, strata.aux_triggers; strata.ts_window)
    nrows = length(cand)
    nrows > 0 || throw(ArgumentError("no raw events for $det in $(length(filekeys)) raw files"))
    nrows < n_det && @warn "detector $det: only $nrows rows in $(length(filekeys)) raw files, quota is $n_det"
    # The FPGA baseline drifts from file to file, so an outlier is one against the baselines
    # of its own file rather than against those of the whole period.
    idx, stratum = select_indices(cand, n_det; strata.exclude, strata.zero_energy, strata.high_energy,
        strata.high_energy_quantile, strata.baseline_outlier, strata.baseline_mad,
        baseline_group=:filekey, rng)
    isempty(idx) && throw(ArgumentError("no candidate rows left for $det after excluding $(strata.exclude)"))
    files = cand.filekey[idx]
    parts = map(unique(files)) do fk
        rows = findall(==(fk), files)
        timestamp = cand.timestamp[idx[rows]]
        wf = read_ldata((waveform_column, :timestamp), data, (DataTier(:raw), fk, timestamp, det))
        wf.timestamp == timestamp ||
            throw(ErrorException("the waveforms read from $fk for $det do not carry the timestamps they were selected by"))
        n = length(rows)
        WaveformSample(getproperty(wf, waveform_column), files[rows], cand.eventnumber[idx[rows]],
            fill(ch, n), fill(det, n), fill(det_type, n), cand.daqenergy[idx[rows]],
            cand.baseline[idx[rows]], timestamp, stratum[rows])
    end
    vcat(parts...)
end

"""
    build_sample(data, filekey, config; rng, chinfo, filekeys) -> WaveformSample

Sample for the period and category of `filekey`: `config.n_max` waveforms
split equally over `config.detectors`, each drawn by `select_indices` from
the candidate rows of `filekeys` and read from `config.waveform_column`.
A detector whose candidate rows fall short of its share of `config.n_max`
contributes all of them and `@warn`s about the shortfall instead of
silently returning fewer rows than the caller might expect.

`config.strata` holds the selection parameters: `aux_triggers` maps flag
names to the detector ids of the aux trigger channels, `ts_window` is the
coincidence window in µs by which those channels' triggers are matched to
the detector's events (125 when absent), `exclude` lists the flags whose
events stay out of the pool, and `zero_energy`, `high_energy`,
`high_energy_quantile`, `baseline_outlier` and `baseline_mad` parameterize
the strata of `select_indices`. Baseline outliers are judged per raw file,
since the FPGA baseline drifts between them. A key left out means no flags,
nothing excluded, or an empty stratum.

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
    strata = (aux_triggers=[Symbol(k) => DetectorId(string(v)) for (k, v) in get(sc, :aux_triggers, PropDict())],
        exclude=Symbol.(get(sc, :exclude, Symbol[])),
        zero_energy=Float64(get(sc, :zero_energy, 0)),
        high_energy=Float64(get(sc, :high_energy, 0)),
        high_energy_quantile=Float64(get(sc, :high_energy_quantile, 0.9)),
        baseline_outlier=Float64(get(sc, :baseline_outlier, 0)),
        baseline_mad=Float64(get(sc, :baseline_mad, 5)),
        ts_window=Float64(get(sc, :ts_window, 125)) * u"µs")
    rows = map(detectors) do det
        i = findfirst(==(det), chinfo.detector)
        isnothing(i) && throw(ArgumentError("detector $det not found in the channel map of $filekey"))
        chinfo[i]
    end
    parts = [_detector_sample(data, filekeys, row, n_det, strata, Symbol(config.waveform_column), rng)
             for row in rows]
    vcat(parts...)
end
export build_sample
