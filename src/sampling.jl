# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

"""
    select_indices(daqenergy, n; zero_energy_fraction, rng) -> Vector{Int}

Sorted row indices, `length(sel) == min(n, length(daqenergy))`. Reserves
`zero_energy_fraction * n` of them for rows with `daqenergy == 0` (events
whose energy the DAQ could not reconstruct) and draws the rest uniformly
without replacement from the other rows; that fraction is only a minimum
when both strata are large enough to supply their share; a stratum too
small for its share is capped at what it has, and the other stratum tops
up to make `n` (or all of `daqenergy`, whichever is smaller).
"""
function select_indices(daqenergy::AbstractVector{<:Real}, n::Integer;
        zero_energy_fraction::Real, rng::AbstractRNG=Random.default_rng())
    0 ≤ zero_energy_fraction ≤ 1 || throw(ArgumentError("zero_energy_fraction must be between 0 and 1, got $zero_energy_fraction"))
    n ≥ 0 || throw(ArgumentError("n must be non-negative, got $n"))
    idx_all = collect(eachindex(daqenergy))
    n ≥ length(idx_all) && return idx_all
    zero_idx = filter(i -> iszero(daqenergy[i]), idx_all)
    other_idx = filter(i -> !iszero(daqenergy[i]), idx_all)
    n_zero = min(round(Int, zero_energy_fraction * n), length(zero_idx))
    n_other = min(n - n_zero, length(other_idx))
    n_zero = min(n - n_other, length(zero_idx))
    sel = vcat(shuffle(rng, zero_idx)[1:n_zero], shuffle(rng, other_idx)[1:n_other])
    sort!(sel)
end
export select_indices

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

_read_raw(f, data::LegendData, sel...) = read_ldata(f, data, DataTier(:raw), sel...)

"""
    _detector_sample(data, filekeys, row, n_det, n_files_max, zero_energy_fraction, waveform_column, rng) -> WaveformSample

`n_det` rows for the detector of `row`, selected by `select_indices` from
the candidate rows of the first `n_files_max` entries of `filekeys` (in the
given order). Warns and returns all candidate rows when there are fewer of
them than `n_det`.
"""
function _detector_sample(data::LegendData, filekeys::AbstractVector{FileKey}, row,
        n_det::Integer, n_files_max::Integer, zero_energy_fraction::Real, waveform_column::Symbol, rng)
    det, ch, det_type = row.detector, row.channel, row.det_type
    # `read_ldata` selects raw-tier channel groups by `DetectorId`, never `ChannelId`;
    # `ch` and `det_type` only ever fill the sample's columns below.
    candidate_filekeys = first(filekeys, min(Int(n_files_max), length(filekeys)))
    per_file = [_read_raw((:daqenergy, :eventnumber), data, fk, det) for fk in candidate_filekeys]
    file_of_row = reduce(vcat, [fill(fk, length(t.daqenergy)) for (fk, t) in zip(candidate_filekeys, per_file)])
    daqenergy = reduce(vcat, (t.daqenergy for t in per_file))
    eventnumber = reduce(vcat, (t.eventnumber for t in per_file))
    nrows = length(daqenergy)
    nrows > 0 || throw(ArgumentError("no raw events for $det in the first $(length(candidate_filekeys)) raw files"))
    nrows < n_det && @warn "detector $det: only $nrows rows in the first $(length(candidate_filekeys)) raw files, quota is $n_det"
    sel = select_indices(daqenergy, n_det; zero_energy_fraction, rng)
    sel_set = Set(sel)
    parts = WaveformSample[]
    for fk in unique(file_of_row[sel])
        rows_fk = findall(==(fk), file_of_row)
        local_sel = findall(in(sel_set), rows_fk)      # positions within this file
        wv = _read_raw((waveform_column,), data, fk, det)
        wvfs = getproperty(wv, waveform_column)[local_sel]
        n = length(local_sel)
        push!(parts, WaveformSample(wvfs, fill(fk, n), Int.(eventnumber[rows_fk[local_sel]]),
            fill(ch, n), fill(det, n), fill(det_type, n), Float64.(daqenergy[rows_fk[local_sel]])))
    end
    vcat(parts...)
end

"""
    build_sample(data, filekey, config; rng, chinfo) -> WaveformSample

Sample for the run of `filekey`: `config.n_max` waveforms split equally over
`config.detectors`, each selected by `select_indices` on `daqenergy` with
`config.strata.zero_energy_fraction`, read from `config.waveform_column`.
For each detector, the candidate rows are restricted to the first
`config.n_files_max` raw files (by `search_disk` order) before selection, so
every selected row comes from one of those files; a detector whose candidate
rows fall short of its share of `config.n_max` contributes all of them and
`@warn`s about the shortfall instead of silently returning fewer rows than
the caller might expect from a run-wide draw.

`chinfo` supplies the `detector`, `channel` and `det_type` columns used to
look up each entry of `config.detectors`; it defaults to the full channel
map of `filekey`. Pass any other table with those three columns to run
without a channel map, e.g. a hand-built `TypedTables.Table` against raw
files that predate or lack channel-map coverage.
"""
function build_sample(data::LegendData, filekey::FileKey, config::PropDict;
        rng::AbstractRNG=Random.default_rng(), chinfo=channelinfo(data, filekey; system=:geds))
    detectors = DetectorId.(config.detectors)
    isempty(detectors) && throw(ArgumentError("config.detectors is empty"))
    n_det = cld(Int(config.n_max), length(detectors))
    dir = data.tier[DataTier(:raw), filekey.category, filekey.period, filekey.run]
    isdir(dir) || throw(ArgumentError("no raw files for $(filekey.period) $(filekey.run) $(filekey.category)"))
    filekeys = search_disk(FileKey, dir)
    isempty(filekeys) && throw(ArgumentError("no raw files for $(filekey.period) $(filekey.run) $(filekey.category)"))
    rows = map(detectors) do det
        i = findfirst(==(det), chinfo.detector)
        isnothing(i) && throw(ArgumentError("detector $det not found in the channel map of $filekey"))
        chinfo[i]
    end
    parts = [_detector_sample(data, filekeys, row, n_det, Int(config.n_files_max),
        Float64(config.strata.zero_energy_fraction), Symbol(config.waveform_column), rng) for row in rows]
    vcat(parts...)
end
export build_sample
