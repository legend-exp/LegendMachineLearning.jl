# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

"""
    select_indices(daqenergy, n; zero_energy_fraction, rng) -> Vector{Int}

Sorted row indices: up to `zero_energy_fraction * n` rows with `daqenergy == 0`
(events whose energy the DAQ could not reconstruct), the remainder drawn
uniformly without replacement from the other rows. Returns every row when
`n ≥ length(daqenergy)`.
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

# `read_ldata` resolves the raw tier from the bare Symbol `:raw` for most
# selector combinations, but not all; retry with `DataTier(:raw)` when the
# Symbol form has no matching method.
function _read_raw(f, data::LegendData, sel...)
    try
        read_ldata(f, data, :raw, sel...)
    catch e
        (e isa MethodError || e isa ArgumentError) || rethrow()
        read_ldata(f, data, DataTier(:raw), sel...)
    end
end

function _detector_sample(data::LegendData, filekeys::AbstractVector{FileKey}, row,
        n_det::Integer, n_files_max::Integer, zero_energy_fraction::Real, waveform_column::Symbol, rng)
    det, ch, det_type = row.detector, row.channel, row.det_type
    # `read_ldata` selects raw-tier channel groups by `DetectorId`, never `ChannelId`;
    # `ch` and `det_type` only ever fill the sample's columns below.
    per_file = [_read_raw((:daqenergy, :eventnumber), data, fk, det) for fk in filekeys]
    file_of_row = reduce(vcat, [fill(fk, length(t.daqenergy)) for (fk, t) in zip(filekeys, per_file)])
    daqenergy = reduce(vcat, (t.daqenergy for t in per_file))
    eventnumber = reduce(vcat, (t.eventnumber for t in per_file))
    nrows = length(daqenergy)
    nrows > 0 || throw(ArgumentError("no raw events for $det in $(length(filekeys)) files"))
    sel = select_indices(daqenergy, n_det; zero_energy_fraction, rng)
    parts = WaveformSample[]
    for fk in unique(file_of_row[sel])
        length(parts) < n_files_max || break
        rows_fk = findall(==(fk), file_of_row)
        local_sel = findall(in(Set(sel)), rows_fk)      # positions within this file
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
`config.strata.zero_energy_fraction`, read from `config.waveform_column` of
at most `config.n_files_max` raw files per detector.

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
    filekeys = search_disk(FileKey, data.tier[DataTier(:raw), filekey.category, filekey.period, filekey.run])
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
