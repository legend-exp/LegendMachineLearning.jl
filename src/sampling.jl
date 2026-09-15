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

function _detector_sample(data::LegendData, filekeys::AbstractVector{FileKey}, det::DetectorId,
        n_det::Integer, n_files_max::Integer, zero_energy_fraction::Real, waveform_column::Symbol, rng)
    chinfo = channelinfo(data, first(filekeys), det)
    ch = chinfo.channel
    # scalar columns of the whole run decide the selection; waveforms are read per file only as needed
    scalars = read_ldata((:daqenergy, :eventnumber), data, :raw, filekeys, ch)
    nrows = length(scalars.daqenergy)
    nrows > 0 || throw(ArgumentError("no raw events for $det in $(length(filekeys)) files"))
    file_of_row = reduce(vcat, [fill(fk, length(read_ldata((:eventnumber,), data, :raw, fk, ch).eventnumber)) for fk in filekeys])
    sel = select_indices(scalars.daqenergy, n_det; zero_energy_fraction, rng)
    parts = WaveformSample[]
    for fk in unique(file_of_row[sel])
        length(parts) < n_files_max || break
        rows_fk = findall(==(fk), file_of_row)
        local_sel = findall(in(Set(sel)), rows_fk)      # positions within this file
        wv = read_ldata((waveform_column,), data, :raw, fk, ch)
        wvfs = getproperty(wv, waveform_column)[local_sel]
        n = length(local_sel)
        push!(parts, WaveformSample(wvfs, fill(fk, n), Int.(scalars.eventnumber[rows_fk[local_sel]]),
            fill(ch, n), fill(det, n), fill(Symbol(chinfo.det_type), n), Float64.(scalars.daqenergy[rows_fk[local_sel]])))
    end
    vcat(parts...)
end

"""
    build_sample(data, filekey, config; rng) -> WaveformSample

Sample for the run of `filekey`: `config.n_max` waveforms split equally over
`config.detectors`, each selected by `select_indices` on `daqenergy` with
`config.strata.zero_energy_fraction`, read from `config.waveform_column`
of at most `config.n_files_max` raw files per detector.
"""
function build_sample(data::LegendData, filekey::FileKey, config::PropDict; rng::AbstractRNG=Random.default_rng())
    detectors = DetectorId.(config.detectors)
    isempty(detectors) && throw(ArgumentError("config.detectors is empty"))
    n_det = cld(Int(config.n_max), length(detectors))
    filekeys = search_disk(FileKey, data.tier[DataTier(:raw), filekey.category, filekey.period, filekey.run])
    isempty(filekeys) && throw(ArgumentError("no raw files for $(filekey.period) $(filekey.run) $(filekey.category)"))
    chinfo = channelinfo(data, filekey; system=:geds)
    for det in detectors
        det in chinfo.detector || throw(ArgumentError("detector $det not found in the channel map of $filekey"))
    end
    parts = [_detector_sample(data, filekeys, det, n_det, Int(config.n_files_max),
        Float64(config.strata.zero_energy_fraction), Symbol(config.waveform_column), rng) for det in detectors]
    vcat(parts...)
end
export build_sample
