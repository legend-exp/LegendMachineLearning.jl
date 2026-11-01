# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

"""
    features_config(config::PropDict) -> HaarFeatures

Feature extractor from `config.features` (`n_samples`, `normalize`, `bl_window = [start, stop]` in µs).
"""
function features_config(config::PropDict)
    f = config.features
    lo, hi = Float64.(f.bl_window)
    return HaarFeatures(;
        n_samples = Int(f.n_samples),
        normalize = Bool(f.normalize),
        bl_window = lo * u"µs" .. hi * u"µs"
    )
end
export features_config

"YAML/PropDict cannot serialize a Matrix; the scan grid goes in as one vector per row."
_grid_rows(m::AbstractMatrix) = [collect(r) for r in eachrow(m)]

"""
    run_qc_sampling(data, training_fk, config; rng, filekeys, kwargs...) -> (sample, cluster, pars, scan)

Build the training sample of the period and category of `training_fk` from
the raw files `filekeys`, scan and fit affinity propagation, write the
`jlqml` sample file (with the rule inputs of the sample rows as its `stats`
table), and return the parameter entry (with `exemplar_labels` initialized to
`UNLABELED`) and the scan report. The entry's `sample` section records, per
detector, the number of scan rows (`n_scan`) and how many of them each rule
selects (`n_matched`). `kwargs` are forwarded to `build_sample` (e.g. `chinfo`
to run without a channel map).
"""
function run_qc_sampling(data::LegendData, training_fk::FileKey, config::PropDict;
    rng::AbstractRNG = Random.default_rng(),
    filekeys::AbstractVector{FileKey} = raw_filekeys(data, training_fk.period, training_fk.category), kwargs...)
    drawn = build_sample(data, training_fk, config; rng, filekeys, kwargs...)
    sample = drawn.sample
    X = features(features_config(config), sample.waveform)
    ap = config.affinity_propagation
    damp = collect(range(Float64(ap.damp.min), Float64(ap.damp.max), length = Int(ap.damp.n)))
    pq = collect(
        range(
            Float64(ap.preference_quantile.min),
            Float64(ap.preference_quantile.max),
            length = Int(ap.preference_quantile.n)
        )
    )
    scan = scan_affinity_propagation(X; damp, preference_quantile = pq, target_n_clusters = Int(ap.target_n_clusters),
        maxiter = Int(ap.maxiter), tol = Float64(ap.tol), rng)
    cluster = fit_affinity_propagation(X; damp = scan.best.damp, preference = scan.best.preference,
        preference_quantile = pq[scan.best.i_preference], maxiter = Int(ap.maxiter), tol = Float64(ap.tol),
        S = scan.S)
    path = qc_sample_filename(data, training_fk)
    mkpath(dirname(path))
    write_sample(path, sample, cluster; drawn.stats)
    pars = PropDict()
    pars.sample = PropDict(:n => length(sample), :detectors => string.(config.detectors),
        :waveform_column => string(config.waveform_column), :n_files => length(filekeys),
        :runs => unique(string(fk.run) for fk in filekeys), :strata => config.strata,
        :n_per_stratum => PropDict((s => count(==(s), sample.stratum) for s in sort(unique(sample.stratum)))...),
        :n_scan => drawn.n_scan, :n_matched => drawn.n_matched,
        :sample_sha256 => file_sha256(path))
    # The entry carries the extractor settings so that the labeling notebook computes the
    # same features as the clustering did.
    pars.features = config.features
    pars.affinity_propagation = cluster_result_to_props(cluster)
    pars.affinity_propagation.scan = PropDict(:damp => scan.damp, :preference_quantile => scan.preference_quantile,
        :preference => scan.preference, :n_clusters => _grid_rows(scan.n_clusters),
        :converged => _grid_rows(scan.converged),
        :iterations => _grid_rows(scan.iterations))
    pars.exemplar_labels = fill(UNLABELED, length(cluster.exemplars))
    return sample, cluster, pars, scan
end
export run_qc_sampling

function _train_from_sample_file(sample_path::AbstractString, training_fk::FileKey, config::PropDict, pars::PropDict;
    rng::AbstractRNG = Random.default_rng())
    digest = file_sha256(sample_path)
    digest == pars.sample.sample_sha256 ||
        throw(
            ArgumentError(
                "sample file changed since it was labeled (SHA-256 $digest vs $(pars.sample.sample_sha256)); rebuild the sample and relabel"
            )
        )
    sample, assignment, exemplars = read_sample(sample_path)
    cluster = cluster_result(pars.affinity_propagation, assignment, exemplars)
    labels = propagate_labels(cluster, Int.(pars.exemplar_labels))
    categories = qc_categories(config)
    for l in unique(labels)
        haskey(categories, l) || throw(ArgumentError("exemplar label $l is not a configured category"))
    end
    fe = features_config(config)
    X = features(fe, sample.waveform)
    mach, rep = train_svm(X, labels, config.svm; groups = sample.det_type, rng)
    metadata = PropDict(:training_filekey => string(training_fk), :period => string(training_fk.period),
        :category => string(training_fk.category), :n_sample => length(sample),
        :detectors => string.(unique(sample.detector)), :labels => labels,
        :created => string(now()), :package_version => string(pkgversion(LegendMachineLearning)))
    model = LegendMLModel(mach, fe, categories, metadata)
    model_path = qc_model_filename(sample_path)
    sha = save_model(model_path, model)
    entry = merge(pars, model_pars_entry(training_fk, sha, rep, model))
    return model, entry, rep
end

"""
    run_qc_training(data, training_fk, config, pars; rng) -> (model, pars_entry, report)

Train the SVM from the labeled sample of `training_fk`, write the JLD2 model
file and return the parameter entry to store under `rpars.ml`. `training_fk`
names the model, independent of which sample rows happened to be drawn.
"""
function run_qc_training(
    data::LegendData,
    training_fk::FileKey,
    config::PropDict,
    pars::PropDict;
    rng::AbstractRNG = Random.default_rng()
)
    return _train_from_sample_file(qc_sample_filename(data, training_fk), training_fk, config, pars; rng)
end
export run_qc_training

"""
    classify_filekey(data, fk, model; detectors, aux_triggers, exclude, ts_window, n_max_per_detector,
        waveform_column, classes, rng) -> TypedTables.Table

Raw events of `fk` of each of `detectors`, classified by `model`: a table
with the columns `detector`, `eventnumber`, `timestamp`, `daqenergy`,
`waveform`, `label`, `margin` (the probability of the label) and
`probabilities` (per event, the probability of each class of `classes`; see
`predict_probabilities`).

`aux_triggers` (`name => detector` pairs or a dictionary of them), `ts_window`
and the flag names in `exclude` select events as in `build_sample`: events
within `ts_window` of a trigger of an excluded aux channel are left out.
Of each detector at most `n_max_per_detector` events are kept, drawn
uniformly with `rng`, in file order. The waveforms are read from the raw
column `waveform_column`.
"""
function classify_filekey(data::LegendData, fk::FileKey, model::LegendMLModel; detectors,
    aux_triggers = PropDict(), exclude = Symbol[], ts_window::Unitful.Time = 125u"µs",
    n_max_per_detector::Integer = typemax(Int), waveform_column::Symbol = :waveform_presummed,
    classes::AbstractVector{<:Integer} = sort(collect(keys(model.categories))), rng::AbstractRNG = Random.default_rng())
    aux = [Symbol(k) => DetectorId(string(v)) for (k, v) in aux_triggers]
    isempty(detectors) && throw(ArgumentError("detectors is empty"))
    excluded_names = Symbol.(exclude)
    for name in excluded_names
        name in first.(aux) || throw(ArgumentError("exclude names $name, which is not an aux trigger"))
    end
    parts = map(DetectorId.(detectors)) do det
        cand = scan_candidates(data, [fk], det, aux; ts_window)
        rows = findall(i -> !any(name -> getproperty(cand, name)[i], excluded_names), eachindex(cand))
        isempty(rows) && throw(ArgumentError("no events of $det left in $fk after excluding $excluded_names"))
        length(rows) > n_max_per_detector && (rows = sort(shuffle(rng, rows)[1:n_max_per_detector]))
        raw = read_ldata((waveform_column, :eventnumber), data, DataTier(:raw), fk, det)
        raw.eventnumber == cand.eventnumber ||
            throw(ErrorException("the waveforms of $det in $fk do not line up with its event table"))
        # A whole-column read leaves the signals encoded; `Vector` decodes each one once, where
        # element access would decode the full trace per sample.
        wvfs = [RDWaveform(w.time, Vector(w.signal)) for w in getproperty(raw, waveform_column)[rows]]
        label, p = predict_probabilities(model, wvfs; classes)
        return (detector = fill(det, length(rows)), eventnumber = cand.eventnumber[rows],
            timestamp = cand.timestamp[rows], daqenergy = cand.daqenergy[rows], waveform = wvfs, label,
            margin = vec(maximum(p, dims = 2)), probabilities = [p[i, :] for i in axes(p, 1)])
    end
    col(name) = reduce(vcat, [getproperty(p, name) for p in parts])
    return Table(detector = col(:detector), eventnumber = col(:eventnumber), timestamp = col(:timestamp),
        daqenergy = col(:daqenergy), waveform = ArrayOfRDWaveforms(col(:waveform)), label = col(:label),
        margin = col(:margin), probabilities = VectorOfSimilarVectors(col(:probabilities)))
end
export classify_filekey
