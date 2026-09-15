# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

"""
    features_config(config::PropDict) -> HaarFeatures

Feature extractor from `config.features` (`n_samples`, `normalize`, `bl_window = [start, stop]` in µs).
"""
function features_config(config::PropDict)
    f = config.features
    lo, hi = Float64.(f.bl_window)
    HaarFeatures(; n_samples=Int(f.n_samples), normalize=Bool(f.normalize), bl_window=lo * u"µs" .. hi * u"µs")
end
export features_config

"""
    run_qc_sampling(data, training_fk, config; rng, kwargs...) -> (sample, cluster, pars, scan)

Build the training sample of the run of `training_fk`, scan and fit affinity
propagation, write the `jlqml` sample file, and return the parameter entry
(with `exemplar_labels` initialized to `UNLABELED`) and the scan report.
`kwargs` are forwarded to `build_sample` (e.g. `chinfo` to run without a
channel map).
"""
function run_qc_sampling(data::LegendData, training_fk::FileKey, config::PropDict; rng::AbstractRNG=Random.default_rng(), kwargs...)
    sample = build_sample(data, training_fk, config; rng, kwargs...)
    X = features(features_config(config), sample.waveform)
    ap = config.affinity_propagation
    damp = collect(range(Float64(ap.damp.min), Float64(ap.damp.max), length=Int(ap.damp.n)))
    pq = collect(range(Float64(ap.preference_quantile.min), Float64(ap.preference_quantile.max), length=Int(ap.preference_quantile.n)))
    scan = scan_affinity_propagation(X; damp, preference_quantile=pq, target_n_clusters=Int(ap.target_n_clusters),
        maxiter=Int(ap.maxiter), tol=Float64(ap.tol))
    cluster = fit_affinity_propagation(X; damp=scan.best.damp, preference=scan.best.preference,
        preference_quantile=pq[scan.best.i_preference], maxiter=Int(ap.maxiter), tol=Float64(ap.tol))
    path = qc_sample_filename(data, training_fk)
    mkpath(dirname(path))
    write_sample(path, sample, cluster)
    pars = PropDict()
    pars.sample = PropDict(:n => length(sample), :detectors => string.(config.detectors),
        :n_files_max => Int(config.n_files_max), :waveform_column => string(config.waveform_column),
        :zero_energy_fraction => Float64(config.strata.zero_energy_fraction), :sample_sha256 => file_sha256(path))
    pars.affinity_propagation = cluster_result_to_props(cluster)
    pars.affinity_propagation.scan = PropDict(:damp => scan.damp, :preference_quantile => scan.preference_quantile,
        :preference => scan.preference, :n_clusters => scan.n_clusters, :converged => scan.converged,
        :iterations => scan.iterations)
    pars.exemplar_labels = fill(UNLABELED, length(cluster.exemplars))
    sample, cluster, pars, scan
end
export run_qc_sampling

function _train_from_sample_file(sample_path::AbstractString, config::PropDict, pars::PropDict; rng::AbstractRNG=Random.default_rng())
    digest = file_sha256(sample_path)
    digest == pars.sample.sample_sha256 ||
        throw(ArgumentError("sample file changed since it was labeled (SHA-256 $digest vs $(pars.sample.sample_sha256)); rebuild the sample and relabel"))
    sample, assignment, exemplars = read_sample(sample_path)
    cluster = cluster_result(pars.affinity_propagation, assignment, exemplars)
    labels = propagate_labels(cluster, Int.(pars.exemplar_labels))
    categories = qc_categories(config)
    for l in unique(labels)
        haskey(categories, l) || throw(ArgumentError("exemplar label $l is not a configured category"))
    end
    fe = features_config(config)
    X = features(fe, sample.waveform)
    mach, rep = train_svm(X, labels, config.svm; groups=sample.det_type, rng)
    training_fk = first(sample.filekey)
    metadata = PropDict(:training_filekey => string(training_fk), :period => string(training_fk.period),
        :category => string(training_fk.category), :n_sample => length(sample),
        :detectors => string.(unique(sample.detector)), :labels => labels,
        :created => string(now()), :package_version => string(pkgversion(LegendMachineLearning)))
    model = LegendMLModel(mach, fe, categories, metadata)
    model_path = qc_model_filename(sample_path)
    sha = save_model(model_path, model)
    entry = merge(pars, model_pars_entry(training_fk, sha, rep, model))
    model, entry, rep
end

"""
    run_qc_training(data, training_fk, config, pars; rng) -> (model, pars_entry, report)

Train the SVM from the labeled sample of `training_fk`, write the JLD2 model
file and return the parameter entry to store under `rpars.ml`.
"""
run_qc_training(data::LegendData, training_fk::FileKey, config::PropDict, pars::PropDict; rng::AbstractRNG=Random.default_rng()) =
    _train_from_sample_file(qc_sample_filename(data, training_fk), config, pars; rng)
export run_qc_training
