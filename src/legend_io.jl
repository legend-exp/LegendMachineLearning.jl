# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

"""
    qc_sample_filename(data, fk) -> String

LH5 file of the training sample: the `jlqml` tier file of the training run's start key.
"""
qc_sample_filename(data::LegendData, fk::FileKey) = data.tier[DataTier(:jlqml), fk]
export qc_sample_filename

"""
    qc_model_filename(sample_path) -> String
    qc_model_filename(data, fk) -> String

JLD2 model file next to the sample file, same stem.
"""
function qc_model_filename(sample_path::AbstractString)
    endswith(sample_path, ".lh5") || throw(ArgumentError("expected an .lh5 sample path, got $sample_path"))
    return string(sample_path[1:(end - 4)], ".jld2")
end
qc_model_filename(data::LegendData, fk::FileKey) = qc_model_filename(qc_sample_filename(data, fk))
export qc_model_filename

"""
    qc_pars_entry_name(fk) -> String

Path of the `rpars.ml` entry of `fk` relative to the props-DB directory.
One file per run and category, so the cal and the phy model of a run keep
their own entry. It names both the file the entry is written to and the
`apply` path of its validity record.
"""
qc_pars_entry_name(fk::FileKey) = "$(fk.period)/$(fk.run)-$(fk.category).yaml"
export qc_pars_entry_name

"""
    qc_ml_config(data, fk) -> PropDict

QC machine-learning config valid for `fk`, the `ml` section of
`dataprod_config(data).qc(fk)`. Throws when that section is absent.
"""
function qc_ml_config(data::LegendData, fk::FileKey)
    qc = dataprod_config(data).qc(fk)
    haskey(qc, :ml) || throw(ArgumentError("dataprod_config(data).qc($fk) has no ml section"))
    qc.ml
end
export qc_ml_config

"""
    qc_ml_pars(data, filekey) -> Union{PropDict, Nothing}

Parameter entry of the QC model valid for `filekey` (`rpars.ml`), resolved
through the validity database. `nothing` when the `rpars/ml` directory does
not exist, or no validity entry of it covers `filekey`.
"""
function qc_ml_pars(data::LegendData, filekey::FileKey)
    db = data.par.rpars.ml
    db isa LegendDataManagement.NoSuchPropsDBEntry && return nothing
    props = try
        db(filekey)
    catch e
        # Only "no entry covers this run" is an absence; everything else is a real error.
        (e isa ArgumentError || e isa ErrorException) && occursin("validity", sprint(showerror, e)) || rethrow()
        return nothing
    end
    return PropDict(props)
end
export qc_ml_pars

"""
    model_pars_entry(training_fk, sha256, train_report, model) -> PropDict

Entry written to `rpars.ml` by the training step. It names the training
filekey (from which the model path follows), the model file digest, the
chosen SVM hyperparameters and the class fractions of the training labels.
"""
function model_pars_entry(training_fk::FileKey, sha256::AbstractString, train_report::NamedTuple, model::LegendMLModel)
    pd = PropDict()
    pd.training_filekey = string(training_fk)
    pd.model_sha256 = String(sha256)
    pd.format_version = MODEL_FORMAT_VERSION
    pd.svm = PropDict(:cost => train_report.best.cost, :gamma => train_report.best.gamma,
        :cv_score => train_report.best.cv_score, :probability => model.machine.model isa ProbabilisticSVC)
    pd.class_fractions = PropDict(class_fractions(model.metadata.labels, model.categories))
    pd.created = string(now())
    return pd
end
export model_pars_entry

"""
    load_qc_model(data, filekey) -> LegendMLModel or missing

Model valid for `filekey`. `missing` when no `rpars.ml` validity entry
covers it. Errors when the resolved entry has no `training_filekey` (it was
not written by the QC training step), was trained in another period, or the
model file digest does not match the entry.
"""
function load_qc_model(data::LegendData, filekey::FileKey)
    pars = qc_ml_pars(data, filekey)
    if pars === nothing
        @info "No QC ML parameter entry valid for $filekey"
        return missing
    end
    haskey(pars, :training_filekey) ||
        throw(
            ArgumentError(
                "rpars.ml entry valid for $filekey has no training_filekey; it was not written by the QC training step"
            )
        )
    training_fk = FileKey(pars.training_filekey)
    training_fk.period == filekey.period ||
        throw(
            ArgumentError(
                "QC model valid for $filekey was trained in $(training_fk.period), not $(filekey.period); train a model for this period"
            )
        )
    return load_model(qc_model_filename(data, training_fk); expected_sha256 = pars.model_sha256)
end
export load_qc_model
