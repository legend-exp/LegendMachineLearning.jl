# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

_logrange(section::PropDict) = collect(logrange(Float64(section.min), Float64(section.max), Int(section.n)))

"""
    svm_grid(svm_config::PropDict) -> (costs, gammas)

Log-spaced tuning grids from the `cost` and `gamma` sections (`min`, `max`, `n`).
"""
svm_grid(svm_config::PropDict) = (_logrange(svm_config.cost), _logrange(svm_config.gamma))
export svm_grid

"""
    class_weights(y) -> Dict{Int,Float64}

Inverse-frequency class weights, normalized so their mean is 1.
"""
function class_weights(y::AbstractVector{<:Integer})
    counts = Dict{Int,Int}()
    for l in y
        counts[Int(l)] = get(counts, Int(l), 0) + 1
    end
    raw = Dict(k => length(y) / c for (k, c) in counts)
    m = mean(values(raw))
    return Dict{Int,Float64}(k => w / m for (k, w) in raw)
end
export class_weights

function _svm_model(cfg::PropDict, cost, gamma)
    return (cfg.probability ? ProbabilisticSVC : SVC)(; cost = Float64(cost), gamma = Float64(gamma))
end

# MLJBase's generic `predict_mode` fallback is `mode.(predict(...))`, which assumes
# probabilistic predictions; a deterministic SVC's `predict` already returns labels.
_predict_mode(mach, Xt) = mach.model isa ProbabilisticSVC ? predict_mode(mach, Xt) : predict(mach, Xt)

function _confusion(classes::AbstractVector{Int}, y_true, y_pred)
    idx = Dict(c => i for (i, c) in enumerate(classes))
    cm = zeros(Int, length(classes), length(classes))
    for (t, p) in zip(y_true, y_pred)
        cm[idx[t], idx[p]] += 1
    end
    return cm
end

"""
    train_svm(X, y, svm_config; groups, rng) -> (machine, report)

Grid search over `cost` × `gamma` with stratified `n_folds` cross-validation
on balanced accuracy, then a refit on all data with the best pair. The report
holds the grid scores, a holdout confusion matrix (20 % stratified split),
per-class and per-group accuracy. `groups` (e.g. detector types) must be
index-aligned with `y`. `class_accuracy[i]` is `missing`, not `0`, when
`classes[i]` has no rows in the holdout split.
"""
function train_svm(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Integer}, svm_config::PropDict;
    groups::AbstractVector = fill(:all, length(y)), rng::AbstractRNG = Random.default_rng())
    Base.require_one_based_indexing(X, y, groups)
    size(X, 1) == length(y) == length(groups) ||
        throw(
            DimensionMismatch(
                "X has $(size(X, 1)) rows, y has $(length(y)) labels, groups has $(length(groups)) entries"
            )
        )
    classes = sort(unique(Int.(y)))
    length(classes) ≥ 2 || throw(ArgumentError("training needs at least two classes, got $(classes)"))
    yint = Int.(y)
    Xt = table(X)
    yc = categorical(yint; levels = classes)
    w = class_weights(yint)

    costs, gammas = svm_grid(svm_config)
    n_folds = Int(svm_config.n_folds)
    cv_score = zeros(Float64, length(costs), length(gammas))
    grid = [(i, j) for i in eachindex(costs) for j in eachindex(gammas)]
    # LIBSVM training is not safe to call concurrently, so the grid runs sequentially.
    for g in eachindex(grid)
        i, j = grid[g]
        mach = machine(_svm_model(svm_config, costs[i], gammas[j]), Xt, yc, w)
        e = evaluate!(mach; resampling = StratifiedCV(; nfolds = n_folds, shuffle = true, rng),
            measure = balanced_accuracy, operation = (svm_config.probability ? predict_mode : predict),
            verbosity = 0)
        cv_score[i, j] = only(e.measurement)
    end
    best_ij = argmax(cv_score)
    best = (cost = costs[best_ij[1]], gamma = gammas[best_ij[2]], cv_score = cv_score[best_ij])

    # holdout confusion matrix with the chosen hyperparameters
    holdout_fraction = 0.2
    train_idx, test_idx = partition(eachindex(yint), 1 - holdout_fraction; stratify = yint, shuffle = true, rng)
    mach_h = machine(_svm_model(svm_config, best.cost, best.gamma), table(X[train_idx, :]),
        categorical(yint[train_idx]; levels = classes), w)
    fit!(mach_h; verbosity = 0)
    y_pred = Int.(unwrap.(_predict_mode(mach_h, table(X[test_idx, :]))))
    y_test = yint[test_idx]
    confusion = _confusion(classes, y_test, y_pred)
    # A class absent from the holdout split has no accuracy to report, not an
    # accuracy of 0.
    class_accuracy = Union{Missing,Float64}[
        (n = sum(confusion[i, :])) == 0 ? missing : confusion[i, i] / n for i in eachindex(classes)]
    group_names = sort(unique(Symbol.(groups)))
    g_test = Symbol.(groups[test_idx])
    for g in group_names
        any(==(g), g_test) ||
            throw(ArgumentError("group $g has no rows in the holdout split; enlarge the sample or drop the group"))
    end
    group_accuracy = [mean(y_pred[g_test .== g] .== y_test[g_test .== g]) for g in group_names]

    mach = machine(_svm_model(svm_config, best.cost, best.gamma), Xt, yc, w)
    fit!(mach; verbosity = 0)
    rep = (cost = costs, gamma = gammas, cv_score, best, classes, confusion, class_accuracy,
        groups = group_names, group_accuracy, holdout_fraction)
    return mach, rep
end
export train_svm

"""
    predict_labels(m::LegendMLModel, X::AbstractMatrix) -> Vector{Int}
"""
predict_labels(m::LegendMLModel, X::AbstractMatrix) = Int.(unwrap.(_predict_mode(m.machine, table(X))))
export predict_labels

(m::LegendMLModel)(wvfs::AbstractVector{<:RDWaveform}) = predict_labels(m, features(m.features, wvfs))

"""
    predict_with_margin(m::LegendMLModel, X_or_wvfs) -> (labels, margin)

Labels and the probability of the predicted class. Only defined for models
trained with `probability = true`.
"""
function predict_with_margin(m::LegendMLModel, X::AbstractMatrix)
    m.machine.model isa ProbabilisticSVC ||
        throw(ArgumentError("margins need a probabilistic model; this model is $(typeof(m.machine.model))"))
    p = predict(m.machine, table(X))
    modes = predict_mode(m.machine, table(X))
    labels = Int.(unwrap.(modes))
    margin = Float64.(pdf.(p, modes))
    return labels, margin
end
function predict_with_margin(m::LegendMLModel, wvfs::AbstractVector{<:RDWaveform})
    return predict_with_margin(m, features(m.features, wvfs))
end
export predict_with_margin
