# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

"""
    preference_grid(X, quantiles; metric=Cityblock()) -> (S, preferences)

Negative pairwise distance matrix `S` of the rows of `X` and the
affinity-propagation preferences at the given quantiles of its upper
triangle.
"""
function preference_grid(X::AbstractMatrix, quantiles::AbstractVector{<:Real}; metric::SemiMetric=Cityblock())
    Base.require_one_based_indexing(X)
    S = -pairwise(metric, X, dims=1)
    n = size(S, 1)
    upper = [S[i, j] for j in 1:n for i in 1:j-1]
    S, quantile(upper, collect(Float64, quantiles))
end

function _run_ap(X, damp, preference, maxiter, tol, metric)
    model = AffinityPropagation(; damp=Float64(damp), preference=Float64(preference),
        maxiter=maxiter, tol=Float64(tol), metric=metric)
    mach = machine(model)
    assignment = predict(mach, table(X))
    Int.(unwrap.(assignment)), report(mach)
end

"""
    scan_affinity_propagation(X; damp, preference_quantile, target_n_clusters=100, maxiter=1000, tol=1e-6, metric=Cityblock())

Run affinity propagation on every (preference, damp) grid point and pick
the converged one whose cluster count is closest to `target_n_clusters`,
fewest iterations on ties. Grid points run on threads.
"""
function scan_affinity_propagation(X::AbstractMatrix;
        damp::AbstractVector{<:Real}, preference_quantile::AbstractVector{<:Real},
        target_n_clusters::Int=100, maxiter::Int=1000, tol::Real=1e-6, metric::SemiMetric=Cityblock())
    Base.require_one_based_indexing(X)
    damp = collect(Float64, damp)
    preference_quantile = collect(Float64, preference_quantile)
    _, preference = preference_grid(X, preference_quantile; metric)
    n_p, n_d = length(preference), length(damp)
    n_clusters = zeros(Int, n_p, n_d)
    converged = falses(n_p, n_d)
    iterations = zeros(Int, n_p, n_d)
    grid = [(p, d) for p in 1:n_p for d in 1:n_d]
    Threads.@threads for (p, d) in grid
        # A grid point that cannot find any exemplar within maxiter makes the
        # underlying algorithm raise AssertionError rather than report
        # converged=false; treat that as non-convergence so one bad grid
        # point does not abort the scan. Any other exception propagates.
        try
            _, rep = _run_ap(X, damp[d], preference[p], maxiter, tol, metric)
            n_clusters[p, d] = length(rep.cluster_labels)
            converged[p, d] = rep.converged
            iterations[p, d] = rep.iterations
        catch e
            e isa AssertionError || rethrow()
            n_clusters[p, d] = 0
            converged[p, d] = false
            iterations[p, d] = maxiter
        end
    end
    any(converged) || throw(ArgumentError("no grid point converged within maxiter = $maxiter"))
    dist = abs.(n_clusters .- target_n_clusters)
    dist[.!converged] .= typemax(Int)
    candidates = findall(==(minimum(dist)), dist)
    best_ci = candidates[argmin(iterations[candidates])]
    best = (i_preference=best_ci[1], i_damp=best_ci[2], damp=damp[best_ci[2]], preference=preference[best_ci[1]])
    (; damp, preference_quantile, preference, n_clusters, converged=Matrix(converged), iterations, best)
end
export scan_affinity_propagation

"""
    fit_affinity_propagation(X; damp, preference, preference_quantile=NaN, maxiter=1000, tol=1e-6, metric=Cityblock()) -> ClusterResult

Fit affinity propagation on the rows of `X` at a single (`damp`,
`preference`) grid point. Cluster `k` is the cluster containing
exemplar `exemplars[k]`.
"""
function fit_affinity_propagation(X::AbstractMatrix; damp::Real, preference::Real,
        preference_quantile::Real=NaN, maxiter::Int=1000, tol::Real=1e-6, metric::SemiMetric=Cityblock())
    assignment, rep = _run_ap(X, damp, preference, maxiter, tol, metric)
    exemplars = collect(Int, rep.exemplars)
    ClusterResult(
        (damp=Float64(damp), preference=Float64(preference), preference_quantile=Float64(preference_quantile),
         maxiter=maxiter, tol=Float64(tol), metric=string(nameof(typeof(metric)))),
        assignment, exemplars, rep.converged, rep.iterations)
end
export fit_affinity_propagation
