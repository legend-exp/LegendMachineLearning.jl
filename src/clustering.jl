# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

"""
    preference_grid(X, quantiles; metric=Cityblock(), rng=Random.default_rng(), n_pairs_max=1_000_000) -> (S, preferences)

Negative pairwise distance matrix `S` of the rows of `X` and the
affinity-propagation preferences at the given quantiles. The quantiles are
estimated from at most `n_pairs_max` distinct off-diagonal pairs `(i, j)`,
`i < j`, drawn with `rng` (all pairs when there are `n_pairs_max` or fewer);
on the sampling branch, pairs are drawn by rejection into a `Set` rather than
enumerated, so no vector of size n(n-1)/2 is ever allocated.
"""
function preference_grid(X::AbstractMatrix, quantiles::AbstractVector{<:Real};
        metric::SemiMetric=Cityblock(), rng::AbstractRNG=Random.default_rng(), n_pairs_max::Int=1_000_000)
    Base.require_one_based_indexing(X)
    S = -pairwise(metric, X, dims=1)
    n = size(S, 1)
    n_pairs = n * (n - 1) ÷ 2
    pairs = if n_pairs ≤ n_pairs_max
        [(i, j) for j in 1:n for i in 1:j-1]
    else
        drawn = Set{Tuple{Int,Int}}()
        while length(drawn) < n_pairs_max
            i = rand(rng, 1:n); j = rand(rng, 1:n)
            i == j || push!(drawn, minmax(i, j))
        end
        drawn
    end
    sampled = [S[i, j] for (i, j) in pairs]
    S, quantile(sampled, collect(Float64, quantiles))
end
export preference_grid

# Clustering.jl's affinityprop prints the full A and R message matrices with
# `@show` before asserting when no exemplar emerges within maxiter; that
# assertion is caught below and treated as non-convergence, so the stdout
# dump is silenced rather than surfaced.
function _run_ap(S::AbstractMatrix, damp::Real, preference::Real, maxiter::Integer, tol::Real)
    S[diagind(S)] .= preference
    result = try
        redirect_stdout(devnull) do
            Clustering.affinityprop(S; damp=Float64(damp), maxiter, tol=Float64(tol), display=:none)
        end
    catch e
        e isa AssertionError || rethrow()
        return Int[], Int[], false, maxiter
    end
    collect(Int, result.assignments), collect(Int, result.exemplars), result.converged, result.iterations
end

"""
    scan_affinity_propagation(X; damp, preference_quantile, target_n_clusters=100, maxiter=1000, tol=1e-6, metric=Cityblock(), rng=Random.default_rng())

Run affinity propagation on every (preference, damp) grid point and pick
the converged one whose cluster count is closest to `target_n_clusters`,
fewest iterations on ties. Grid points share one similarity matrix and run
sequentially: each cell's working set is four n×n matrices, so cells cannot
run concurrently without either racing on the shared matrix or multiplying
that memory by the number of threads. The shared matrix's diagonal is
overwritten with the grid point's preference before each call, so it never
holds a specific preference once the scan returns.
"""
function scan_affinity_propagation(X::AbstractMatrix;
        damp::AbstractVector{<:Real}, preference_quantile::AbstractVector{<:Real},
        target_n_clusters::Int=100, maxiter::Int=1000, tol::Real=1e-6, metric::SemiMetric=Cityblock(),
        rng::AbstractRNG=Random.default_rng())
    Base.require_one_based_indexing(X)
    damp = collect(Float64, damp)
    preference_quantile = collect(Float64, preference_quantile)
    S, preference = preference_grid(X, preference_quantile; metric, rng)
    n_p, n_d = length(preference), length(damp)
    n_clusters = zeros(Int, n_p, n_d)
    converged = falses(n_p, n_d)
    iterations = zeros(Int, n_p, n_d)
    for d in 1:n_d, p in 1:n_p
        _, exemplars, conv, iters = _run_ap(S, damp[d], preference[p], maxiter, tol)
        n_clusters[p, d] = length(exemplars)
        converged[p, d] = conv
        iterations[p, d] = iters
    end
    any(converged) || throw(ArgumentError("no grid point converged within maxiter = $maxiter"))
    dist = abs.(n_clusters .- target_n_clusters)
    dist[.!converged] .= typemax(Int)
    candidates = findall(==(minimum(dist)), dist)
    best_ci = candidates[argmin(iterations[candidates])]
    best = (i_preference=best_ci[1], i_damp=best_ci[2], damp=damp[best_ci[2]], preference=preference[best_ci[1]])
    (; damp, preference_quantile, preference, n_clusters, converged=Matrix(converged), iterations, best, S)
end
export scan_affinity_propagation

"""
    fit_affinity_propagation(X; damp, preference, preference_quantile=NaN, maxiter=1000, tol=1e-6, metric=Cityblock(), S=nothing) -> ClusterResult

Fit affinity propagation on the rows of `X` at a single (`damp`,
`preference`) grid point. Cluster `k` is the cluster containing
exemplar `exemplars[k]`. `S`, when given, is the shared similarity matrix
from `scan_affinity_propagation`, reused instead of recomputed; its diagonal
is overwritten with `preference` on every call.
"""
function fit_affinity_propagation(X::AbstractMatrix; damp::Real, preference::Real,
        preference_quantile::Real=NaN, maxiter::Int=1000, tol::Real=1e-6, metric::SemiMetric=Cityblock(),
        S::Union{Nothing,AbstractMatrix}=nothing)
    S = S === nothing ? -pairwise(metric, X, dims=1) : S
    assignment, exemplars, conv, iters = _run_ap(S, damp, preference, maxiter, tol)
    isempty(exemplars) && throw(ArgumentError(
        "affinity propagation found no exemplar for damp = $damp, preference = $preference within maxiter = $maxiter"))
    ClusterResult(
        (damp=Float64(damp), preference=Float64(preference), preference_quantile=Float64(preference_quantile),
         maxiter=maxiter, tol=Float64(tol), metric=string(nameof(typeof(metric)))),
        assignment, exemplars, conv, iters)
end
export fit_affinity_propagation
