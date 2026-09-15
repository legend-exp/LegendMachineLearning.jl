# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

using Test
using LegendMachineLearning
using LegendMachineLearning: scan_affinity_propagation, fit_affinity_propagation, preference_grid
using StableRNGs

# three well separated blobs in 8 dimensions, 30 points each
function blobs(rng)
    centers = [zeros(8), fill(20.0, 8), vcat(fill(-20.0, 4), fill(20.0, 4))]
    X = vcat((c' .+ randn(rng, 30, 8) for c in centers)...)
    truth = repeat(1:3, inner=30)
    X, truth
end

@testset "affinity propagation" begin
    rng = StableRNG(1)
    X, truth = blobs(rng)

    @testset "preference_grid" begin
        S, prefs = preference_grid(X, [0.1, 0.5])
        @test size(S) == (90, 90)
        @test all(S .<= 0)
        @test issorted(prefs)
    end

    @testset "scan" begin
        r = scan_affinity_propagation(X; damp=[0.5, 0.9], preference_quantile=[0.05, 0.5],
            target_n_clusters=3, maxiter=500)
        @test size(r.n_clusters) == (2, 2)
        @test size(r.converged) == (2, 2)
        @test size(r.iterations) == (2, 2)
        @test r.n_clusters[r.best.i_preference, r.best.i_damp] == 3
        @test r.converged[r.best.i_preference, r.best.i_damp]
        @test r.best.damp in r.damp
        @test r.best.preference in r.preference
        @test size(r.S) == (90, 90)
    end

    @testset "fit" begin
        r = scan_affinity_propagation(X; damp=[0.5, 0.9], preference_quantile=[0.05, 0.5],
            target_n_clusters=3, maxiter=500)
        c = fit_affinity_propagation(X; damp=r.best.damp, preference=r.best.preference, maxiter=500)
        @test c isa ClusterResult
        @test c.converged
        @test length(c.exemplars) == 3
        @test length(c.assignment) == 90
        # each exemplar belongs to its own cluster
        @test all(c.assignment[c.exemplars[k]] == k for k in eachindex(c.exemplars))
        # cluster assignment matches the ground truth up to relabeling
        @test length(unique(zip(c.assignment, truth))) == 3
        @test c.hyperparameters.damp == r.best.damp

        # reusing the scan's shared similarity matrix gives the same fit
        c_shared = fit_affinity_propagation(X; damp=r.best.damp, preference=r.best.preference, maxiter=500, S=r.S)
        @test c_shared.assignment == c.assignment
        @test c_shared.exemplars == c.exemplars
    end

    @testset "no converged grid point" begin
        @test_throws "no grid point converged" scan_affinity_propagation(X; damp=[0.5],
            preference_quantile=[0.5], target_n_clusters=3, maxiter=1)
        # Clustering.jl's `@show A R` dump before its zero-exemplar assertion must
        # not reach stdout.
        mktemp() do path, io
            redirect_stdout(io) do
                try
                    scan_affinity_propagation(X; damp=[0.5], preference_quantile=[0.5], target_n_clusters=3, maxiter=1)
                catch e
                    e isa ArgumentError || rethrow()
                end
            end
            flush(io)
            @test isempty(read(path, String))
        end
    end
end
