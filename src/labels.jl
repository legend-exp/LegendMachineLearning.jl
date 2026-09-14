# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

"Label value of an exemplar that has not been assigned a category yet."
const UNLABELED = -1

"""
    propagate_labels(cluster::ClusterResult, exemplar_labels) -> Vector{Int}

Label of every sample row, taken from the label of its cluster's exemplar.
Throws if any exemplar is still `UNLABELED`: training must never run on
partially labeled clusters.
"""
function propagate_labels(cluster::ClusterResult, exemplar_labels::AbstractVector{<:Integer})
    axes(exemplar_labels, 1) == axes(cluster.exemplars, 1) ||
        throw(DimensionMismatch("$(length(exemplar_labels)) exemplar labels for $(length(cluster.exemplars)) exemplars"))
    missing_idx = findall(==(UNLABELED), exemplar_labels)
    isempty(missing_idx) || throw(ArgumentError("exemplars $(missing_idx) are unlabeled"))
    [Int(exemplar_labels[k]) for k in cluster.assignment]
end
export propagate_labels

"""
    qc_categories(config::PropDict) -> Dict{Int,Symbol}

Category table from the `categories` section of the QC ML config.
"""
qc_categories(config::PropDict) = Dict{Int,Symbol}(Int(k) => Symbol(v) for (k, v) in config.categories)
export qc_categories

"""
    class_fractions(labels, categories) -> Dict{Int,Float64}

Fraction of each configured category among `labels`; categories that do
not occur get 0. Throws on labels outside the table.
"""
function class_fractions(labels::AbstractVector{<:Integer}, categories::Dict{Int,Symbol})
    counts = Dict{Int,Int}(k => 0 for k in keys(categories))
    for l in labels
        haskey(counts, l) || throw(ArgumentError("label $l is not a configured category"))
        counts[l] += 1
    end
    n = length(labels)
    Dict{Int,Float64}(k => c / n for (k, c) in counts)
end
export class_fractions
