# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

"""
    LegendMachineLearning

Machine learning algorithms for the LEGEND experiment.
"""
module LegendMachineLearning

using LinearAlgebra: diagind
using Random: Random, AbstractRNG, shuffle
using Statistics: mean, quantile
using StatsBase: sample
using Dates: now
using SHA: sha256

using Adapt: adapt
using MLDataDevices: MLDataDevices
using Lux: Lux
using ADTypes: ADTypes, AbstractADType
import DifferentiationInterface as DI

using Unitful: Unitful, @u_str
using IntervalSets: ClosedInterval, leftendpoint, rightendpoint, (..)
using ArraysOfArrays: flatview, VectorOfSimilarArrays
using RadiationDetectorSignals: RDWaveform, ArrayOfRDWaveforms
using RadiationDetectorDSP: signalstats, shift_waveform, multiply_waveform
using LegendDSP: HaarAveragingFilter
using Distances: SemiMetric, Cityblock, pairwise
using CategoricalArrays: categorical, unwrap
using CategoricalDistributions: pdf
using MLJBase: machine, fit!, predict, predict_mode, evaluate!, StratifiedCV,
    serializable, restore!, partition
using MLJModelInterface: table
using Clustering: Clustering
using MLJLIBSVMInterface: SVC, ProbabilisticSVC
using StatisticalMeasures: balanced_accuracy
using JLD2: jldsave, jldopen
using PropDicts: PropDict
using TypedTables: Table
using LegendDataManagement: LegendDataManagement, LegendData, FileKey, DataPeriod, DataCategory,
    ChannelId, DetectorId, DataTier, runinfo, channelinfo, read_ldata, search_disk
using LegendHDF5IO: lh5open

include("types.jl")
include("features.jl")
include("labels.jl")
include("clustering.jl")
include("training.jl")
include("serialization.jl")
include("sampling.jl")
include("legend_io.jl")
include("pipeline.jl")

end # module
