# This file is a part of ML-based-LAr-veto, licensed under the MIT License (MIT).

using LegendHDF5IO, LegendDataTypes, LegendDataManagement, LegendDataManagement.LDMUtils, LegendEventAnalysis
using IntervalSets, PropertyFunctions, TypedTables, PropDicts, StatsBase
using Unitful, Format, Dates, Measurements
using Printf
using Random
using ProgressMeter, TimerOutputs
using ParallelProcessingTools
using Distributed
using YAML
using StructArrays: StructArrays, StructArray, StructVector
using ArraysOfArrays: VectorOfVectors
using Logging: global_logger
using TerminalLoggers: TerminalLogger
using ArgParse

global_logger(TerminalLogger())

# Include helper functions
include("io.jl")
include("extraction.jl")
include("validity.jl")
include("balancing.jl")
include("normalization.jl")
include("prediction.jl")
include("plotting.jl")
include("report.jl")

# Non-ML feature assembly (uses LH5, YAML — no CUDA/Lux dependency)
include("features.jl")

# ML-dependent modules (models, training, model_io) are loaded on-demand
# via src/ml_setup.jl when a processor that needs them is included.
# This avoids expensive CUDA/Lux precompilation for CPU-only pipeline steps.

# GC
GC.gc()
