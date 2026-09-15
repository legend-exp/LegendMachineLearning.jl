# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

import Test
import LegendMachineLearning

Test.@testset "Package LegendMachineLearning" begin
    include("test_aqua.jl")
    include("test_explicit_imports.jl")
    include("test_features.jl")
    include("test_types_labels.jl")
    include("test_clustering.jl")
    include("test_training.jl")
    include("test_serialization.jl")
    include("test_sampling.jl")
    include("test_legend_io.jl")
    include("test_pipeline.jl")
    include("test_docs.jl")
    Test.@test isempty(Test.detect_ambiguities(LegendMachineLearning))
end
