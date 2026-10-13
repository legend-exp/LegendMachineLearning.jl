# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

import Test
import ExplicitImports
import LegendMachineLearning

Test.@testset "ExplicitImports" begin
    Test.@test ExplicitImports.check_no_implicit_imports(LegendMachineLearning) === nothing
    # Lux, ADTypes and DifferentiationInterface support planned neural-net
    # training; the module imports their names now so later code can use them
    # without touching this import list.
    Test.@test ExplicitImports.check_no_stale_explicit_imports(LegendMachineLearning;
        ignore = (:Lux, :adapt, :MLDataDevices, :ADTypes, :AbstractADType, :DI)) === nothing
end
