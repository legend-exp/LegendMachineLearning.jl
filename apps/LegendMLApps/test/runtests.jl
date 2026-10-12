# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

using Test
using LegendMLApps.QCML: parse_args, NOTEBOOK

@testset "LegendMLApps" begin
    @test isfile(NOTEBOOK)

    mktempdir() do dir
        config = joinpath(dir, "config.json")
        write(config, "{}")
        noenv = Dict{String,String}()

        @testset "defaults and values" begin
            o = parse_args(["--config", config], noenv)
            @test o == (; help=false, config, period=nothing, category="cal", port=10004, host="127.0.0.1")
            o = parse_args(["--config=$config", "--period", "p18", "--category=phy", "--port", "10010",
                "--host", "0.0.0.0"], noenv)
            @test (o.period, o.category, o.port, o.host) == ("p18", "phy", 10010, "0.0.0.0")
            # the config falls back to LEGEND_DATA_CONFIG
            @test parse_args(String[], Dict("LEGEND_DATA_CONFIG" => config)).config == config
            @test parse_args(["--help"], noenv).help
        end

        @testset "fail fast" begin
            @test_throws "unknown option --perod" parse_args(["--config", config, "--perod", "p18"], noenv)
            @test_throws "unexpected argument p18" parse_args(["--config", config, "p18"], noenv)
            @test_throws "option --period needs a value" parse_args(["--config", config, "--period"], noenv)
            @test_throws "option --period needs a value" parse_args(["--period", "--config", config], noenv)
            @test_throws "pass --config PATH or set LEGEND_DATA_CONFIG" parse_args(String[], noenv)
            @test_throws "does not exist" parse_args(["--config", joinpath(dir, "missing.json")], noenv)
            @test_throws "invalid period 18" parse_args(["--config", config, "--period", "18"], noenv)
            @test_throws "invalid category ath" parse_args(["--config", config, "--category", "ath"], noenv)
            @test_throws "invalid port 0" parse_args(["--config", config, "--port", "0"], noenv)
            @test_throws "invalid port x" parse_args(["--config", config, "--port", "x"], noenv)
        end
    end
end
