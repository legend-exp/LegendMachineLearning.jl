# This file is a part of LegendMachineLearning.jl, licensed under the MIT License (MIT).

"""
    LegendMLApps

Command-line apps around LegendMachineLearning. Each app is a submodule with
its own `@main` entry point.
"""
module LegendMLApps

"""
    QCML

Entry point of the `legend-qcml` app: serves the QC labeling, training and
check notebook `notebooks/qcml.jl` with Pluto.
"""
module QCML

using LegendDataManagement: LegendData, DataPeriod, DataCategory
using LegendMachineLearning: training_filekey, qc_pars_entry_path, qc_ml_periods
import Pluto

const NOTEBOOK = normpath(@__DIR__, "..", "notebooks", "qcml.jl")

const USAGE = """
    usage: legend-qcml [--config PATH] [--period PERIOD] [--category CATEGORY] [--port PORT] [--host HOST]

    Serve the QC machine-learning notebook (labeling, training, check) with Pluto.

      --config PATH        LEGEND data config (default: \$LEGEND_DATA_CONFIG)
      --period PERIOD      period selected at start, e.g. p18 (default: the last period with an entry)
      --category CATEGORY  cal or phy (default: cal)
      --port PORT          port of the Pluto server (default: 10004)
      --host HOST          address the server listens on (default: 127.0.0.1)
      --help               show this message
    """

const CATEGORIES = ("cal", "phy")

"""
    parse_args(args, env=ENV) -> NamedTuple

Options of `legend-qcml` from the command-line arguments `args`, as
`(; help, config, period, category, port, host)`. Options take their value as
the next argument or after `=`. `config` falls back to
`env["LEGEND_DATA_CONFIG"]`; `period` is `nothing` when not given. Throws an
`ArgumentError` on an unknown option, a missing value, a missing config
file, or an invalid period, category or port.
"""
function parse_args(args::AbstractVector{<:AbstractString}, env = ENV)
    values = Dict{String,String}()
    help = false
    i = firstindex(args)
    while i <= lastindex(args)
        arg = args[i]
        if arg in ("--help", "-h")
            help = true
        elseif startswith(arg, "--")
            name, value = occursin('=', arg) ? split(arg, '='; limit = 2) : (arg, nothing)
            name in ("--config", "--period", "--category", "--port", "--host") ||
                throw(ArgumentError("unknown option $name"))
            if value === nothing
                i < lastindex(args) && !startswith(args[i + 1], "--") ||
                    throw(ArgumentError("option $name needs a value"))
                i += 1
                value = args[i]
            end
            values[name[3:end]] = value
        else
            throw(ArgumentError("unexpected argument $arg"))
        end
        i += 1
    end
    help && return (; help, config = nothing, period = nothing, category = "cal", port = 10004, host = "127.0.0.1")
    config = get(values, "config", get(env, "LEGEND_DATA_CONFIG", ""))
    isempty(config) && throw(ArgumentError("no data config: pass --config PATH or set LEGEND_DATA_CONFIG"))
    isfile(config) || throw(ArgumentError("data config $config does not exist"))
    period = get(values, "period", nothing)
    period === nothing || occursin(r"^p\d{2}$", period) ||
        throw(ArgumentError("invalid period $period, expected e.g. p18"))
    category = get(values, "category", "cal")
    category in CATEGORIES ||
        throw(ArgumentError("invalid category $category, expected one of $(join(CATEGORIES, ", "))"))
    port = tryparse(Int, get(values, "port", "10004"))
    port !== nothing && 1 <= port <= 65535 || throw(ArgumentError("invalid port $(values["port"])"))
    host = get(values, "host", "127.0.0.1")
    return (; help, config = abspath(expanduser(config)), period, category, port, host)
end

"""
    entry_period(config, period, category) -> String

`period`, checked to have an `rpars.ml` entry of `category` for its training
run, or the last period with such an entry when `period` is `nothing`.
"""
function entry_period(config::AbstractString, period, category::AbstractString)
    l200 = withenv(() -> LegendData(:l200), "LEGEND_DATA_CONFIG" => config)
    available = string.(qc_ml_periods(l200, category))
    isempty(available) &&
        throw(ArgumentError("no rpars.ml entry of category $category under the par directory of $config"))
    period === nothing && return last(available)
    fk = training_filekey(l200, DataPeriod(period), DataCategory(category))
    path = qc_pars_entry_path(l200, fk)
    isfile(path) || throw(
        ArgumentError("period $period has no entry $path; periods with a $category entry: $(join(available, ", "))")
    )
    return period
end

function (@main)(args)
    opts = try
        o = parse_args(args)
        o.help ? o : merge(o, (; period = entry_period(o.config, o.period, o.category)))
    catch e
        e isa ArgumentError || rethrow()
        println(stderr, "legend-qcml: ", e.msg, "\n\n", USAGE)
        return 1
    end
    if opts.help
        print(USAGE)
        return 0
    end
    ENV["LEGEND_DATA_CONFIG"] = opts.config
    ENV["QCML_PERIOD"] = opts.period
    ENV["QCML_CATEGORY"] = opts.category
    # Pluto saves the notebook when it opens it; an installed (read-only) package
    # tree gets a writable copy.
    notebook = if uperm(dirname(NOTEBOOK)) & 0x02 != 0
        NOTEBOOK
    else
        cp(NOTEBOOK, joinpath(mktempdir(), basename(NOTEBOOK)))
    end
    println("""
        legend-qcml: $(opts.category) $(opts.period) with $(opts.config)
        Notebook: $notebook
        Open http://localhost:$(opts.port)/ and pick the running notebook.
        From another machine, tunnel the port first: ssh -L $(opts.port):localhost:$(opts.port) $(gethostname())
        """)
    Pluto.run(; notebook, port = opts.port, host = opts.host, launch_browser = false,
        require_secret_for_open_links = false, require_secret_for_access = false,
        threads = string(Threads.nthreads()))
    return 0
end

end # module QCML

end # module LegendMLApps
