# This file is a part of ML-based-LAr-veto, licensed under the MIT License (MIT).
# Configuration parsing - Juleana compatible

function get_argparse()
    settings = ArgParseSettings(
        prog = "ML-based LAr Veto",
        description = "Neural network training for LAr veto",
        version = "1.0",
        add_version = true
    )
    @add_arg_table settings begin
        "--config", "-c"
            help = "Path to config file (YAML)"
            arg_type = String
            required = true
        "--group", "-g"
            help = "ML grouping to process"
            arg_type = String
            default = ""
        "--runmode", "--rm"
            dest_name = "runmode"
            arg_type = String
        "--only"
            help = "Comma-separated list of processors to run (overrides YAML `enabled` flags). Mutually exclusive with --skip."
            arg_type = String
            default = ""
        "--skip"
            help = "Comma-separated list of processors to exclude (overrides YAML `enabled` flags). Mutually exclusive with --only."
            arg_type = String
            default = ""
    end
    parse_args(settings)
end

function get_processingconfig()
    parsed_args = get_argparse()
    processing_config = readlprops(parsed_args["config"])
    processing_config.parsed_args = parsed_args
    
    # Project root = directory containing the config file's parent
    project_root = dirname(dirname(parsed_args["config"]))
    
    # Load data_config.yaml (default: config/data_config.yaml in project root)
    data_config_path = get(processing_config, :data_config, joinpath(project_root, "config", "data_config.yaml"))
    @info "Loading data config from $data_config_path"
    data_config = readlprops(data_config_path)
    
    # Resolve relative paths against project root
    for key in (:config, :metadata, :generated)
        if haskey(data_config.paths, key)
            p = string(data_config.paths[key])
            if !isabspath(p)
                data_config.paths[key] = joinpath(project_root, p)
            end
        end
    end
    
    # Store base paths
    processing_config.paths = data_config.paths
    
    # Generate derived paths from base paths
    processing_config.paths.groupings = joinpath(data_config.paths.config, "ml_groupings.yaml")
    processing_config.paths.output = PropDict(
        :tier => joinpath(data_config.paths.generated, "tier"),
        :model => joinpath(data_config.paths.generated, "model"),
        :plots => joinpath(data_config.paths.generated, "plots"),
        :logs => joinpath(data_config.paths.generated, "logs"),
        :reports => joinpath(data_config.paths.generated, "reports"),
        :geometry => joinpath(data_config.paths.generated, "geometry")
    )
    
    # Define metadata categories (order matters for pipeline)
    processing_config.metadata_categories = [:geometry, :extraction, :balancing, :hpo, :training, :prediction]
    
    # Set LEGEND_DATA_CONFIG environment variable
    ENV["LEGEND_DATA_CONFIG"] = data_config.legend_data_config
    @info "LEGEND_DATA_CONFIG set to $(data_config.legend_data_config)"
    
    # Environment variables from config
    if haskey(processing_config, :env_variables)
        for (key, val) in processing_config.env_variables
            ENV[String(key)] = string(val)
        end
    end
    
    # Load LegendData
    @info "Loading LegendData"
    l200 = LegendData(:l200)
    
    # Get ML group(s) — supports a single name or a list, plus a comma-
    # separated CLI override.
    group_names = if !isempty(parsed_args["group"])
        String.(strip.(split(parsed_args["group"], ",")))
    elseif haskey(processing_config.datasets, :active_groups)
        ag = processing_config.datasets.active_groups
        ag isa AbstractVector ? String.(ag) : [String(ag)]
    else
        # `active_group` (singular) historically held one name, but accept a
        # list too so a YAML typo (`active_group: [a, b]`) just works.
        ag = get(processing_config.datasets, :active_group, "mlgroup001")
        ag isa AbstractVector ? String.(ag) : [String(ag)]
    end
    filter!(!isempty, group_names)
    isempty(group_names) && error("No ML groups configured (datasets.active_group/active_groups)")
    processing_config.group_names = group_names
    # Keep `group_name` populated for any legacy reader (uses the first one).
    processing_config.group_name  = first(group_names)
    @info "Active ML group(s): $group_names"

    # Get processors sorted by rank
    possible_steps = filter!(x -> x != :default, Symbol.(keys(processing_config.processors)))
    possible_steps = sort(possible_steps, by = s -> processing_config.processors[s].rank)
    processing_config.possible_steps = possible_steps
    processing_config.enabled_steps = possible_steps[[processing_config.processors[s].enabled for s in possible_steps]]

    # CLI overrides: --only / --skip beat the YAML `enabled` flags
    only_str = strip(get(parsed_args, "only", ""))
    skip_str = strip(get(parsed_args, "skip", ""))
    if !isempty(only_str) && !isempty(skip_str)
        error("--only and --skip are mutually exclusive")
    end
    _parse_step_list(s) = Symbol.(filter!(!isempty, strip.(split(s, ","))))
    if !isempty(only_str)
        only_syms = _parse_step_list(only_str)
        unknown = setdiff(Set(only_syms), Set(possible_steps))
        !isempty(unknown) && error("Unknown processor(s) in --only: $(collect(unknown)). Available: $possible_steps")
        processing_config.enabled_steps = filter(s -> s in only_syms, possible_steps)
        @info "CLI --only override → enabled_steps = $(processing_config.enabled_steps)"
    elseif !isempty(skip_str)
        skip_syms = _parse_step_list(skip_str)
        unknown = setdiff(Set(skip_syms), Set(possible_steps))
        !isempty(unknown) && error("Unknown processor(s) in --skip: $(collect(unknown)). Available: $possible_steps")
        processing_config.enabled_steps = filter(s -> !(s in skip_syms), processing_config.enabled_steps)
        @info "CLI --skip override → enabled_steps = $(processing_config.enabled_steps)"
    end

    return l200, processing_config, group_names
end
