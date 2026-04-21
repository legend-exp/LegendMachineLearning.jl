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
    processing_config.metadata_categories = [:geometry, :extraction, :balancing, :training, :prediction]
    
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
    
    # Get ML group
    group_name = if !isempty(parsed_args["group"])
        parsed_args["group"]
    else
        get(processing_config.datasets, :active_group, "mlgroup001")
    end
    processing_config.group_name = group_name
    @info "Active ML group: $group_name"
    
    # Get processors sorted by rank
    possible_steps = filter!(x -> x != :default, Symbol.(keys(processing_config.processors)))
    possible_steps = sort(possible_steps, by = s -> processing_config.processors[s].rank)
    processing_config.possible_steps = possible_steps
    processing_config.enabled_steps = possible_steps[[processing_config.processors[s].enabled for s in possible_steps]]
    
    return l200, processing_config, group_name
end
