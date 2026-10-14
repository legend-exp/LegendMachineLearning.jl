# This file is a part of ML-based-LAr-veto, licensed under the MIT License (MIT).
# Validity parsing for metadata configurations

"""
    get_valid_config(metadata_path::String, category::Symbol, group_name::String)

Find the valid config file for a given group and category.
Looks up validity.yaml in the category folder to find which config applies.

Returns the full path to the config file, or nothing if not found.
"""
function get_valid_config(metadata_path::String, category::Symbol, group_name::String)
    category_dir = joinpath(metadata_path, string(category))
    validity_file = joinpath(category_dir, "validity.yaml")
    
    if !isfile(validity_file)
        @warn "No validity.yaml found in $category_dir"
        return nothing
    end
    
    validity = YAML.load_file(validity_file)
    
    # Search for group in validity mappings
    for (config_file, valid_groups) in validity
        if group_name in valid_groups
            config_path = joinpath(category_dir, config_file)
            if isfile(config_path)
                @debug "Found valid config for $group_name in $category: $config_file"
                return config_path
            else
                @warn "Config file $config_file listed in validity.yaml but not found"
            end
        end
    end
    
    @warn "No valid config found for group $group_name in category $category"
    return nothing
end
export get_valid_config

"""
    load_metadata_config(processing_config::PropDict, category::Symbol)

Load the metadata config for the current group and category.
Uses validity lookup to find the correct config file.
"""
function load_metadata_config(processing_config::PropDict, category::Symbol)
    metadata_path = processing_config.paths.metadata
    group_name = processing_config.group_name
    
    config_path = get_valid_config(metadata_path, category, group_name)
    
    if isnothing(config_path)
        return nothing
    end
    
    @info "Loading $category config from $(basename(config_path))"
    YAML.load_file(config_path)
end
export load_metadata_config

"""
    get_all_metadata_configs(processing_config::PropDict)

Load all metadata configs for the current group.
Returns a Dict with category => config mappings.
"""
function get_all_metadata_configs(processing_config::PropDict)
    configs = Dict{Symbol, Any}()
    
    for category in processing_config.metadata_categories
        config = load_metadata_config(processing_config, category)
        if !isnothing(config)
            configs[category] = config
        end
    end
    
    configs
end
export get_all_metadata_configs
