# Model I/O — save, load, and find model files (JLD2)
#
# Provides:
#   save_model(path, ps, st, metadata)
#   load_model(path) → (ps, st, metadata)
#   find_model(dir; selection) → path

"""Save model parameters, states (in testmode), and metadata to JLD2."""
function save_model(path::String, ps, st, metadata::Dict)
    mkpath(dirname(path))
    jldsave(path; ps=ps, st=Lux.testmode(st), metadata=metadata)
    @info "  Model saved: $path"
end

"""Load parameters, states, and metadata from a JLD2 model file."""
function load_model(path::String)
    data = JLD2.load(path)
    return data["ps"], data["st"], data["metadata"]
end

"""
    find_model(model_dir; selection=1) → String

Find a `.jld2` model file. `selection=1` = newest, `selection=2` = second-newest.
"""
function find_model(model_dir::String; selection::Int=1)
    jld2_files = filter(f -> endswith(f, ".jld2"), readdir(model_dir; join=true))
    isempty(jld2_files) && error("No .jld2 model files found in $model_dir")
    sort!(jld2_files)
    n = length(jld2_files)
    idx = n - selection + 1
    1 <= idx <= n || error("selection=$selection out of range — only $n models in $model_dir")
    return jld2_files[idx]
end


