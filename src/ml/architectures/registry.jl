# Architecture registry — central dispatch for model construction
#
# Provides:
#   MODEL_REGISTRY                        — Dict{String, Function}
#   register_architecture!(name, builder) — register a new architecture
#   unwrap_config(raw)                    — extract arch name from YAML dict
#   build_model(arch_name, cfg, n_sipm)   — construct model from registry
#   build_model(raw_yaml, n_sipm)         — construct model from raw YAML
#
# Each architecture lives in its own file under architectures/ and calls
# register_architecture!() at the bottom.  The YAML top-level key IS the
# architecture name:
#
#   mlp_detector_concat:
#     input: ...
#     architecture: ...
#     training: ...

const MODEL_REGISTRY = Dict{String, Function}()

"""Register a model builder: `register_architecture!("name", builder_fn)`."""
function register_architecture!(name::String, builder::Function)
    MODEL_REGISTRY[name] = builder
end

# ── Config unwrapping ────────────────────────────────────────────────────────

"""
    unwrap_config(raw::Dict) → (arch_name::String, cfg::Dict)

Extract the single top-level key as architecture name and return the inner dict.
"""
function unwrap_config(raw::Dict)
    ks = collect(keys(raw))
    length(ks) == 1 || error(
        "Training YAML must have exactly one top-level key (architecture name), got: $ks")
    arch_name = String(first(ks))
    cfg = raw[arch_name]
    cfg isa Dict || error("Value under '$arch_name' must be a dict, got $(typeof(cfg))")
    return arch_name, cfg
end

# ── Public build API ─────────────────────────────────────────────────────────

"""
    build_model(raw_yaml::Dict, n_sipm::Int) → (model, layout::Symbol)

Construct a model from a raw YAML dict (training path).
"""
function build_model(raw_yaml::Dict, n_sipm::Int)
    arch_name, cfg = unwrap_config(raw_yaml)
    return build_model(arch_name, cfg, n_sipm)
end

"""
    build_model(arch_name::String, cfg::Dict, n_sipm::Int) → (model, layout::Symbol)

Construct a model by name + config (used by both training and prediction).
"""
function build_model(arch_name::String, cfg::Dict, n_sipm::Int)
    haskey(MODEL_REGISTRY, arch_name) || error(
        "Unknown architecture: '$arch_name'. Registered: $(collect(keys(MODEL_REGISTRY)))")
    return MODEL_REGISTRY[arch_name](cfg, n_sipm)
end
