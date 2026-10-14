# Optimizer construction from config
#
# Provides:
#   build_optimizer(cfg::Dict) → Optimisers.AbstractRule

"""
    build_optimizer(cfg::Dict) → AbstractRule

Construct an optimizer from the `training` block of a model config.
Reads: `learning_rate`, `weight_decay`, `optimizer` (adamw|adam|rmsprop).
"""
function build_optimizer(cfg::Dict)
    tc = cfg["training"]
    lr = Float32(tc["learning_rate"])
    wd = Float32(get(tc, "weight_decay", 0f0))
    name = lowercase(get(tc, "optimizer", "adamw"))

    name == "adamw"   && return Optimisers.AdamW(lr, (0.9f0, 0.999f0), wd)
    name == "adam"    && return (wd > 0 ?
        Optimisers.OptimiserChain(Optimisers.WeightDecay(wd), Optimisers.Adam(lr)) :
        Optimisers.Adam(lr))
    name == "rmsprop" && return Optimisers.RMSProp(lr)
    error("Unknown optimizer: '$name' — supported: adamw, adam, rmsprop")
end
