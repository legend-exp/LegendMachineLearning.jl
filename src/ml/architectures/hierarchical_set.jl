# HierarchicalSetModel — Two-level set model: Triggers → SiPMs → Event
#
# Architecture:
#   Per SiPM (shared weights):
#     Triggers [time, pe] → φ-MLP → Pool → sipm_vec
#     sipm_vec ⊕ geometry  → SiPM MLP → D_sipm
#   Event level:
#     Pool over 55 SiPM vectors → event_vec ⊕ HPGe embedding → Tail → logit
#
# Requires: layers/blocks.jl, layers/pooling.jl, layers/set_ops.jl
#
# TODO: Implement — see docs/model-architectures.md §3

# register_architecture!("hierarchical_set", _build_hierarchical_set)
