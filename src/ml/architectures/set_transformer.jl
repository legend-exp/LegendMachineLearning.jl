# FlatSetTransformer — Set Transformer over all triggers
#
# Architecture:
#   All triggers [time, pe, cos, dz, angle] → embed → ISAB × L → PMA → event_vec
#   event_vec ⊕ HPGe embedding → Tail MLP → logit
#
# Uses induced set attention blocks (ISAB) for O(N·M) complexity.
#
# Requires: layers/blocks.jl, layers/attention.jl, layers/set_ops.jl
#
# TODO: Implement — see docs/model-architectures.md §4

# register_architecture!("set_transformer", _build_set_transformer)
