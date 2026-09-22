# ML Architecture Guide

> How the ML code in `src/ml/` is structured, and how to add new model architectures.

---

## Directory Layout

```
src/
├── ml_setup.jl              ← Single entry point (include from processors)
├── features.jl              ← Feature loading, assembly, SiPM ordering (non-ML)
└── ml/
    ├── imports.jl            ← Package imports + _ML_READY / _GPU_READY flags
    ├── device.jl             ← GPU/CPU selection, gpu_sync_gc(), _snap()
    ├── losses.jl             ← Loss functions + LOSS_REGISTRY
    ├── optimizers.jl         ← Optimizer construction from YAML config
    ├── evaluation.jl         ← eval_loop() for validation / test metrics
    ├── training_loop.jl      ← train_model() — warmup, cosine LR, early stopping
    ├── model_io.jl           ← save_model(), load_model(), find_model()
    ├── layers/
    │   ├── blocks.jl         ← dense_block(), mlp_block(), det_embedding()
    │   ├── pooling.jl        ← (stub) MeanPool, MaxPool, AttentionPool, PMA
    │   ├── attention.jl      ← (stub) MultiheadAttention, ISAB
    │   └── set_ops.jl        ← (stub) Variable-length batching, masking
    └── architectures/
        ├── registry.jl       ← MODEL_REGISTRY, build_model(), unwrap_config()
        ├── det_concat_mlp.jl ← DetConcatMLP (two-branch MLP, current default)
        ├── hierarchical_set.jl  ← (stub) HierarchicalSetModel
        └── set_transformer.jl   ← (stub) FlatSetTransformer
```

## Include Order

`ml_setup.jl` includes everything in dependency order.  The `_ML_READY` guard
ensures it only runs once, even when multiple processors `include` it.

```
imports.jl → device.jl → losses.jl → optimizers.jl
→ layers/{blocks, pooling, attention, set_ops}
→ architectures/{registry, det_concat_mlp, hierarchical_set, set_transformer}
→ evaluation.jl → training_loop.jl → model_io.jl
```

## Key Concepts

### 1. Architecture Registry

Every model architecture registers itself via:

```julia
register_architecture!("my_model_name", _build_my_model)
```

The builder function signature must be:

```julia
function _build_my_model(cfg::Dict, n_sipm::Int) -> (model, layout::Symbol)
```

- `cfg` is the parsed YAML config (everything under the architecture key)
- `n_sipm` is the number of SiPM channels
- `model` is a Lux model (any `AbstractLuxLayer`)
- `layout` is `:by_channel` or `:flat` (how SiPM features are arranged)

The YAML top-level key determines which architecture is used:

```yaml
mlp_detector_concat:      # ← this key selects the architecture
  input: { ... }
  architecture: { ... }
  training: { ... }
```

### 2. Shared Building Blocks (layers/blocks.jl)

All architectures should use these shared helpers:

| Function | Purpose |
|---|---|
| `resolve_activation(name)` | String → activation function (`relu`, `gelu`, `swish`, `tanh`, `sigmoid`) |
| `dense_block(in, out, act; dp, bn)` | Single dense layer with optional dropout + BatchNorm |
| `mlp_block(dims, act; dp, bn)` | Chain of dense blocks from a list of widths |
| `det_embedding(hpge_cfg, act; bn)` | HPGe detector feature embedding (MLP or identity) |

This avoids duplicating MLP construction logic across architectures.

### 3. Loss Functions (losses.jl)

Losses are registered in `LOSS_REGISTRY`. Currently only BCE on logits:

```julia
LOSS_REGISTRY["bce"] = _bce   # numerically stable BCE from logits
```

To add a new loss, define it and register:

```julia
function _my_loss(ŷ, y)
    # ŷ are raw logits (before sigmoid), y are 0/1 labels
    ...
end
LOSS_REGISTRY["my_loss"] = _my_loss
```

Then set `training.loss: my_loss` in the YAML config.

### 4. Optimizers (optimizers.jl)

`build_optimizer(cfg)` reads `training.optimizer` and `training.weight_decay`
from the YAML config. Supported: `adamw`, `adam`, `rmsprop`.

### 5. Training Loop (training_loop.jl)

`train_model(model, ps, st, train_dl, val_dl, cfg, dev_fn)` handles:
- Warmup phase (first batch, tracks compilation time)
- Cosine annealing LR schedule
- Early stopping with configurable patience
- CSV logging of per-epoch metrics
- Returns `(best_ps, best_st, best_epoch, best_val_loss)`

### 6. Evaluation (evaluation.jl)

`eval_loop(model, ps, st, loader, dev_fn)` computes mean loss + accuracy
over any DataLoader. Used for validation during training and final test metrics.

`_eval_loop` is an alias for backward compatibility with `process_training.jl`.

### 7. Model I/O (model_io.jl)

- `save_model(path, ps, st, metadata)` — JLD2 with metadata dict
- `load_model(path)` → `(ps, st, metadata)`
- `find_model(dir)` / `find_latest_model(dir)` — locate saved model files

---

## How to Add a New Architecture

### Step 1: Create the architecture file

```
src/ml/architectures/my_new_model.jl
```

### Step 2: Define the Lux model

```julia
if _ML_READY

using Lux: AbstractLuxContainerLayer, Chain, Dense

struct MyNewModel{A,B} <: AbstractLuxContainerLayer{(:encoder, :head)}
    encoder::A
    head::B
end

function Lux.apply(m::MyNewModel, x, ps, st)
    h, st_enc = m.encoder(x, ps.encoder, st.encoder)
    y, st_head = m.head(h, ps.head, st.head)
    return y, (encoder=st_enc, head=st_head)
end

function _build_my_new_model(cfg::Dict, n_sipm::Int)
    ac = cfg["architecture"]
    # ... use mlp_block(), dense_block(), resolve_activation() ...
    model = MyNewModel(encoder, head)
    layout = :by_channel
    return model, layout
end

register_architecture!("my_new_model", _build_my_new_model)

end  # if _ML_READY
```

### Step 3: Add the include to `ml_setup.jl`

Add after the existing architecture includes:

```julia
include(joinpath(_ml_dir, "architectures", "my_new_model.jl"))
```

### Step 4: Create a YAML config

```yaml
my_new_model:
  input:
    sipm:
      features: [sipm_pe_sums_prompt_scaled, sipm_pe_sums_delayed_scaled]
      layout: by_channel
    hpge:
      enabled: true
      features: [scaled_angle]
      output_dim: 16
  architecture:
    hidden_widths: [128, 64]
    activation: relu
    dropout: 0.1
    batch_norm: true
    # ... model-specific keys ...
  training:
    epochs: 100
    batch_size: 512
    learning_rate: 1e-3
    optimizer: adamw
    weight_decay: 1e-4
    early_stopping_patience: 20
    lr_scheduler: cosine
    lr_min: 1e-6
```

### Step 5: Run training

No changes to `process_training.jl` needed — the registry dispatches automatically.

---

## Planned Models

### HierarchicalSetModel (`hierarchical_set.jl`)

Two-level set architecture: per-trigger MLP → per-SiPM aggregation → event-level head.
See `docs/model-architectures.md` §3 for physics motivation.

### FlatSetTransformer (`set_transformer.jl`)

Induced Set Attention Blocks (ISAB) + Pooling by Multihead Attention (PMA)
over all triggers as a flat set. See `docs/model-architectures.md` §4.

---

## Configuration-Driven Design

Everything is driven by YAML config:
- **Which model** → top-level YAML key
- **Which features** → `input.sipm.features` / `input.hpge.features`
- **Architecture params** → `architecture.*`
- **Training params** → `training.*`
- **Which loss** → `training.loss` (default: `bce`)
- **Which optimizer** → `training.optimizer`

No hardcoded feature lists, no if/else chains for model selection.
