# Training Flow

## Overview

```
jlnormml/{train, val, test}
  │
  │  load config-driven features (sipm + det)
  │  assemble feature matrices
  │  optional: SiPM ordering permutation
  │  train Lux neural network
  │
  ▼
generated/model/{group}/{arch}/{group}_{arch}_{timestamp}.jld2
```

## Input

**Source:** `jlnormml/{train,val,test}` — reads `:<tier>` table + `sipm_detector_ids`

**Features loaded** (driven by model config, e.g. `mlp_detector_concat`):

| Type | Feature columns read | Shape per event |
|------|---------------------|-----------------|
| SiPM | `sipm_pe_sums_scaled`, `sipm_pe_sums_prompt_scaled`, `sipm_pe_sums_delayed_scaled`, `cos_proximity`, `scaled_delta_z` | VoV → Matrix (n_sipms each) |
| Det | `scaled_angle`, `scaled_z_center` | scalar each |
| Label | `label` | Int8 → Float32 |

### Feature Assembly (`assemble_features`)

```
x_sipm = Matrix{Float32}  (n_sipm_features × n_sipms, n_samples)
x_det  = Matrix{Float32}  (n_det_features, n_samples)
y      = Vector{Float32}  (n_samples,)
```

Optional: SiPM ordering permutation sorts SiPMs by physical position
(IB top → IB bottom → OB top → OB bottom, by angle within each group).

## Processing

- Architecture: `mlp_detector_concat` (DetConcatMLP)
- Framework: Lux.jl + Optimisers.jl
- GPU: CUDA (if available)
- Training loop with early stopping, LR scheduling
- Metrics: accuracy, loss per split

## Output

**File:** `generated/model/{group}/{arch}/{group}_{arch}_{timestamp}.jld2`

| JLD2 Key | Content |
|----------|---------|
| `ps` | Model parameters (trained weights) |
| `st` | Model state (BatchNorm running stats etc.) |
| `metadata` | Dict with: `config` (full model config), `architecture_name`, `timestamp_utc`, `input_dims` (sipm_channels, sipm_features, det_features), `sipm_ordering` (permutation + enabled flag), `sipm_detector_ids`, `final_metrics` (train/val/test loss+acc) |

**No LH5 output** — only JLD2 model file.
