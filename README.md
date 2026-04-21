# ML-based LAr Veto

Machine-learning pipeline for liquid argon (LAr) SiPM-based veto classification in the LEGEND-200 experiment. Replaces the classical global 4×4 cut with a geometry-aware neural network. Built in Julia using Lux.jl for GPU-accelerated training (NVIDIA A100 / CUDA).

## Motivation

The conventional LAr veto uses a global **4×4 cut**: veto if the summed PE across all 58 SiPMs is ≥ 4 **or** the SiPM multiplicity is ≥ 4 within a coincidence window. This ignores the geometric relationship between the triggered HPGe detector and the SiPM positions.

This package trains an MLP that takes per-SiPM photoelectron counts (in prompt and delayed time windows), the cosine proximity of each SiPM to the HPGe, and the HPGe position encoding as input. LAr scintillation has a singlet (prompt, ~6 ns) and triplet (delayed, ~1.6 µs) component — the prompt/delayed PE ratio helps distinguish true coincidence from random. The output is a continuous veto probability ∈ [0, 1]. The cut threshold is tuned to match the forced-trigger acceptance of the 4×4 cut.

## Processing Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         ML-based LAr Veto Pipeline                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    legend-metadata (channelinfo, diodes)                    │
│  │  1. GEOMETRY │◄── legend-pygeom (extra_meta: rod lengths, string coords)  │
│  └──────┬──────┘                                                             │
│         │  HPGe geometry YAML: (x, y, z) per detector, string layout         │
│         │  SiPM geometry YAML: barrel, fiber, position, rawid                │
│         ▼                                                                    │
│  ┌──────────────┐    jlevt tier (LEGEND event data)                          │
│  │ 2. EXTRACTION │◄── LegendData (read_ldata)                               │
│  └──────┬───────┘                                                            │
│         │  Parallel workers (Distributed.jl): read jlevt in batches          │
│         │  Apply event filters (ljl_propfunc): QC, multiplicity, energy      │
│         │  Select configured keys (spms + geds subsets)                       │
│         │  Compute windowed PE sums per SiPM:                                │
│         │    • Full window:    [t0 − 1 µs, t0 + 5 µs]                       │
│         │    • Prompt window:  [t0 − 1 µs, t0 + 1 µs]                       │
│         │    • Delayed window: [t0 − 1 µs, t0 + 5 µs]                       │
│         │  Filter DC-tagged triggers + PE threshold                          │
│         │  Output: jlext tier (filtered events + :wpe group)                 │
│         ▼                                                                    │
│  ┌──────────────┐                                                            │
│  │ 3. BALANCING  │                                                           │
│  └──────┬───────┘                                                            │
│         │  Read windowed PE from jlext                                       │
│         │  ML PE-sum filter: keep events in [pe_lo, pe_hi] for training      │
│         │  Balance sub500keV (label=1) vs forcedtrigger (label=0) counts     │
│         │  Assign HPGe detector IDs to forcedtrigger events                  │
│         │  Output: jlbalml (balanced, for training)                          │
│         │          jlbal   (all events, for inference)                        │
│         ▼                                                                    │
│  ┌────────────────┐                                                          │
│  │ 4. NORMALIZATION│                                                         │
│  └──────┬─────────┘                                                          │
│         │  Step 1 — Scaling: clip → log1p → global min-max to [0, 1]         │
│         │    Separate scaling for full / prompt / delayed PE matrices         │
│         │    Global min/max computed across both datasets                     │
│         │  Step 2 — Geometry features:                                       │
│         │    cos_proximity(HPGe, SiPM_i) per event × SiPM                    │
│         │    HPGe angle_norm, z_center_norm (position encoding)              │
│         │  Step 3 — Label + train/val/test split (jlnormml)                  │
│         │           or single output (jlnorm)                                │
│         │  Output: jlnormml/{train,val,test}.lh5                             │
│         │          jlnorm/all.lh5                                            │
│         ▼                                                                    │
│  ┌──────────────┐    GPU: NVIDIA A100 (CUDA.jl + cuDNN)                      │
│  │ 5. TRAINING   │◄── AD: Zygote.jl                                         │
│  └──────┬───────┘                                                            │
│         │  Build DetConcatMLP (two-branch architecture)                      │
│         │  SiPM ordering: IB_top → IB_bottom → OB_top → OB_bottom           │
│         │  Train with AdamW + cosine LR schedule + early stopping            │
│         │  Loss: numerically stable BCE on raw logits                        │
│         │  Output: model/<group>/<group>_mlp_<timestamp>.jld2                │
│         ▼                                                                    │
│  ┌──────────────┐                                                            │
│  │ 6. PREDICTION │                                                           │
│  └──────┬───────┘                                                            │
│         │  Load latest model + jlnorm(ml) data                               │
│         │  Hard classification rules:                                        │
│         │    pe_sum == 0     → pred = 0.0  (no veto)                         │
│         │    pe_sum ≥ thresh → pred = 1.0  (definite veto)                   │
│         │    otherwise       → model(sipm_features, det_features)            │
│         │  Classical 4×4 comparison: veto if sum_pe≥4 OR mult≥4              │
│         │  Threshold tuning: match FT survival of 4×4 cut                    │
│         │  Energy spectrum plots (K40, K42 regions)                          │
│         │  Output: jlpred(ml) tier + plots + threshold YAML                  │
│         ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐         │
│  │  pred_ml ∈ [0, 1]  →  veto if pred_ml > threshold_ft           │         │
│  │  Threshold tuned so that FT acceptance = 4×4 FT acceptance      │         │
│  └──────────────────────────────────────────────────────────────────┘         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Tier Flow Summary

```
jlevt (LEGEND raw events)
  │
  ├─ process_extraction ──► jlext (filtered + windowed PE)
  │
  ├─ process_balancing  ──► jlbalml (balanced, training)
  │                     ──► jlbal   (all events, inference)
  │
  ├─ process_normalization ► jlnormml/{train,val,test}.lh5
  │                        ► jlnorm/all.lh5
  │
  ├─ process_training   ──► model.jld2 (DetConcatMLP weights)
  │
  └─ process_prediction ──► jlpredml (labeled splits)
                         ──► jlpred   (all events, final output)
```

## Model Architecture

**DetConcatMLP** — two-branch MLP that fuses SiPM light-sensor features with HPGe position encoding.

### Input Features

#### SiPM Branch (58 × 3 = 174 features in `prompt_delayed` mode)

Per SiPM channel, interleaved (`by_channel` layout):

| Feature | Description |
|---------|-------------|
| `pe_scaled_prompt[ch_i]` | Normalized PE sum in prompt window [t0 − 1 µs, t0 + 1 µs] |
| `pe_scaled_delayed[ch_i]` | Normalized PE sum in delayed window [t0 − 1 µs, t0 + 5 µs] |
| `cos_prox[ch_i]` | Cosine of angle between HPGe position and SiPM *i* position |

Layout: `[pe_p₁, pe_d₁, cos₁, pe_p₂, pe_d₂, cos₂, …, pe_p₅₈, pe_d₅₈, cos₅₈]`

In `full_window` mode (legacy): 58 × 2 = 116 features (`[pe₁, cos₁, …]`).

#### Detector Branch (2 features)

| Feature | Description |
|---------|-------------|
| `ged_angle_norm` | Normalized angular position of triggered HPGe |
| `ged_z_center_norm` | Normalized vertical center position of triggered HPGe |

#### Label

| Value | Meaning |
|-------|---------|
| 1 | **Signal** — sub-500 keV HPGe trigger (physics event) |
| 0 | **Background** — forced trigger (no HPGe energy deposit) |

### Network Structure

```
    SiPM Input (174)                    Det Input (2)
         │                                   │
    [ prefix ]                          Dense(2 → 64)
  (optional layers)                     Activation
         │                              Dense(64 → 24)
         │                                   │
         └──────────── VCat ─────────────────┘
                         │
                    198 features
                         │
                  Dense(198 → 512)
                  Norm + Activation + Dropout
                         │
                  Dense(512 → 256)
                  Norm + Activation + Dropout
                         │
                  Dense(256 → 128)
                  Norm + Activation + Dropout
                         │
                  Dense(128 → 64)
                  Norm + Activation + Dropout
                         │
                  Dense(64 → 1)
                         │
                    raw logit
                         │
                     σ(logit)
                         │
                 prediction ∈ [0, 1]
```

**Key design choices:**

- **Two-branch architecture**: SiPM light pattern and HPGe geometry are processed separately before fusion. The detector embedding learns a representation of where the HPGe sits in the cryostat, allowing the tail to interpret the light pattern relative to the detector position.
- **Prompt + delayed PE windows**: Exploits the singlet/triplet time structure of LAr scintillation. True coincidence events have characteristic prompt/delayed ratios; random coincidence events do not.
- **Cosine proximity**: Encodes the geometric relationship (3D angle) between each SiPM and the triggered HPGe. SiPMs closer to the interaction point are expected to detect more light.
- **LayerNorm** over BatchNorm: LayerNorm normalizes per-sample, producing simpler gradients that compile fast with Zygote AD.
- **BCE on raw logits**: Numerically stable binary cross-entropy avoids gradient saturation.

### Prediction Rules

The final veto decision combines model output with rule-based hard overrides:

| Condition | Decision | Rationale |
|-----------|----------|-----------|
| `event_sum_pe == 0` | `pred = 0.0` (no veto) | No light detected — nothing to veto |
| `event_sum_pe ≥ pe_hard_veto` | `pred = 1.0` (veto) | Strong light — clearly correlated |
| otherwise | model prediction | Ambiguous region where ML adds value |

Classical comparison: **4×4 cut** — veto if `sum_pe ≥ 4` **or** `multiplicity ≥ 4`.

The ML threshold is selected so that **forced-trigger acceptance matches the 4×4 cut acceptance**.

## Project Structure

```
├── main.jl                       # Entry point — loads config, runs enabled processors
├── run_flow.sh                   # SLURM script: CPU pipeline (geometry→extraction→balancing→norm)
├── run_training.sh               # SLURM script: GPU training (A100)
├── config/
│   ├── data_config.yaml          # Data paths, LEGEND config path
│   ├── ml_groupings.yaml         # Period/run definitions per ML group
│   └── processing_config.yaml    # Enable/disable processors, rank, workers
├── metadata/                     # Per-group configs with validity lookup
│   ├── geometry/                 # Generated geometry configs
│   ├── extraction/               # Event filters, key selection, time windows
│   ├── balancing/                # PE range filter, balancing mode
│   ├── normalization/            # Scaling method, geometry augmentation
│   ├── training/                 # Model architecture, hyperparameters
│   └── prediction/               # Thresholds, hard classification rules
├── processors/
│   ├── process_geometry.jl       # HPGe/SiPM geometry YAML generation
│   ├── process_extraction.jl     # jlevt → jlext (filter + windowed PE)
│   ├── process_balancing.jl      # jlext → jlbal(ml) (balance + det ID assignment)
│   ├── process_normalization.jl  # jlbal(ml) → jlnorm(ml) (scale + geometry + split)
│   ├── process_training.jl       # jlnormml → model.jld2 (Lux + Zygote + CUDA)
│   └── process_prediction.jl     # model + jlnorm(ml) → jlpred(ml) + plots
├── src/
│   ├── startup.jl                # Package imports, module includes
│   ├── config.jl                 # Config parsing, LegendData setup
│   ├── validity.jl               # Metadata validity lookup
│   ├── geometry.jl               # HPGe/SiPM position computation
│   ├── extraction.jl             # Event filter parsing, key selection, distributed I/O
│   ├── balancing.jl              # PreparedDataset, windowed PE computation
│   ├── normalization.jl          # Scaling methods (log1p_minmax)
│   ├── model_builder.jl          # DetConcatMLP construction
│   └── prediction.jl             # Model loading, inference, feature assembly
└── generated/
    ├── geometry/                 # HPGe + SiPM geometry YAMLs
    ├── tier/                     # Data tiers (jlext, jlbal, jlbalml, jlnorm, jlnormml, jlpred, jlpredml)
    ├── model/                    # Saved models (.jld2)
    ├── plots/                    # QC + evaluation plots
    ├── reports/                  # Extraction reports (Markdown)
    └── logs/                     # SLURM stdout/stderr
```

## Requirements

- **Julia** ≥ 1.11
- **GPU** (training): NVIDIA A100 (CUDA.jl + cuDNN)
- **Key packages**: Lux.jl, Zygote.jl, CUDA.jl, LegendDataManagement.jl, LegendHDF5IO.jl, JLD2.jl

## SLURM Resource Configuration

1 Julia thread = 1 CPU core. SLURM allocates cores, Julia distributes them across master + workers.

### `run_flow.sh` — CPU-only pipeline

| Parameter | Location | Description |
|-----------|----------|-------------|
| `--cpus-per-task` | run_flow.sh | Total CPU cores from SLURM |
| `--mem` | run_flow.sh | Total RAM |
| `--time` | run_flow.sh | Wall time |
| `JULIA_NUM_THREADS` | run_flow.sh | Threads for Julia master process |
| `n_workers` | processing_config.yaml | Workers spawned via `addprocs()` for extraction |
| `threads_per_worker` | processing_config.yaml | Threads per extraction worker |

**CPU budget:** `n_workers × threads_per_worker + JULIA_NUM_THREADS ≤ cpus-per-task`

**RAM budget:** auto-split as `mem / (n_workers + 1)` per process (via `--heap-size-hint`)

| Step | Processes | Threads | Example (72 CPUs, 7w × 9tpw + 9 master) |
|------|-----------|---------|------------------------------------------|
| Extraction | master + n_workers | master: `JULIA_NUM_THREADS`, each worker: `threads_per_worker` | 8 processes, 7×9 + 9 = 72 cores |
| All others | master only | `JULIA_NUM_THREADS` | 1 process, 9 cores (rest idle) |

### `run_training.sh` — GPU training

| Parameter | Location | Description |
|-----------|----------|-------------|
| `--cpus-per-task` | run_training.sh | CPU cores (data loading) |
| `--gres=gpu:a100:1` | run_training.sh | GPU allocation |
| `--mem` | run_training.sh | Total RAM |
| `--time` | run_training.sh | Wall time |
| `JULIA_NUM_THREADS` | run_training.sh | Set to `SLURM_CPUS_PER_TASK` (all cores) |

Training runs single-process — no `addprocs`. All CPUs go to the master for data loading; the GPU handles forward/backward passes.