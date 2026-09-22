# Normalization Flow

## Overview

```
jlbalml/{sub500keV, forcedtrigger}          jlbal/{sub500keV, forcedtrigger}
  │                                            │
  │  scale PE: clip[0,15] → log1p → minmax    │  same scaling (shared params)
  │  augment geometry features                 │  augment geometry features  
  │  scale triggers (PE + time)                │  scale triggers
  │  add label: sub500keV=1, FT=0             │  add label: sub500keV=1, FT=0
  │  combine + shuffle                         │  combine + shuffle
  │  split 70/20/10                            │
  ▼                                            ▼
jlnormml/{train, val, test}                 jlnorm/all
                                               │
                                          jlbal/physics
                                               │  scale independently
                                               │  no label
                                               ▼
                                          jlnorm/physics
```

## Input

**Source:** `jlbal` and `jlbalml` — reads `:<tier>` table + `sipm_detector_ids`

**Config:** `metadata/normalization/mlgroup001.yaml`

**Geometry files:**
- `generated/geometry/relative/{group}_relative_geometry.yaml`
- `generated/geometry/HPGe/{group}_hpge_geometry.yaml`
- `generated/geometry/SiPM/{group}_sipm_geometry.yaml`

## Processing

### PE Scaling Pipeline (global across both datasets per tier)

```
raw PE (Float64) → clip [0, 15] → log1p → minmax [0, 1]
```

MinMax params computed globally over sub500keV + forcedtrigger combined.

Applied to: `sipm_pe_sums`, `sipm_pe_sums_prompt`, `sipm_pe_sums_delayed`

### Geometry Augmentation

Per event, lookup `assigned_ged_detector` → geometry YAML:

| Feature | Type | Source |
|---------|------|--------|
| `cos_proximity` | VoV (per SiPM) | relative geometry: cos of angular distance HPGe↔SiPM |
| `scaled_delta_z` | VoV (per SiPM) | relative geometry: normalized z-distance HPGe↔SiPM |
| `scaled_angle` | scalar | HPGe geometry: angle / 360 |
| `scaled_z_center` | scalar | HPGe geometry: normalized z position |

### Trigger Scaling

| Feature | Transform |
|---------|-----------|
| `trig_pe_scaled` | same clip→log1p→minmax as PE columns (global params) |
| `trig_time_scaled` | linear → [-1, 1] over [-10 µs, +10 µs] |
| `trig_det_ids` | pass-through (VoV{UInt32}) |
| `trig_count` | count of triggers per event (Int32) |

### Labeling + Combine + Shuffle

1. sub500keV → `label = 1` (signal)
2. forcedtrigger → `label = 0` (background)
3. Concatenate all columns
4. Random shuffle (same permutation for all columns)

### Split (jlnormml only)

| Split | Fraction |
|-------|----------|
| train | 70% |
| test | 20% |
| val | 10% |

### Physics (pass-through → jlnorm only)

- Scaled independently (same pipeline, fresh global params from that dataset alone)
- **No label column**
- Not combined with training data

## Output

**Files:** `generated/tier/jlnorm(ml)/{group}/l200-{group}-{split}-tier_jlnorm(ml).lh5`

| LH5 Key | Columns |
|---------|---------|
| `:<tier>` | **Scaled PE (VoV{Float32}):** `sipm_pe_sums_scaled`, `sipm_pe_sums_prompt_scaled`, `sipm_pe_sums_delayed_scaled` |
| | **Geometry (VoV{Float32}):** `cos_proximity`, `scaled_delta_z` |
| | **Geometry (Float32):** `scaled_angle`, `scaled_z_center` |
| | **Triggers:** `trig_pe_scaled` (VoV), `trig_time_scaled` (VoV), `trig_det_ids` (VoV), `trig_count` (Int32) |
| | **Pass-through:** `assigned_ged_detector` (UInt32), `event_sum_pe` (Float32), `event_multiplicity` (Int32), `event_sum_pe_prompt`, `event_multiplicity_prompt`, `event_sum_pe_delayed`, `event_multiplicity_delayed`, `ged_energy_keV` |
| | **Label:** `label` (Int8) — **not present for physics** |
| `sipm_detector_ids` | Vector{UInt32} |

**Key point:** `event_sum_pe` and `event_multiplicity` are **unscaled pass-through values** from jlbal — used later by prediction for the 4×4 classifier.

**Datasets produced:**

| Tier | Datasets | Labeled |
|------|----------|---------|
| jlnormml | `train`, `val`, `test` | yes (sub500keV=1, FT=0) |
| jlnorm | `all` | yes (combined sub500keV + FT) |
| jlnorm | `physics` | **no** |
