# Balancing Flow

## Overview

```
jlext/{sub500keV, forcedtrigger, physics}
  │
  │  read :wpe + :triggers + sipm_detector_ids
  │
  ├──► jlbalml (training)
  │      PE filter: 0.5 < event_sum_pe < 1e10
  │      Balance: downsample/upsample FT to match sub500keV count
  │      FT gets sampled HPGe detector IDs
  │
  ├──► jlbal (inference)
  │      ALL events (no PE filter, no balancing)
  │      FT gets sampled HPGe detector IDs
  │
  └──► jlbal/physics (pass-through)
         ALL events, real detector IDs
```

## Input

**Source:** `jlext/{sub500keV, forcedtrigger}` — reads `:wpe` group + `:triggers` + `sipm_detector_ids`

**Config:** `metadata/balancing/mlgroup001.yaml`
- `ml_valid_pe_range: [0.5, 1e10]`
- `pass_through: [physics]`

## Processing

### jlbalml (Training)

1. **PE filter:** Keep only events where `0.5 < event_sum_pe < 1e10`
   - Removes zero-PE events and very low PE events
   - sub500keV: `n_sub_ml` events pass
   - FT: `n_ft_ml` events pass

2. **Balance:** Target count = `n_sub_ml`
   - If `n_ft_ml ≥ n_target`: random downsample FT
   - If `n_ft_ml < n_target`: upsample FT with duplicates

3. **Detector ID assignment:**
   - sub500keV: keeps real `ged_detector_id`
   - FT: `sample_detector_ids()` — random from sub500keV distribution (prevents model from learning "no detector = FT")

### jlbal (Inference)

1. **No PE filter** — all events included (including zero-PE)
2. **No balancing** — original counts preserved
3. **Detector ID assignment:** same as jlbalml (FT gets sampled IDs)

### Physics (Pass-through)

- Written to jlbal only, with real `ged_detector_id`

## Output

**Files:** `generated/tier/jlbal(ml)/{group}/l200-{group}-{dataset}-tier_jlbal(ml).lh5`

| LH5 Key | Columns |
|---------|---------|
| `:<tier>` | `sipm_pe_sums` (VoV), `sipm_pe_sums_prompt` (VoV), `sipm_pe_sums_delayed` (VoV), `event_sum_pe`, `event_multiplicity`, `event_sum_pe_prompt`, `event_multiplicity_prompt`, `event_sum_pe_delayed`, `event_multiplicity_delayed`, **`assigned_ged_detector`**, `ged_energy_keV`, `ged_t0_us`, `trigger_det_ids` (VoV), `trigger_times_us` (VoV), `trigger_pe_vals` (VoV) |
| `sipm_detector_ids` | sorted Vector{UInt32} |

**Datasets produced:**

| Tier | Datasets | Events |
|------|----------|--------|
| jlbalml | `sub500keV`, `forcedtrigger` | PE-filtered, balanced |
| jlbal | `sub500keV`, `forcedtrigger`, `physics` | All events |
