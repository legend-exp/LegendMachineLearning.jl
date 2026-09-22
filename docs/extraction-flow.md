# Extraction Flow

## Overview

```
jlevt (LEGEND raw, per FileKey)
  │
  │  filter events (per dataset config)
  │  compute windowed PE sums (3 windows)
  │  collect raw triggers in ±10 µs window
  │  remove NaN/Inf events
  │
  ▼
jlext/{sub500keV, forcedtrigger, physics}
```

## Input

**Source:** `jlevt` tier (`:phy` category), read via `read_ldata(l200, :jlevt, filekeys)`

**Config:** `metadata/extraction/mlgroup001.yaml`

| Dataset | Event Filter | Reference |
|---------|-------------|-----------|
| `sub500keV` | valid physics, E < 500 keV, mult == 1 | `ged_t0` |
| `forcedtrigger` | forced trigger only | fixed center 48 µs |
| `physics` | valid physics, 25 keV ≤ E < 5500 keV, mult == 1 | `ged_t0` |

## Processing

### Windowed PE Sums (`_windowed_pe`)

Three time windows relative to t₀ (or fixed center for FT):

| Window | Range |
|--------|-------|
| **full** | [t₀ − 2 µs, t₀ + 6 µs] |
| **prompt** | [t₀ − 2 µs, t₀ + 2 µs] |
| **delayed** | [t₀ + 2 µs, t₀ + 6 µs] |

**Per trigger:** Skip if DC (`trig_dc=true`) → skip if `PE < 0.6` → skip if outside window → add PE to SiPM sum.

Sub-threshold triggers (< 0.6 PE) are **completely ignored** — not zeroed, not stored anywhere.

**Per SiPM:** `pe_sums[d]` = sum of all passing triggers on that SiPM in the window.

**Per event:**
- `event_sum_pe` = sum over all SiPMs
- `event_multiplicity` = count of SiPMs with PE ≥ 0.6

### Raw Triggers (unique window)

Window: [t₀ − 10 µs, t₀ + 10 µs]. Same PE threshold (≥ 0.6) and DC filter.
Stored as three VoV columns: detector ID, time relative to t₀ (µs), PE value.

## Output

**Files:** `generated/tier/jlext/{group}/l200-{group}-{dataset}-tier_jlext.lh5`

| LH5 Key | Columns |
|---------|---------|
| `:jlext` | Original jlevt table columns (spms.detector, spms.trig_max_trap_cal, spms.trig_pos_trap, spms.trig_max_trap_is_dc, geds.*) |
| `:wpe` | `sipm_pe_sums` (VoV), `sipm_pe_sums_prompt` (VoV), `sipm_pe_sums_delayed` (VoV), `event_sum_pe`, `event_multiplicity`, `event_sum_pe_prompt`, `event_multiplicity_prompt`, `event_sum_pe_delayed`, `event_multiplicity_delayed`, `ged_detector_id`, `ged_energy_keV`, `ged_t0_us` |
| `:triggers` | `det_id` (VoV), `time_rel_us` (VoV), `pe` (VoV) |
| `sipm_detector_ids` | sorted Vector{UInt32}, length = n_sipms |
| `:metadata` | extraction_name, group_name, filter, generated_at, event counts |

**Datasets produced:** `sub500keV`, `forcedtrigger`, `physics`
