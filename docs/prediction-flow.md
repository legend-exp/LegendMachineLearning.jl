# Prediction Flow

## Overview

```
jlnormml/{train, val, test}                jlnorm/{all, physics}
  │                                            │
  │  load model from JLD2                      │  load model from JLD2
  │  assemble features                         │  assemble features
  │  predict_with_rules                        │  predict_with_rules
  │  compute_pred_4x4                          │  compute_pred_4x4
  │                                            │
  ▼                                            ▼
jlpredml/{train, val, test}                jlpred/{all, physics}
                                               │
                                               │  (physics only)
                                               │  apply binary veto threshold
                                               │  rewrite with veto_ml column
                                               ▼
                                          jlpred/physics (with veto_ml)
```

## Input

**Model:** `generated/model/{group}/{arch}/{latest}.jld2`

**Data:**
- `jlnormml/{train,val,test}` → jlpredml
- `jlnorm/{all,physics,...}` → jlpred

**Config:** `metadata/prediction/mlgroup001.yaml`
- `threshold_method`: `ft_matched_4x4_survival` or `k40_matched_4x4_survival`
- `hard_classification.pe_hard_veto`: 15.0

## Processing

### Prediction Rules (`predict_with_rules`)

```
if event_sum_pe == 0        → pred_ml = 0.0  (no light at all)
if event_sum_pe ≥ 15 (hard) → pred_ml = 1.0  (definite veto)
else                        → pred_ml = σ(model_output)  (NN inference)
```

### Classical 4×4 Veto (`compute_pred_4x4`)

```
pred_4x4 = (event_sum_pe ≥ 4 OR event_multiplicity ≥ 4) ? 1 : 0
```

Uses the **unscaled** `event_sum_pe` and `event_multiplicity` passed through from extraction.

### Survival CDF (from jlpred/all)

**Only computed when `data.has_label == true`** (i.e., the "all" dataset which has labels).

```
mask0 = label .< 0.5   (FT events, label=0)
mask1 = label .>= 0.5  (sub500keV events, label=1)

sf_4x4_label0 = count(pred_4x4[mask0] == 0) / count(mask0)   ← FT survival of 4×4
```

**FT-matched threshold:** Find ML threshold `t` where `survival_FT(t) == sf_4x4_label0`.

**⚠️ Important:** The "all" dataset in jlnorm comes from **jlbal** (not jlbalml).
jlbal has **ALL events** (no PE filter), so zero-PE events are included.
These zero-PE events have `pred_ml = 0.0` and `pred_4x4 = 0` (both survive).
This means the 4×4 survival fraction includes zero-PE events in both numerator and denominator.

### K40 Threshold (from physics data)

Scan ML thresholds to match K40 survival fraction of 4×4 veto.
Only computed on unlabeled physics data (`!data.has_label`).

### Binary Veto (physics only)

```
active_threshold = threshold_method == "k40" ? threshold_k40 : threshold_ft
veto_ml = pred_ml ≥ active_threshold ? 1 : 0
```

Physics files are **rewritten** with the `veto_ml` column added.

## Output

**Files:** `generated/tier/jlpred(ml)/{group}/l200-{group}-{split}-tier_jlpred(ml).lh5`

| LH5 Key | Columns |
|---------|---------|
| `:<tier>` | `sipm_pe_scaled` (VoV), `sipm_cos_prox` (VoV), `ged_angle_norm`, `ged_z_center_norm`, `event_sum_pe`, `event_multiplicity`, **`pred_ml`** (Float32), **`pred_4x4`** (Int8) |
| | Optional: `label` (Int8, only if labeled), `ged_energy_keV`, `sipm_pe_scaled_prompt` (VoV), `event_sum_pe_prompt`, `event_multiplicity_prompt`, `sipm_pe_scaled_delayed` (VoV), `event_sum_pe_delayed`, `event_multiplicity_delayed` |
| | Physics only: **`veto_ml`** (Int8) |
| `sipm_detector_ids` | Vector{UInt32} |

**Column renaming** (normalization → prediction):

| jlnorm column | jlpred column |
|--------------|---------------|
| `sipm_pe_sums_scaled` | `sipm_pe_scaled` |
| `cos_proximity` | `sipm_cos_prox` |
| `scaled_angle` | `ged_angle_norm` |
| `scaled_z_center` | `ged_z_center_norm` |
| `sipm_pe_sums_prompt_scaled` | `sipm_pe_scaled_prompt` |
| `sipm_pe_sums_delayed_scaled` | `sipm_pe_scaled_delayed` |

**Datasets produced:**

| Tier | Datasets | Labeled | Has veto_ml |
|------|----------|---------|-------------|
| jlpredml | `train`, `val`, `test` | yes | no |
| jlpred | `all` | yes | no |
| jlpred | `physics` | no | **yes** |

## Plots & Reports

- `{suffix}_survival_cdf.png` — from jlpred/all (labeled)
- `{suffix}_prediction_histogram.png` — from jlpred/all (labeled)
- `{suffix}_{ds}_energy_spectrum.png` — physics only
- `{suffix}_{ds}_k40_k42_spectrum.png` — physics only
- `{suffix}.md` — prediction report with thresholds
