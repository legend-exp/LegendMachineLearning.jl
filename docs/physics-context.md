# Physics Context: LAr Scintillation Veto for LEGEND-200

## 1. Liquid Argon as Active Veto

LEGEND-200 operates bare HPGe detectors immersed in liquid argon (LAr). The LAr
serves as both coolant and active veto medium: energy depositions from background
particles produce scintillation light in the VUV range (128 nm), which is
wavelength-shifted and detected by an array of silicon photomultipliers (SiPMs)
surrounding the detector strings.

The fundamental veto principle: if an event deposits energy in an HPGe detector
*and* produces coincident scintillation light in the LAr, the event is likely
background (not 0νββ). Events with no LAr signal are more likely signal-like.

## 2. LAr Scintillation Physics

### Excimer Formation and Decay

When ionizing radiation passes through liquid argon, it produces excited argon
dimers (excimers, Ar₂*) in two electronic states:

- **Singlet state** (¹Σ): fast decay, τ_s ≈ 6 ns
- **Triplet state** (³Σ): slow decay, τ_t ≈ 1.6 μs

Both states emit 128 nm VUV photons. The total scintillation yield is
~40,000 photons/MeV, but light collection efficiency in LEGEND-200 is O(1%)
due to geometry, wavelength shifting losses, and SiPM quantum efficiency.

### Prompt/Delayed Ratio (P/D)

The ratio of prompt (singlet) to delayed (triplet) light depends on the
ionization density of the particle, which determines the singlet-to-triplet
population ratio:

| Particle | Typical P/D ratio | dE/dx |
|----------|-------------------|-------|
| Electrons/γ | ~0.3 (more triplet) | Low |
| α particles | ~1.2–3.0 (more singlet) | Very high |
| Nuclear recoils | ~0.7–1.0 | High |
| Muons | ~0.3 (MIP-like) | Low |

This P/D ratio is a powerful discriminant between particle types and is one
reason why the prompt and delayed time windows carry different physical
information. However, the exact boundary between "prompt" and "delayed" is not
sharp — it is a continuous exponential transition. Current analysis uses
configurable time windows (prompt: [−2, +2] μs, delayed: [+2, +6] μs), but
these are operational choices, not fundamental physics boundaries.

### Light Propagation

Scintillation light propagates through the LAr volume with:
- Rayleigh scattering length ~55 cm at 128 nm (longer after shifting)
- Absorption by impurities (O₂, N₂, H₂O) — typically >2 m absorption length
- Reflection/absorption at detector surfaces

Key consequence: **SiPMs near the energy deposition see more light**. The
angular proximity between SiPM and source location (relative to the HPGe
detector string) is a strong predictor of light yield.

## 3. True vs. Random Coincidences

### True Coincidence (Signal for Veto)

A particle deposits energy both in the HPGe (triggering the event) and in the
surrounding LAr (producing scintillation light). Characteristics:

- **Temporal structure**: Prompt peak at t₀ (HPGe trigger time) followed by
  exponential delayed tail with τ ≈ 1.6 μs
- **Spatial correlation**: SiPMs geometrically close to the HPGe see
  significantly more light than distant ones
- **P/D ratio**: Depends on particle type (see table above)
- **Multiplicity**: Multiple SiPMs fire, concentrated near the HPGe

Examples: Ar-39 β-decay near an HPGe (most common background below 500 keV),
external γ-rays Compton-scattering in both LAr and HPGe, α-decays on surfaces.

### Random Coincidence (No Veto)

SiPM triggers that happen to fall within the analysis time window around an HPGe
event purely by chance, without any physical connection. Characteristics:

- **Temporally flat**: Uniform distribution across the time window, no structure
  around t₀
- **Spatially uncorrelated**: Triggers distributed uniformly across all SiPMs,
  no preference for SiPMs near the HPGe
- **No meaningful P/D**: Because there is no single energy deposition, prompt
  and delayed windows sample from the same flat background
- **Low multiplicity**: Typically individual SiPM dark counts or distant
  scintillation events

Sources: SiPM dark counts (~100 Hz/mm² at LAr temperature), far-away
radioactive decays, electronic noise, afterpulsing.

## 4. Training Datasets and Label Noise

### Forced Trigger (FT) — Background Proxy

Events triggered externally at a fixed time (48 μs), regardless of HPGe
activity. Since there is no HPGe energy deposition:

- **~100% random coincidences** — any SiPM activity is purely accidental
- Clean negative label (label = 0, "no veto needed")
- No assigned HPGe detector → one is randomly sampled for geometry features

### Sub-500 keV — Signal Proxy (Noisy!)

Physics events with HPGe energy < 500 keV, dominated by Ar-39 β-decays:

- **Mix of true AND random coincidences** — NOT 100% true coincidences
- Ar-39 β-particles have very short range in LAr (~mm), so the scintillation
  is localized near the HPGe that triggered
- However, many low-energy events have very little LAr light (below SiPM
  threshold), making them indistinguishable from random coincidences
- Used as positive label (label = 1, "veto signal") despite containing
  significant label noise

### Implications for Model Training

The label noise in the sub-500 keV dataset is a fundamental challenge:

1. A naive supervised model learns "sub-500 keV events look like X" rather than
   "true LAr coincidences look like X"
2. The model may learn to classify random coincidences that happen to be in the
   sub-500 keV set as "signal"
3. This motivates:
   - Noise-robust training techniques (label smoothing, mixup)
   - Anomaly detection approaches (train only on FT, flag deviations)
   - Physics-informed architectures that encode what true coincidences
     *should* look like

### Physics Data (Inference Only)

Full physics events (25–5500 keV), not used for training. The trained model is
applied to these events to produce veto decisions. This is the ultimate test:
the model must correctly identify LAr coincidences in events it has never seen
during training.

## 5. Detector Geometry

### HPGe Detectors

- ~60 enriched germanium detectors arranged in strings
- Each string hangs vertically in the LAr volume
- Detectors identified by name (e.g., V00050A) and numeric ID (UInt32)
- Key geometric features per detector:
  - `angle_deg`: Azimuthal angle in degrees (position around the array)
  - `z_center`: Vertical position (mm)
  - `string_id`: Which string the detector is on
  - `position_in_string`: Vertical position within its string

### SiPM Array

- 55 SiPMs in the current ML group (mlgroup001, period p16)
- Arranged at two heights (top/bottom) on inner barrel (IB) and outer barrel (OB)
- 4 groups: IB_top, IB_bottom, OB_top, OB_bottom
- Each SiPM identified by name (e.g., S007) and numeric ID (UInt32)
- Key geometric features:
  - `angle_deg`: Azimuthal angle
  - `z_mm`: Vertical position

### Relative Geometry (SiPM ↔ HPGe Pairs)

Pre-computed for every (SiPM, HPGe) pair:
- `cos_proximity`: Cosine of the 3D angle between SiPM and HPGe positions,
  viewed from the array center. Range [-1, 1]. A key predictor: SiPMs with
  high cos_proximity to the triggering HPGe are expected to see more light.
- `scaled_delta_z`: Normalized vertical distance |z_SiPM − z_HPGe|.
  Range [0, 1].

### Azimuthal Symmetry

The LEGEND-200 geometry has approximate azimuthal symmetry — the physics does
not depend on absolute angle, only on relative angles between detectors. This
is already encoded by using `cos_proximity` rather than absolute positions.
Models should respect this symmetry.

## 6. Available Data Per Event

### Summed PE Windows (Current Model Input)

Per SiPM (55 values per event), summed over all triggers:
- `sipm_pe_sums`: Full window [−2, +6] μs
- `sipm_pe_sums_prompt`: Prompt window [−2, +2] μs
- `sipm_pe_sums_delayed`: Delayed window [+2, +6] μs

Also event-level aggregates (sum/multiplicity over all SiPMs) for each window.

### Raw Trigger Sequences (Stored but Unused by Current Model)

Per event, variable-length list of individual SiPM triggers within ±10 μs
around the HPGe t₀ reference time:
- `trigger_det_ids`: Which SiPM fired (UInt32 ID)
- `trigger_times_us`: Time relative to t₀ (μs), range [-10, +10]
- `trigger_pe_vals`: Calibrated PE value of the trigger (≥ 0.6 PE threshold)

Each trigger represents a single photon detection (or dark count) by a single
SiPM at a specific time. Typical events have 0–100+ triggers. The ±10 μs
window is wider than the summed PE windows to capture the full temporal context.

Trigger-level data preserves:
- **Individual PE values** (lost in summing)
- **Precise timing** (lost in window integration)
- **Which SiPM fired when** (lost in per-SiPM sums)
- **Temporal clustering** (distinguishable from uniform background)

This information is the basis for more advanced model architectures.

### HPGe Information

Per event (scalar):
- `ged_energy_keV`: Energy deposited in HPGe
- `ged_t0_us`: HPGe trigger time (reference for all SiPM timing)
- `assigned_ged_detector`: Which HPGe triggered (determines geometry lookup)
