# Training Strategies for LAr Veto Classification

## Table of Contents

1. [The Label Noise Problem](#1-the-label-noise-problem)
2. [Supervised Learning with Noise Robustness](#2-supervised-learning-with-noise-robustness)
3. [Anomaly Detection / Semi-Supervised](#3-anomaly-detection--semi-supervised)
4. [Contrastive Learning](#4-contrastive-learning)
5. [Hybrid and Ensemble Approaches](#5-hybrid-and-ensemble-approaches)
6. [Evaluation and Comparison](#6-evaluation-and-comparison)
7. [Recommendations](#7-recommendations)

---

## 1. The Label Noise Problem

### Dataset Composition

| Dataset | Label | True Coincidences | Random Coincidences |
|---------|-------|:-----------------:|:-------------------:|
| **ForcedTrigger** | 0 (no veto) | ~0% | ~100% |
| **Sub-500 keV** | 1 (veto) | Significant fraction | Also significant |

The sub-500 keV dataset is dominated by Ar-39 β-decays, which produce true LAr
scintillation near the HPGe. However:

- **Many events have very little LAr light** — Ar-39 β-particles have short
  range (~mm in LAr), and low-energy decays produce few photons
- **Low-light events are indistinguishable from random coincidences** at the
  SiPM level
- **The sub-500 keV label is a proxy**: "this event has low HPGe energy, so it
  is *likely* Ar-39 background" — not "this event definitely has a true LAr
  coincidence"

### Consequences for Supervised Training

A naive binary cross-entropy loss treats all labels as ground truth:

```
L = -[y·log(p) + (1-y)·log(1-p)]
```

When `y=1` for a sub-500 keV event that is actually a random coincidence,
the model is pushed to classify random coincidences as "veto." This:

1. **Reduces veto efficiency** — the model learns noise patterns
2. **Increases false positive rate on physics data**
3. **Limits the performance ceiling** regardless of model complexity

---

## 2. Supervised Learning with Noise Robustness

### 2.1 Label Smoothing

Replace hard labels with soft targets:

```
y_smooth = y · (1 - α) + (1 - y) · β

Example: α = 0.2, β = 0.05
  Sub-500 keV: y = 1 → 0.8
  ForcedTrigger: y = 0 → 0.05
```

**Rationale**: Acknowledges that sub-500 keV events are not 100% true
coincidences. The smoothing parameter α encodes our uncertainty about the
label quality.

**Asymmetric smoothing** is preferred: ForcedTrigger labels are very clean
(β ≈ 0.01–0.05), but sub-500 keV labels have significant noise (α ≈ 0.1–0.3).

Implementation:
```julia
function smoothed_bce(logits, labels; α=0.2, β=0.05)
    soft_labels = labels .* (1 - α) .+ (1 .- labels) .* β
    return Flux.logitbinarycrossentropy(logits, soft_labels)
end
```

### 2.2 Mixup Augmentation

Interpolate between pairs of training examples:

```
x̃ = λ · x_i + (1-λ) · x_j
ỹ = λ · y_i + (1-λ) · y_j
where λ ~ Beta(α_mix, α_mix), typically α_mix ∈ [0.1, 0.4]
```

**For trigger-level data**: Mixup is non-trivial because inputs are
variable-length sets. Options:

1. **Feature-level mixup**: After the encoder (at the fixed-size embedding
   level), interpolate embeddings before the tail MLP
2. **Manifold mixup**: Interpolate at a random hidden layer
3. **CutMix for sets**: Randomly replace a fraction of triggers from one event
   with triggers from another event, mix labels proportionally

Option 1 is simplest and most practical.

### 2.3 Confidence Learning (Cleanlab)

Use a trained model to *identify mislabeled examples*, then retrain on the
cleaned dataset:

1. Train initial model with standard cross-entropy
2. Compute per-sample predicted probabilities via cross-validation
3. Identify "confident errors": samples where the model's prediction
   consistently disagrees with the label
4. Remove or re-label these samples
5. Retrain on cleaned dataset

For the LAr veto problem:
- Sub-500 keV events predicted as 0 with high confidence → likely
  *random coincidences mislabeled as signal*
- These are exactly the events degrading supervised training

Implementation: Use the `confident_joint` approach from Northcutt et al. (2021).

### 2.4 Focal Loss

Down-weight easy examples, focus on hard ones:

```
L_focal = -α_t · (1 - p_t)^γ · log(p_t)
where p_t = p if y=1, else 1-p
```

With γ=2, correctly-classified examples contribute little to the loss.
This helps with label noise because noisy examples are typically "hard"
(model is uncertain), but: it can also amplify noise by focusing on
genuinely mislabeled examples.

**Use with caution** — combine with label smoothing to mitigate.

### 2.5 Symmetric Cross-Entropy

Combine forward and reverse cross-entropy for noise robustness:

```
L_SCE = α · CE(p, y) + β · CE(y, p)
      = α · [-y·log(p) - (1-y)·log(1-p)]
        + β · [-p·log(y_clip) - (1-p)·log(1-y_clip)]
```

The reverse term penalizes overconfident predictions, providing natural
regularization against noisy labels.

---

## 3. Anomaly Detection / Semi-Supervised

### 3.1 Autoencoder on ForcedTrigger

**Key insight**: ForcedTrigger data is ~100% random coincidences. If we
learn the distribution of random coincidences, anything that deviates is
a potential true coincidence (veto candidate).

#### Variational Autoencoder (VAE)

```
Encoder: trigger features → μ, log(σ²)  (latent dim Z, e.g., 16)
Decoder: z ~ N(μ, σ²) → reconstructed trigger features

Loss = reconstruction_error + KL(q(z|x) || p(z))
     = MSE(x, x̂) + KL divergence
```

**Training**: Only on ForcedTrigger events
**Inference**: Compute reconstruction error for all events
- Low error → event looks like random coincidence → no veto
- High error → event deviates from random → potential true coincidence → veto

#### For Trigger-Level Data

The VAE encoder could be a Set Transformer or DeepSets architecture:

```
Triggers → Set Encoder → μ, σ (latent, Z-dim)
z ~ N(μ, σ) → Set Decoder → reconstructed trigger features

Set Decoder: z → repeat for each SiPM → predicted trigger statistics
```

The reconstruction target could be:
- Per-SiPM PE sums (simpler, matches current features)
- Per-SiPM trigger count and mean PE (intermediate)
- Full trigger set (hardest, requires generating variable-length output)

Practical recommendation: Use per-SiPM aggregate reconstruction (55D output)
with a trigger-level encoder (to leverage fine-grained input information).

### 3.2 One-Class Classification

Train a model to enclose ForcedTrigger data in a compact hypersphere in
embedding space (Deep SVDD / OC-SVM on learned features):

```
Encoder: event → embedding ∈ ℝ^D
Loss = mean(||f(x) - c||²)     where c is the centroid
```

Events mapped far from c are anomalies (potential true coincidences).

### 3.3 Normalizing Flow

Learn an invertible mapping from data space to a simple base distribution
(e.g., Gaussian). The log-likelihood under the learned distribution directly
measures how "typical" an event is.

```
p(x) = p_base(f(x)) · |det(∂f/∂x)|
```

Low p(x) → anomalous event → potential veto candidate.

Advantage over VAE: Exact likelihood computation (no approximation).
Disadvantage: Architecture constraints (must be invertible), harder to implement.

### 3.4 Advantages of Anomaly Detection

1. **No label noise problem** — trained only on (clean) ForcedTrigger
2. **Learns what random coincidences look like**, which is well-defined
3. **Complementary to supervised** — errors are likely uncorrelated
4. **Physically interpretable**: reconstruction error = "how much does this
   event deviate from random coincidence expectations?"

### 3.5 Disadvantages

1. **Doesn't learn what true coincidences look like** — only what they
   are NOT (not random)
2. **Any deviation from random triggers anomaly** — including detector
   glitches, LAr purity variations, etc.
3. **Threshold selection**: Need to choose anomaly score cutoff for veto,
   which requires some labeled data anyway
4. **May miss subtle true coincidences** that look almost random

---

## 4. Contrastive Learning

### 4.1 Supervised Contrastive Learning

Learn an embedding where true and random coincidences are separated:

```
Anchor: event x
Positive: event from same class as x
Negative: event from different class

Loss = -log[exp(sim(z, z+)/τ) / Σ_k exp(sim(z, z_k)/τ)]
where sim(a,b) = a·b / (||a||·||b||), τ is temperature
```

Despite using labels, contrastive learning is more robust to label noise
than cross-entropy because:
- It learns relative similarities, not absolute predictions
- Noisy positives that look like negatives naturally cluster with negatives
- No sharp decision boundary → graceful degradation with noise

### 4.2 Self-Supervised Pretraining

Learn event representations without any labels, then fine-tune:

**Augmentations for trigger data** (for SimCLR-style pretraining):
- Time jitter: Add small Gaussian noise to trigger times
- PE jitter: Scale trigger PE values by random factor ~N(1, 0.1)
- Trigger dropout: Randomly remove a fraction of triggers
- SiPM masking: Zero out triggers from random SiPMs

```
View 1: augment(event) → encoder → z₁
View 2: augment(event) → encoder → z₂
Contrastive loss: pull z₁, z₂ together, push away from other events
```

After pretraining, the encoder produces representations that capture event
structure. Fine-tune with a small labeled set.

### 4.3 Advantages

- More robust to label noise than standard supervised
- Can leverage unlabeled physics data for pretraining
- Learned embeddings are useful for visualization and analysis
- Can combine with downstream supervised or anomaly detection

### 4.4 Disadvantages

- More complex training pipeline
- Requires careful augmentation design (must be physically meaningful)
- Contrastive learning needs large batch sizes for good negatives
- Two-stage training (pretrain + fine-tune) is slower

---

## 5. Hybrid and Ensemble Approaches

### 5.1 Score Fusion

Train both a supervised classifier and an anomaly detector, combine scores:

```
p_supervised = Classifier(event)       ∈ [0, 1]
a_anomaly    = AnomalyDetector(event)  ∈ ℝ (higher = more anomalous)

p_veto = f(p_supervised, a_anomaly)
```

Combination strategies:
- **Weighted average**: `p = α·p_supervised + (1-α)·normalize(a_anomaly)`
- **Learned combination**: Train a small MLP on `[p_supervised, a_anomaly]`
  using a small, manually-verified validation set
- **Threshold intersection**: Veto if `p_supervised > τ₁ AND a_anomaly > τ₂`

### 5.2 Teacher-Student with Anomaly Scores

1. Train anomaly detector on ForcedTrigger
2. Score all sub-500 keV events
3. Use anomaly score as soft label: `y_soft = normalize(a_anomaly)`
4. Train supervised model with soft labels instead of hard 0/1

This effectively uses the anomaly detector to clean the sub-500 keV labels.

### 5.3 Multi-Task Learning

Train a single network with multiple outputs:

```
Event → Encoder → Shared representation
                ├→ Classification head → veto probability
                ├→ Reconstruction head → reconstruct per-SiPM PE sums
                └→ P/D prediction head → predict prompt/delayed ratio

Total loss = α·L_classify + β·L_reconstruct + γ·L_pd_predict
```

The reconstruction task (self-supervised) regularizes the encoder and helps
learn meaningful representations even with noisy classification labels.

---

## 6. Evaluation and Comparison

### Metrics

For the LAr veto problem, standard accuracy is insufficient. Key metrics:

1. **Signal efficiency** (at fixed background acceptance): "What fraction of
   true coincidences does the veto catch, when allowing X% of random events
   to leak through?"

2. **Background rejection** (at fixed signal efficiency): "How much background
   is rejected when requiring Y% of signals to be vetoed?"

3. **ROC AUC** and **PR AUC**: Overall discrimination performance.

4. **Physics validation**: Apply to known calibration sources (Th-228, K-42)
   where the expected LAr signature is well-understood.

### Handling Label Noise in Evaluation

Since test set labels are also noisy, evaluation requires:

1. **Rely on physics knowledge**: Calibration source events have known
   properties (Th-228: strong LAr signal; K-42: moderate)
2. **ForcedTrigger specificity**: If the model classifies >X% of FT events as
   "veto," something is wrong (FT should be ~100% random)
3. **Compare to simple cuts**: Does the ML model outperform the traditional
   energy × multiplicity cut?
4. **Stability across periods**: Does the model transfer across data-taking
   periods with different SiPM configurations?

### Ablation Studies

For each model, evaluate:
1. With/without geometry features → How important is spatial information?
2. With/without temporal features → How important is timing beyond P/D sum?
3. With/without the HPGe embedding → How important is HPGe position?
4. Sub-500 keV vs. higher energy test events → Does performance depend on
   energy (which correlates with LAr light yield)?

---

## 7. Recommendations

### Phase 1: Immediate (Strong Baseline)

**Noise-Robust Supervised + Physics-Prior Features**
- Extend current MLP with richer per-SiPM features (P/D ratio, temporal
  stats, multiplicity)
- Use label smoothing (α=0.2 for sub-500 keV, β=0.05 for FT)
- Fast to implement, provides strong baseline

### Phase 2: Trigger-Level Model

**Hierarchical Set Model + Label Smoothing**
- First model that uses raw trigger data
- Physics-motivated hierarchy (trigger → SiPM → event)
- Compare against Phase 1 to quantify value of trigger-level information

### Phase 3: Anomaly Detection

**VAE on ForcedTrigger**
- Train only on clean (random) data
- Compare anomaly score distribution on sub-500 keV vs. FT
- Use reconstruction error as complementary veto signal

### Phase 4: Ensemble

**Supervised + Anomaly Fusion**
- Combine Phase 2 and Phase 3 scores
- Should be strictly better than either alone
- Final production model for physics analysis

### Phase 5: Advanced (If Needed)

**Set Transformer + Contrastive Pretraining**
- Most flexible model with self-supervised pretraining
- Only pursue if Phase 2–4 leave significant room for improvement
- Requires more compute and careful tuning

### Summary

```
Phase 1: MLP + Physics Features + Label Smoothing     [fast, strong baseline]
Phase 2: Hierarchical Set + Label Smoothing            [trigger-level, physics]
Phase 3: VAE on ForcedTrigger                          [anomaly detection]
Phase 4: Ensemble (Phase 2 + Phase 3)                  [best of both worlds]
Phase 5: Set Transformer + Contrastive (if needed)     [maximum flexibility]
```

---

## References

- Northcutt, C. G. et al. (2021). "Confident Learning: Estimating Uncertainty
  in Dataset Labels." JAIR.
- Zhang, H. et al. (2018). "mixup: Beyond Empirical Risk Minimization." ICLR.
- Kingma, D. P. & Welling, M. (2014). "Auto-Encoding Variational Bayes." ICLR.
- Chen, T. et al. (2020). "A Simple Framework for Contrastive Learning of
  Visual Representations." ICML (SimCLR).
- Khosla, P. et al. (2020). "Supervised Contrastive Learning." NeurIPS.
- Wang, Y. et al. (2019). "Symmetric Cross Entropy for Robust Learning with
  Noisy Labels." ICCV.
- Ruff, L. et al. (2018). "Deep One-Class Classification." ICML (Deep SVDD).
