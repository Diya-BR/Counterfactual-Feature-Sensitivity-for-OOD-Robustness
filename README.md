# Counterfactual Feature Sensitivity for OOD Robustness

**Can we tell, using only in-distribution data, that a model relies on a spurious feature, before that reliance breaks it under distribution shift?**

We propose a **counterfactual Spurious Sensitivity score**: the fraction of a model's predictions that flip when one candidate feature is changed and everything else is held fixed. Scores are computed and **frozen** on training/validation data only. Afterwards we test whether they predict the model's later out-of-distribution (OOD) accuracy drop, and compare against five baselines.

Experiments use **Colored MNIST** (exact counterfactuals, known ground truth) across multiple architectures, seeds, dataset sizes and correlation strengths. Waterbirds is planned as real-data validation.

---

## Research questions

| | Question |
|---|---|
| **RQ1** | Does a pre-shift score identify the spurious feature among decoys (blind detection)? |
| **RQ2** | Does the score predict the size of the later OOD drop, **beyond** what plain label correlation already explains? |
| **RQ3** | Does the result hold on real data (Waterbirds), where counterfactuals are approximate? |

---

## Project status

| Notebook | Content | Status |
|---|---|---|
| `Notebook-0.ipynb` | **Shared core** (Stage A): data, interventions, six scores, freeze protocol, OOD evaluation, analysis, smoke test. Main LR grid. | ✅ Final (design-based registry + MI + permutation importance) |
| `Notebook-1.ipynb` | Logistic regression (Stage B): 40-run grid on the final core, all six scores frozen, smoke test, realism check for all three features | ✅ Pre-shift scores frozen; OOD evaluation deferred to the merge notebook |
| `Notebook-2.ipynb` | Random forest | ⏳ Pending |
| `Notebook-3.ipynb` | **MLP** (Stage B): 40-run grid on the final core, all six scores frozen, smoke test, realism check, blind detection, regularization × augmentation sweep (60 runs, separate freeze), hard blind detection | ✅ Run on final core (core SHA matches `results/lr/core.sha256`) |
| `Notebook-4.ipynb` | **Small CNN** (Stage B): 40-run grid, blind detection, realism check, IG diagnostic | ✅ Run on final core |
| Waterbirds | Frozen ResNet features + linear head; multiple counterfactual methods | ⏳ Pending |

---

## Experimental setup

### Colored MNIST
- **Task:** digit parity (even = 0, odd = 1). Images are 3 × 28 × 28.
- **Spurious feature, `color`:** each digit is drawn in red **or** green. With correlation *c*, colour agrees with the label with probability (1 + *c*)/2.
- **Decoys, `marker_tl` and `marker_br`:** 3 × 3 blue corner patches, independent of the label (correlation 0). A good score should ignore them.
- **Null control:** a do-nothing intervention that must always score exactly 0.

| Environment | Colour–label correlation | Colour matches label |
|---|---|---|
| E1 (train) | 0.50 / 0.70 / 0.85 / 0.95 | 75% – 97.5% |
| E2 | 0.75 | 87.5% |
| E3 | 0.50 | 75% |
| E4 | 0.25 | 62.5% |
| E5 | 0.00 | 50% (colour useless) |
| **REV** | **−0.95** | **2.5% (colour points the wrong way)** |

### Exact counterfactuals
- **Colour:** swap the red and green channels. Same digit, same pixels, same label; only the colour changes.
- **Markers:** toggle the blue patch.

Both are exactly reversible, so counterfactual images are not off-manifold. A **realism check** confirms this: a classifier cannot tell swapped images from natural ones with the same label and colour (AUROC ≈ 0.5), while a deliberately corrupted swap is caught (AUROC ≈ 1.0).

### Grid (per architecture)
4 training correlations × 2 sizes (5,000 / 20,000) × 5 seeds = **40 runs**. Every architecture uses the **same seeds and therefore identical E1 data**, so runs are paired across models.

---

## Scores compared

All scores are computed **before any shift**, from E1 training/validation data only.

| Score | What it measures | Uses the model? |
|---|---|---|
| **Spurious sensitivity** (ours) | Fraction of predictions that flip under the feature's counterfactual | ✅ |
| Label correlation | \|corr(label, feature)\| in training data | ❌ data only |
| Mutual information | MI(label; feature) in training data | ❌ data only |
| Stability | Accuracy where the feature agrees with the label minus where it disagrees | ✅ |
| SHAP (attribute-level) | Shapley value of each feature intervention on P(true class) | ✅ |
| Permutation importance | Accuracy drop when the feature is shuffled across validation examples (grouped, 5 repeats) | ✅ |

**Ground truth:** a feature is spurious if its *designed* training correlation exceeds 0.5 (strictly). So colour at 0.5 is *not* labelled spurious, and colour-randomised runs are not spurious.

---

## Freeze protocol (no peeking)

```
run_grid()  →  freeze_scores()  →  ood_evaluate()  →  analyze()
 train +        merge, SHA-256,      refuses to run       Spearman, partial ρ,
 score on       read-only, refuses   unless the frozen    cluster-bootstrap CIs,
 E1 only        to overwrite         hash still matches   Holm, mixed model,
                                                          controls, blind detection
```

OOD test data cannot be opened until the scores are frozen and the hash verifies. The smoke test checks this mechanically.

**Main target:** `rev_drop` = in-distribution test accuracy − accuracy on the reversed-colour set. Also reported: `mean_drop` (E2–E5) and `worst_drop`.

---

## Results so far

> These are results from Colored MNIST only, with three architectures (LR, CNN, MLP). RF and Waterbirds are pending, so treat everything below as preliminary.

### Controls
In **every** run of all three models, colour sensitivity beats both decoys and the null, and null sensitivity is exactly **0.000** (including all 60 MLP sweep runs).

Realism check (natural vs counterfactual images with the same label and feature value; 0.5 = indistinguishable):

| Check | AUROC |
|---|---|
| Colour swap (Notebook 4, CNN classifier) | 0.50–0.52 at every training correlation |
| Positive control: swap plus a faint artifact | 1.00 |
| Colour / marker_tl / marker_br (Notebook 1) | 0.48–0.53 for all three features at every training correlation |
| Colour / marker_tl / marker_br (Notebook 3, MLP classifier) | 0.50–0.54 for all three features at every training correlation; positive control 1.00 |

### RQ2: does the frozen score predict the reversed-colour drop? (40 runs each)

| Score | LR ρ | LR partial ρ \| label corr | CNN ρ | CNN partial ρ \| label corr | MLP ρ | MLP partial ρ \| label corr |
|---|---|---|---|---|---|---|
| **Sensitivity** | **0.989** | **0.914** | **0.994** | **0.976** | **0.984** | **0.972** |
| SHAP | 0.992 | 0.935 | 0.995 | 0.976 | 0.978 | 0.957 |
| Permutation importance | 0.979 | 0.825 | 0.990 | 0.969 | 0.985 | 0.931 |
| Stability | 0.950 | 0.576 | 0.951 | 0.860 | 0.958 | 0.911 |
| Label correlation | 0.934 | — | 0.880 | — | 0.885 | — |
| Mutual information | 0.934 | −0.083 | 0.879 | −0.084 | 0.885 | −0.023 |

ρ is the Spearman correlation between the frozen score and `rev_drop`; the partial ρ controls for label correlation. Sources: LR from `Notebook-0` (`runs_mi_perm_v2`); CNN from `results/cnn/runs_cnn`; MLP from `results/mlp/runs_mlp`.

- The model-based scores (sensitivity, SHAP, permutation importance) all predict the drop well beyond label correlation.
- Mutual information adds nothing beyond label correlation. That is expected: with two binary variables it is essentially a rescaling of the same quantity.

### Same data, different models (the cleanest RQ2 test)
For each of the 40 (corr, size, seed) settings, LR, CNN and MLP see **identical training data**, so label correlation and MI are identical by construction and cannot tell them apart.

| Pair | Spearman ρ (Δ sensitivity vs Δ `rev_drop`) | Pairs where the higher-sensitivity model dropped more |
|---|---|---|
| CNN − LR | **0.978** (p ≈ 10⁻²⁷) | **40 / 40** |
| MLP − LR | **0.952** (p ≈ 10⁻²¹) | **40 / 40** |
| MLP − CNN | **0.961** (p ≈ 10⁻²²) | **38 / 40** (both misses are near-ties: sensitivity gap ≤ 0.02, drop gap ≤ 0.004) |

| Train corr | n | LR `rev_drop` | CNN `rev_drop` | MLP `rev_drop` |
|---|---|---|---|---|
| 0.50 | 5,000 | 0.121 | 0.027 | 0.058 |
| 0.50 | 20,000 | 0.110 | 0.011 | 0.027 |
| 0.70 | 5,000 | 0.216 | 0.063 | 0.112 |
| 0.70 | 20,000 | 0.206 | 0.031 | 0.058 |
| 0.85 | 5,000 | 0.345 | 0.128 | 0.252 |
| 0.85 | 20,000 | 0.331 | 0.068 | 0.113 |
| 0.95 | 5,000 | 0.576 | **0.805** | 0.635 |
| 0.95 | 20,000 | 0.516 | 0.208 | 0.283 |

The CNN usually relies on colour far less than LR, because it can learn digit shape. The exception is a strong shortcut with little data (0.95, n = 5,000), where it relies on colour *more*. The MLP sits between the two. Sensitivity tracks all of these directions.

### Within a training correlation (MLP)
Holding the training correlation fixed (10 runs per level), does the frozen colour score still rank runs by their later `rev_drop`? Mean Spearman over the four levels:

| Score | Within corr | Within corr **and** size (5 seeds per cell) |
|---|---|---|
| **Sensitivity** | **0.977** | **0.906** |
| SHAP | 0.976 | 0.900 |
| Permutation importance | 0.860 | 0.675 |
| Stability | 0.848 | 0.375 |
| Label correlation | −0.109 | 0.238 |
| Mutual information | −0.115 | 0.238 |

Data-only scores carry no signal once the correlation level is fixed; sensitivity and SHAP still do, even inside a single (corr, size) cell. Source: `results/mlp/within_level_spearman_mlp.csv`.

**Dataset size (MLP).** At the same training correlation, the MLP trained on 20,000 images relies on colour about half as much as the one trained on 5,000 (colour sensitivity at 0.95: **0.68 → 0.30**), and its reversed-colour drop halves with it (0.635 → 0.283). Label correlation is identical across the two sizes. Source: `results/mlp/colour_scores_by_size_mlp.csv`.

### Interventions on training (MLP sweep: 3 L2 levels × {none, colour randomisation}, corr 0.7 and 0.95, n = 5,000, 5 seeds)
- **Colour randomisation** (swap colour on half the training images) collapses colour sensitivity (0.95: **0.68 → 0.04**; 0.70: 0.13 → 0.03), and the reversed-colour drop disappears with it (0.95: **0.64 → ≈ 0**). This is the direct check that the score measures reliance: remove the reliance and both the score and the failure go away.
- **L2 regularization** at the core's levels (`alpha` 1e-5 to 1e-2) barely changes either (0.95: sensitivity 0.61–0.68).
- Pooled over the 60 sweep runs, permutation importance gives ρ = 0.98 (partial 0.91) and sensitivity ρ = 0.86 (partial 0.53). The partial ρ is lower here because colour randomisation also changes the training-data label correlation, so label correlation already explains part of the sweep. Source: `results/mlp/sweep_summary_mlp.csv`, `results/mlp/runs_mlp_sweep/`.

### RQ1: blind detection
Almost every score achieves top-1 = 1.0 and AUROC = 1.0, on the grid runs and on the 10 generated tasks where the shortcut is sometimes a marker. The exceptions are stability for the CNN (top-1 0.97 and AUROC 0.99 on the grid, AUROC 0.985 on the 10 tasks) and sensitivity for the MLP (AUROC 0.998 on the 10 tasks). This version is **too easy to separate methods**.

**Hard blind detection (MLP, 30 tasks, `results/mlp/hard_blind_detection_mlp.csv`).** Two shortcuts with *close* random strengths (stronger 0.75–0.90, the other up to 0.10 below) plus one decoy. Ground truth is not the designed strength but **actual damage**: after scoring, each attribute is reversed on test digits one at a time, and the most harmful one is the target. (The designed-stronger shortcut was the most harmful in 28 of 30 tasks.)

| Score | Top-1 (most harmful) | Strong vs weak order | AUROC shortcut vs decoy |
|---|---|---|---|
| Label correlation | 1.000 | 1.000 | 1.0 |
| Mutual information | 1.000 | 1.000 | 1.0 |
| **Sensitivity** | **0.967** | **0.967** | **1.0** |
| SHAP | 0.933 | 0.933 | 1.0 |
| Permutation importance | 0.933 | 0.933 | 1.0 |
| Stability | 0.800 | 0.800 | 1.0 |

Data-only scores win this test, which is the tautology problem in its clearest form: on synthetic data, the attribute that does the most damage is almost always the one most correlated in the sample, so ranking by observed correlation is nearly an oracle. Among model-based scores, sensitivity is best. Detection is not where model-based scores add value; predicting *how much* a given model will fail (RQ2, paired and within-level results) is.

---

## Limitations and open issues

- **Our score is not uniquely best.** SHAP and permutation importance perform about as well. The supported claim is that *model-based* reliance scores predict OOD failure where data-only statistics cannot.
- **Correlation level drives much of the pooled ρ.** Within-stratum and paired analyses are the stronger evidence.
- **Blind detection saturates** (all methods perfect on the 10-task version). The hard version (MLP) separates methods, but data-only scores still win it on synthetic data (tautology).
- **Mixed-effects model** has few groups and boundary warnings. When pooling architectures, use model as a fixed effect plus the cluster bootstrap.
- **Synthetic data:** exact counterfactuals do not exist on real images, hence the planned Waterbirds multi-method comparison.
- Three architectures have OOD results so far (LR main grid in Notebook 0, the CNN, and the MLP). RF and Waterbirds are still to go through OOD evaluation and the pooled analysis.
- **MLP regularization levels are mild:** the core's L2 range (1e-5 to 1e-2) barely changes colour reliance, so the sweep says little about regularization in general.

---

## Repository layout

```
.
├── Notebook-0.ipynb            # Shared core (Stage A) + main LR grid
├── Notebook-1.ipynb            # LR grid on the final core: frozen pre-shift scores + realism check
├── Notebook-3.ipynb            # MLP (Stage B) + reg × aug sweep + hard blind detection
├── Notebook-4.ipynb            # Small CNN (Stage B)
├── cnn_model.py                # CNN wrapper (needed to unpickle CNN models)
└── results/
    ├── mlp/
    │   ├── runs_mlp/                          # main 40-run grid (same files as runs_cnn, plus
    │   │                                      #   analysis_spearman_{mean,rev}_drop.csv)
    │   ├── runs_mlp_sweep/                    # 60-run reg × aug sweep, frozen separately
    │   ├── blind_detection_mlp.csv            # 10 tasks, all six scores
    │   ├── hard_blind_detection_mlp.csv       # 30 hard tasks (+ _summary_mlp.csv)
    │   ├── within_level_spearman_mlp.csv
    │   ├── colour_scores_by_size_mlp.csv
    │   ├── sweep_summary_mlp.csv / sweep_colour_scores_mlp.csv
    │   ├── realism_check_mlp.csv / reg_check_mlp.csv
    │   ├── weight_regions_mlp.csv             # exploratory first-layer weight view
    │   └── core.sha256                        # matches results/lr/core.sha256
    └── cnn/
        ├── runs_cnn/
        │   ├── scores_frozen.csv              # frozen pre-shift scores
        │   ├── scores_frozen.manifest.json    # SHA-256 + config at freeze time
        │   ├── run_meta.csv                   # per-run accuracy and settings
        │   ├── ood_drops.csv / ood_drops_long.csv
        │   └── analysis_spearman.csv
        ├── blind_detection_cnn_v2.csv         # 10 tasks, all six scores (use this one)
        ├── blind_detection_cnn.csv            # superseded (four scores only)
        ├── realism_check_cnn.csv
        └── ig_regions_cnn.csv                 # exploratory gradient view
```

Model `.pkl` files are not committed (size). They are in each notebook's Kaggle output.

---

## How to run (Kaggle)

1. Open the notebook on Kaggle. Turn on **Internet** (for the MNIST download) and, for the CNN, **GPU T4 x1**.
2. Run the smoke test first: in `Notebook-4`, click the Part 4 cell, choose **Run before**, then run it. It should print `CNN SMOKE TEST PASSED`.
3. Commit with **Save Version → Save & Run All**. The committed version is the freeze.
4. Each notebook writes to its own `out_dir`. Never reuse a frozen directory; start a new one for any re-run.

**Reproducibility:** fixed seeds throughout; deterministic cuDNN for the CNN; CNN hyperparameters were chosen on E1 validation accuracy only, before any OOD data was seen, and are recorded in the freeze manifest.
