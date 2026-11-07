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
| `Notebook-3.ipynb` | MLP | ⏳ Pending |
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

> These are results from Colored MNIST only, with two architectures. RF, MLP and Waterbirds are pending, so treat everything below as preliminary.

### Controls
In **every** run of both models, colour sensitivity beats both decoys and the null, and null sensitivity is exactly **0.000**.

Realism check (natural vs counterfactual images with the same label and feature value; 0.5 = indistinguishable):

| Check | AUROC |
|---|---|
| Colour swap (Notebook 4, CNN classifier) | 0.50–0.52 at every training correlation |
| Positive control: swap plus a faint artifact | 1.00 |
| Colour / marker_tl / marker_br (Notebook 1) | 0.48–0.53 for all three features at every training correlation |

### RQ2: does the frozen score predict the reversed-colour drop? (40 runs each)

| Score | LR ρ | LR partial ρ \| label corr | CNN ρ | CNN partial ρ \| label corr |
|---|---|---|---|---|
| **Sensitivity** | **0.989** | **0.914** | **0.994** | **0.976** |
| SHAP | 0.992 | 0.935 | 0.995 | 0.976 |
| Permutation importance | 0.979 | 0.825 | 0.990 | 0.969 |
| Stability | 0.950 | 0.576 | 0.951 | 0.860 |
| Label correlation | 0.934 | — | 0.880 | — |
| Mutual information | 0.934 | −0.083 | 0.879 | −0.084 |

ρ is the Spearman correlation between the frozen score and `rev_drop`; the partial ρ controls for label correlation. Sources: LR from `Notebook-0` (`runs_mi_perm_v2`); CNN from `results/cnn/runs_cnn`.

- The model-based scores (sensitivity, SHAP, permutation importance) all predict the drop well beyond label correlation.
- Mutual information adds nothing beyond label correlation. That is expected: with two binary variables it is essentially a rescaling of the same quantity.

### Same data, different models (the cleanest RQ2 test)
For each of the 40 (corr, size, seed) settings, LR and CNN see **identical training data**, so label correlation and MI are identical by construction and cannot tell them apart.

- The difference in sensitivity (CNN − LR) predicts the difference in reversed-colour drop with **Spearman ρ = 0.978** (p ≈ 10⁻²⁷).
- In **all 40 pairs**, the model with the higher sensitivity is the one that dropped more.

| Train corr | n | LR `rev_drop` | CNN `rev_drop` |
|---|---|---|---|
| 0.50 | 5,000 | 0.121 | 0.027 |
| 0.50 | 20,000 | 0.110 | 0.011 |
| 0.70 | 5,000 | 0.216 | 0.063 |
| 0.70 | 20,000 | 0.206 | 0.031 |
| 0.85 | 5,000 | 0.345 | 0.128 |
| 0.85 | 20,000 | 0.331 | 0.068 |
| 0.95 | 5,000 | 0.576 | **0.805** |
| 0.95 | 20,000 | 0.516 | 0.208 |

The CNN usually relies on colour far less than LR, because it can learn digit shape. The exception is a strong shortcut with little data (0.95, n = 5,000), where it relies on colour *more*. Sensitivity tracks both directions.

### RQ1: blind detection
Almost every score achieves top-1 = 1.0 and AUROC = 1.0, on the grid runs and on the 10 generated tasks where the shortcut is sometimes a marker. The one exception is stability for the CNN: top-1 0.97 and AUROC 0.99 on the grid, AUROC 0.985 on the 10 tasks. The task is currently **too easy to separate methods**; harder decoys are needed.

---

## Limitations and open issues

- **Our score is not uniquely best.** SHAP and permutation importance perform about as well. The supported claim is that *model-based* reliance scores predict OOD failure where data-only statistics cannot.
- **Correlation level drives much of the pooled ρ.** Within-stratum and paired analyses are the stronger evidence.
- **Blind detection saturates** (all methods perfect).
- **Mixed-effects model** has few groups and boundary warnings. When pooling architectures, use model as a fixed effect plus the cluster bootstrap.
- **Synthetic data:** exact counterfactuals do not exist on real images, hence the planned Waterbirds multi-method comparison.
- Only two architectures have OOD results so far (LR main grid in Notebook 0, and the CNN). Notebook 1's frozen LR scores, RF, MLP and Waterbirds are still to go through OOD evaluation and the pooled analysis.

---

## Repository layout

```
.
├── Notebook-0.ipynb            # Shared core (Stage A) + main LR grid
├── Notebook-1.ipynb            # LR grid on the final core: frozen pre-shift scores + realism check
├── Notebook-4.ipynb            # Small CNN (Stage B)
├── cnn_model.py                # CNN wrapper (needed to unpickle CNN models)
└── results/
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
