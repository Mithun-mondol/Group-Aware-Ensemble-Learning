# Group-Aware Ensemble Learning for Concrete Strength Prediction

Data and code for:

> Mondol, M., Saha, T., Mondol, P., Bala, A. K., Billah, M. M., & Haque, R.
> *Group-Aware Ensemble Learning for Concrete Strength Prediction: A
> Dual-Validation Framework that Quantifies the Generalisation Gap.*
> Submitted to **Results in Engineering** (RINENG-D-26-11517).

The paper measures what happens to a concrete-strength model when the
validation protocol stops letting it see its own design class: standard 5-fold
cross-validation gives R² = 0.9237 on 28-day strength, and GroupKFold by design
class gives 0.8594 on the same data and the same model — a generalisation gap
of 6.43 percentage points. For slump the same comparison moves R² from ≈ 0.65
to below zero.

Everything needed to reproduce that is in this repository.

---

## Contents

### Data

| File | What it is |
|---|---|
| `Dhaka-Sylhet_mix_design.xlsx` | The QC ledger: 842 batch records from the SASEC Dhaka–Sylhet Corridor Road Investment Project. 683 survive the preprocessing in §3.2 (drop rows missing a critical column, remove physically impossible strength regression, de-duplicate identical mixes). |
| `validation_fold_assignments.csv` | The exact fold assignment used for every result in the paper, under both schemes, including the leave-one-class-out definitions. |

### Analysis

| File | What it produces |
|---|---|
| `concrete_strength_pipeline.py` | The main pipeline: nested cross-validation, Optuna search, the four-learner weighted ensemble, TreeSHAP, the Pareto search, and the code-compliance analysis. Main-text Tables 5–10 and Figures 2–9. |
| `rerun_R2.py` | The corrected re-run driver. Jobs A1–A2 (published protocol), C1–C3 (the early-stopping correction, isolated one step at a time), E1–E6 (secondary targets), F1–F2 (one-hot encoding), G1 (cement-brand grouping), R5a/R5b (the matched 26-feature comparison). |
| `extras_R2.py` | Blocks X1–X6: the anchor ablation with its full metric set, winsorisation sensitivity, the bootstrap replicates, the per-sample TreeSHAP matrices, the repeated class-to-fold assignments, and the fixed-parameter out-of-fold predictions. |
| `extras2_R2.py` | Blocks X7–X9: permutation importance, full-data-fit mean \|SHAP\|, and the no-anchor arm of the fold-assignment experiment. Supplementary Tables S4, S5 and S6b. |
| `mtl_R2.py` | The MTL-PINN baseline: jobs M1–M4, the five-seed spread, and the physics-penalty (λ) sweep. |
| `figures_R2.py` | Regenerates every data figure from the stored out-of-fold predictions, so no figure requires refitting a model. |
| `dataset.py` | The 683-record loader, shared by `figures_R2.py` so the figures and the tables cannot disagree about which rows are in the dataset. |
| `make_requirements.py` | Regenerates `requirements.txt` from the live environment. |

### Reproducing the correction

| File | What it is |
|---|---|
| `pipeline_patch.diff` | The exact diff between the pipeline as first released and the corrected version now in this repository. |
| `patch_pipeline.py` | The script that applies it. |
| `results/` | Every result CSV the scripts above produce, so the numbers in the paper can be checked without re-running anything. |

---

## Reproducing the results

```bash
pip install -r requirements.txt

python concrete_strength_pipeline.py     # main tables and figures
python rerun_R2.py                       # corrected protocol, all jobs
python extras_R2.py                      # supplementary blocks X1-X6
python extras2_R2.py                     # supplementary blocks X7-X9
python mtl_R2.py --jobs ALL              # MTL-PINN baseline
python figures_R2.py                     # figures, from stored predictions
```

All randomised components are seeded at 42.

---

## Two things worth knowing before you re-run this

**1. Early stopping was leaking, and it is fixed here.**

The version of `concrete_strength_pipeline.py` first released with this
repository fitted LightGBM and CatBoost with `eval_set` set to the very fold the
model was then scored on, in nine places. The number of boosting rounds was
therefore selected on the evaluation data. The effect is small — the headline
28-day figures move from R² = 0.9251 / 0.8527 to 0.9237 / 0.8594 — but it is
real, and it is the kind of optimism the paper exists to argue against. Every
number in the published version of the paper is computed under the corrected
protocol, in which no model is shown, at fit time, any row on which it is later
evaluated, and the inner cross-validation loop is group-aware whenever the outer
loop is. `pipeline_patch.diff` is the change.

**2. `tf.random.set_seed()` does not seed Keras 3.**

The neural baseline appeared to be nondeterministic: repeated runs at a fixed
seed differed by up to 18 percentage points of R². It was not nondeterminism.
`tf.random.set_seed()` — the conventional call, and the one this pipeline used —
no longer resets the generator that Keras 3 draws initializer and dropout
randomness from. Under Keras 2 and `tf.keras` it was correct; the results
silently stopped being reproducible when the environment moved to Keras 3, with
no error and no warning.

The fix is one call:

```python
import keras
keras.utils.set_random_seed(seed)    # seeds Python, NumPy, TensorFlow and Keras
```

With it, repeated runs of the baseline are bitwise identical. Without it they
are not, and nothing tells you so. Any study whose deep-learning results were
seeded the conventional way and whose environment has since moved to Keras 3
has this problem.

Tree-ensemble results were never affected — they do not go through Keras.

---

## Environment

`requirements.txt` pins the versions used. The results in the paper were
produced under Python 3.13 with TensorFlow 2.21 and Keras 3.15. Tree-ensemble
results are reproducible under that pinned environment at seed 42. Re-running
the published protocol two library generations later, on identical fold
assignments, reproduced standard 5-fold R² to within 0.0005 and GroupKFold R²
to within 0.0042 — reported as a measured property rather than asserted as
reproducibility in general.

## Licence and contact

Please cite the paper if you use this data or code.
Questions: mithun.mondol@gmail.com
