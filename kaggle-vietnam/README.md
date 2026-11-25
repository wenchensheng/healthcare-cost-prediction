# Kaggle: Predictive Modeling for Healthcare Costs in Vietnam

**Community competition · RMSE · 12th of 78 teams (as of Nov 25, 2026) · Best public leaderboard score: 2.25342**

Predicting one year of individual healthcare cost from Vietnamese social
health insurance records. This folder documents a complete competition run:
five submissions, a −16% RMSE improvement from first to last, and the
structural findings that produced almost all of it. The final training script
is in [`code/`](code/); the full experiment log is in
[`RESULTS.md`](RESULTS.md).

## The problem and the data

- **Task:** predict each insured person's total healthcare cost for the year
  (`y`), scored by RMSE on the original scale.
- **Data:** 25,873 training rows × 32 columns, 8,625 test rows × 27 columns —
  demographics, employment and income components, insurance details, and
  utilization history.
- **The catch:** four utilization columns in the training data (outpatient /
  inpatient visit counts and costs: `FRE_NGOAITRU`, `FRE_NOITRU`,
  `COST_NGOAITRU`, `COST_NOITRU`) are *components of the target itself* and
  are deliberately absent from the test set. Using them as features would be
  target leakage, so they are excluded from the feature set throughout — they
  appear only inside strictly out-of-fold household statistics (below), never
  as a row's own features.

## Three findings that drove the score

### 1. The target has an exact closed form — and it is mostly zeros

`y = log1p(COST_NGOAITRU + COST_NOITRU)` holds exactly (99.54% of rows match
to <1e-9; the rest is float precision). So the target is a log-transformed
sum of outpatient + inpatient cost, and:

- **81.7% of rows are exactly zero** (no utilization at all). Only 18.3% of
  rows carry any cost — but they account for **75.4% of the total MSE**.
- Error decomposition showed the bottleneck is *whether* a person uses care
  at all, not *how much* they spend once they do: the positive-only regressor
  already reached RMSE 1.5868, while a perfect `P(y>0)` classifier would put
  the hurdle model's theoretical RMSE near 0.68. All later work therefore
  concentrated on the classification side.

### 2. Households are split across train and test

The `ID` is a household prefix plus a member index. There are 9,398
households; 5,383 appear on both sides of the split, and **94.01% of test
rows have at least one household member in the training data**. Within-household
correlation of `y` is substantial (split-half r = 0.3967) — family members'
utilization is real, usable signal.

Household features were the single biggest gain (local CV 2.5241 → 2.3544):

- **Static household features:** household size, head-of-household age /
  occupation / income, household income aggregates, member index.
- **"Mate" features (38 columns):** other members' `y` statistics (mean, max,
  sum, positive count and share, at several smoothing levels), cost-scale
  statistics, members' utilization aggregates, head/spouse features,
  adult/child splits.

Leakage discipline is what makes these legitimate: for training rows the
statistics are **leave-one-out** (a row never sees its own `y`); inside
cross-validation they are recomputed **within each fold**; test rows use the
full training set as the reference. A row's own utilization columns are never
used.

### 3. ID prefix segments are geographic / institutional blocks

ID prefixes behave like regional or institutional blocks: for 3-character
prefixes (324 groups), the between-group standard deviation of mean `y` is
0.746, and 5-character prefix positive rates range from 0.00 to 0.47.
Target-encoding these segments (fit rows encoded with inner out-of-fold
predictions, smoothing 3/10/30, prefix lengths 3–10) raised the hurdle
classifier's AUC from 0.8359 to 0.8453 and cut CV by a further ~0.02.

## Submission history

| Version | Approach | Local CV RMSE | Public LB | Standing |
|---|---|---|---|---|
| v1 | HistGradientBoosting on `log1p(y)` | 2.6712 | 2.69057 | — |
| v2 | HistGradientBoosting, Poisson loss, raw scale | 2.5441 | 2.54565 | — |
| v3 | Calibrated weighted blend of tournament models | 2.5421 | 2.54577 | 59th of 78 |
| v4 | Household + segment features, hurdle ensemble blend | 2.3006 | **2.28194** | **23rd of 78** |
| v6 | Household age-composition features + CatBoost hurdle blend | ≈2.276 | **2.25342** | **12th of 78** |

*(Standing as of Nov 25, 2026; the leaderboard is still open and positions
can move.)*

### Update — v6 (Nov 25, 2026)

A further iteration round found one more piece of signal the earlier
feature set had missed: **household age composition** — who else is in the
household and where the target person sits in its age order (household
age-band counts, within-household age position, the youngest member's
age). Adding those features, plus a CatBoost hurdle model for blend
diversity, brought local CV to ≈2.276 and the public leaderboard to
**2.25342 — 12th of 78 teams** (as of Nov 25, 2026).

## Final model (v4)

An equal-weight blend (0.2 each) of five components, all refit on the full
training set:

1. **Hurdle ensemble:** the mean of four hurdle configurations on the full
   feature set (each = HistGradientBoosting classifier for `P(y>0)` ×
   HistGradientBoosting L2 regressor on positive rows).
2. The strongest single hurdle configuration from that ensemble, counted
   again (the greedy blend's actual selection).
3. Poisson-loss HistGradientBoosting on the full feature set.
4. L2-loss HistGradientBoosting on the full feature set.
5. A hurdle model on the earlier (v1) segment-encoding feature set.

Feature inventory: 47 base dummy features + household static features +
38 household "mate" features + segment target encodings and segment-level
feature aggregates. Only columns present in the test set are used as raw
features.

## Validation discipline

- One protocol throughout: **5-fold KFold (shuffled, seed 42), RMSE on the
  original scale, predictions clipped at 0.** Every candidate model in
  [`RESULTS.md`](RESULTS.md) was judged on this protocol and nothing else.
- The leaderboard was used sparingly — five submissions total — and local CV
  tracked it closely (v2: CV 2.5441 vs LB 2.54565; v3: 2.5421 vs 2.54577;
  v4: 2.3006 vs 2.28194, where the household/segment features generalized
  slightly *better* on the public split than CV suggested).
- The blend's equal weights were confirmed two independent ways: greedy
  selection on out-of-fold predictions landed on 0.2 × 5, and an honest
  meta-CV stacking estimate (2.3005) agreed with the blend's CV (2.3006).

## What did not work

Documented in full in [`RESULTS.md`](RESULTS.md). In brief: LightGBM with a
Tweedie objective (2.4609); a LightGBM classifier inside the hurdle (2.3616);
stacked household-propensity features (no gain over the raw household
statistics); allocating a predicted household total across members (dead end,
household-total R² = 0.02); an inpatient-mixture regressor (no gain);
isotonic and scalar calibration of the classifier probability (no gain).

## What's next

- Two-level random effects over segments × households (EM-style shrinkage
  beyond fixed smoothing constants).
- Modeling the positive-cost distribution by cost tiers — observed costs
  spike at round VND amounts (20k/50k/100k/200k/500k).
- A count × severity joint model built on the visit-frequency structure.
- The classifier AUC (~0.845) appears close to the information limit of the
  available features: whether an individual seeks care in a given year has a
  large irreducible random component, which bounds how far any model on this
  feature set can go.

## Code

| File | What it does |
|---|---|
| [`code/final_fit.py`](code/final_fit.py) | Refits the winning v4 blend on the full training set and writes the submission file. Expects `data/train.csv`, `data/test.csv`, `data/sample_submission.csv` alongside it. |
| [`code/hh_features.py`](code/hh_features.py) | Household static features and segment target encodings, with the out-of-fold machinery. |
| [`code/hh_mate_features2.py`](code/hh_mate_features2.py) | Household "mate" features (v2): other members' outcome and utilization statistics, leave-one-out safe. |

Competition data is not redistributed here; see the
[competition page](https://www.kaggle.com/competitions/predictive-modeling-for-healthcare-costs-in-vietnam)
for the data and rules. This was a Kaggle **community competition** (no
prizes or medals) — entered as an end-to-end applied ML exercise in a domain
I work in professionally: healthcare cost and utilization data.
