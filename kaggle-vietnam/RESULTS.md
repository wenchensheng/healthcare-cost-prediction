# Experiment Log — Vietnam Healthcare Cost Competition (v4 round)

All results use one fixed protocol: **5-fold KFold (shuffled, seed 42),
RMSE on the original scale, predictions clipped at ≥ 0.** The anchor
baseline — the previous champion (Poisson HistGradientBoosting on base
features) — was reproduced locally at CV 2.5438 before any new work started
(kernel CV 2.5441).

## Headline result

- **Final candidate CV RMSE = 2.3006** (baseline 2.5441, **−0.2435 / −9.6%**).
- Submitted as v4: **public leaderboard 2.28194, 23rd of 78 teams**
  (as of Nov 25, 2026), up from 59th.
- Historical CV→LB offsets: v2 CV 2.5441 → LB 2.54565 (+0.0016);
  v3 CV 2.5421 → LB 2.54577 (+0.0037); v4 CV 2.3006 → LB 2.28194 (−0.0187) —
  the household/segment features generalized better on the public split
  than CV estimated.

## The three structural findings behind the gains

1. **`y = log1p(COST_NGOAITRU + COST_NOITRU)` holds exactly** (99.54% of
   rows match to <1e-9; the remainder is float precision), and `y > 0` ⟺
   either cost column is positive, without exception.
2. **Households (ID prefix) are split across train/test:** 9,398
   households, 5,383 on both sides; **94.01% of test rows have a household
   mate in train.** Split-half within-household correlation of `y` is
   0.3967. Household features (other members' `y` / utilization statistics,
   always computed out-of-fold / leave-one-out) were the largest single
   gain: 2.5241 → 2.3544.
3. **ID prefix segments are geographic/institutional blocks:** 3-character
   segments (324 groups) have a between-group SD of mean `y` of 0.746;
   5-character segment positive rates span 0.00–0.47. Segment target
   encoding (fit within fold; fit rows via inner OOF) raised classifier
   AUC from 0.8359 to 0.8453 and cut CV by a further ~0.02.

Error structure (which set the strategy): **75.4% of MSE comes from the
18.3% of rows with `y > 0`**; the positive-row regressor was already decent
(RMSE 1.5868 on positives), so the bottleneck is the `P(y>0)` classifier —
with a perfect classifier the hurdle's theoretical RMSE is ≈ 0.68. All
effort therefore went into classification-side signal.

## Candidate results (all on the fixed 5-fold CV protocol)

| Candidate | CV RMSE | Note |
|---|---|---|
| Baseline Poisson HistGBM (base features) | 2.5438 | = v2 champion, reproduced locally |
| M1 + household static X features (size / head / income aggregates) | 2.5241 | −0.020 |
| M2 + household `y` features v1 (LOO mean / max / positive share) | 2.3544 | **−0.170, biggest single step** |
| M3 HistGBM Poisson (extended mate features: cost scale + FRE/COST) | 2.3482 | |
| M3 LightGBM Tweedie 1.5 | 2.4609 | rejected |
| M3 LightGBM Poisson | 2.3516 | |
| M3 Hurdle (Poisson positive regressor) | 2.3393 | |
| M3 HistGBM L2 (squared loss, all rows) | 2.3420 | |
| M3 Hurdle (L2 positive regressor) | 2.3395 | classifier AUC 0.8317 |
| M4 HistGBM Poisson (mate features v2: relation splits, multi-smoothing) | 2.3383 | |
| M4 Hurdle L2 | 2.3359 | |
| M4 Hurdle (LightGBM classifier) | 2.3616 | rejected |
| Hurdle + tuned classifier (leaves 15 / lr .05 / L2 = 1) | 2.3275 | AUC 0.8359 |
| M5 Hurdle + stacked household-propensity features (nested OOF) | 2.3353 | no gain, rejected |
| Household-total allocation (diagnostic) | — | household-total R² = 0.02, dead end |
| M6 Hurdle + segment target encoding (k = 3,4,5,7) | 2.3083 | AUC 0.8453 |
| M6 Hurdle + inpatient-mixture regressor | 2.3086 | no gain |
| M6 Poisson + segment TE | 2.3240 | |
| M7 Hurdle (segment TE multi-smoothing + segment X aggregates) | 2.3076 | |
| M7 Poisson / M7 L2 | 2.3183 / 2.3173 | |
| M8 Hurdle config ensemble (mean of 4 classifier configs) | 2.3060 | best single model |
| Stacking (NNLS/Ridge, honest meta-CV estimate) | 2.3005 | |
| **Final blend (greedy equal weights 0.2 × 5)** | **2.3006** | cross-validated against stacking |

The blend weights were selected greedily on the same out-of-fold
predictions, which carries mild optimism; two things keep the estimate
honest: the greedy search landed on exactly equal weights (0.2 × 5), and an
independent meta-CV stacking estimate (2.3005) agrees. Isotonic calibration
of the classifier probability (2.3091) and scalar calibration both gave no
gain and were not used.

## Final blend composition

Equal weights (0.2 each), every component refit on the full training set
before predicting the test set:

1. M8 hurdle config ensemble: mean of four hurdle configurations
   (classifier leaf sizes 15/31/15/7) on M7 features — HistGradientBoosting
   classifier + L2 positive-row regressor.
2. The cfgB configuration (leaves 31 / lr .04) counted a second time — the
   greedy blend's actual selection.
3. M7 Poisson HistGradientBoosting.
4. M7 L2 HistGradientBoosting.
5. M6 hurdle (segment-TE v1 feature set).

Feature inventory (raw features use only columns present in the test set;
the four utilization columns appear solely as *other household members'*
statistics, never as a row's own features):

- Base dummy features (47 columns) + household static features (household
  size, head's age/occupation/income, household income aggregates, member
  index, …).
- Mate features v2 (38 columns): other members' `y` mean / max / sum /
  positive count and share (multiple smoothing levels), cost-scale
  statistics, members' FRE/COST aggregates, head/spouse singled out,
  adult/child splits. Training rows: leave-one-out; inside CV: computed
  within fold; test rows: full-train reference set.
- Segment features: ID-prefix target encodings for `y` and positive rate
  at prefix lengths k ∈ {3,4,5,7,8,10} (smoothing 3/10/30, fit rows via
  inner OOF) + segment X aggregates (segment mean age / income /
  employment rate / size, computed on train+test features only).

## Submission self-checks (all passed)

- `submission_v4_candidate.csv`: 8,625 rows × 2 columns, columns
  `[ID, y.hat]`; IDs aligned row-by-row with the sample submission; no
  NaNs; all predictions ≥ 0.
- Prediction mean 1.1635 (train `y` mean 1.1919), median 0.598, max 7.371 —
  the normal shrinkage of a conditional-mean model (train `y` max 14.15).

## Open questions and next directions

- Classifier AUC 0.845 looks close to the information ceiling of the
  available features: whether an individual uses care in a given year
  carries substantial unobservable randomness.
- Untried directions (uncertain payoff): two-level segment × household
  random effects (EM-style); modeling the positive-cost distribution by
  cost tiers (spikes at 20k/50k/100k/200k/500k VND); a count × severity
  joint model using the visit-frequency structure.
- Discipline note: this entire round was iterated on local CV only; the
  leaderboard received exactly one submission (v4) at the end.

## Postscript — v6 round (Nov 25, 2026)

A later iteration round found one more signal source the feature set above
had missed: **household age-composition features** (household age-band
counts, the target's position in the household age order, the youngest
member's age), plus a CatBoost hurdle model added to the blend for
diversity. Honest local CV ≈ 2.276, versus 2.3006 for the v4 blend logged
above. Submitted as v6: **public leaderboard 2.25342, 12th of 78 teams**
(as of Nov 25, 2026). The v4 results above stand unchanged — they are the
record of the v4 round.

## Postscript — v7 round (Nov 25, 2026)

Two further rounds (7–8) attacked the classification ceiling from the
target's own structure. Round 7 showed the outpatient component alone is
more predictable than any-cost overall (AUC 0.8599 for P(outpatient > 0)
versus the ~0.850 any-cost ceiling every earlier classifier had hit);
round 8 built a joint four-state model (no cost / outpatient only /
inpatient only / both) with state-conditional magnitude models, whose
member (standalone OOF RMSE 2.2812) joined the full model pool under an
NNLS/Ridge meta-blend. A dedicated inpatient feature block gave no gain
(P(inpatient > 0) AUC +0.0007, noise) and was rejected. Honest meta-CV
2.2688, versus 2.2757 for the v6 recipe. Submitted as v7: **public
leaderboard 2.24397, 1st of 78 teams** (as of Nov 25, 2026). The v4 and
v6 results above stand unchanged — they are the record of those rounds.
