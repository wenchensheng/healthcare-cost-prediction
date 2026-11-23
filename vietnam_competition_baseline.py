"""STAGED baseline v1 for Kaggle competition:
'Predictive Modeling for Healthcare Costs in Vietnam'
(slug: predictive-modeling-for-healthcare-costs-in-vietnam, Community, metric: RMSE)

STATUS: staged, NOT run. It can only run after the account joins the competition
(user must open the competition page, click 'Join Competition' and accept the
rules) and this kernel is pushed with the competition attached as a data source
(competitionDataSources: ["predictive-modeling-for-healthcare-costs-in-vietnam"]).

Competition files (confirmed via the API data-list endpoint, 2026-11-12):
  train.csv             2,914,543 bytes
  test.csv                866,490 bytes
  sample_submission.csv   231,885 bytes
  descriptions.xlsx        10,995 bytes  <- exact column definitions live here

Approach (v1, deliberately simple):
  - Predict the TOTAL annual healthcare cost directly (primary target).
    (v2 idea, not implemented: frequency x severity decomposition using the
    inpatient/outpatient cost and frequency columns present in train.)
  - log1p-transform the target, one-hot all non-numeric feature columns,
    HistGradientBoostingRegressor, 5-fold CV, report RMSE on the original
    scale (the competition metric is RMSE).
  - Retrain on all of train and write /kaggle/working/submission.csv in the
    sample_submission column format. Writing the file is NOT a submission;
    nothing is sent to the leaderboard by this script.
"""
import glob
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

COMP_SLUG = "predictive-modeling-for-healthcare-costs-in-vietnam"

# TODO (resolve on first real run): the exact target column name is defined in
# descriptions.xlsx, which cannot be inspected until the competition is joined.
# The loader tries these candidates in order; if none matches it prints every
# train column and stops with instructions instead of guessing.
TARGET_CANDIDATES = [
    "total_cost", "total_annual_cost", "total_healthcare_cost",
    "total", "cost", "y", "target",
]

# Other observed-target columns in train (per the competition overview). They
# must never be used as features for the total-cost target (leakage), so the
# loader drops them from X if present. TODO: confirm exact names from
# descriptions.xlsx on the first real run.
AUX_TARGET_CANDIDATES = [
    "inpatient_cost", "outpatient_cost",
    "inpatient_frequency", "outpatient_frequency",
    "inpatient_visits", "outpatient_visits",
]

t0 = time.time()


def find_file(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, f"{name} not found under /kaggle/input (competition attached?)"
    return hits[0]


# ---------- Load ----------
train = pd.read_csv(find_file("train.csv"))
test = pd.read_csv(find_file("test.csv"))
sample_sub = pd.read_csv(find_file("sample_submission.csv"))
print("train shape:", train.shape, "| test shape:", test.shape)
print("train columns:", list(train.columns))
print("sample_submission columns:", list(sample_sub.columns))
print(train.head())

# ---------- Resolve target ----------
target = next((c for c in TARGET_CANDIDATES if c in train.columns), None)
if target is None:
    raise SystemExit(
        "TARGET NOT RESOLVED. Train columns are printed above. "
        "Check descriptions.xlsx, add the real total-cost column name to "
        "TARGET_CANDIDATES, and re-run. No guessing was done."
    )
print("target column:", target)

# ID column = first column of the sample submission (assumed to be an id that
# also exists in train/test). TODO: confirm against descriptions.xlsx.
id_col = sample_sub.columns[0]
print("id column (from sample_submission):", id_col)

drop_cols = {target, id_col} | {c for c in AUX_TARGET_CANDIDATES if c in train.columns}
feature_cols = [c for c in train.columns if c not in drop_cols]
print("feature columns:", feature_cols)

X = pd.get_dummies(train[feature_cols], dummy_na=True)
X_test = pd.get_dummies(test[feature_cols], dummy_na=True).reindex(
    columns=X.columns, fill_value=0)
y = train[target].to_numpy(dtype=float)
y_log = np.log1p(y)

# ---------- 5-fold CV (metric: RMSE, original scale) ----------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_log = np.zeros(len(train))
for fold, (tr, te) in enumerate(kf.split(X)):
    m = HistGradientBoostingRegressor(random_state=42)
    m.fit(X.iloc[tr], y_log[tr])
    oof_log[te] = m.predict(X.iloc[te])
    fr = float(np.sqrt(mean_squared_error(np.expm1(y_log[te]), np.expm1(oof_log[te]))))
    print(f"fold {fold}: RMSE (original scale) = {fr:.2f}")

pred_oof = np.expm1(oof_log)
rmse = float(np.sqrt(mean_squared_error(y, pred_oof)))
mae = float(mean_absolute_error(y, pred_oof))
r2 = float(r2_score(y, pred_oof))
print(f"CV_RMSE_ORIGINAL_SCALE {rmse:.2f}")
print(f"CV_MAE_ORIGINAL_SCALE {mae:.2f}")
print(f"CV_R2_ORIGINAL_SCALE {r2:.4f}")

# ---------- Retrain on full train, write submission file (not submitted) ------
final = HistGradientBoostingRegressor(random_state=42)
final.fit(X, y_log)
test_pred = np.expm1(final.predict(X_test))
test_pred = np.clip(test_pred, 0, None)

sub = sample_sub.copy()
sub[sub.columns[-1]] = test_pred  # prediction goes in the sample's target column
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("wrote /kaggle/working/submission.csv with shape", sub.shape)

wall = time.time() - t0
print(f"WALL_SECONDS {wall:.1f}")
with open("/kaggle/working/metrics.json", "w") as f:
    json.dump({
        "competition": COMP_SLUG, "target": target,
        "cv_rmse_original_scale": rmse, "cv_mae_original_scale": mae,
        "cv_r2_original_scale": r2, "wall_seconds_inside_kernel": wall,
    }, f, indent=2)
print("wrote /kaggle/working/metrics.json")
