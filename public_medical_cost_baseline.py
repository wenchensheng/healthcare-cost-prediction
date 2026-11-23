"""Baseline: predict individual medical charges (public 'Medical Cost Personal' dataset).

Pipeline: log1p target, one-hot categoricals, HistGradientBoostingRegressor,
5-fold cross-validation, metrics reported on the original charges scale.
Also writes metrics.json to /kaggle/working so output retrieval can be verified.
"""
import glob
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

t0 = time.time()

# ---------- Load ----------
candidates = glob.glob("/kaggle/input/**/insurance.csv", recursive=True)
print("candidate dataset paths:", candidates)
assert candidates, "insurance.csv not found under /kaggle/input"
df = pd.read_csv(candidates[0])

# ---------- Brief EDA ----------
print("shape:", df.shape)
print("dtypes:\n", df.dtypes)
print("first rows:\n", df.head())
print("charges describe:\n", df["charges"].describe())
print("missing per column:\n", df.isna().sum())
print("smoker counts:\n", df["smoker"].value_counts())
print("mean charges by smoker:\n", df.groupby("smoker")["charges"].mean())

# ---------- Features / target ----------
y_log = np.log1p(df["charges"].to_numpy(dtype=float))
X = pd.get_dummies(
    df.drop(columns=["charges"]),
    columns=["sex", "smoker", "region"],
    drop_first=False,
)
print("feature matrix:", X.shape, "columns:", list(X.columns))

# ---------- 5-fold CV ----------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_log = np.zeros(len(df))
fold_rmses = []
for fold, (tr, te) in enumerate(kf.split(X)):
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X.iloc[tr], y_log[tr])
    oof_log[te] = model.predict(X.iloc[te])
    fold_rmse = float(np.sqrt(mean_squared_error(
        np.expm1(y_log[te]), np.expm1(oof_log[te]))))
    fold_rmses.append(fold_rmse)
    print(f"fold {fold}: RMSE (original scale) = {fold_rmse:.2f}")

pred = np.expm1(oof_log)
true = df["charges"].to_numpy(dtype=float)
rmse = float(np.sqrt(mean_squared_error(true, pred)))
mae = float(mean_absolute_error(true, pred))
r2 = float(r2_score(true, pred))

print(f"CV_RMSE_ORIGINAL_SCALE {rmse:.2f}")
print(f"CV_MAE_ORIGINAL_SCALE {mae:.2f}")
print(f"CV_R2_ORIGINAL_SCALE {r2:.4f}")

wall = time.time() - t0
print(f"WALL_SECONDS {wall:.1f}")

metrics = {
    "dataset": "mirichoi0218/insurance (Medical Cost Personal)",
    "rows": int(df.shape[0]),
    "model": "HistGradientBoostingRegressor (log1p target, one-hot cats)",
    "cv_folds": 5,
    "fold_rmse_original_scale": fold_rmses,
    "cv_rmse_original_scale": rmse,
    "cv_mae_original_scale": mae,
    "cv_r2_original_scale": r2,
    "wall_seconds_inside_kernel": wall,
}
with open("/kaggle/working/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("wrote /kaggle/working/metrics.json")
