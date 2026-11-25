"""Final: refit the winning blend on FULL train, predict test, write
submission_v4_candidate.csv. Blend (from OOF greedy, equal weights):
  0.2 * M8_hurdle_ens4 (mean of hurdle cfgA-D on M7 features)
+ 0.2 * M8_cfgB hurdle (M7 features)
+ 0.2 * M7_poisson
+ 0.2 * M7_L2
+ 0.2 * M6_hurdle (M6 features: segTE k=3,4,5,7 smooth10)
Mate features for test rows use the full-train reference set; fit rows use
LOO mate features on full train (mirrors the CV procedure)."""
import json, time
import numpy as np, pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.model_selection import KFold
import hh_features as HF
import hh_mate_features2 as MF2

SEED = 42
LEAK = ["FRE_NGOAITRU", "FRE_NOITRU", "COST_NGOAITRU", "COST_NOITRU"]
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
sample = pd.read_csv("data/sample_submission.csv")
y = train["y"].to_numpy(float)
z = (y > 0).astype(float)
ids = train["ID"].to_numpy(); ids_te = test["ID"].to_numpy()
feature_cols = [c for c in train.columns if c not in {"y", "ID"} | set(LEAK) and c in test.columns]
tr_static, te_static, tr_hh, tr_mem, te_hh, te_mem = HF.build_static_hh(train, test)

def dummies(a, b):
    X = pd.get_dummies(a, dummy_na=True)
    Xt = pd.get_dummies(b, dummy_na=True).reindex(columns=X.columns, fill_value=0)
    return X.astype(np.float32), Xt.astype(np.float32)

X1, X1_te = dummies(pd.concat([train[feature_cols], tr_static], axis=1),
                    pd.concat([test[feature_cols], te_static], axis=1))
train_full = HF.split_id(train)
ref_all = train_full[["HH", "MEM", "ID", "AGE", "y"] + LEAK].copy()
gmean = float(y.mean()); base_rate = float(z.mean())
HH = tr_hh.to_numpy(); HH_te = te_hh.to_numpy()
t0 = time.time()

# ---- mate features: fit rows LOO on full train; test rows from full train ----
m_fit = MF2.mate_features2(ref_all, tr_hh.to_numpy(), tr_mem.to_numpy(), tgt_id=ids, gmean=gmean, base_rate=base_rate)
m_te = MF2.mate_features2(ref_all, te_hh.to_numpy(), te_mem.to_numpy(), gmean=gmean, base_rate=base_rate)
print(f"mate features done ({time.time()-t0:.0f}s)")

# ---- segment TE helpers ----
def te_apply(fit_seg, fit_val, app_seg, smooth, prior):
    df = pd.DataFrame({"s": fit_seg, "v": fit_val})
    st = df.groupby("s")["v"].agg(["mean", "count"])
    enc = (st["mean"] * st["count"] + prior * smooth) / (st["count"] + smooth)
    return pd.Series(app_seg).map(enc).fillna(prior).to_numpy()

def seg_block(smooths, fit_oof=True):
    """Returns (fit_block, test_block) for the given (k, smooth) list.
    Fit rows: inner-OOF TE. Test rows: TE fit on full train."""
    seg_tr = {k: np.array([h[:k] for h in HH]) for k, _ in smooths}
    seg_te = {k: np.array([h[:k] for h in HH_te]) for k, _ in smooths}
    n = len(train)
    fit_cols, te_cols = [], []
    inner = KFold(5, shuffle=True, random_state=999)
    for k, sm in smooths:
        ey = np.zeros(n); ez = np.zeros(n)
        for itr, iva in inner.split(np.arange(n)):
            ey[iva] = te_apply(seg_tr[k][itr], y[itr], seg_tr[k][iva], sm, gmean)
            ez[iva] = te_apply(seg_tr[k][itr], z[itr], seg_tr[k][iva], sm, base_rate)
        fit_cols += [ey, ez]
        te_cols.append(te_apply(seg_tr[k], y, seg_te[k], sm, gmean))
        te_cols.append(te_apply(seg_tr[k], z, seg_te[k], sm, base_rate))
        # log count (from full train) for both
        vc = pd.Series(seg_tr[k]).value_counts()
        fit_cols.append(np.log1p(pd.Series(seg_tr[k]).map(vc).to_numpy()))
        te_cols.append(np.log1p(pd.Series(seg_te[k]).map(vc).fillna(0).to_numpy()))
    return (np.column_stack(fit_cols).astype(np.float32),
            np.column_stack(te_cols).astype(np.float32))

# M6 seg block: k in {3,4,5,7}, smooth 10, with log-count (matches exp6 layout:
# per k -> TE_y, TE_z, logcnt) -- exp6 appended [ey, ez, logcnt] per k. Same here.
seg6_fit, seg6_te = seg_block([(3, 10.0), (4, 10.0), (5, 10.0), (7, 10.0)])
# M7 seg block: exp7 layout per (k,sm) -> [TE_y, TE_z] only, plus seg X-agg.
SMOOTHS7 = [(3, 3.0), (4, 10.0), (5, 10.0), (7, 10.0), (8, 30.0), (10, 30.0)]
def seg_block7():
    seg_tr = {k: np.array([h[:k] for h in HH]) for k, _ in SMOOTHS7}
    seg_te = {k: np.array([h[:k] for h in HH_te]) for k, _ in SMOOTHS7}
    n = len(train)
    fit_cols, te_cols = [], []
    inner = KFold(5, shuffle=True, random_state=999)
    for k, sm in SMOOTHS7:
        ey = np.zeros(n); ez = np.zeros(n)
        for itr, iva in inner.split(np.arange(n)):
            ey[iva] = te_apply(seg_tr[k][itr], y[itr], seg_tr[k][iva], sm, gmean)
            ez[iva] = te_apply(seg_tr[k][itr], z[itr], seg_tr[k][iva], sm, base_rate)
        fit_cols += [ey, ez]
        te_cols.append(te_apply(seg_tr[k], y, seg_te[k], sm, gmean))
        te_cols.append(te_apply(seg_tr[k], z, seg_te[k], sm, base_rate))
    return (np.column_stack(fit_cols).astype(np.float32),
            np.column_stack(te_cols).astype(np.float32))
seg7_fit, seg7_te = seg_block7()

both = pd.concat([HF.add_income(train_full)[["HH", "AGE", "TOTAL_INCOME", "CO_LAM_VIEC"]],
                  HF.add_income(HF.split_id(test))[["HH", "AGE", "TOTAL_INCOME", "CO_LAM_VIEC"]]],
                 ignore_index=True)
def seg_x_agg(hh_array):
    out = []
    for k in [3, 5]:
        seg = np.array([h[:k] for h in hh_array])
        tmp = both.assign(seg=np.array([h[:k] for h in both["HH"]]))
        g = tmp.groupby("seg")
        m_age = g["AGE"].mean(); m_inc = g["TOTAL_INCOME"].mean()
        m_work = g["CO_LAM_VIEC"].mean(); sz = g.size()
        s = pd.Series(seg)
        out += [s.map(m_age).to_numpy(), np.log1p(s.map(m_inc).to_numpy()),
                s.map(m_work).to_numpy(), np.log1p(s.map(sz).to_numpy())]
    return np.column_stack(out).astype(np.float32)
segx_fit = seg_x_agg(HH); segx_te = seg_x_agg(HH_te)
print(f"seg blocks done ({time.time()-t0:.0f}s)")

X7_fit = np.hstack([X1.to_numpy(), m_fit.to_numpy(), seg7_fit, segx_fit]).astype(np.float32)
X7_te = np.hstack([X1_te.to_numpy(), m_te.to_numpy(), seg7_te, segx_te]).astype(np.float32)
X6_fit = np.hstack([X1.to_numpy(), m_fit.to_numpy(), seg6_fit]).astype(np.float32)
X6_te = np.hstack([X1_te.to_numpy(), m_te.to_numpy(), seg6_te]).astype(np.float32)
print("X7:", X7_fit.shape, "| X6:", X6_fit.shape)

pos = y > 0
CFG = {
    "cfgA": {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15, "l2_regularization": 1.0},
    "cfgB": {"learning_rate": 0.04, "max_iter": 400, "max_leaf_nodes": 31, "l2_regularization": 1.0},
    "cfgC": {"learning_rate": 0.03, "max_iter": 600, "max_leaf_nodes": 15, "l2_regularization": 1.0},
    "cfgD": {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 7, "l2_regularization": 1.0},
}
def hurdle_pred(Xf, Xt, kw):
    clf = HistGradientBoostingClassifier(random_state=SEED, **kw).fit(Xf, pos.astype(int))
    reg = HistGradientBoostingRegressor(random_state=SEED).fit(Xf[pos], y[pos])
    return clf.predict_proba(Xt)[:, 1] * np.clip(reg.predict(Xt), 0, None)

preds = {}
for name, kw in CFG.items():
    preds[name] = hurdle_pred(X7_fit, X7_te, kw)
    print(f"hurdle {name} done ({time.time()-t0:.0f}s)", flush=True)
ens4 = np.mean([preds[k] for k in CFG], axis=0)
mp = HistGradientBoostingRegressor(loss="poisson", random_state=SEED).fit(X7_fit, y)
preds["M7_poisson"] = np.clip(mp.predict(X7_te), 0, None)
ml = HistGradientBoostingRegressor(random_state=SEED).fit(X7_fit, y)
preds["M7_L2"] = np.clip(ml.predict(X7_te), 0, None)
preds["M6_hurdle"] = hurdle_pred(X6_fit, X6_te, CFG["cfgA"])
print(f"all members done ({time.time()-t0:.0f}s)")

final = 0.2 * (ens4 + preds["cfgB"] + preds["M7_poisson"] + preds["M7_L2"] + preds["M6_hurdle"])
final = np.clip(final, 0, None)

sub = sample.copy()
assert (sub["ID"].to_numpy() == ids_te).all(), "sample_submission ID order mismatch"
sub["y.hat"] = final
sub.to_csv("submission_v4_candidate.csv", index=False)

# ---- self-checks ----
chk = pd.read_csv("submission_v4_candidate.csv")
assert chk.shape == (8625, 2), chk.shape
assert list(chk.columns) == ["ID", "y.hat"], list(chk.columns)
assert (chk["ID"].to_numpy() == ids_te).all()
assert chk["y.hat"].notna().all() and (chk["y.hat"] >= 0).all()
print("\nsubmission_v4_candidate.csv OK:", chk.shape)
print(chk["y.hat"].describe())
print("pred mean:", float(final.mean()), "| train y mean:", gmean,
      "| share pred>0.01:", float((final > 0.01).mean()))
json.dump({"members": {k: float(np.mean(v)) for k, v in preds.items()},
           "final_mean": float(final.mean()), "final_max": float(final.max())},
          open("final_fit_summary.json", "w"), indent=2)
