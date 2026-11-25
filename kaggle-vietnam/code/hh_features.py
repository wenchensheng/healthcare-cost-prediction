"""Household feature builder for the Vietnam healthcare-cost competition.

Households (ID prefix before '_') are split across train/test: 94% of test
rows have a household-mate in train. Two feature families:
  * X-only household features: computed on combined train+test features
    (mirrors test-time availability; no y involved).
  * y-based household features: computed from a reference set that owns y
    (train fold / full train), always excluding the target row itself.
"""
import numpy as np, pandas as pd

INCOME_COLS = [
    "TIEN_LUONG_12T", "TIEN_LE_TET", "THUONG_PHU_CAP",
    "LUONG_CONG_VIEC_2", "TIEN_LE_TET_2", "THUONG_PHU_CAP_2",
    "LUONG_CONG_VIEC_3", "TIEN_TRO_CAP_THAT_NGHIEP",
    "TIEN_TRO_CAP_THOI_VIEC_1_LAN",
]

def split_id(df):
    parts = df["ID"].str.rsplit("_", n=1)
    df = df.copy()
    df["HH"] = parts.str[0]
    df["MEM"] = parts.str[1].astype(int)
    return df

def add_income(df):
    df = df.copy()
    inc = df[INCOME_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df["TOTAL_INCOME"] = inc.sum(axis=1)
    df["INCOME_SOURCE_COUNT"] = (inc > 0).sum(axis=1)
    return df

def build_static_hh(train, test):
    """Return (train_static, test_static) DataFrames of X-only HH features,
    row-aligned with inputs. Also returns per-row HH / MEM."""
    tr = add_income(split_id(train))
    te = add_income(split_id(test))
    tr["IS_TRAIN"] = 1; te["IS_TRAIN"] = 0
    both = pd.concat([tr, te], ignore_index=True)

    g = both.groupby("HH")
    both["HH_N_ALL"] = g["ID"].transform("size")
    both["HH_N_TRAIN"] = g["IS_TRAIN"].transform("sum")
    both["HH_N_TEST"] = both["HH_N_ALL"] - both["HH_N_TRAIN"]
    both["HH_INCOME_SUM"] = g["TOTAL_INCOME"].transform("sum")
    both["HH_INCOME_MAX"] = g["TOTAL_INCOME"].transform("max")
    both["HH_AGE_MEAN"] = g["AGE"].transform("mean")
    both["HH_AGE_MAX"] = g["AGE"].transform("max")
    both["HH_N_WORK"] = g["CO_LAM_VIEC"].transform(lambda s: s.fillna(0).sum())
    both["HH_N_CHILD"] = g["AGE"].transform(lambda s: (s < 15).sum())
    both["HH_N_ELDER"] = g["AGE"].transform(lambda s: (s >= 60).sum())
    both["HH_ANY_SALARY"] = (both["HH_INCOME_SUM"] > 0).astype(int)
    both["OWN_INCOME_SHARE"] = np.where(
        both["HH_INCOME_SUM"] > 0, both["TOTAL_INCOME"] / both["HH_INCOME_SUM"], 0.0)
    both["IS_HEAD"] = (both["MEM"] == 1).astype(int)
    both["IS_SPOUSE"] = (both["MEM"] == 2).astype(int)
    both["MEM_FRAC"] = both["MEM"] / both["HH_N_ALL"]

    # head (MEM==1) features joined onto every member row
    head = both[both["MEM"] == 1].set_index("HH")
    head_cols = {
        "AGE": "HEAD_AGE", "GENDER": "HEAD_GENDER",
        "TOTAL_INCOME": "HEAD_INCOME", "CO_THE_BAO_HIEM": "HEAD_HAS_CARD",
        "LOAI_THE_1": "HEAD_LOAI_THE_1", "LOAI_THE_2": "HEAD_LOAI_THE_2",
        "MA_NGHE_NGHIEP": "HEAD_OCC", "CO_LAM_VIEC": "HEAD_WORKS",
        "MARRIED_STATUS": "HEAD_MARRIED",
    }
    for src, dst in head_cols.items():
        both[dst] = both["HH"].map(head[src])
    both["HAS_HEAD_ROW"] = both["HH"].isin(head.index).astype(int)

    static_cols = [
        "HH_N_ALL", "HH_N_TRAIN", "HH_N_TEST", "HH_INCOME_SUM", "HH_INCOME_MAX",
        "HH_AGE_MEAN", "HH_AGE_MAX", "HH_N_WORK", "HH_N_CHILD", "HH_N_ELDER",
        "HH_ANY_SALARY", "OWN_INCOME_SHARE", "IS_HEAD", "IS_SPOUSE", "MEM_FRAC",
        "HEAD_AGE", "HEAD_GENDER", "HEAD_INCOME", "HEAD_HAS_CARD",
        "HEAD_LOAI_THE_1", "HEAD_LOAI_THE_2", "HEAD_OCC", "HEAD_WORKS",
        "HEAD_MARRIED", "HAS_HEAD_ROW",
    ]
    tr_out = both[both["IS_TRAIN"] == 1].sort_index()
    te_out = both[both["IS_TRAIN"] == 0].sort_index()
    # concat preserved original order within each split (ignore_index + stable order)
    tr_static = tr_out[static_cols].reset_index(drop=True)
    te_static = te_out[static_cols].reset_index(drop=True)
    return tr_static, te_static, tr["HH"].reset_index(drop=True), tr["MEM"].reset_index(drop=True), te["HH"].reset_index(drop=True), te["MEM"].reset_index(drop=True)


def hh_y_features(ref_hh, ref_mem, ref_y, tgt_hh, tgt_mem, tgt_id=None, ref_id=None,
                  gmean=None, smooth=2.0):
    """y-based household features for target rows, from a reference set.

    If tgt_id/ref_id are given, a target row's own reference row (same ID)
    is excluded from its household stats (LOO mode). Otherwise the target
    rows are assumed absent from the reference set.
    Returns DataFrame aligned with target rows.
    """
    ref_y = np.asarray(ref_y, dtype=float)
    if gmean is None:
        gmean = float(ref_y.mean())
    # per-household member lists from the reference set
    hh_members = {}
    for i in range(len(ref_y)):
        hh_members.setdefault(ref_hh[i], []).append(
            (ref_mem[i], ref_y[i], ref_id[i] if ref_id is not None else None))
    n_t = len(tgt_hh)
    n_mates = np.zeros(n_t); mean = np.full(n_t, gmean); mean_s = np.zeros(n_t)
    sm = np.zeros(n_t); mx = np.zeros(n_t); npos = np.zeros(n_t)
    head = np.full(n_t, np.nan); spouse = np.full(n_t, np.nan)
    for r in range(n_t):
        members = hh_members.get(tgt_hh[r], [])
        if tgt_id is not None:
            members = [m for m in members if m[2] != tgt_id[r]]
        if not members:
            mean_s[r] = (0.0 + smooth * gmean) / (0.0 + smooth)
            continue
        ys = np.array([m[1] for m in members])
        n_mates[r] = len(ys)
        mean[r] = ys.mean()
        mean_s[r] = (ys.sum() + smooth * gmean) / (len(ys) + smooth)
        sm[r] = ys.sum(); mx[r] = ys.max(); npos[r] = (ys > 0).sum()
        for mem, yv, _ in members:
            if mem == 1: head[r] = yv
            elif mem == 2: spouse[r] = yv
    out = pd.DataFrame({
        "HHY_N_MATES": n_mates,
        "HHY_HAS_MATE": (n_mates > 0).astype(int),
        "HHY_MEAN": mean,
        "HHY_MEAN_SMOOTH": mean_s,
        "HHY_SUM": sm,
        "HHY_MAX": mx,
        "HHY_NPOS": npos,
        "HHY_SHARE_POS": np.where(n_mates > 0, npos / np.maximum(n_mates, 1), 0.0),
        "HHY_HEAD": np.where(np.isnan(head), -1.0, head),
        "HHY_HEAD_MISSING": np.isnan(head).astype(int),
        "HHY_SPOUSE": np.where(np.isnan(spouse), -1.0, spouse),
        "HHY_SPOUSE_MISSING": np.isnan(spouse).astype(int),
    })
    return out
