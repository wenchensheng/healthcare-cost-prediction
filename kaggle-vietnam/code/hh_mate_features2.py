"""Mate features v2: v1 outputs + relation splits (adult/child by age and by
MEM), multiple smoothing levels for share-pos / mean-y, log transforms of
utilization sums, head/spouse utilization totals."""
import numpy as np, pandas as pd

def mate_features2(ref, tgt_hh, tgt_mem, tgt_id=None, gmean=1.1919, base_rate=0.18336):
    """ref needs columns: HH, MEM, ID, AGE, y, FRE_NGOAITRU, FRE_NOITRU,
    COST_NGOAITRU, COST_NOITRU."""
    members = {}
    for hh, mem, rid, age, yv, fo, fi, co, ci in zip(
            ref["HH"], ref["MEM"], ref["ID"], ref["AGE"], ref["y"],
            ref["FRE_NGOAITRU"], ref["FRE_NOITRU"],
            ref["COST_NGOAITRU"], ref["COST_NOITRU"]):
        members.setdefault(hh, []).append(
            (int(mem), rid, float(age), float(yv), float(fo), float(fi), float(co), float(ci)))
    n_t = len(tgt_hh)
    F = {}
    def col(name, fill=0.0):
        F[name] = np.full(n_t, fill); return F[name]
    n_mates = col("HHY_N_MATES")
    mean = col("HHY_MEAN", gmean); mean_s = col("HHY_MEAN_SMOOTH", gmean)
    sm = col("HHY_SUM"); mx = col("HHY_MAX"); npos = col("HHY_NPOS"); share = col("HHY_SHARE_POS")
    share_s1 = col("HHY_SHARE_POS_S1", base_rate); share_s5 = col("HHY_SHARE_POS_S5", base_rate)
    mean_s1 = col("HHY_MEAN_S1", gmean); mean_s5 = col("HHY_MEAN_S5", gmean)
    lcmean = col("HHC_LMEAN"); lcmax = col("HHC_LMAX"); lcsum = col("HHC_LSUM")
    fo_mean = col("HHF_FREOUT_MEAN"); fo_max = col("HHF_FREOUT_MAX"); fo_sum = col("HHF_FREOUT_SUM")
    fo_lsum = col("HHF_FREOUT_LSUM"); fi_mean = col("HHF_FREIN_MEAN"); any_fi = col("HHF_ANY_FREIN")
    lcpv = col("HHF_LCPV"); fre_tot_l = col("HHF_FRE_TOT_LSUM")
    head_y = col("HHY_HEAD", -1.0); spouse_y = col("HHY_SPOUSE", -1.0)
    head_miss = col("HHY_HEAD_MISSING", 1.0); spouse_miss = col("HHY_SPOUSE_MISSING", 1.0)
    head_fo = col("HHF_HEAD_FREOUT", -1.0); head_fi = col("HHF_HEAD_FREIN", -1.0)
    spouse_fo = col("HHF_SPOUSE_FREOUT", -1.0)
    ad_mean = col("HHY_MEAN_ADULT", gmean); ad_share = col("HHY_SHARE_ADULT", base_rate); ad_n = col("HHY_N_ADULT")
    ch_mean = col("HHY_MEAN_CHILD", gmean); ch_share = col("HHY_SHARE_CHILD", base_rate); ch_n = col("HHY_N_CHILD_M")
    mem12_mean = col("HHY_MEAN_MEM12", gmean); mem12_share = col("HHY_SHARE_MEM12", base_rate)
    has_mate = col("HHY_HAS_MATE")
    for r in range(n_t):
        ms = members.get(tgt_hh[r], [])
        if tgt_id is not None:
            ms = [m for m in ms if m[1] != tgt_id[r]]
        if not ms:
            continue
        ys = np.array([m[3] for m in ms]); costs = np.expm1(ys)
        fo = np.array([m[4] for m in ms]); fi = np.array([m[5] for m in ms])
        tot_cost = np.array([m[6] + m[7] for m in ms])
        n = len(ys); has_mate[r] = 1; n_mates[r] = n
        mean[r] = ys.mean(); sm[r] = ys.sum(); mx[r] = ys.max()
        npos[r] = (ys > 0).sum(); share[r] = (ys > 0).mean()
        mean_s[r] = (ys.sum() + 2 * gmean) / (n + 2)
        mean_s1[r] = (ys.sum() + 1 * gmean) / (n + 1)
        mean_s5[r] = (ys.sum() + 5 * gmean) / (n + 5)
        share_s1[r] = ((ys > 0).sum() + 1 * base_rate) / (n + 1)
        share_s5[r] = ((ys > 0).sum() + 5 * base_rate) / (n + 5)
        lcmean[r] = np.log1p(costs.mean()); lcmax[r] = np.log1p(costs.max()); lcsum[r] = np.log1p(costs.sum())
        fo_mean[r] = fo.mean(); fo_max[r] = fo.max(); fo_sum[r] = fo.sum()
        fo_lsum[r] = np.log1p(fo.sum()); fi_mean[r] = fi.mean(); any_fi[r] = float((fi > 0).any())
        tf = (fo + fi).sum(); fre_tot_l[r] = np.log1p(tf)
        lcpv[r] = np.log1p(tot_cost.sum() / tf) if tf > 0 else 0.0
        ages = np.array([m[2] for m in ms]); mems = np.array([m[0] for m in ms])
        ad = ages >= 18; chd = ~ad
        if ad.any():
            ad_n[r] = ad.sum(); ad_mean[r] = ys[ad].mean(); ad_share[r] = (ys[ad] > 0).mean()
        if chd.any():
            ch_n[r] = chd.sum(); ch_mean[r] = ys[chd].mean(); ch_share[r] = (ys[chd] > 0).mean()
        m12 = mems <= 2
        if m12.any():
            mem12_mean[r] = ys[m12].mean(); mem12_share[r] = (ys[m12] > 0).mean()
        for mem, rid, age, yv, fov, fiv, cov, civ in ms:
            if mem == 1:
                head_y[r] = yv; head_miss[r] = 0; head_fo[r] = fov; head_fi[r] = fiv
            elif mem == 2:
                spouse_y[r] = yv; spouse_miss[r] = 0; spouse_fo[r] = fov
    return pd.DataFrame(F)
