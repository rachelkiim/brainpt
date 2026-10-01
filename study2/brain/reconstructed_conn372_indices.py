# -*- coding: utf-8 -*-
"""Reconstruct the 31 coupling indices and test whether the 372/15/10 accounting reproduces."""
import io, zipfile, itertools, warnings
import numpy as np, pandas as pd
from scipy import stats

import os

OUT = os.path.dirname(os.path.abspath(__file__))  # results are written here
DATA = os.path.join(OUT, "..", "..", "data", "study2")
ANALYSIS_ZIP = os.path.join(DATA, "BrainPT_analysis_data.zip")
BEHAVIOUR_ZIP = os.path.join(DATA, "analysis.zip")

warnings.filterwarnings("ignore")
PKG = OUT
z = zipfile.ZipFile(ANALYSIS_ZIP)

a = pd.read_csv(io.BytesIO(z.read("01_chronic_raw/chronic_conn_allpairs.csv")))
nm = pd.read_csv(io.BytesIO(zipfile.ZipFile(BEHAVIOUR_ZIP)
                            .read("analysis/analysis_results/ttest_gonogo_rt_change.csv")))
grp = {str(r["name"]): r["group"] for _, r in nm.iterrows()}
neural = set(pd.read_csv(os.path.join(DATA, "quantified", "study2_eeg_combined.csv"),
                         encoding="utf-8-sig").participant.unique())
a = a[a.initials.isin(neural)].copy()
print(f"participants {a.initials.nunique()} | pairs {a.pair.nunique()} | bands {a.band.nunique()} | conditions {a.condition.nunique()}")

CH = sorted({c for p in a.pair.unique() for c in p.split("-")})
def region(c):
    if c.startswith("AF") or c.startswith("F") and not c.startswith("FC"): return "F"
    if c.startswith("FC"): return "FC"
    return c[0]
REG = ["F", "FC", "O", "P", "T"]
REGPAIRS = [f"{x}-{y}" for i, x in enumerate(REG) for y in REG[i:]]
PRESPEC = {"F3-P7", "P8-F4", "AF3-P7", "P8-AF4"}   # pair names as stored in the data
print(f"regions {REG} -> {len(REGPAIRS)} region pairs")

a["c1"] = a.pair.str.split("-").str[0]; a["c2"] = a.pair.str.split("-").str[1]
a["r1"] = a.c1.map(region); a["r2"] = a.c2.map(region)
a["regpair"] = [f"{min(x,y,key=REG.index)}-{max(x,y,key=REG.index)}" for x, y in zip(a.r1, a.r2)]

def indices(d):
    """One (participant, visit, condition, band) group -> 31 indices"""
    out = {"global": d.wpli.mean()}
    for ch in CH:
        out[f"node_{ch}"] = d[(d.c1 == ch) | (d.c2 == ch)].wpli.mean()
    for rp in REGPAIRS:
        out[f"reg_{rp}"] = d[d.regpair == rp].wpli.mean()
    out["prespec_FP"] = d[d.pair.isin(PRESPEC)].wpli.mean()
    return out

rows = []
for (vis, ini, cond, band), d in a.groupby(["visit", "initials", "condition", "band"]):
    rows.append(dict(visit=vis, ini=ini, condition=cond, band=band, **indices(d)))
W = pd.DataFrame(rows)
IDX = [c for c in W.columns if c not in ("visit", "ini", "condition", "band")]
print(f"indices = {len(IDX)}   (manuscript 31 -> {'match' if len(IDX)==31 else 'MISMATCH'})")

res = []
for cond in ["baseline", "shortform", "brainpt", "task"]:
    for band in ["Theta", "Alpha", "Beta"]:
        sub = W[(W.condition == cond) & (W.band == band)]
        p1 = sub[sub.visit == "V1"].set_index("ini")
        p2 = sub[sub.visit == "V2"].set_index("ini")
        common = sorted(set(p1.index) & set(p2.index))
        for m in IDX:
            ch_ = (p2.loc[common, m] - p1.loc[common, m])
            g = pd.Series([grp.get(i) for i in common], index=common)
            e = ch_[g == "Experimental"].dropna(); c = ch_[g == "Control"].dropna()
            if len(e) < 2 or len(c) < 2: continue
            t, p = stats.ttest_ind(e, c, equal_var=True)
            res.append(dict(condition=cond, band=band, index=m, n_exp=len(e), n_con=len(c),
                            exp_change=e.mean(), con_change=c.mean(), t=t, p=p,
                            favors_exp=bool(e.mean() > c.mean())))
R = pd.DataFrame(res)
print(f"\ntotal tests = {len(R)}   (manuscript 372 -> {'match' if len(R)==372 else 'MISMATCH'})")

sig = R[R.p < .05]
print(f"significant = {len(sig)}   (change-score t; the manuscript's 15 is the ANCOVA count, see reconstructed_conn372_checks.py)")
print(f"expected by chance = {0.05*len(R):.1f}   (manuscript about 19)")

print(f"\n{'condition':<11}{'band':<7}{'sig':>5}{'favours exp':>12}/31")
for cond in ["baseline", "shortform", "brainpt", "task"]:
    for band in ["Theta", "Alpha", "Beta"]:
        cell = R[(R.condition == cond) & (R.band == band)]
        s = (cell.p < .05).sum(); f = cell.favors_exp.sum()
        mark = "  ←" if s >= 5 else ""
        print(f"{cond:<11}{band:<7}{s:>5}{f:>12}/{len(cell)}{mark}")

bb = R[(R.condition == "brainpt") & (R.band == "Beta")]
print(f"\nbeta x brainpt cell: {int((bb.p<.05).sum())} significant (change-score t; ANCOVA gives 10, manuscript 10), "
      f"favouring experimental {int(bb.favors_exp.sum())}/{len(bb)} (manuscript 31/31)")
R.to_csv(os.path.join(PKG, "conn372_rebuilt.csv"), index=False, encoding="utf-8-sig")
print(f"\nWritten: conn372_rebuilt.csv")
