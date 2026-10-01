# -*- coding: utf-8 -*-
"""
Using the reconstructed 372 definition: (1) check the internal test behind permutation
P=.052, (2) check the effect sizes d=0.60-0.96, (3) check the 91-pair paragraph.

Note: the original script does not exist. This file implements a definition worked
backwards from the reported numbers - it is a reconstruction, not a recovery.
"""
import io, zipfile, warnings
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

CH = sorted({c for p in a.pair.unique() for c in p.split("-")})
REG = ["F", "FC", "O", "P", "T"]
def region(c):
    if c.startswith("FC"): return "FC"
    if c.startswith("AF") or c.startswith("F"): return "F"
    return c[0]
REGPAIRS = [f"{x}-{y}" for i, x in enumerate(REG) for y in REG[i:]]
PRESPEC = ["F3-P7", "P8-F4", "AF3-P7", "P8-AF4"]   # pair names as stored in the data
a["c1"] = a.pair.str.split("-").str[0]; a["c2"] = a.pair.str.split("-").str[1]
a["regpair"] = [f"{min(region(x),region(y),key=REG.index)}-{max(region(x),region(y),key=REG.index)}"
                for x, y in zip(a.c1, a.c2)]

rows = []
for (vis, ini, cond, band), d in a.groupby(["visit", "initials", "condition", "band"]):
    r = {"visit": vis, "ini": ini, "condition": cond, "band": band, "global": d.wpli.mean()}
    for ch in CH: r[f"node_{ch}"] = d[(d.c1 == ch) | (d.c2 == ch)].wpli.mean()
    for rp in REGPAIRS: r[f"reg_{rp}"] = d[d.regpair == rp].wpli.mean()
    r["prespec_FP"] = d[d.pair.isin(PRESPEC)].wpli.mean()
    rows.append(r)
W = pd.DataFrame(rows)
IDX = [c for c in W.columns if c not in ("visit", "ini", "condition", "band")]
CONDS = ["baseline", "shortform", "brainpt", "task"]; BANDS = ["Theta", "Alpha", "Beta"]

# Participant and group vectors, shared by every cell
inis = sorted(set(W[W.visit == "V1"].ini) & set(W[W.visit == "V2"].ini))
G0 = np.array([grp.get(i) == "Experimental" for i in inis], float)
n = len(inis)
print(f"participants {n} (experimental {int(G0.sum())} / control {int(n-G0.sum())})")

# Load y1, y2 for each (cell, index)
Y1, Y2, KEY = [], [], []
for cond in CONDS:
    for band in BANDS:
        s = W[(W.condition == cond) & (W.band == band)]
        p1 = s[s.visit == "V1"].set_index("ini").reindex(inis)
        p2 = s[s.visit == "V2"].set_index("ini").reindex(inis)
        for m in IDX:
            Y1.append(p1[m].values.astype(float)); Y2.append(p2[m].values.astype(float))
            KEY.append((cond, band, m))
Y1 = np.array(Y1); Y2 = np.array(Y2)          # (372, n)
print(f"test units {Y1.shape[0]} | missing {int(np.isnan(Y1).sum()+np.isnan(Y2).sum())}")

# ---- Frisch-Waugh-Lovell residualisation: M = I - P[1,y1] ----
def ancova_t(Y1, Y2, g):
    """t for the ANCOVA group coefficient, per row. g is (n,) or (P,n). Missing dropped per row."""
    K = Y1.shape[0]
    g = np.atleast_2d(g).astype(float)
    out = np.full((K, g.shape[0]), np.nan)
    for k in range(K):
        y1, y2 = Y1[k], Y2[k]
        ok = np.isfinite(y1) & np.isfinite(y2)
        m = int(ok.sum())
        if m < 5:
            continue
        y1o, y2o, go = y1[ok], y2[ok], g[:, ok]
        X = np.column_stack([np.ones(m), y1o])
        M = np.eye(m) - X @ np.linalg.pinv(X)
        yt = M @ y2o
        SSY = float(y2o @ yt)
        gm = go @ M
        gMg = np.einsum("ij,ij->i", gm, go)
        gy = go @ yt
        with np.errstate(divide="ignore", invalid="ignore"):
            RSS = SSY - gy**2 / gMg
            t = np.where((gMg > 1e-12) & (RSS > 1e-12),
                         gy / np.sqrt(np.maximum(gMg, 1e-30))
                         * np.sqrt((m - 3) / np.maximum(RSS, 1e-30)), np.nan)
        out[k] = t
    return out                          # (K, P)

NEFF = np.array([int((np.isfinite(Y1[k]) & np.isfinite(Y2[k])).sum()) for k in range(len(KEY))])
T_obs = ancova_t(Y1, Y2, G0)[:, 0]
P_obs = 2 * stats.t.sf(np.abs(T_obs), NEFF - 3)
res = pd.DataFrame(KEY, columns=["condition", "band", "index"])
res["n"] = NEFF; res["t"] = T_obs; res["p"] = P_obs
CH_ = Y2 - Y1
e, c = CH_[:, G0 == 1], CH_[:, G0 == 0]
res["exp_change"] = np.nanmean(e, 1); res["con_change"] = np.nanmean(c, 1)
res["favors_exp"] = res.exp_change > res.con_change
ne = np.isfinite(e).sum(1); nc = np.isfinite(c).sum(1)
sp = np.sqrt(((ne-1)*np.nanvar(e, 1, ddof=1) + (nc-1)*np.nanvar(c, 1, ddof=1)) / (ne+nc-2))
res["d_change"] = (np.nanmean(e, 1) - np.nanmean(c, 1)) / sp
sig = res[res.p < .05]
print(f"significant {len(sig)} | beta x brainpt {int(((sig.condition=='brainpt')&(sig.band=='Beta')).sum())}")

# ================= (2) effect sizes =================
print("\n" + "=" * 100); print("(2) Effect sizes of the 10 significant indices in beta x BrainPT"); print("=" * 100)
bb = res[(res.condition == "brainpt") & (res.band == "Beta") & (res.p < .05)].copy()
bb = bb.sort_values("d_change", ascending=False)
print(f"\n{'index':<14}{'t(ANCOVA)':>11}{'p':>9}{'d(change,pooled SD)':>20}")
for _, r in bb.iterrows():
    print(f"{r['index']:<14}{r.t:>11.3f}{r.p:>9.4f}{r.d_change:>18.3f}")
print(f"\n  d range: {bb.d_change.min():.2f} - {bb.d_change.max():.2f}   (manuscript 0.60-0.96)")
match = abs(bb.d_change.min()-0.60) < .06 and abs(bb.d_change.max()-0.96) < .06
print(f"  verdict: {'match' if match else 'MISMATCH'}")

# ================= (1) permutation test =================
print("\n" + "=" * 100); print("(1) Permutation test: group labels shuffled 3000 times"); print("=" * 100)
rng = np.random.default_rng(20260924)
NPERM = 3000
Gp = np.array([rng.permutation(G0) for _ in range(NPERM)])     # (3000, n)

cellidx = {}
for i, (cond, band, m) in enumerate(KEY):
    cellidx.setdefault((cond, band), []).append(i)
BB = cellidx[("brainpt", "Beta")]

def perm_counts(method):
    if method == "ancova":
        T = ancova_t(Y1, Y2, Gp)                                # (372, 3000)
        P = 2 * stats.t.sf(np.abs(T), (NEFF - 3)[:, None])
    else:
        ch = Y2 - Y1
        V = np.nan_to_num(ch, nan=0.0)
        Msk = np.isfinite(ch).astype(float)                      # (K, n)
        G = Gp.astype(float)                                     # (P, n)
        ne = Msk @ G.T; nc = Msk @ (1 - G).T                     # (K, P)
        se_ = (V * Msk) @ G.T; sc_ = (V * Msk) @ (1 - G).T
        me = se_ / ne; mc = sc_ / nc
        qe = (V**2 * Msk) @ G.T; qc = (V**2 * Msk) @ (1 - G).T
        ve = (qe - ne * me**2) / np.maximum(ne - 1, 1)
        vc = (qc - nc * mc**2) / np.maximum(nc - 1, 1)
        dfree = ne + nc - 2
        sp2 = ((ne - 1) * ve + (nc - 1) * vc) / np.maximum(dfree, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (me - mc) / np.sqrt(sp2 * (1 / ne + 1 / nc))
            P = 2 * stats.t.sf(np.abs(tt), dfree)
    S = np.nan_to_num(P, nan=1.0) < .05
    bb_cnt = S[BB, :].sum(0)
    mx = np.array([max(S[ix, j].sum() for ix in cellidx.values()) for j in range(NPERM)])
    return bb_cnt, mx

for method, label in [("ancova", "ANCOVA (same as observed)"), ("change", "change-score t (differs from observed)")]:
    bbc, mx = perm_counts(method)
    p_cell = (bbc >= 10).mean()
    p_max = (mx >= 10).mean()
    print(f"\n  {label}")
    print(f"    P(that cell >= 10)                 = {p_cell:.4f}")
    print(f"    P(max over the 12 cells >= 10)     = {p_max:.4f}   <- corrects for cell selection")
    print(f"    null means: that cell {bbc.mean():.2f}, max cell {mx.mean():.2f}")

res.to_csv(os.path.join(PKG, "reconstructed_conn372_results.csv"), index=False, encoding="utf-8-sig")

# ================= (3) 91-pair channel level =================
print("\n" + "=" * 100); print("(3) 91 pairs, channel level: beta x BrainPT"); print("=" * 100)
s = a[(a.band == "Beta") & (a.condition == "brainpt")]
p1 = s[s.visit == "V1"].pivot_table(index="initials", columns="pair", values="wpli")
p2 = s[s.visit == "V2"].pivot_table(index="initials", columns="pair", values="wpli")
common = sorted(set(p1.index) & set(p2.index))
gg = np.array([grp.get(i) == "Experimental" for i in common], float)
pairs = sorted(set(p1.columns) & set(p2.columns))
A1 = p1.loc[common, pairs].values.T; A2 = p2.loc[common, pairs].values.T
tt = ancova_t(A1, A2, gg)[:, 0]
pp = 2 * stats.t.sf(np.abs(tt), len(common) - 3)
chp = A2 - A1
ee, cc = chp[:, gg == 1], chp[:, gg == 0]
spp = np.sqrt(((ee.shape[1]-1)*ee.var(1, ddof=1) + (cc.shape[1]-1)*cc.var(1, ddof=1)) / (ee.shape[1]+cc.shape[1]-2))
dd = (ee.mean(1) - cc.mean(1)) / spp
P91 = pd.DataFrame(dict(pair=pairs, t=tt, p=pp, d=dd, favors_exp=ee.mean(1) > cc.mean(1)))
S91 = P91[P91.p < .05]
print(f"  pairs {len(P91)} | participants {len(common)}")
print(f"  significant pairs {len(S91)} (manuscript 17), favouring experimental {int(S91.favors_exp.sum())} (manuscript 16)")
print(f"  expected by chance {0.05*len(P91):.1f} (manuscript 4.6)")
print(f"  d range {S91.d.min():.2f} - {S91.d.max():.2f} (manuscript -0.14 to 1.69)")
pre = P91[P91.pair.isin(PRESPEC)]
print(f"  the four pre-specified pairs, p: " + " · ".join(f"{r.pair} {r.p:.2f}" for _, r in pre.iterrows())
      + f"   (manuscript .09-.39, none among the 17 -> {'confirmed' if not pre.pair.isin(S91.pair).any() else 'MISMATCH'})")
print(f"  top significant pairs: {sorted(S91.sort_values('p').pair.head(12))}")
P91.to_csv(os.path.join(PKG, "reconstructed_conn91_beta_brainpt.csv"), index=False, encoding="utf-8-sig")
print("\nWritten: reconstructed_conn372_results.csv, reconstructed_conn91_beta_brainpt.csv")
