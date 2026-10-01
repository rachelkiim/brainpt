# -*- coding: utf-8 -*-
"""F4 cluster p: which null configuration yields .010 or .02."""
import io, zipfile, warnings
import numpy as np, pandas as pd
from scipy import stats

import os

OUT = os.path.dirname(os.path.abspath(__file__))  # results are written here
DATA = os.path.join(OUT, "..", "..", "data", "study1")
FIGURE_ZIP = os.path.join(DATA, "BrainPT_Figure_data.zip")

warnings.filterwarnings("ignore")
PKG = OUT

z = zipfile.ZipFile(FIGURE_ZIP)
raw = pd.ExcelFile(io.BytesIO(z.read("Figure_A2.xlsx"))).parse("01_raw")
freqs = raw.frequency_Hz.values
X = raw.drop(columns=["frequency_Hz"]).values
n = X.shape[1]; df = n - 1
t = X.mean(axis=1) / (X.std(axis=1, ddof=1) / np.sqrt(n))
phi = np.corrcoef(t[:-1], t[1:])[0, 1]
OBS = {"8.5-10.5": (5, 17.422), "11.5-12.0": (2, 5.282), "17.0-18.5": (4, 10.734)}
NSIM = 20000

def clus(mask, tv):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]: j += 1
            out.append((j - i + 1, float(np.abs(tv[i:j+1]).sum()))); i = j + 1
        else: i += 1
    return out

def sim(marginal, thresh, use_max, seed=1):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((NSIM, len(freqs)))
    x = np.empty_like(e); x[:, 0] = e[:, 0]; s = np.sqrt(1 - phi**2)
    for k in range(1, len(freqs)):
        x[:, k] = phi * x[:, k-1] + s * e[:, k]
    tt = x if marginal == "normal" else stats.t.ppf(
        np.clip(stats.norm.cdf(x), 1e-12, 1-1e-12), df)
    m = np.abs(tt) > thresh
    E, M = [], []
    for i in range(NSIM):
        cs = clus(m[i], tt[i])
        if not cs:
            if use_max: E.append(0); M.append(0.0)
            continue
        if use_max:
            E.append(max(c[0] for c in cs)); M.append(max(c[1] for c in cs))
        else:
            E += [c[0] for c in cs]; M += [c[1] for c in cs]
    return np.array(E), np.array(M)

TC_T = stats.t.ppf(0.975, df)      # 2.3646
TC_Z = stats.norm.ppf(0.975)       # 1.9600
TC_T1 = stats.t.ppf(0.95, df)      # one-sided, 1.8946

VAR = [
    ("A  t(7) marginal | |t|>2.365 | max statistic",      "t",      TC_T,  True),
    ("B  normal marginal | |z|>1.960 | max statistic",       "normal", TC_Z,  True),
    ("C  normal marginal | |z|>2.365 | max statistic",       "normal", TC_T,  True),
    ("D  t(7) marginal | |t|>1.960 | max statistic",       "t",      TC_Z,  True),
    ("E  t(7) marginal | |t|>2.365 | all clusters",   "t",      TC_T,  False),
    ("F  normal marginal | |z|>1.960 | all clusters",   "normal", TC_Z,  False),
    ("G  t(7) marginal | one-sided 1.895 | max statistic", "t",      TC_T1, True),
]

print("=" * 112)
print(f"F4 cluster p: sweep over null configurations  (phi={phi:.4f}, {NSIM:,} draws)")
print("Earlier draft: mass 17.42, corrected P=.02   |   decision memo: extent .039, mass .010")
print("Current manuscript reports configuration A: sign-flip P=.09, AR(1) P=.06 (see f4_cluster_correction.py)")
print("=" * 112)
hdr = f"\n{'configuration':<42}" + "".join(f"{k+' ext':>13}{k+' mass':>13}" for k in ["8.5-10.5"])
print(f"\n{'configuration':<42}{'ext p':>9}{'mass p':>9}   {'(11.5-12.0)':>20}   {'(17.0-18.5)':>20}")
rows = []
for name, marg, th, mx in VAR:
    E, M = sim(marg, th, mx)
    out = {}
    for k, (e0, m0) in OBS.items():
        out[k] = ((E >= e0).mean(), (M >= m0).mean())
    a = out["8.5-10.5"]; b = out["11.5-12.0"]; c = out["17.0-18.5"]
    print(f"{name:<42}{a[0]:>9.4f}{a[1]:>9.4f}   {b[0]:>9.4f}{b[1]:>10.4f}   {c[0]:>9.4f}{c[1]:>10.4f}")
    rows.append(dict(setting=name, ext_p_main=a[0], mass_p_main=a[1],
                     ext_p_2=b[0], mass_p_2=b[1], ext_p_3=c[0], mass_p_3=c[1]))

print("\n  Proximity to the target values (main cluster 8.5-10.5 Hz)")
for r in rows:
    hits = []
    if abs(r["mass_p_main"] - .010) < .004: hits.append("mass~.010 (memo)")
    if abs(r["mass_p_main"] - .02) < .005:  hits.append("mass~.02 (earlier draft)")
    if abs(r["ext_p_main"] - .039) < .006:  hits.append("extent~.039 (memo)")
    if abs(r["ext_p_main"] - .02) < .005:   hits.append("extent~.02 (earlier draft)")
    if hits: print(f"    {r['setting']}  →  {' · '.join(hits)}")
if not any(abs(r["mass_p_main"]-.010)<.004 or abs(r["mass_p_main"]-.02)<.005
           or abs(r["ext_p_main"]-.039)<.006 or abs(r["ext_p_main"]-.02)<.005 for r in rows):
    print("    no configuration yields .010 / .02 / .039")

pd.DataFrame(rows).to_csv(os.path.join(PKG, "f4_null_variants.csv"), index=False, encoding="utf-8-sig")
print(f"\nWritten: f4_null_variants.csv")
