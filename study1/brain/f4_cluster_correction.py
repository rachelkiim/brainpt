# -*- coding: utf-8 -*-
"""F4 spectrum cluster correction, recomputed from Figure_A2.xlsx. No EDF or ICA needed."""
import io, zipfile, itertools, warnings
import numpy as np, pandas as pd
from scipy import stats

import os

OUT = os.path.dirname(os.path.abspath(__file__))  # results are written here
DATA = os.path.join(OUT, "..", "..", "data", "study1")
FIGURE_ZIP = os.path.join(DATA, "BrainPT_Figure_data.zip")

warnings.filterwarnings("ignore")
PKG = OUT

z = zipfile.ZipFile(FIGURE_ZIP)
xl = pd.ExcelFile(io.BytesIO(z.read("Figure_A2.xlsx")))
raw = xl.parse("01_raw")
ana = xl.parse("02_analysis")

freqs = raw.frequency_Hz.values
X = raw.drop(columns=["frequency_Hz"]).values      # (59 bins, 8 participants)
n = X.shape[1]
df = n - 1
TCRIT = stats.t.ppf(0.975, df)

print("=" * 100); print("1. Recover the t series and check the data"); print("=" * 100)
t_raw = X.mean(axis=1) / (X.std(axis=1, ddof=1) / np.sqrt(n))
t_sheet = ana.mean_dB.values / ana.sem_dB.values
print(f"  participants {n} | bins {len(freqs)} | df={df} | t_crit={TCRIT:.4f}")
print(f"  t from raw vs t from mean/SEM sheet, max difference: {np.abs(t_raw-t_sheet).max():.6f}")
t = t_raw

def lag1(v):
    return np.corrcoef(v[:-1], v[1:])[0, 1]
r1 = lag1(t)
print(f"  lag-1 autocorrelation: {r1:.4f}   (memo says 0.811 -> {'match' if abs(r1-0.811)<0.01 else 'MISMATCH'})")

p_unc = 2 * stats.t.sf(np.abs(t), df)
sig = p_unc < 0.05
print(f"  uncorrected significant bins: {sig.sum()} / {len(freqs)}   (expected 11 -> {'match' if sig.sum()==11 else 'MISMATCH'})")
print(f"  agrees with the sheet's significant column: {bool((sig == ana.significant.values).all())}")

def clusters(mask, tv):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append(dict(lo=freqs[i], hi=freqs[j], extent=j - i + 1,
                            mass=float(np.abs(tv[i:j+1]).sum()),
                            min_p=float(p_unc[i:j+1].min())))
            i = j + 1
        else:
            i += 1
    return out

obs = clusters(sig, t)
print("\n  Observed clusters")
for c in obs:
    print(f"    {c['lo']:>5.1f}–{c['hi']:<5.1f} Hz  extent={c['extent']}  "
          f"mass={c['mass']:.3f}  min uncorrected p={c['min_p']:.4f}")

# ---------------- AR(1) null distribution ----------------
def ar1_null(phi, nbin, nsim, seed):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((nsim, nbin))
    x = np.empty_like(e)
    x[:, 0] = e[:, 0]
    s = np.sqrt(1 - phi ** 2)
    for k in range(1, nbin):
        x[:, k] = phi * x[:, k-1] + s * e[:, k]
    # Map normal marginals onto t(df) marginals, preserving the autocorrelation
    tt = stats.t.ppf(np.clip(stats.norm.cdf(x), 1e-12, 1-1e-12), df)
    m = np.abs(tt) > TCRIT
    max_ext = np.zeros(nsim); max_mass = np.zeros(nsim)
    for i in range(nsim):
        cs = clusters(m[i], tt[i])
        if cs:
            max_ext[i] = max(c["extent"] for c in cs)
            max_mass[i] = max(c["mass"] for c in cs)
    return max_ext, max_mass

NSIM = 20000
print("\n" + "=" * 100); print(f"2. AR(1) null, phi={r1:.3f}, {NSIM:,} draws x 3 seeds"); print("=" * 100)
rows = []
for seed in (1, 2, 3):
    me, mm = ar1_null(r1, len(freqs), NSIM, seed)
    for c in obs:
        rows.append(dict(seed=seed, band=f"{c['lo']}-{c['hi']}", extent=c["extent"], mass=c["mass"],
                         p_extent=float((me >= c["extent"]).mean()),
                         p_mass=float((mm >= c["mass"]).mean())))
R = pd.DataFrame(rows)
print(f"\n{'band':<14}{'extent':>7}{'mass':>9}   " + "".join(f"{'p_ext s'+str(s):>11}" for s in (1,2,3))
      + "   " + "".join(f"{'p_mass s'+str(s):>12}" for s in (1,2,3)))
for c in obs:
    key = f"{c['lo']}-{c['hi']}"
    g = R[R.band == key]
    print(f"{key+' Hz':<14}{c['extent']:>7}{c['mass']:>9.3f}   "
          + "".join(f"{v:>11.4f}" for v in g.p_extent) + "   "
          + "".join(f"{v:>12.4f}" for v in g.p_mass))
print("\n  Seed-to-seed spread")
for c in obs:
    g = R[R.band == f"{c['lo']}-{c['hi']}"]
    print(f"    {c['lo']}-{c['hi']} Hz   extent {g.p_extent.max()-g.p_extent.min():.4f}   "
          f"mass {g.p_mass.max()-g.p_mass.min():.4f}")

# ---------------- Exhaustive sign-flip test (secondary) ----------------
print("\n" + "=" * 100); print(f"3. Secondary test: exhaustive sign-flip, 2^{n} = {2**n} relabelings (keeps the real covariance)"); print("=" * 100)
signs = np.array(list(itertools.product([1, -1], repeat=n)))
Xs = X[None, :, :] * signs[:, None, :]
tm = Xs.mean(axis=2) / (Xs.std(axis=2, ddof=1) / np.sqrt(n))
mask = np.abs(tm) > TCRIT
me = np.zeros(len(signs)); mm = np.zeros(len(signs))
for i in range(len(signs)):
    cs = clusters(mask[i], tm[i])
    if cs:
        me[i] = max(c["extent"] for c in cs); mm[i] = max(c["mass"] for c in cs)
print(f"\n{'band':<14}{'extent':>7}{'mass':>9}{'p_extent':>11}{'p_mass':>11}")
for c in obs:
    label = "{}-{} Hz".format(c["lo"], c["hi"])
    pe = (me >= c["extent"]).mean()
    pm = (mm >= c["mass"]).mean()
    print(f"{label:<14}{c['extent']:>7}{c['mass']:>9.3f}{pe:>11.4f}{pm:>11.4f}")

R.to_csv(os.path.join(PKG, "f4_cluster_ar1.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(obs).to_csv(os.path.join(PKG, "f4_clusters_observed.csv"), index=False, encoding="utf-8-sig")
print(f"\nWritten: f4_cluster_ar1.csv, f4_clusters_observed.csv")
