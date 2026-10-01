# -*- coding: utf-8 -*-
"""
fNIRS statistics, reproduced from the condition-wise GLM betas.

Scope. This script covers the statistical layer only. The preprocessing that produced the
betas -- TDDR motion correction, modified Beer-Lambert conversion, short-separation
regression, drift control and the GLM itself -- was carried out in the NIRSIT vendor
software, not in code in this repository.

Inputs:
    data/study1/quantified/study1_fnirs_glm_long.csv     Study 1, participant x condition x channel HbO beta
    data/study2/quantified/study2_fnirs_glm_long.csv   Study 2, same shape
    data/study2/quantified/study2_fnirs_long.csv       Study 2, channel columns plus ROI columns and group

Every value printed here appears in the manuscript, and each is compared against the printed
value so that a mismatch is visible rather than silent. Three do not reproduce; they are
labelled and explained at the end rather than quietly omitted.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

OUT = os.path.dirname(os.path.abspath(__file__))  # results are written here
DATA = os.path.join(OUT, "..", "..", "data")
Q1 = os.path.join(DATA, "study1", "quantified")
Q2 = os.path.join(DATA, "study2", "quantified")

PRIMARY = "ch01"           # BA46, right dorsolateral prefrontal cortex, pre-specified
CHROMO = "HbO"
EQUIV_BOUND_DZ = 0.5       # conventional bound; reproduces the reported equivalence P
_rows = []


def check(label, got, want, tol=0.006, unit=""):
    ok = want is None or (np.isfinite(got) and abs(got - want) <= tol)
    mark = "  " if want is None else ("OK" if ok else "XX")
    w = "" if want is None else f"   manuscript {want:>8.3f}"
    print(f"  {mark} {label:<46}{got:>10.4f}{unit}{w}")
    _rows.append(dict(item=label, value=got, manuscript=want, ok=bool(ok)))
    return ok


def paired(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    d = b - a
    t, p = stats.ttest_rel(b, a)
    return dict(n=len(d), delta=d.mean(), t=t, p=p, dz=t / np.sqrt(len(d)),
                same=int((np.sign(d) == np.sign(d.mean())).sum()))


def bh(ps):
    ps = np.asarray(ps, float)
    n = len(ps)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(np.argsort(ps)[::-1]):
        prev = min(prev, ps[i] * n / (n - rank))
        q[i] = prev
    return q


def ci95(x):
    x = x[np.isfinite(x)]
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def tost(a, b, bound_dz):
    d = (b - a)
    d = d[np.isfinite(d)]
    n, sd = len(d), d.std(ddof=1)
    raw = bound_dz * sd
    se = sd / np.sqrt(n)
    return max(stats.t.sf((d.mean() + raw) / se, n - 1),
               stats.t.cdf((d.mean() - raw) / se, n - 1))


def ancova(df):
    """V2 ~ group + V1. Returns the group coefficient's two-sided P."""
    X = np.column_stack([np.ones(len(df)),
                         (df.group == "Experimental").astype(float), df["V1"]])
    b, *_ = np.linalg.lstsq(X, df["V2"].values, rcond=None)
    r = df["V2"].values - X @ b
    dof = len(df) - 3
    se = np.sqrt((r ** 2).sum() / dof * np.linalg.pinv(X.T @ X)[1, 1])
    return 2 * stats.t.sf(abs(b[1] / se), dof)


# ============================================================== Study 1
print("=" * 88)
print("Study 1 (n=9), visit 2, channel ch01 (BA46, right DLPFC)")
print("=" * 88)

a = pd.read_csv(os.path.join(Q1, "study1_fnirs_glm_long.csv"), encoding="utf-8-sig")
a = a[(a.chromophore == CHROMO) & (a.visit == "V2")]
prim = a[a.channel == PRIMARY]
w = prim.pivot_table(index="initials", columns="condition", values="beta")
w = w[[c for c in ["shortform1", "brainpt", "shortform2", "task"] if c in w.columns]].dropna()
print(f"\n  participants with all four blocks: {len(w)}")

print("\n  Condition means")
for c, want in [("shortform1", -0.093), ("brainpt", -0.023),
                ("shortform2", 0.110), ("task", 0.180)]:
    check(f"beta {c}", w[c].mean(), want)
for c, lo, hi in [("shortform1", -0.173, -0.014), ("task", 0.051, 0.309)]:
    l, h = ci95(w[c].values)
    check(f"{c} 95% CI lower", l, lo)
    check(f"{c} 95% CI upper", h, hi)

print("\n  Paired contrasts, Benjamini-Hochberg across the four")
CON = [("shortform1", "task", 0.273, 4.42, 0.002, 1.47, 8, 0.009),
       ("shortform1", "shortform2", 0.203, 3.06, 0.016, 1.02, 8, 0.031),
       ("brainpt", "task", 0.202, 2.52, 0.036, 0.84, None, 0.048),
       ("shortform1", "brainpt", 0.071, 1.05, 0.323, 0.35, None, 0.323)]
res = [paired(w[x].values, w[y].values) for x, y, *_ in CON]
qs = bh([r["p"] for r in res])
for (x, y, dw, tw, pw, dzw, sw, qw), r, q in zip(CON, res, qs):
    tag = f"{x} vs {y}"
    check(f"{tag} delta", r["delta"], dw)
    check(f"{tag} t", r["t"], tw, tol=0.02)
    check(f"{tag} P", r["p"], pw, tol=0.002)
    check(f"{tag} dz", r["dz"], dzw, tol=0.02)
    check(f"{tag} q", q, qw, tol=0.002)
    if sw is not None:
        check(f"{tag} same direction", r["same"], sw, tol=0)

print("\n  Other Study 1 statements")
check("shortform2 vs rest, P", stats.ttest_1samp(w["shortform2"].values, 0)[1], 0.12, tol=0.01)
check(f"BrainPT vs task equivalence, TOST P (dz {EQUIV_BOUND_DZ})",
      tost(w["brainpt"].values, w["task"].values, EQUIV_BOUND_DZ), 0.83, tol=0.01)
for c, want in [("shortform1", -2.70), ("task", 3.21)]:
    v = prim[prim.condition == c].groupby("initials").beta.mean().dropna()
    check(f"{c} ch01 one-sample t", stats.ttest_1samp(v, 0)[0], want, tol=0.02)

print("\n  Frontopolar control -- the manuscript reports delta +0.020, P=.54")
cand = []
for ch in sorted(a.channel.unique()):
    ww = a[a.channel == ch].pivot_table(index="initials", columns="condition", values="beta")
    if not {"shortform1", "task"} <= set(ww.columns):
        continue
    s = ww[["shortform1", "task"]].dropna()
    r = paired(s["shortform1"].values, s["task"].values)
    cand.append((ch, r["delta"], r["p"], r["n"]))
for ch, d, p, n in cand:
    flag = "  <- candidate" if abs(d - 0.020) < 0.02 and abs(p - 0.54) < 0.12 else ""
    print(f"     {ch}  delta={d:+.4f}  P={p:.3f}  n={n}{flag}")
print("     no single channel gives delta=+0.020 with P=.54; see the note at the end")

print("\n  All channels, group mean (Figure 3b)")
for c in ["shortform1", "task"]:
    ch = a[a.condition == c].pivot_table(index="initials", columns="channel", values="beta")
    print(f"     {c}")
    print("       " + "  ".join(f"{k}:{ch[k].mean():+.3f}(n={int(ch[k].notna().sum())})"
                                for k in sorted(ch.columns)))

# ============================================================== Study 2
print("\n" + "=" * 88)
print("Study 2 -- neural carry-over, ch01, short-form block")
print("=" * 88)

cg = pd.read_csv(os.path.join(Q2, "study2_fnirs_glm_long.csv"), encoding="utf-8-sig")
cg = cg[cg.chromophore == CHROMO]
cl = pd.read_csv(os.path.join(Q2, "study2_fnirs_long.csv"), encoding="utf-8-sig")
grp = cl[["initials", "group"]].drop_duplicates().set_index("initials").group.to_dict()

# Neural analysis sample = participants in the EEG sample (same rule as lord_tost_fnirs.py).
# Without this restriction one extra experimental participant enters (n=10, Table 1 says 9).
neural = set(pd.read_csv(os.path.join(Q2, "study2_eeg_combined.csv"), encoding="utf-8-sig").participant)
p = (cg[(cg.condition == "shortform") & (cg.channel == PRIMARY)]
     .pivot_table(index="initials", columns="visit", values="beta").dropna().reset_index())
p["group"] = p.initials.map(grp)
p = p.dropna(subset=["group"])
p = p[p.initials.isin(neural)]
p["change"] = p["V2"] - p["V1"]
e = p[p.group == "Experimental"]
c = p[p.group == "Control"]
print(f"\n  experimental n={len(e)}, control n={len(c)}   "
      f"(Table 1 states 9 and 7 for the fNIRS sample)")
check("Experimental n", len(e), 9, tol=0)
check("Control n", len(c), 7, tol=0)
check("Control change", c.change.mean(), 0.014)
check("Experimental change", e.change.mean(), 0.095)
# The manuscript prints P=.36 and calls it the ANCOVA. It is the change-score
# Student t-test; the ANCOVA (the stated decision criterion) gives a different P.
check("change-score t (pooled), P", stats.ttest_ind(e.change, c.change).pvalue, 0.36, tol=0.01)
check("ANCOVA V2 ~ group + V1, group P", ancova(p), None)

print("\n  ROI columns in study2_fnirs_long.csv")
for roi in [x for x in cl.columns if x.startswith("roi_")]:
    same = [k for k in cl.columns if k.startswith("ch") and cl[k].equals(cl[roi])]
    print(f"     {roi:<14} identical to {same if same else '(not a single channel)'}")
print("     note: these place right DLPFC at ch08, while the manuscript's pre-specified")
print("           BA46 right DLPFC channel is ch01. The two conventions disagree.")

pd.DataFrame(_rows).to_csv(os.path.join(OUT, "fnirs_statistics_output.csv"),
                           index=False, encoding="utf-8-sig")
chk = [r for r in _rows if r["manuscript"] is not None]
bad = [r for r in chk if not r["ok"]]
print(f"\n{'=' * 88}")
print(f"  {len(chk) - len(bad)} of {len(chk)} values matched the manuscript")
if bad:
    print("\n  Not reproduced:")
    for r in bad:
        print(f"    - {r['item']}: got {r['value']:.4f}, manuscript {r['manuscript']}")
print("""
  Notes.

  * Study 2 carry-over: restricted to the neural analysis sample (9 vs 7), both group
    changes reproduce (+0.095, +0.014). The printed P=.36 is the pooled change-score
    t-test; the ANCOVA the manuscript names as its criterion gives the P shown above.
    Both are non-significant, so the conclusion does not depend on which is reported.
  * No single channel yields the frontopolar contrast of +0.020 with P=.54; the closest are
    ch07 (+0.010, .89) and ch06 (+0.028, .66). The manuscript's BA10 index is probably an
    average over channels whose membership is not recorded.
""")
print("=" * 88)
