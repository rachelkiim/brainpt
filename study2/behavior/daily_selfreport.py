# -*- coding: utf-8 -*-
"""Study 2 daily self-report (take-home survey): item means per group, behavioural sample
(13 experimental + 8 control), compared with the manuscript."""
import os, io, zipfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "study2")
Z = zipfile.ZipFile(os.path.join(DATA, "analysis.zip"))
rd = lambda n: pd.read_csv(io.BytesIO(Z.read(n)))

# Behavioural sample: experimental = paired short-form analysis, control = pre-test table
exp = rd("analysis/analysis_results/paired_exp_shortform.csv")
pre = rd("analysis/analysis_results/ttest_pre_shortform.csv")
EXP = sorted(int(x) for x in exp.participant_num.unique())
CON = sorted(int(x) for x in pre[pre.group.astype(str).str.lower().str.startswith("control")].participant_num.unique())
GRP = {n: "experimental" for n in EXP}
GRP.update({n: "control" for n in CON})
print(f"experimental {len(EXP)}: {EXP}")
print(f"control      {len(CON)}: {CON}")

th = pd.read_excel(io.BytesIO(Z.read("analysis/googleform/BRAIN-PT takehome.xlsx")), engine="openpyxl")
idcol = [c for c in th.columns if "성함" in c][0]
items = [c for c in th.columns if c.startswith("오늘 하루")]
LABEL = ["sustained_attention", "emotional_stability", "impulse_control",
         "plan_adherence", "overall_satisfaction"]
# Participants are labelled S2-nn; nn is the behavioural participant number
th["num"] = pd.to_numeric(th[idcol].astype(str).str.strip().str.extract(r"^S2-(\d+)$")[0], errors="coerce")

before_n = th.num.nunique()
th["group"] = th.num.map(GRP)
dropped = sorted(set(th[th.group.isna()].num.dropna().astype(int)))
th = th[th.group.notna()].copy()
print(f"\nrespondents {before_n} -> behavioural sample {th.num.nunique()} (excluded {dropped})")
print(f"responses {len(th)}: experimental {th[th.group == 'experimental'].num.nunique()} participants "
      f"({(th.group == 'experimental').sum()}), control {th[th.group == 'control'].num.nunique()} "
      f"({(th.group == 'control').sum()})")

rows = []
for lab, c in zip(LABEL, items):
    v = pd.to_numeric(th[c], errors="coerce")
    r = {"item": lab}
    for g in ("experimental", "control"):
        vv = v[th.group == g].dropna()
        r[f"{g}_n_resp"] = len(vv)
        r[f"{g}_n_part"] = th.loc[vv.index, "num"].nunique()
        r[f"{g}_mean"] = vv.mean()
        r[f"{g}_sd"] = vv.std(ddof=1)
    r["diff_exp_minus_con"] = r["experimental_mean"] - r["control_mean"]
    rows.append(r)
d = pd.DataFrame(rows)
d.to_csv(os.path.join(HERE, "daily_selfreport_descriptives.csv"), index=False, encoding="utf-8-sig")

# Manuscript values (experimental, control)
WANT = {"sustained_attention": (3.481, 3.664), "emotional_stability": (3.702, 3.513),
        "impulse_control": (3.569, 3.807), "plan_adherence": (3.414, 3.849),
        "overall_satisfaction": (3.696, 3.723)}
n_ok = 0
print(f"\n{'item':<22}{'experimental':>14}{'control':>14}{'diff':>9}   manuscript")
for _, r in d.iterrows():
    we, wc = WANT[r["item"]]
    ok = abs(r.experimental_mean - we) < .0005 and abs(r.control_mean - wc) < .0005
    n_ok += ok
    print(f"{r['item']:<22}{r.experimental_mean:>7.3f}({r.experimental_sd:.2f})"
          f"{r.control_mean:>7.3f}({r.control_sd:.2f}){r.diff_exp_minus_con:>9.3f}   "
          f"{we:.3f} / {wc:.3f}  {'OK' if ok else '!!'}")
print(f"\nlower in experimental: {int((d.diff_exp_minus_con < 0).sum())}/5 items")
print(f"{n_ok}/5 items match the manuscript")
