# BrainPT

**BrainPT, a Cognitive Mini-Game App for Short-Form Video Use in Young Adults: Neural and Behavioral Pilot Study**

Yunjeong Jeong, Chaewon Kim, Sungmin Ha, Seoyoung Choi, Sanghoon Han · Department of Psychology, Yonsei University

BrainPT is an iOS app that interrupts short-form viewing with a brief set of cognitive tasks once daily Instagram + YouTube use reaches 30 minutes. Instead of blocking use, it targets the neural state during viewing.

<p align="center">
  <img src="figures/brainpt_app.png" width="860" alt="BrainPT app screens">
</p>

## Results

**Study 1 (n=9): a single insertion moves the prefrontal state.**
Right dorsolateral prefrontal activation (fNIRS) fell below rest during short-form viewing and rose above it during goal-directed tasks (−0.093 vs +0.180, P=.002). The viewing block right after a BrainPT insertion moved toward the task state (+0.110, P=.02).

<p align="center">
  <img src="figures/study1_insertion.png" width="860" alt="Study 1 prefrontal activation by block">
</p>

**Study 2 (13 vs 8): daily short-form use fell after 2 weeks.**
Use dropped from 143.5 to 50.9 min/day in the experimental group and rose from 104.8 to 132.9 in the control group (difference −120.7 min, P=.004, d=−1.35). ADHD self-report and resting neural measures did not differ between groups.

<p align="center">
  <img src="figures/study2_shortform_use.png" width="520" alt="Daily short-form use before and after 2 weeks">
</p>

**BrainPT engages the brain differently from viewing.** Before the intervention (n=18), beta-band coupling rose across the scalp during BrainPT and fell during short-form viewing.

<p align="center">
  <img src="figures/study2_beta_coupling.png" width="760" alt="Beta-band coupling by condition">
</p>

All analyses are exploratory; Study 2 is nonrandomized and its primary outcome is self-reported. Full paper figures are in [`figures/paper/`](figures/paper).

## Code

`common/` holds the EEG and fNIRS pipelines and statistics, `study1/` and `study2/` the study-specific analyses. Participant data are not included (available from the corresponding author on reasonable request); scripts expect them under `data/study1/` and `data/study2/`.

```bash
pip install -r requirements.txt
cd common/eeg && python reproduce_eeg.py
```

## Citation

> Jeong Y, Kim C, Ha S, Choi S, Han S. BrainPT, a cognitive mini-game app for short-form video use in young adults: neural and behavioral pilot study. *[journal, year — to be added]*.
