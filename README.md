# BrainPT

Analysis code for *BrainPT: a cognitive mini-game app that intervenes in young adults'
short-form video use — a neural and behavioural pilot study*.

- **Study 1** — within-subject, two lab visits (n=9). EEG and fNIRS during short-form viewing,
  a single BrainPT insertion, and goal-directed tasks.
- **Study 2** — non-randomised pre-post design, experimental (daily BrainPT use for two weeks)
  vs control. Behavioural outcomes, app logs, EEG and fNIRS at both visits.

## Layout

```
common/
  eeg/          preprocessing pipeline, wPLI, entropy, CFC, EEG reproduction and Lord/TOST checks
  fnirs/        fNIRS statistics, reproduction and Lord/TOST checks
study1/
  behavior/     BART preprocessing notebook
  brain/        aperiodic (FOOOF) analysis, F4 spectrum cluster correction
study2/
  behavior/     main behavioural notebook (study2.ipynb), daily self-report, app-log (firebase/)
  brain/        block-split power, extra metrics, full connectivity, 372-test connectivity checks
data/           not included — see below
```

## Data

Participant data are not part of this repository. The scripts expect them under `data/`
in the layout described in `data/README.md` (Study 1 and Study 2 folders with `behavior/`,
`markers/`, `preprocessed/`, `quantified/`, `neural/`).

## Running

```bash
pip install -r requirements.txt
cd common/eeg   && python reproduce_eeg.py      # Study 1/2 EEG values against the manuscript
cd common/fnirs && python reproduce_fnirs.py    # fNIRS values against the manuscript
```

Each script resolves `data/` relative to its own location, so run it from its folder.
The statistics scripts need only numpy, scipy and pandas. Preprocessing from raw EDF
(`common/eeg/eeg_pipeline.py`) additionally needs mne, pyprep, mne-icalabel and torch,
and the raw recordings, which are not distributed.

`study2/behavior/firebase/download.py` exports app logs from Firestore and needs a service
account key (`serviceAccountKey.json`, ignored by git).

## Known issues

- **Bad-channel cap is non-deterministic.** `detect_bad_channels` keeps
  `list(set(bads))[:3]` when more than half the channels are flagged. Set order depends on the
  process hash seed, so the three interpolated channels — and the resulting band power —
  change between runs for recordings where the cap fires. Recordings where it does not fire
  reproduce exactly. Condition ordering (short-form < BrainPT < task) held in every re-run.
- **Band edges are inclusive at both ends**, so 4, 8 and 12 Hz bins count in two bands.
- The 372-test connectivity scripts (`study2/brain/reconstructed_conn372_*.py`) rebuild the
  index definitions from the reported numbers; the original script was not preserved.
