"""wPLI over all channel pairs, with cached preprocessing."""
import numpy as np, os, re, pickle, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'common', 'eeg'))
from wpli_analysis import wpli, load_marker
from eeg_pipeline import run_preprocessing, EEG_CHANNELS
from itertools import combinations

BANDS={'Theta':(4,8),'Alpha':(8,12),'Beta':(12,30)}
PAIRS=list(combinations(range(14),2))  # 91 pairs

def process_full(edf, marker_path, from_csv=False, trim_s=30, trim_e=20):
    if from_csv:
        from csv_loader import run_preprocessing_from_csv
        cleaned, orig_n = run_preprocessing_from_csv(edf)
    else:
        cleaned, orig_n = run_preprocessing(edf)
    rows = load_marker(marker_path)
    fs=cleaned.info['sfreq']; scale=cleaned.n_times/orig_n
    ev={}
    for lat,dur,typ in rows:
        s=max(0,round(lat*scale)); e=min(cleaned.n_times, round((lat+dur)*scale))
        ev[typ]=(s,e)
    if 'baseline' in ev:
        s,e=ev['baseline']; ev['baseline']=(s+int(trim_s*fs), e-int(trim_e*fs))
    data=cleaned.get_data()
    out={}
    for cond,(s,e) in ev.items():
        if (e-s)/fs < 15: continue
        seg=data[:,s:e]
        for (ia,ib) in PAIRS:
            nm=f"{EEG_CHANNELS[ia]}-{EEG_CHANNELS[ib]}"
            for bn,br in BANDS.items():
                out[(cond,nm,bn)]=wpli(seg[ia],seg[ib],fs,br)
    return out
