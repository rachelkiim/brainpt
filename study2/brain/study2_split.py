"""Study 2 short-form / task block power, split into thirds."""
import numpy as np, sys, os, re, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'common', 'eeg'))
from wpli_analysis import load_marker
from eeg_pipeline import run_preprocessing, EEG_CHANNELS, local_band_power, BANDS

def process(edf, mk, from_csv=False, trim_s=30, trim_e=20, nsplit=3):
    if from_csv:
        from csv_loader import run_preprocessing_from_csv
        cleaned,orig_n=run_preprocessing_from_csv(edf)
    else:
        cleaned,orig_n=run_preprocessing(edf)
    rows=load_marker(mk); fs=cleaned.info['sfreq']; scale=cleaned.n_times/orig_n
    ev={}
    for lat,dur,typ in rows:
        s=max(0,round(lat*scale)); e=min(cleaned.n_times,round((lat+dur)*scale))
        ev[typ]=(s,e)
    if 'baseline' not in ev: return {}
    bs,be=ev['baseline']; bs+=int(trim_s*fs); be-=int(trim_e*fs)
    data=cleaned.get_data()
    # Baseline power per channel
    base={}
    for ci,ch in enumerate(EEG_CHANNELS):
        for bn,br in BANDS.items():
            base[(ch,bn)]=local_band_power(data[ci,bs:be],fs,br)
    out={}
    for cond in ['shortform','task']:
        if cond not in ev: continue
        s,e=ev[cond]; tot=e-s
        if tot/fs < 60: continue
        part=tot//nsplit
        for k in range(nsplit):
            ps=s+k*part; pe=s+(k+1)*part if k<nsplit-1 else e
            for ci,ch in enumerate(EEG_CHANNELS):
                for bn,br in BANDS.items():
                    p=local_band_power(data[ci,ps:pe],fs,br)
                    b=base[(ch,bn)]
                    out[(f"{cond}_p{k+1}",ch,bn)] = 10*np.log10(p/b) if (p and b and b>0) else np.nan
    return out
