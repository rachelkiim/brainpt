"""Loader for sessions with no EDF, only the merged EmotivPRO CSV export (14 EEG channels, 128 Hz)."""
import numpy as np
import pandas as pd
import mne
mne.set_log_level('WARNING')

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eeg_pipeline import EEG_CHANNELS, detect_bad_channels

def load_emotivpro_csv_as_raw(csv_path):
    df = pd.read_csv(csv_path, skiprows=1)
    eeg_cols = [f'EEG.{ch}' for ch in EEG_CHANNELS]
    missing = [c for c in eeg_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 없는 채널: {missing}")

    data_uv = df[eeg_cols].values.T  # (14, n_times), same scale as EDF (uV)
    data_v = data_uv / 1e6  # to volts, as when reading EDF

    sfreq = 128.0
    info = mne.create_info(ch_names=EEG_CHANNELS, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data_v, info, verbose=False)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='warn')
    return raw


def run_preprocessing_from_csv(csv_path, seed=97):
    """Same as eeg_pipeline.run_preprocessing, but from the CSV export."""
    raw = load_emotivpro_csv_as_raw(csv_path)
    orig_n_times = raw.n_times

    raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', verbose=False)

    bads = detect_bad_channels(raw)
    raw.info['bads'] = bads
    print(f"  Bad channels detected: {bads}")

    good_raw = raw.copy().drop_channels(bads) if bads else raw.copy()
    good_raw.set_eeg_reference('average', verbose=False)

    n_components = min(len(good_raw.ch_names) - 1, 10)
    ica = mne.preprocessing.ICA(n_components=n_components, method='infomax',
                                  fit_params=dict(extended=True), random_state=seed, max_iter=500)
    ica.fit(good_raw, verbose=False)

    try:
        from mne_icalabel import label_components
        nyquist = raw.info['sfreq'] / 2.0
        h_freq_wide = min(100.0, nyquist - 1.0)
        wide_raw = raw.copy().drop_channels(bads) if bads else raw.copy()
        wide_raw.filter(l_freq=1.0, h_freq=h_freq_wide, fir_design='firwin', verbose=False)
        wide_raw.set_eeg_reference('average', verbose=False)
        ic_labels = label_components(wide_raw, ica, method='iclabel')
        labels = ic_labels['labels']
        probs = ic_labels['y_pred_proba']
        exclude = [i for i, (l, p) in enumerate(zip(labels, probs))
                   if l in ('eye blink', 'muscle artifact', 'line noise', 'channel noise') and p >= 0.80]
        print(f"  ICLabel labels: {labels}")
        print(f"  Excluded components: {exclude}")
    except Exception as e:
        print(f"  ICLabel failed ({e}), skipping IC removal")
        exclude = []

    ica.exclude = exclude
    cleaned = good_raw.copy()
    ica.apply(cleaned, verbose=False)

    if bads:
        bad_raw = raw.copy().pick(bads)
        bad_raw.set_montage(mne.channels.make_standard_montage('standard_1020'), on_missing='warn')
        cleaned.add_channels([bad_raw], force_update_info=True)
        cleaned.info['bads'] = bads
        cleaned.interpolate_bads(reset_bads=True, verbose=False)
    cleaned.reorder_channels(EEG_CHANNELS)

    return cleaned, orig_n_times
