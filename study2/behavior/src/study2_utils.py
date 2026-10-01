"""Reusable core functions for the Study 2 analysis notebook."""
import numpy as np
import pandas as pd
from scipy import stats

CONTROL_GROUP = [13, 14, 17, 18, 20, 21, 22, 23]
EXPERIMENTAL_GROUP = [1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 15, 16, 19]
EXCLUDED_PARTICIPANTS = {3, 5}  # dropped out


def assign_group(participant_num):
    if participant_num in CONTROL_GROUP:
        return 'Control'
    if participant_num in EXPERIMENTAL_GROUP:
        return 'Experimental'
    return 'Unknown'


def ttest(a, b, labels=('A', 'B'), paired=False):
    a = pd.to_numeric(pd.Series(a), errors='coerce').dropna().astype(float).values
    b = pd.to_numeric(pd.Series(b), errors='coerce').dropna().astype(float).values
    t, p = (stats.ttest_rel if paired else stats.ttest_ind)(a, b)
    d = (a.mean() - b.mean()) / np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    print(f"{labels[0]}: {a.mean():.1f}±{a.std():.1f}(n={len(a)})  {labels[1]}: {b.mean():.1f}±{b.std():.1f}(n={len(b)})  t={t:.3f} p={p:.4f} d={d:.3f} {'*' if p < 0.05 else 'ns'}")


def analyze_bart(df, participant_num):
    bart_data = df[['maxPumps', 'nPumps', 'earnings', 'thisRow.t']].dropna(subset=['nPumps'])
    if len(bart_data) == 0:
        return None

    trials = []
    current_trial = []
    for idx, row in bart_data.iterrows():
        if row['nPumps'] == 1:
            if current_trial:
                trials.append(current_trial[-1])
            current_trial = [row]
        else:
            current_trial.append(row)
    if current_trial:
        trials.append(current_trial[-1])

    trial_df = pd.DataFrame(trials)
    total_earnings = trial_df['earnings'].sum()
    mean_pumps = trial_df['nPumps'].mean()
    total_trials = len(trial_df)
    explosions = (trial_df['nPumps'] >= trial_df['maxPumps']).sum()
    explosion_rate = explosions / total_trials if total_trials > 0 else 0

    # mean pump latency (intra-trial only: t_diff < 2s)
    bart_t = bart_data[bart_data['thisRow.t'].notna()].copy()
    bart_t['t_diff'] = bart_t['thisRow.t'].diff()
    mean_pump_latency = bart_t[bart_t['t_diff'] < 2.0]['t_diff'].mean()

    return {
        'participant_num': participant_num,
        'total_earnings': total_earnings,
        'mean_pumps': mean_pumps,
        'explosion_rate': explosion_rate,
        'mean_pump_latency': mean_pump_latency
    }


def analyze_emotion_gonogo(df, participant_num):
    """Analyze the Emotion Go-NoGo task."""
    task_data = df[df['FaceE'].notna() & df['Pract.corr'].isna()].copy()
    if len(task_data) == 0:
        return None

    all_resp_cols = [c for c in df.columns if 'key_resp' in c and '.keys' in c]
    task_data['response'] = None
    task_data['rt'] = None

    for col in all_resp_cols:
        rt_col = col.replace('.keys', '.rt')
        mask = task_data[col].notna()
        task_data.loc[mask, 'response'] = task_data.loc[mask, col]
        if rt_col in df.columns:
            task_data.loc[mask, 'rt'] = task_data.loc[mask, rt_col]

    task_data['correct'] = 0
    go_mask = task_data['CorrectAnsP'] == 'space'
    nogo_mask = task_data['CorrectAnsP'].isna()
    task_data.loc[go_mask & (task_data['response'] == 'space'), 'correct'] = 1
    task_data.loc[nogo_mask & task_data['response'].isna(), 'correct'] = 1

    accuracy = task_data['correct'].mean()
    go_trials = task_data[go_mask]
    nogo_trials = task_data[nogo_mask]
    go_accuracy = go_trials['correct'].mean() if len(go_trials) > 0 else 0
    nogo_accuracy = nogo_trials['correct'].mean() if len(nogo_trials) > 0 else 0
    go_rt = go_trials[go_trials['rt'].notna()]['rt']
    mean_rt = go_rt.mean() if len(go_rt) > 0 else 0

    return {
        'participant_num': participant_num,
        'accuracy': accuracy,
        'go_accuracy': go_accuracy,
        'nogo_accuracy': nogo_accuracy,
        'mean_rt': mean_rt
    }
