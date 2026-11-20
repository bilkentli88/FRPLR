"""Reproduce accuracy and paired cluster-bootstrap contrasts in revised ACL Results.

Usage:
    python acl_accuracy_reanalysis.py /path/to/extracted/causal_factorization_private_update

Requires numpy and pandas. Reuses already stored predictions; does not run an LLM.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SOURCES = {
    'Qwen3-4B': 'data/qwen/qwen_harmonized_robustness_v1.jsonl',
    'Gemma 3 4B': 'data/gemma/gemma_harmonized_robustness_v1.jsonl',
    'Phi-4-mini': 'data/phi/phi_factorial_confirmatory_v1.jsonl',
    'Mistral-7B': 'data/mistral/mistral_factorial_confirmatory_v1.jsonl',
}
LOCALIZATION = {
    'Qwen localization': 'data/qwen/v062_geometry_localization_4k8k.jsonl',
    'Gemma localization': 'data/gemma/v072_gemma_geometry_localization_4k8k.jsonl',
}
INDEX = ['trial_index', 'semantic_seed', 'filler_language', 'target_chat_tokens']

def bootstrap_by_trial(wide: pd.DataFrame, values: pd.Series, seed: int, reps: int=10_000):
    clusters = values.groupby(wide['trial_index']).mean().sort_index().to_numpy(float)
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(clusters), size=(reps,len(clusters)))
    samples = clusters[sample_indices].mean(axis=1)
    return (float(clusters.mean()), *[float(x) for x in np.quantile(samples,[0.025,0.975])])

def main(root: Path):
    for i, (name, rel) in enumerate(SOURCES.items()):
        frame = pd.read_json(root/rel, lines=True)
        print('\n'+name, 'independent seeds',frame.trial_index.nunique())
        print('accuracy (%):', (frame.groupby('factorial_code').correct.mean()*100).round(2).to_dict())
        wide = frame.pivot(index=INDEX,columns='factorial_code',values='correct').reset_index()
        for to_,from_ in [('CE','CC'),('FE','CE'),('FE','CC')]:
            mean,lo,hi=bootstrap_by_trial(wide,wide[to_]-wide[from_],20260828+i*100+len(f'{to_}-{from_}'))
            print(f'  {to_}-{from_}: {mean*100:+.2f} pp [{lo*100:+.2f}, {hi*100:+.2f}]')
        if name == 'Gemma 3 4B':
            ce=frame.loc[frame.factorial_code=='CE']
            print('CE predictions:',ce.predicted_digit.value_counts().to_dict())
    for name,rel in LOCALIZATION.items():
        frame=pd.read_json(root/rel,lines=True)
        print('\n'+name, 'independent seeds',frame.trial_index.nunique())
        print('accuracy (%):', (frame.groupby('geometry_code').correct.mean()*100).round(2).to_dict())

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('repo_root',type=Path)
    args=parser.parse_args()
    main(args.repo_root)
