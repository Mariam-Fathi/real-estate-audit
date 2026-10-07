"""Part 3 contrast: the common way to measure leakage, a random split vs a property-grouped split.

The two splits produce different test sets, so their difference mixes leakage with test-set sampling noise.
Run from the repo root:  python experiments/run_naive_split.py [n_seeds]
Writes results/leakage_naive.csv.
"""
import sys
import time

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, 'src')
from reaudit import leakage  # noqa: E402
from reaudit.data import load_raw  # noqa: E402


def main(n_seeds=5, only=None):
    """only: optional list of model names to (re)run; their rows replace earlier results for those models."""
    df = leakage.prepare(load_raw())
    y = df['y'].to_numpy()
    rows = []
    for seed in range(n_seeds):
        random_split = leakage.split_with_twins(df, seed)[:2]
        grouped_split = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(df, groups=df['pid']))
        for name, model in leakage.models(seed).items():
            if only and name not in only:
                continue
            if name == 'ridge':
                continue
            X = leakage.design_matrix(df, name)
            for split_name, (tr, te) in [('random', random_split), ('grouped', grouped_split)]:
                t = time.time()
                model.fit(X.iloc[tr], y[tr])
                rows.append({'seed': seed, 'model': name, 'split': split_name,
                             **leakage.metrics(y[te], model.predict(X.iloc[te]))})
                print(f'seed {seed} {name:17s} {split_name:7s} R2 {rows[-1]["r2"]:.4f} [{time.time() - t:.0f}s]',
                      flush=True)
    out = pd.DataFrame(rows)
    if only:
        out = pd.concat([pd.read_csv('results/leakage_naive.csv').query('model not in @only'), out])
    out.to_csv('results/leakage_naive.csv', index=False)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5, sys.argv[2].split(',') if len(sys.argv) > 2 else None)
