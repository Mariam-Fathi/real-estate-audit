"""Part 3 experiment: duplicate-property leakage in price models.

Run from the repo root:  python experiments/run_leakage.py [n_seeds]
Writes results/leakage_metrics.csv (one row per seed x model x keep_share x test subset)
and results/leakage_records.parquet (per-record errors for seed 0, for the paired analysis).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, 'src')
from reaudit import leakage                    # noqa: E402
from reaudit.data import load_raw              # noqa: E402

DOSE_MODEL = 'random forest'
KEEP_SHARES = [0.0, 0.25, 0.5, 0.75, 1.0]


def main(n_seeds=5, only=None):
    """only: optional list of model names to (re)run; their rows replace earlier results for those models."""
    os.makedirs('results', exist_ok=True)
    df = leakage.prepare(load_raw())
    y = df['y'].to_numpy()
    sizes = df.groupby('pid')['pid'].transform('size')
    print(f'{len(df):,} records, {df.pid.nunique():,} properties, '
          f'{(sizes > 1).mean():.1%} of records share a property with another record', flush=True)

    rows, records = [], []
    for seed in range(n_seeds):
        train_idx, test_idx, twin = leakage.split_with_twins(df, seed)
        test_seen = np.isin(df['pid'].to_numpy()[test_idx], df['pid'].to_numpy()[train_idx])
        print(f'seed {seed}: {twin.sum():,} twins in train, {test_seen.mean():.1%} of test rows have one', flush=True)
        for name, model in leakage.models(seed).items():
            if only and name not in only:
                continue
            X = leakage.design_matrix(df, name)
            shares = KEEP_SHARES if name == DOSE_MODEL else [0.0, 1.0]
            for share in shares:
                t = time.time()
                rows_tr = leakage.training_rows(train_idx, twin, share, seed)
                model.fit(X.iloc[rows_tr], y[rows_tr])
                p = model.predict(X.iloc[test_idx])
                for subset, mask in [('all test rows', np.ones(len(test_idx), bool)),
                                     ('property seen in train', test_seen),
                                     ('property not seen', ~test_seen)]:
                    rows.append({'seed': seed, 'model': name, 'keep_share': share, 'subset': subset,
                                 'n_train': len(rows_tr), 'n_test': int(mask.sum()),
                                 **leakage.metrics(y[test_idx][mask], p[mask])})
                if seed == 0 and share in (0.0, 1.0):
                    records.append(pd.DataFrame({'model': name, 'keep_share': share, 'row': test_idx,
                                                 'seen': test_seen, 'abs_err': np.abs(p - y[test_idx])}))
                print(f'  {name:17s} keep {share:.2f}  R2 {rows[-3]["r2"]:.4f}  [{time.time() - t:.0f}s]', flush=True)

    new_rows, new_records = pd.DataFrame(rows), pd.concat(records)
    if only:
        new_rows = pd.concat([pd.read_csv('results/leakage_metrics.csv').query('model not in @only'), new_rows])
        new_records = pd.concat([pd.read_parquet('results/leakage_records.parquet').query('model not in @only'),
                                 new_records])
    new_rows.to_csv('results/leakage_metrics.csv', index=False)
    new_records.to_parquet('results/leakage_records.parquet', index=False)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5, sys.argv[2].split(',') if len(sys.argv) > 2 else None)
