"""Part 2 experiment: plant known errors, run original and corrected detectors, score them.

Run from the repo root:  python experiments/run_validation.py [n_seeds]
Writes results/validation_*.csv.
"""
import os
import sys
import time

import pandas as pd

sys.path.insert(0, 'src')
from reaudit import detectors, legacy                     # noqa: E402
from reaudit.corrupt import corrupt                       # noqa: E402
from reaudit.data import load_raw                         # noqa: E402
from reaudit.evaluate import flag_metrics, recall_by_subtype, score_metrics   # noqa: E402

IF_THRESHOLD = 0.5   # sklearn's default rule: anomaly when -score_samples > 0.5 (offset_ for contamination='auto')


def run_flag_detectors(df, seed):
    """{name: (family, boolean flags)}"""
    out = {
        'original: placeholder dates': ('invalid_date', legacy.placeholder_dates(df)),
        'corrected: invalid dates': ('invalid_date', detectors.invalid_dates(df)),
        'original: same price, different dates': ('duplicate', legacy.same_price_different_dates(df)),
        'original: price manipulation': ('duplicate', legacy.price_manipulation(df)),
        'corrected: listing duplicates': ('duplicate', detectors.duplicates(df)),
    }
    z = detectors.price_robust_z(df)
    iso = detectors.price_isolation_forest(df, seed=seed)
    iso_p = detectors.price_isolation_forest(df, seed=seed, price_only=True)
    out['corrected: robust z (|z| > 3.5)'] = ('price_unit', z > detectors.MODIFIED_Z_THRESHOLD)
    out['ML baseline: isolation forest'] = ('price_unit', iso > IF_THRESHOLD)
    out['ML baseline: isolation forest (price features)'] = ('price_unit', iso_p > IF_THRESHOLD)
    scores = {'corrected: robust z (|z| > 3.5)': z, 'ML baseline: isolation forest': iso,
              'ML baseline: isolation forest (price features)': iso_p}
    return out, scores


def main(n_seeds=5):
    os.makedirs('results', exist_ok=True)
    raw = load_raw()

    t = time.time()
    base_flags, _ = run_flag_detectors(raw, seed=0)
    base = pd.DataFrame([{'detector': k, 'family': fam, 'flags_on_unmodified_data': int(f.sum())}
                         for k, (fam, f) in base_flags.items()])
    base.to_csv('results/validation_base_flags.csv', index=False)
    print(base.to_string(index=False), f'\n[{time.time() - t:.0f}s]', flush=True)

    rows, subtype_rows = [], []
    for seed in range(n_seeds):
        t = time.time()
        df, labels = corrupt(raw, seed)
        flags, scores = run_flag_detectors(df, seed)
        for name, (fam, f) in flags.items():
            m = flag_metrics(f, labels, fam, preexisting=base_flags[name][1])
            if name in scores:
                m.update(score_metrics(scores[name], labels, fam))
            rows.append({'seed': seed, 'detector': name, 'family': fam, **m})
            for sub, r in recall_by_subtype(f, labels, fam).items():
                subtype_rows.append({'seed': seed, 'detector': name, 'family': fam, 'subtype': sub, 'recall': r})
        print(f'seed {seed} done [{time.time() - t:.0f}s]', flush=True)

    pd.DataFrame(rows).to_csv('results/validation_metrics.csv', index=False)
    pd.DataFrame(subtype_rows).to_csv('results/validation_recall_by_subtype.csv', index=False)
    summary = pd.DataFrame(rows).groupby(['family', 'detector'])[['precision', 'precision_upper', 'recall', 'f1']].mean()
    print(summary.round(3).to_string())


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
