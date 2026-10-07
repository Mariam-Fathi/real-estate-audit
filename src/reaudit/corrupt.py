"""Plant known errors into the dataset so detectors can be scored against ground truth.

Each error is planted in a disjoint set of records. Returns the corrupted frame and a label table with
one row per positive record: (index, family, subtype). For a planted duplicate, both the copy and its
source are positives, because a detector can only recognise the pair.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .detectors import LISTING_KEY

SENTINELS = ['1900-01-01', '0000-00-00', '1970-01-01', 'unknown', 'N/A', '9999-12-31']


@dataclass
class Spec:
    sentinel_dates: int = 3000
    typo_years: int = 3000          # half +1000 years (2019 -> 3019), half transposed digits (2015 -> 2051)
    exact_duplicates: int = 2000
    near_duplicates: int = 4000     # half one attribute blanked, half price re-keyed within +/-0.5%
    price_factors: dict = field(default_factory=lambda: {10: 1000, 100: 1000, 1000: 1000, 0.001: 1000})


def _take(rng, pool, n, used):
    pool = np.setdiff1d(pool, np.fromiter(used, dtype=np.int64, count=len(used)))
    pick = rng.choice(pool, size=n, replace=False)
    used.update(pick.tolist())
    return pick


def corrupt(df, seed, spec=None):
    spec = spec or Spec()
    rng = np.random.default_rng(seed)
    out = df.copy()
    labels = []
    used = set()
    label = lambda idx, fam, sub: labels.extend((i, fam, sub) for i in idx)

    dated = out.index[out['prev_sold_date'].str.fullmatch(r'\d{4}-\d{2}-\d{2}', na=False)].to_numpy()

    years = out.loc[dated, 'prev_sold_date'].str[:4].astype(int)
    transposable = years[(years.astype(str).str[:2] + years.astype(str).str[3] + years.astype(str).str[2])
                         .astype(int) > 2026].index.to_numpy()

    # invalid dates -------------------------------------------------------------------------------
    idx = _take(rng, dated, spec.sentinel_dates, used)
    vals = rng.choice(SENTINELS, size=len(idx))
    out.loc[idx, 'prev_sold_date'] = vals
    labels.extend((i, 'invalid_date', f'sentinel {v}') for i, v in zip(idx, vals, strict=True))

    n_plus = spec.typo_years // 2
    idx = _take(rng, dated, n_plus, used)
    s = out.loc[idx, 'prev_sold_date']
    out.loc[idx, 'prev_sold_date'] = (s.str[:4].astype(int) + 1000).astype(str) + s.str[4:]
    label(idx, 'invalid_date', 'year +1000')
    idx = _take(rng, transposable, spec.typo_years - n_plus, used)
    s = out.loc[idx, 'prev_sold_date']
    out.loc[idx, 'prev_sold_date'] = s.str[:2] + s.str[3] + s.str[2] + s.str[4:]
    label(idx, 'invalid_date', 'transposed year')

    # duplicates ---------------------------------------------------------------------------------
    keyed = out.index[out[LISTING_KEY].notna().all(axis=1)].to_numpy()
    copies = []
    next_id = out.index.max() + 1

    def add_copies(src, subtype, mutate=None):
        nonlocal next_id
        c = out.loc[src].copy()
        if mutate is not None:
            mutate(c)
        c.index = pd.RangeIndex(next_id, next_id + len(c))
        next_id += len(c)
        copies.append(c)
        label(src, 'duplicate', subtype)
        label(c.index, 'duplicate', subtype)

    add_copies(_take(rng, keyed, spec.exact_duplicates, used), 'exact copy')

    n_blank = spec.near_duplicates // 2
    def blank_one(c):
        cols = rng.choice(['bed', 'bath', 'acre_lot', 'house_size'], size=len(c))
        for col in np.unique(cols):
            c.loc[cols == col, col] = np.nan
    add_copies(_take(rng, keyed, n_blank, used), 'one attribute blanked', blank_one)

    priced = np.intersect1d(keyed, out.index[out['price'] >= 1000])
    def rekey_price(c):
        c['price'] = (c['price'] * (1 + rng.choice([-1, 1], len(c)) * rng.uniform(0.001, 0.005, len(c)))).round()
    add_copies(_take(rng, priced, spec.near_duplicates - n_blank, used), 'price re-keyed (<=0.5%)', rekey_price)

    # price unit errors ---------------------------------------------------------------------------
    priced = out.index[out['price'] > 0].to_numpy()
    for factor, n in spec.price_factors.items():
        idx = _take(rng, priced, n, used)
        out.loc[idx, 'price'] = out.loc[idx, 'price'] * factor
        label(idx, 'price_unit', f'x{factor:g}')

    out = pd.concat([out, *copies])
    labels = pd.DataFrame(labels, columns=['index', 'family', 'subtype'])
    return out, labels
