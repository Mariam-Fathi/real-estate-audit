"""Corrected detectors.

invalid_dates / duplicates return boolean Series (True = flagged); the price detectors return an anomaly
score (higher = more anomalous) so they can be evaluated across thresholds.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

MIN_DATE = pd.Timestamp('1900-01-02')           # 1900-01-01 is a common sentinel
MAX_DATE = pd.Timestamp('2026-12-31')           # no sale can be later than the analysis year
EPOCH_SENTINELS = {'1900-01-01', '1970-01-01'}  # parseable dates that are almost always defaults

LISTING_KEY = ['street', 'city', 'state', 'zip_code', 'brokered_by', 'status']
COMPARE = ['bed', 'bath', 'acre_lot', 'house_size']
PRICE_TOLERANCE = 0.01
MODIFIED_Z_THRESHOLD = 3.5                      # Iglewicz & Hoaglin (1993)


def invalid_dates(df):
    """Non-missing prev_sold_date that is unparseable, a known sentinel, or outside [MIN_DATE, MAX_DATE]."""
    s = df['prev_sold_date']
    d = pd.to_datetime(s, format='%Y-%m-%d', errors='coerce')
    present = s.notna()
    return present & (d.isna() | s.isin(EPOCH_SENTINELS) | (d < MIN_DATE) | (d > MAX_DATE))


def duplicates(df):
    """Records that share a listing (same address, broker and status) with compatible attributes.

    Attributes are compatible when equal or when either side is missing; prices may differ by
    PRICE_TOLERANCE. A for_sale and a sold record of one property differ in status, so they are not matched.
    """
    cols = LISTING_KEY + COMPARE + ['price', 'prev_sold_date']
    g = df.loc[df[LISTING_KEY].notna().all(axis=1), cols]
    multi = g.duplicated(subset=LISTING_KEY, keep=False)
    g = g[multi].reset_index(names='rid')
    pairs = g.merge(g, on=LISTING_KEY, suffixes=('_a', '_b'))
    pairs = pairs[pairs.rid_a < pairs.rid_b]

    ok = pd.Series(True, index=pairs.index)
    for c in COMPARE + ['prev_sold_date']:
        a, b = pairs[f'{c}_a'], pairs[f'{c}_b']
        ok &= (a == b) | a.isna() | b.isna()
    pa, pb = pairs['price_a'], pairs['price_b']
    ok &= ((pa - pb).abs() <= PRICE_TOLERANCE * np.fmax(pa, pb)) | pa.isna() | pb.isna()

    hit = pairs[ok]
    flagged = pd.Index(hit.rid_a).union(pd.Index(hit.rid_b))
    return pd.Series(df.index.isin(flagged), index=df.index)


MAD_FLOOR_QUANTILE = 0.05


def _robust_z(v, groups, fallback, min_n=20, mad_floor_quantile=None):
    """Modified z-score of v against its group's median/MAD; falls back to a coarser group when small.

    mad_floor_quantile: if set, a group's MAD is raised to at least this quantile of the MADs of all groups
    with >= min_n values. Without it, a group where most values are identical has MAD ~ 0 and every other
    value gets an enormous z. The floor is computed from the data itself, never from labels.
    """
    def stats(by):
        med = v.groupby(by).transform('median')
        mad = (v - med).abs().groupby(by).transform('median')
        n = v.groupby(by).transform('count')
        return med, mad, n
    med, mad, n = stats(groups)
    fmed, fmad, _ = stats(fallback)
    use_fallback = (n < min_n) | (mad == 0) | mad.isna()
    med, mad = med.where(~use_fallback, fmed), mad.where(~use_fallback, fmad)
    if mad_floor_quantile is not None:
        group_mad = (v - v.groupby(groups).transform('median')).abs().groupby(groups).median()
        group_n = v.groupby(groups).count()
        mad = mad.clip(lower=group_mad[group_n >= min_n].quantile(mad_floor_quantile))
    return 0.6745 * (v - med) / mad


def price_robust_z(df, mad_floor_quantile=MAD_FLOOR_QUANTILE):
    """|modified z| of log price-per-sqft within zip code (state fallback); log price when size is missing.

    mad_floor_quantile=None reproduces the Part 2 detector, which had no MAD floor.
    """
    price = df['price'].where(df['price'] > 0)
    size = df['house_size'].where(df['house_size'] > 0)
    lp = np.log(price)
    lppsf = lp - np.log(size)
    zip_ = df['zip_code'].fillna(-1)
    state = df['state'].fillna('?')
    z = _robust_z(lppsf, zip_, state, mad_floor_quantile=mad_floor_quantile).abs()
    z = z.fillna(_robust_z(lp, zip_, state, mad_floor_quantile=mad_floor_quantile).abs())
    return z.where(df['price'].isna() | (df['price'] > 0), np.inf).fillna(0)


def price_isolation_forest(df, seed=0, price_only=False):
    """Isolation Forest on log price, log price/sqft (both also relative to zip median) and size features.

    price_only=True keeps just the four price features, to separate the model from the feature choice.
    """
    price = df['price'].where(df['price'] > 0)
    lp = np.log(price)
    lppsf = lp - np.log(df['house_size'].where(df['house_size'] > 0))
    zip_ = df['zip_code'].fillna(-1)
    X = pd.DataFrame({
        'lp': lp, 'lppsf': lppsf,
        'lp_rel': lp - lp.groupby(zip_).transform('median'),
        'lppsf_rel': lppsf - lppsf.groupby(zip_).transform('median'),
        'bed': df['bed'], 'bath': df['bath'], 'lacre': np.log1p(df['acre_lot']),
    })
    if price_only:
        X = X[['lp', 'lppsf', 'lp_rel', 'lppsf_rel']]
    for c in list(X):
        X[f'{c}_na'] = X[c].isna().astype(float)
        X[c] = X[c].fillna(X[c].median())
    model = IsolationForest(n_estimators=100, max_samples=256, random_state=seed, n_jobs=-1).fit(X)
    return pd.Series(-model.score_samples(X), index=df.index)
