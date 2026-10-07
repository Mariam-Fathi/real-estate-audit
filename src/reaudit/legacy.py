"""The original series' detectors, vectorised with identical logic (see notebook 01).

Each returns a boolean Series over df.index: True = record flagged.
"""
import pandas as pd

PROPERTY_KEY = ['brokered_by', 'price', 'bed', 'bath', 'acre_lot',
                'street', 'city', 'state', 'zip_code', 'house_size']
PLACEHOLDERS = ['####', '0000-00-00', '1900-01-01', 'nan', 'null', 'none',
                'missing', 'unknown', 'n/a', 'na']


def _dates(df):
    return pd.to_datetime(df['prev_sold_date'], format='%Y-%m-%d', errors='coerce')


def placeholder_dates(df):
    # The original ran astype(str) on an object column, which under pandas 2 turns NaN into the string 'nan'.
    # pandas 3 keeps NaN missing instead, so the conversion is spelled out to reproduce the bug on any version.
    s = df['prev_sold_date'].astype('string').fillna('nan').str.lower()
    return s.isin(PLACEHOLDERS) | s.str.contains('####', regex=False)


def _span_flags(g, key):
    gid = g.groupby(key).ngroup()                                       # dropna=True, as in the original
    d = _dates(g)
    agg = pd.DataFrame({'gid': gid, 'd': d}).groupby('gid')['d'].agg(['size', 'count', 'min', 'max'])
    keep = agg[(agg['size'] > 1) & (agg['count'] > 1) & ((agg['max'] - agg['min']).dt.days > 365)].index
    return gid.isin(keep)


def same_price_different_dates(df):
    dup = df[df.duplicated(subset=PROPERTY_KEY, keep=False)].dropna(subset=PROPERTY_KEY)
    return _span_flags(dup, PROPERTY_KEY).reindex(df.index, fill_value=False)


def price_manipulation(df):
    key = ['street', 'city', 'state', 'zip_code', 'house_size']
    g = df.dropna(subset=key)
    gid = g.groupby(key).ngroup()
    g = g[gid.groupby(gid).transform('size') > 1].assign(gid=gid, d=_dates(g))
    dated = g[g['d'].notna()]
    a = dated.groupby('gid').agg(nd=('price', 'size'), pmin=('price', 'min'), pmax=('price', 'max'),
                                 pna=('price', lambda s: s.isna().sum()), dmin=('d', 'min'), dmax=('d', 'max'))
    keep = a[(a.nd > 1) & (a.pmin == a.pmax) & (a.pna == 0) & ((a.dmax - a.dmin).dt.days > 365)].index
    return g['gid'].isin(keep).reindex(df.index, fill_value=False)
