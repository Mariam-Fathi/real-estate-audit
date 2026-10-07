"""Part 3: how duplicate records of one property inflate a price model's measured accuracy.

Design: one random train/test split per seed. Training rows whose property also appears in the test set are
"twins". Every condition keeps the test set fixed and removes the same number of training rows: a share
(1 - f) of the twins plus f x (number of twins) random non-twin rows. f = 1 is an ordinary random split
(all twins kept), f = 0 is a leak-free split. Training size is identical in every condition, so any
difference in test error is caused by the twins.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROPERTY_KEY = ['street', 'zip_code', 'city', 'state']
FEATURES = ['bed', 'bath', 'acre_lot', 'house_size', 'zip_code', 'state_code', 'status_code', 'sold_year']
PRICE_RANGE = (10_000, 50_000_000)


def property_ids(df):
    """One id per physical property (encoded street + zip + city + state); records missing any of these get
    their own id, since they cannot be matched to anything."""
    missing = df[PROPERTY_KEY].isna().any(axis=1).to_numpy()
    pid = np.empty(len(df), dtype=np.int64)
    pid[~missing] = df[~missing].groupby(PROPERTY_KEY).ngroup().to_numpy()
    pid[missing] = (pid[~missing].max() + 1 if (~missing).any() else 0) + np.arange(missing.sum())
    return pd.Series(pid, index=df.index, name='pid')


def prepare(raw):
    df = raw[raw['price'].between(*PRICE_RANGE)].copy()
    df['pid'] = property_ids(df)
    df['y'] = np.log(df['price'])
    df['state_code'] = df['state'].astype('category').cat.codes
    df['status_code'] = df['status'].astype('category').cat.codes
    df['sold_year'] = pd.to_datetime(df['prev_sold_date'], format='%Y-%m-%d', errors='coerce').dt.year
    return df.reset_index(drop=True)


def split_with_twins(df, seed, test_size=0.2):
    """Random record-level split. Returns (train_idx, test_idx, twin_mask over train_idx)."""
    rng = np.random.default_rng(seed)
    is_test = rng.random(len(df)) < test_size
    test_idx = np.flatnonzero(is_test)
    train_idx = np.flatnonzero(~is_test)
    twin = np.isin(df['pid'].to_numpy()[train_idx], df['pid'].to_numpy()[test_idx])
    return train_idx, test_idx, twin


def training_rows(train_idx, twin, keep_share, seed):
    """Drop (1 - keep_share) of the twins and keep_share x n_twins random non-twins; size is constant."""
    rng = np.random.default_rng(seed + 10_000)
    twins, others = train_idx[twin], train_idx[~twin]
    n = len(twins)
    n_keep = int(round(keep_share * n))
    kept_twins = rng.permutation(twins)[:n_keep]
    kept_others = rng.permutation(others)[:len(others) - n_keep]
    return np.sort(np.concatenate([kept_twins, kept_others]))


def models(seed):
    return {
        'ridge': make_pipeline(StandardScaler(), Ridge()),
        # early stopping off: its random validation split would add noise unrelated to the twins
        'gradient boosting': HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=63, early_stopping=False,
                                                           random_state=seed),
        'random forest': RandomForestRegressor(n_estimators=60, max_features=0.5, n_jobs=-1, random_state=seed),
    }


def design_matrix(df, model_name):
    X = df[FEATURES]
    if model_name == 'gradient boosting':          # handles missing values natively
        return X
    missing = {f'{c}_missing': X[c].isna().astype(np.float32) for c in FEATURES}
    X = X.fillna(X.median())
    if model_name == 'ridge':
        # a linear model needs skewed sizes on a log scale and categories one-hot encoded
        X = X.assign(**{c: np.log1p(X[c]) for c in ['bed', 'bath', 'acre_lot', 'house_size']})
        dummies = pd.get_dummies(X[['state_code', 'status_code']].astype('category'), dtype=np.float32)
        X = pd.concat([X.drop(columns=['state_code', 'status_code']), dummies], axis=1)
    return X.assign(**missing).astype(np.float32)


def metrics(y, p):
    abs_log = np.abs(p - y)
    return {'r2': r2_score(y, p), 'mae_log': abs_log.mean(),
            'median_ape': np.median(np.abs(np.exp(p - y) - 1))}
