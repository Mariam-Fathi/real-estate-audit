import os

import numpy as np
import pandas as pd

PROPERTY_KEY = ['street', 'zip_code', 'city', 'state']   # the encoded street has no unit number

CANDIDATE_PATHS = [
    '/kaggle/input/usa-real-estate-dataset/realtor-data.zip.csv',
    'data/raw/realtor-data.zip.csv',
    '../data/raw/realtor-data.zip.csv',
]


def load_raw(path=None):
    """Load the dataset with prev_sold_date kept as text, so invalid values stay visible."""
    path = path or next(p for p in CANDIDATE_PATHS if os.path.exists(p))
    return pd.read_csv(path, dtype={'prev_sold_date': 'string'})


def property_ids(df):
    """One id per physical property (encoded street + zip + city + state); records missing any of these get
    their own id, since they cannot be matched to anything."""
    missing = df[PROPERTY_KEY].isna().any(axis=1).to_numpy()
    pid = np.empty(len(df), dtype=np.int64)
    pid[~missing] = df[~missing].groupby(PROPERTY_KEY).ngroup().to_numpy()
    pid[missing] = (pid[~missing].max() + 1 if (~missing).any() else 0) + np.arange(missing.sum())
    return pd.Series(pid, index=df.index, name='pid')
