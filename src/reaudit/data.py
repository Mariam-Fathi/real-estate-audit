import glob
import os

import numpy as np
import pandas as pd

PROPERTY_KEY = ['street', 'zip_code', 'city', 'state']   # the encoded street has no unit number

FILENAME = 'realtor-data.zip.csv'
CANDIDATE_PATHS = [f'data/raw/{FILENAME}', f'../data/raw/{FILENAME}']
SEARCH_ROOTS = ['/kaggle/input']    # Kaggle's mount layout differs between notebook versions, so search it


def find_dataset(candidates=CANDIDATE_PATHS, search_roots=SEARCH_ROOTS):
    """Path of the listings CSV: a known local path first, otherwise the first match under a search root."""
    for path in candidates:
        if os.path.exists(path):
            return path
    for root in search_roots:
        hits = sorted(glob.glob(os.path.join(root, '**', FILENAME), recursive=True))
        if hits:
            return hits[0]
    raise FileNotFoundError(f'{FILENAME} not found in {candidates} or under {search_roots}. Attach the USA Real '
                            'Estate Dataset (ahmedshahriarsakib) as an input, or download it to data/raw/.')


def load_raw(path=None):
    """Load the dataset with prev_sold_date kept as text, so invalid values stay visible."""
    return pd.read_csv(path or find_dataset(), dtype={'prev_sold_date': 'string'})


def property_ids(df):
    """One id per physical property (encoded street + zip + city + state); records missing any of these get
    their own id, since they cannot be matched to anything."""
    missing = df[PROPERTY_KEY].isna().any(axis=1).to_numpy()
    pid = np.empty(len(df), dtype=np.int64)
    pid[~missing] = df[~missing].groupby(PROPERTY_KEY).ngroup().to_numpy()
    pid[missing] = (pid[~missing].max() + 1 if (~missing).any() else 0) + np.arange(missing.sum())
    return pd.Series(pid, index=df.index, name='pid')
