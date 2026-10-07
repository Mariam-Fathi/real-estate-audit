import os

import pandas as pd

CANDIDATE_PATHS = [
    '/kaggle/input/usa-real-estate-dataset/realtor-data.zip.csv',
    'data/raw/realtor-data.zip.csv',
    '../data/raw/realtor-data.zip.csv',
]


def load_raw(path=None):
    """Load the dataset with prev_sold_date kept as text, so invalid values stay visible."""
    path = path or next(p for p in CANDIDATE_PATHS if os.path.exists(p))
    return pd.read_csv(path, dtype={'prev_sold_date': 'string'})
