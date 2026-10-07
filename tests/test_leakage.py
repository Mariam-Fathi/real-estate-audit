import numpy as np
import pandas as pd
import pytest

from reaudit import leakage


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 2000
    street = rng.integers(0, 1500, n).astype(float)          # ~25% of records share a property
    street[:20] = np.nan
    return pd.DataFrame({'street': street, 'zip_code': 78701.0, 'city': 'Austin', 'state': 'Texas'})


def test_property_ids_group_same_address_and_isolate_missing(df):
    pid = leakage.property_ids(df)
    known = df.street.notna()
    assert pid[known].groupby(df.street[known]).nunique().max() == 1
    assert pid[~known].is_unique and not pid[~known].isin(pid[known]).any()


def test_twins_are_exactly_train_rows_whose_property_is_in_test(df):
    df = df.assign(pid=leakage.property_ids(df))
    tr, te, twin = leakage.split_with_twins(df, seed=1)
    assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == len(df)
    test_pids = set(df.pid.iloc[te])
    assert all((p in test_pids) == t for p, t in zip(df.pid.iloc[tr], twin, strict=True))


@pytest.mark.parametrize('share', [0.0, 0.25, 0.5, 1.0])
def test_training_size_is_constant_and_twin_share_is_controlled(df, share):
    df = df.assign(pid=leakage.property_ids(df))
    tr, te, twin = leakage.split_with_twins(df, seed=1)
    rows = leakage.training_rows(tr, twin, share, seed=1)
    assert len(rows) == len(tr) - twin.sum()
    kept_twins = np.isin(rows, tr[twin]).sum()
    assert kept_twins == round(share * twin.sum())
    assert np.isin(rows, tr).all()
