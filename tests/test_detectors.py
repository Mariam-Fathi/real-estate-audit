import numpy as np
import pandas as pd
import pytest

from reaudit import detectors, legacy
from reaudit.corrupt import Spec, corrupt
from reaudit.evaluate import flag_metrics


def frame(rows):
    cols = ['brokered_by', 'status', 'price', 'bed', 'bath', 'acre_lot', 'street',
            'city', 'state', 'zip_code', 'house_size', 'prev_sold_date']
    df = pd.DataFrame(rows, columns=cols)
    df['prev_sold_date'] = df['prev_sold_date'].astype('string')
    return df


HOUSE = [1.0, 'for_sale', 300000.0, 3.0, 2.0, 0.2, 10.0, 'Austin', 'Texas', 78701.0, 1500.0, '2015-06-01']


def with_(**kw):
    row = dict(zip(['brokered_by', 'status', 'price', 'bed', 'bath', 'acre_lot', 'street',
                    'city', 'state', 'zip_code', 'house_size', 'prev_sold_date'], HOUSE))
    row.update(kw)
    return list(row.values())


# --- the bug found in Part 1 ----------------------------------------------------------------------
def test_original_placeholder_detector_flags_plain_missing_values():
    df = frame([with_(prev_sold_date=None), with_()])
    assert legacy.placeholder_dates(df).tolist() == [True, False]


def test_corrected_date_detector_ignores_missing_values():
    df = frame([with_(prev_sold_date=None), with_()])
    assert not detectors.invalid_dates(df).any()


@pytest.mark.parametrize('value', ['1900-01-01', '1970-01-01', '0000-00-00', 'unknown', 'N/A',
                                   '9999-12-31', '3019-04-02', '2051-03-01'])
def test_invalid_dates_are_flagged(value):
    assert detectors.invalid_dates(frame([with_(prev_sold_date=value)])).all()


@pytest.mark.parametrize('value', ['1901-01-01', '2022-04-15', '2026-04-08'])
def test_plausible_dates_pass(value):
    assert not detectors.invalid_dates(frame([with_(prev_sold_date=value)])).any()


# --- duplicates -------------------------------------------------------------------------------------
def test_lifecycle_pair_is_not_a_duplicate():
    df = frame([with_(), with_(status='sold', prev_sold_date='2022-04-15')])
    assert not detectors.duplicates(df).any()
    assert legacy.same_price_different_dates(df).all()     # ...but the original detector flags it


def test_exact_copy_is_a_duplicate_but_original_misses_it():
    df = frame([with_(), with_()])
    assert detectors.duplicates(df).all()
    assert not legacy.same_price_different_dates(df).any()  # identical dates -> span 0 -> never flagged


def test_near_copy_with_missing_attribute_is_a_duplicate():
    df = frame([with_(), with_(house_size=np.nan), with_(street=11.0)])
    assert detectors.duplicates(df).tolist() == [True, True, False]


def test_price_tolerance():
    df = frame([with_(), with_(price=301000.0), with_(price=310000.0)])
    flags = detectors.duplicates(df)
    assert flags.tolist() == [True, True, False]


def test_original_groupby_drops_records_with_missing_key():
    df = frame([with_(bed=np.nan), with_(bed=np.nan, status='sold', prev_sold_date='2022-04-15')])
    assert not legacy.same_price_different_dates(df).any()


# --- price ------------------------------------------------------------------------------------------
def test_robust_z_flags_unit_error():
    rng = np.random.default_rng(0)
    rows = [with_(price=float(p), street=float(i), house_size=1500.0)
            for i, p in enumerate(rng.normal(300000, 20000, 60).round())]
    rows.append(with_(price=300.0, street=999.0))          # $300 -- a /1000 unit error
    z = detectors.price_robust_z(frame(rows))
    assert z.iloc[-1] > detectors.MODIFIED_Z_THRESHOLD
    assert (z.iloc[:-1] < detectors.MODIFIED_Z_THRESHOLD).all()


def test_zero_price_scores_infinite():
    z = detectors.price_robust_z(frame([with_(price=0.0)] + [with_(street=float(i)) for i in range(25)]))
    assert np.isinf(z.iloc[0])


# --- corruption + metrics ---------------------------------------------------------------------------
@pytest.fixture
def small():
    rng = np.random.default_rng(1)
    n = 3000
    return frame([with_(street=float(i), price=float(rng.integers(100, 900)) * 1000,
                        zip_code=float(78700 + i % 5),
                        prev_sold_date=f'{rng.integers(1990, 2022)}-0{rng.integers(1, 9)}-15')
                  for i in range(n)])


def test_corruption_is_reproducible_and_disjoint(small):
    spec = Spec(sentinel_dates=20, typo_years=20, exact_duplicates=10, near_duplicates=10,
                price_factors={10: 10, 0.001: 10})
    a, la = corrupt(small, seed=3, spec=spec)
    b, lb = corrupt(small, seed=3, spec=spec)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(la, lb)
    assert len(a) == len(small) + 20
    assert la.groupby('index')['family'].nunique().max() == 1
    assert detectors.invalid_dates(a).sum() == 40


def test_flag_metrics():
    flags = pd.Series([True, True, False, False])
    labels = pd.DataFrame({'index': [0, 2], 'family': 'x', 'subtype': 's'})
    m = flag_metrics(flags, labels, 'x')
    assert (m['tp'], m['fp'], m['fn']) == (1, 1, 1)
    assert m['precision'] == m['recall'] == 0.5
