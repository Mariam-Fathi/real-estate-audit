import json

import numpy as np
import pandas as pd
import pytest

from reaudit import audit, contract
from reaudit.cli import main

COLUMNS = ['brokered_by', 'status', 'price', 'bed', 'bath', 'acre_lot', 'street',
           'city', 'state', 'zip_code', 'house_size', 'prev_sold_date']


@pytest.fixture
def listings():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        'brokered_by': rng.integers(1, 20, n).astype(float),
        'status': rng.choice(['for_sale', 'sold'], n),
        'price': rng.lognormal(np.log(300000), 0.2, n).round(),
        'bed': rng.integers(1, 6, n).astype(float),
        'bath': rng.integers(1, 4, n).astype(float),
        'acre_lot': 0.2,
        'street': np.arange(n, dtype=float),
        'city': 'Austin', 'state': 'Texas',
        'zip_code': 78701.0,
        'house_size': rng.normal(1800, 200, n).round(),
        'prev_sold_date': '2015-06-01',
    })
    df.loc[df.status == 'sold', 'prev_sold_date'] = '2022-03-01'
    df.loc[0, 'prev_sold_date'] = '3019-04-02'          # invalid date
    df.loc[1, 'price'] = 0.0                            # zero price
    df.loc[2, 'price'] = df.loc[2, 'price'] * 1000      # unit error
    df.loc[3, 'bed'] = 99.0                             # violates an expectation
    df = pd.concat([df, df.iloc[[10]]], ignore_index=True)   # exact duplicate listing (rows 10 and 200)
    df['prev_sold_date'] = df['prev_sold_date'].astype('string')
    return df[COLUMNS]


def test_structure_rejects_unknown_status(listings):
    bad = listings.assign(status=listings.status.where(listings.index > 0, 'pending'))
    with pytest.raises(contract.ContractError, match='status'):
        contract.enforce_structure(bad)


def test_structure_rejects_missing_column(listings):
    with pytest.raises(contract.ContractError):
        contract.enforce_structure(listings.drop(columns='bed'))


def test_expectation_report_counts_violations(listings):
    report = contract.expectation_report(contract.enforce_structure(listings))
    counts = dict(zip(report.column + ' ' + report.check, report.violations, strict=True))
    assert counts == {'price greater_than(0)': 1, 'bed in_range(0, 50)': 1}


def test_audit_flags_each_planted_problem(listings):
    flags, summary = audit.run(listings)
    by_row = flags['flags'].to_dict()
    assert by_row[0] == 'invalid_date'
    assert by_row[1] == 'zero_price'
    assert by_row[2] == 'price_outlier'
    assert by_row[10] == by_row[200] == 'listing_duplicate'
    assert summary['records'] == len(listings)
    assert summary['flags']['listing_duplicate'] == 2


def test_cli_writes_report(listings, tmp_path):
    src = tmp_path / 'listings.csv'
    listings.to_csv(src, index=False)
    assert main(['audit', str(src), '--out', str(tmp_path / 'out')]) == 0
    summary = json.loads((tmp_path / 'out' / 'listings_summary.json').read_text())
    assert summary['flags']['zero_price'] == 1
    report = (tmp_path / 'out' / 'listings_report.md').read_text()
    assert '| `zero_price` | 1 |' in report


def test_cli_fails_cleanly_on_contract_violation(listings, tmp_path, capsys):
    src = tmp_path / 'bad.csv'
    listings.assign(status='pending').to_csv(src, index=False)
    assert main(['audit', str(src), '--out', str(tmp_path)]) == 2
    assert 'does not match' in capsys.readouterr().err
