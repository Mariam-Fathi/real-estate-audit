"""Run the full audit on a listings DataFrame: contract, detectors and a summary."""
from datetime import datetime, timezone

import pandas as pd

from . import contract, detectors
from .data import property_ids

FLAG_DESCRIPTIONS = {
    'invalid_date': 'prev_sold_date is unparseable, a known sentinel, or outside 1900-01-02 .. 2026-12-31',
    'listing_duplicate': 'same address, broker and status as another record, with compatible attributes',
    'zero_price': 'price is 0',
    'price_outlier': 'price per sq ft (or price) is extreme for its zip code (absolute modified z > 3.5, MAD floored)',
}
REPORT_COLUMNS = ['status', 'price', 'bed', 'bath', 'house_size', 'acre_lot', 'city', 'state', 'zip_code',
                  'prev_sold_date']


def run(df):
    """Returns (flags, summary): flags has one row per flagged record; summary is JSON-serialisable."""
    df = contract.enforce_structure(df)
    expectations = contract.expectation_report(df)

    z = detectors.price_robust_z(df)
    flags = pd.DataFrame({
        'invalid_date': detectors.invalid_dates(df),
        'listing_duplicate': detectors.duplicates(df),
        'zero_price': df['price'] == 0,
        'price_outlier': (z > detectors.MODIFIED_Z_THRESHOLD) & (df['price'] != 0),
    }, index=df.index)

    pid = property_ids(df)
    repeats = pid.map(pid.value_counts()) > 1
    missing_date = df['prev_sold_date'].isna().groupby(df['status']).mean()

    flagged = flags.any(axis=1)
    out = df.loc[flagged, REPORT_COLUMNS].assign(price_z=z[flagged].round(2))
    out.insert(0, 'flags', flags[flagged].apply(lambda r: ';'.join(r.index[r]), axis=1))
    out.index.name = 'row'

    summary = {
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'records': len(df),
        'records_flagged': int(flagged.sum()),
        'flags': {k: int(v) for k, v in flags.sum().items()},
        'expectations': expectations.to_dict(orient='records'),
        'context': {
            'records_sharing_an_address': int(repeats.sum()),
            'missing_prev_sold_date_by_status': {k: round(float(v), 4) for k, v in missing_date.items()},
        },
    }
    return out, summary


def markdown(summary, source):
    lines = [f'# Data quality audit: `{source}`', '',
             f"{summary['records']:,} records · {summary['records_flagged']:,} flagged "
             f"({summary['records_flagged'] / summary['records']:.2%}) · generated {summary['generated_at']}", '',
             '## Flags', '', '| Flag | Records | Meaning |', '|---|---:|---|']
    lines += [f'| `{k}` | {v:,} | {FLAG_DESCRIPTIONS[k]} |' for k, v in summary['flags'].items()]
    lines += ['', '## Contract expectations', '']
    if summary['expectations']:
        lines += ['| Column | Check | Violations |', '|---|---|---:|']
        lines += [f"| {e['column']} | `{e['check']}` | {e['violations']:,} |" for e in summary['expectations']]
    else:
        lines.append('All expectations hold.')
    ctx = summary['context']
    lines += ['', '## Context (not errors)', '',
              f"- {ctx['records_sharing_an_address']:,} records share their address with another record, usually "
              'a `for_sale` and a `sold` record of the same home. Group by address before property-level modelling.',
              '- Share of records with no `prev_sold_date`, by status (missing is expected for unsold homes): '
              + ', '.join(f'{k} {v:.1%}' for k, v in ctx['missing_prev_sold_date_by_status'].items()), '']
    return '\n'.join(lines)
