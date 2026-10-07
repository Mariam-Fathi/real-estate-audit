"""Data contract for the realtor.com listings file.

Two levels:
- STRUCTURE: what any version of the file must satisfy before an audit makes sense (columns, types, allowed
  categories). A violation stops the audit.
- EXPECTATIONS: what a clean record satisfies. Violations are counted and reported, never fatal: the point of
  the audit is to find them.
"""
import pandas as pd
import pandera.pandas as pa

STATUSES = ['for_sale', 'sold', 'ready_to_build']
DATE_PATTERN = r'^\d{4}-\d{2}-\d{2}$'

_num = dict(nullable=True, coerce=True)

STRUCTURE = pa.DataFrameSchema(
    {
        'brokered_by': pa.Column(float, **_num),
        'status': pa.Column(str, pa.Check.isin(STATUSES)),
        'price': pa.Column(float, **_num),
        'bed': pa.Column(float, **_num),
        'bath': pa.Column(float, **_num),
        'acre_lot': pa.Column(float, **_num),
        'street': pa.Column(float, **_num),
        'city': pa.Column(str, nullable=True),
        'state': pa.Column(str, nullable=True),
        'zip_code': pa.Column(float, **_num),
        'house_size': pa.Column(float, **_num),
        'prev_sold_date': pa.Column(pd.StringDtype(), nullable=True, coerce=True),
    },
    strict=True,
    name='listings structure',
)

EXPECTATIONS = pa.DataFrameSchema(
    {
        'price': pa.Column(float, [pa.Check.gt(0),
                                   pa.Check.le(1e9)], nullable=True),
        'bed': pa.Column(float, pa.Check.in_range(0, 50), nullable=True),
        'bath': pa.Column(float, pa.Check.in_range(0, 50), nullable=True),
        'acre_lot': pa.Column(float, pa.Check.ge(0), nullable=True),
        'house_size': pa.Column(float, [pa.Check.gt(0),
                                        pa.Check.le(100_000)], nullable=True),
        'zip_code': pa.Column(float, pa.Check.in_range(0, 99_999), nullable=True),
        'prev_sold_date': pa.Column(pd.StringDtype(), pa.Check.str_matches(DATE_PATTERN),
                                    nullable=True),
    },
    checks=[
        pa.Check(lambda d: ~((d['status'] == 'sold') & d['prev_sold_date'].isna()),
                 name='sold record has a sale date', element_wise=False),
        pa.Check(lambda d: ~((d['status'] == 'ready_to_build') & d['prev_sold_date'].notna()),
                 name='ready_to_build record has no previous sale', element_wise=False),
    ],
    name='listings expectations',
)


class ContractError(ValueError):
    pass


def enforce_structure(df):
    """Coerce types and validate structure; raises ContractError listing every violation."""
    try:
        return STRUCTURE.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        summary = e.failure_cases.groupby(['column', 'check'], dropna=False).size().rename('violations')
        raise ContractError(f'input does not match the listings structure:\n{summary.to_string()}') from None


def expectation_report(df):
    """One row per expectation: how many records violate it (0 rows when all pass)."""
    try:
        EXPECTATIONS.validate(df, lazy=True)
        return pd.DataFrame(columns=['column', 'check', 'violations'])
    except pa.errors.SchemaErrors as e:
        fc = e.failure_cases
        fc = fc[fc['failure_case'].notna() | fc['column'].isna()]   # nullable columns report no NaN failures
        report = (fc.groupby(['column', 'check'], dropna=False)['index'].nunique()
                    .rename('violations').reset_index())
        report['column'] = report['column'].fillna('(row)')
        return report.sort_values('violations', ascending=False, ignore_index=True)
