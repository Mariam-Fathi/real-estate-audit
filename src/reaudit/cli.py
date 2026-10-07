"""Command line entry point:  reaudit audit <listings.csv> [--out reports/]"""
import argparse
import json
import sys
import time
from pathlib import Path

from . import audit
from .contract import ContractError
from .data import load_raw


def main(argv=None):
    parser = argparse.ArgumentParser(prog='reaudit', description='Data-quality audit for realtor.com listings.')
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit', help='validate a listings CSV and flag suspect records')
    a.add_argument('path', help='listings CSV (same columns as the Kaggle USA Real Estate Dataset)')
    a.add_argument('--out', default='reports', help='output directory (default: reports/)')
    args = parser.parse_args(argv)

    t = time.time()
    df = load_raw(args.path)
    try:
        flags, summary = audit.run(df)
    except ContractError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.path).name.split('.')[0]
    flags.to_csv(out / f'{stem}_flags.csv')
    (out / f'{stem}_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out / f'{stem}_report.md').write_text(audit.markdown(summary, Path(args.path).name), encoding='utf-8')

    print(f"{summary['records']:,} records, {summary['records_flagged']:,} flagged "
          f"({time.time() - t:.0f}s) -> {out / (stem + '_report.md')}")
    for k, v in summary['flags'].items():
        print(f'  {k:18s} {v:>8,}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
