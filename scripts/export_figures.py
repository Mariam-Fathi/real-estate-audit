"""Copy selected charts from the executed notebooks into docs/figures/ for the README.

Run from the repo root after executing the notebooks:  python scripts/export_figures.py
"""
import base64
from pathlib import Path

import nbformat

# (notebook, index of the chart within that notebook, output name)
FIGURES = [
    ('01_root_cause_analysis.ipynb', 2, 'broker_volume.png'),
    ('02_detector_validation.ipynb', 0, 'original_vs_corrected.png'),
    ('02_detector_validation.ipynb', 1, 'price_detectors.png'),
    ('03_duplicate_leakage.ipynb', 0, 'leakage_dose_response.png'),
    ('03_duplicate_leakage.ipynb', 1, 'leakage_mechanism.png'),
]


def charts(path):
    nb = nbformat.read(path, as_version=4)
    return [o['data']['image/png'] for c in nb.cells for o in c.get('outputs', []) if 'image/png' in o.get('data', {})]


def main():
    out = Path('docs/figures')
    out.mkdir(parents=True, exist_ok=True)
    for notebook, i, name in FIGURES:
        (out / name).write_bytes(base64.b64decode(charts(Path('notebooks') / notebook)[i]))
        print(f'{notebook} chart {i} -> {out / name}')


if __name__ == '__main__':
    main()
