"""Builds notebooks/02_detector_validation.ipynb. Run from the repo root: python notebooks/build_02.py

Requires results/validation_*.csv from experiments/run_validation.py.
"""
import nbformat as nbf

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# USA Real Estate Data Quality Audit — Part 2: Validating detectors against planted errors

**Author:** Mariam Fathi · **Data:** USA Real Estate Dataset (2,226,382 listings) · **Code:** `src/reaudit`, `experiments/run_validation.py`

[Part 1](01_root_cause_analysis.ipynb) showed that the original detectors flagged valid records. Showing they are wrong
is not the same as showing a replacement is right. This part measures both on data where the answer is known:
I plant errors into the real dataset, run every detector, and score its flags against the planted labels.

## Design

| Family | Planted per seed | Subtypes | Original detector | Corrected detector |
|---|---:|---|---|---|
| Invalid sale date | 6,000 | 6 sentinel values (`1900-01-01`, `0000-00-00`, `1970-01-01`, `unknown`, `N/A`, `9999-12-31`); year +1000 (2019→3019); transposed year (2015→2051) | placeholder list | strict parse + known sentinels + plausible range 1900-01-02 … 2026-12-31 |
| Duplicate listing | 6,000 copies (12,000 records incl. sources) | exact copy; one of bed/bath/lot/size blanked; price re-keyed within ±0.5% | same price, different dates; price manipulation | same address + broker + **status**, attributes equal or missing, price within 1% |
| Price unit error | 4,000 | price ×10, ×100, ×1000, ÷1000 | — | modified z-score of log price/sqft within zip code (state fallback), threshold 3.5 (Iglewicz & Hoaglin, 1993) |
| | | | | ML baseline: Isolation Forest, sklearn default threshold |

- Errors go into disjoint records of the **real** dataset, so detectors face real noise, not a clean synthetic table.
- 5 random seeds; intervals are 95% t-intervals across seeds.
- **Precision is a range.** Every unplanted record is scored as a negative, so a flag on an error that was *already in
  the data* counts against the detector (lower bound). The upper bound excludes records the detector also flags
  on the unmodified data. For the original detectors only the lower bound is meaningful: Part 1 showed their flags on
  unmodified data are valid records.
""")

md("## 1. Setup")
code(r"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, '../src')
from reaudit import detectors
from reaudit.data import load_raw

m = pd.read_csv('../results/validation_metrics.csv')
sub = pd.read_csv('../results/validation_recall_by_subtype.csv')
base = pd.read_csv('../results/validation_base_flags.csv')
SEEDS = m.seed.nunique()
print(f'{SEEDS} seeds, {m.detector.nunique()} detectors')

BLUE, ORANGE, GRAY = '#2a78d6', '#eb6834', '#b4b2ab'
plt.rcParams.update({
    'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#c9c8c3', 'axes.grid': True, 'grid.color': '#ecebe7', 'grid.linewidth': 0.8,
    'axes.axisbelow': True, 'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'font.size': 10.5,
})


def ci(x):
    x = np.asarray(x, dtype=float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), h


def fmt(x):
    mean, h = ci(x)
    return f'{mean:.3f} ± {h:.3f}'
""")

md("## 2. Results")
code(r"""
ORDER = ['original: placeholder dates', 'corrected: invalid dates',
         'original: same price, different dates', 'original: price manipulation', 'corrected: listing duplicates',
         'corrected: robust z (|z| > 3.5)', 'ML baseline: isolation forest',
         'ML baseline: isolation forest (price features)']
rows = []
for det in ORDER:
    g = m[m.detector == det]
    original = det.startswith('original')
    rows.append({
        'family': g.family.iloc[0], 'detector': det,
        'flags / seed': f'{g.flagged.mean():,.0f}',
        'precision (lower)': fmt(g.precision),
        'precision (upper)': '—' if original else fmt(g.precision_upper),
        'recall': fmt(g.recall), 'F1': fmt(g.f1),
    })
table = pd.DataFrame(rows).set_index(['family', 'detector'])
table
""")
code(r"""
s = m.groupby('detector')[['precision', 'precision_upper', 'recall', 'f1']].mean()
assert s.loc['corrected: invalid dates', 'recall'] == 1.0
assert s.loc['corrected: listing duplicates', 'recall'] == 1.0
assert s.loc['original: placeholder dates', 'precision'] < 0.01
assert s.loc[['original: same price, different dates', 'original: price manipulation'], 'recall'].max() < 0.05
assert s.loc['corrected: robust z (|z| > 3.5)', 'precision_upper'] > 0.8
""")
code(r"""
pairs = [('Invalid dates', 'original: placeholder dates', 'corrected: invalid dates'),
         ('Duplicates', 'original: same price, different dates', 'corrected: listing duplicates')]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
for ax, metric in zip(axes, ['precision', 'recall']):
    y = np.arange(len(pairs))
    for j, (who, color) in enumerate([(1, GRAY), (2, BLUE)]):
        vals = [m[m.detector == p[who]][metric].mean() for p in pairs]
        bars = ax.barh(y + (0.2 if j == 0 else -0.2), vals, height=0.36, color=color,
                       label='original' if j == 0 else 'corrected')
        for b, v in zip(bars, vals):
            ax.text(v + 0.02, b.get_y() + b.get_height() / 2, f'{v:.3f}', va='center', fontsize=9.5)
    ax.set_yticks(y, [p[0] for p in pairs]); ax.set_xlim(0, 1.18); ax.invert_yaxis()
    ax.set_title(metric.capitalize() + (' (lower bound)' if metric == 'precision' else ''))
    ax.grid(axis='y', visible=False)
axes[1].legend(loc='upper right', bbox_to_anchor=(1.0, 1.32), ncol=2, frameon=False)
fig.suptitle(f'Original vs corrected detectors on planted errors (mean of {SEEDS} seeds)',
             x=0.01, ha='left', fontweight='bold', fontsize=13)
plt.tight_layout(); plt.show()
""")

md(r"""
**Reading the table**

- **Invalid dates.** The original detector flags ~738,000 records to catch a third of the planted errors (precision
  0.003): the missing-value bug from Part 1 dominates, and its list misses `1970-01-01`, `9999-12-31` and every typo year.
  The corrected detector finds every planted error. This family is the easiest by construction (see Limitations).
- **Duplicates.** The original detectors find **under 4%** of planted duplicates. They can't see an exact copy at all,
  since identical dates give a time span of 0 days, and the few hits are copies of a record that already belonged to a `for_sale` + `sold` pair
  (300 of 300 source records in seed 0). The corrected detector finds all of them; its lower-bound precision of 0.72 comes from the 4,815 records it
  also flags in the unmodified data (§4).
- **Price unit errors.** At the a-priori threshold of 3.5, the robust z-score catches 92% of planted errors. Its
  lower-bound precision is low because 4% of real listings have genuinely extreme prices per square foot, but 84% of
  its flags in corrupted data are either planted errors or records it already flags in the real data.
""")

md("### Recall by error subtype")
code(r"""
pivot = (sub.groupby(['family', 'subtype', 'detector'])['recall'].mean()
         .unstack('detector')[[d for d in ORDER if d in sub.detector.unique()]])
pivot.style.format('{:.3f}', na_rep='').background_gradient(cmap='Blues', vmin=0, vmax=1, axis=None)
""")
md(r"""
The hardest case is the **×10 price error**: 21% are missed. A tenfold price lands inside the natural spread of some zip
codes (e.g. a luxury home in a mostly modest area), so it is not separable from a real price without more context.
""")

md(r"""
## 3. Domain statistic vs. generic anomaly detection

For the price family I compare *rankings* rather than one threshold: average precision (area under the precision–recall
curve) and R-precision (the share of planted errors among the top-k scores, k = number planted).
""")
code(r"""
price = ['corrected: robust z (|z| > 3.5)', 'ML baseline: isolation forest',
         'ML baseline: isolation forest (price features)']
labels_short = ['Robust z (zip-relative)', 'Isolation Forest (all features)', 'Isolation Forest (price features)']
res = pd.DataFrame([{'detector': lab, 'AP': ci(m[m.detector == d].average_precision)[0],
                     'AP ±': ci(m[m.detector == d].average_precision)[1],
                     'R-precision': ci(m[m.detector == d].r_precision)[0],
                     'R-precision ±': ci(m[m.detector == d].r_precision)[1],
                     'flags at default threshold': m[m.detector == d].flagged.mean()}
                    for d, lab in zip(price, labels_short)])
assert res.AP.iloc[2] > 5 * res.AP.iloc[1]          # features matter more than the model
assert res['R-precision'].iloc[2] - res['R-precision'].iloc[0] < 0.1

fig, ax = plt.subplots(figsize=(7.5, 2.8))
y = np.arange(len(res))
ax.barh(y, res.AP, xerr=res['AP ±'], color=[BLUE, GRAY, BLUE], height=0.55,
        error_kw=dict(ecolor='#52514e', capsize=3, lw=1))
for i, r in res.iterrows():
    ax.text(r.AP + r['AP ±'] + 0.015, i, f'{r.AP:.3f}', va='center')
ax.set_yticks(y, res.detector); ax.invert_yaxis(); ax.set_xlim(0, 0.8)
ax.set_xlabel('Average precision (95% CI across seeds)')
ax.set_title('Zip-relative price features carry the signal, not the model')
ax.grid(axis='y', visible=False)
plt.tight_layout(); plt.show()
res.round(3)
""")
md(r"""
Three results:

1. **Features matter more than the model.** The same Isolation Forest goes from AP 0.08 to 0.59 when it is limited
   to the four price features, two of which are *relative to the zip-code median*. The extra features (bedrooms,
   bathrooms, lot size) make it isolate unusual houses, and an unusual house is not a price error.
2. **With the right features, the two methods rank errors about equally well.** The forest has the higher average
   precision (0.59 vs 0.41), but its R-precision is close (0.58 vs 0.55) and it varies much more between seeds.
3. **Only the z-score gives a usable alert rule as it stands.** At its default threshold the forest flags about 15% of the
   dataset (340,000 records); the z-score's a-priori threshold flags 4% and comes with an explanation a reviewer can
   check ("price per sq ft is far above the zip-code median"). The forest would need a threshold calibrated on
   held-out labels before it could be used for alerts.
""")

md(r"""
## 4. What the corrected detectors find in the real data

Flags on the *unmodified* dataset are not scored (there is no ground truth), but they are the reason to build the
detectors. Each group below is a candidate for review.
""")
code(r"""
raw = load_raw('../data/raw/realtor-data.zip.csv')
bad_dates = raw[detectors.invalid_dates(raw)]
dup = raw[detectors.duplicates(raw)]
z = detectors.price_robust_z(raw)
print(f'invalid dates      : {len(bad_dates)} records -> {bad_dates.prev_sold_date.tolist()}')
print(f'listing duplicates : {len(dup):,} records in {dup.groupby(detectors.LISTING_KEY).ngroups:,} groups')
print(f'price |z| > 3.5    : {(z > 3.5).sum():,} records ({(z > 3.5).mean():.1%}); price = 0: {(raw.price == 0).sum()}')
assert len(bad_dates) == 2 and len(dup) == 4_815
""")
code(r"""
cols = detectors.COMPARE + ['price', 'prev_sold_date']
exact = dup.duplicated(subset=detectors.LISTING_KEY + cols, keep=False).sum()
filled = dup[cols].notna().sum(axis=1)
print(f'exact copies among flagged duplicates: {exact}')
print('status of flagged duplicates:', dup.status.value_counts().to_dict())
dup.sort_values(['street', 'brokered_by']).head(6)[['status', 'street', 'brokered_by'] + cols]
""")
code(r"""
zero = raw[raw.price == 0]
print('status of price = 0 listings:', zero.status.value_counts().to_dict())
raw.assign(z=z).sort_values('z', ascending=False).query('price > 0').head(5)[
    ['status', 'price', 'house_size', 'bed', 'bath', 'city', 'state', 'z']]
""")
md(r"""
- **Dates:** two real errors, a sale in the year 3019 and a `1970-01-01` (the Unix epoch, a classic default).
- **Duplicates:** none of the 4,815 flagged records are exact copies. They are the same listing entered twice with one
  copy missing fields, the same pattern as the *one attribute blanked* subtype. About 1,800 are `ready_to_build`
  listings, which may be several floor plans offered at one development address rather than duplicates; that ambiguity
  is why precision is reported as a range.
- **Prices:** the top flags are `price = 0`, mostly on `ready_to_build` listings. Placeholder values do exist in this
  dataset, but in the **price** column, not the date column the original series searched.
- **A weakness in the z-score:** among positive prices, the top flags have z above 1,000 for ordinary homes (a
  $29,000 house in Watts, OK). In those zip codes about 30% of listings share one identical price, so the median
  absolute deviation is about 0.0005 on the log scale and any other price looks extreme. A floor on the MAD would fix
  this. It is left for the package step rather than tuned here, so the threshold in this experiment stays a-priori.
""")

md(r"""
## 5. Limitations

1. **The same person designed the errors and the detectors.** The date family especially is solved almost by
   construction: the corrected detector's rules (strict format, sentinel list, plausible range) were written knowing what
   would be planted. The duplicate and price families are less circular (the tolerance and the 3.5 threshold were fixed
   before the experiment, and the price detector still misses 21% of ×10 errors), but none of these results show how the
   detectors would do on error types I didn't imagine.
2. **Planted errors are cleaner than real ones.** A real duplicate may differ in several fields at once; a real price
   error may be a transposed digit rather than a power of ten.
3. **Precision depends on how many errors are planted.** With 4,000 price errors among 2.2M records, even a good
   detector's precision is bounded by the real extreme prices it also flags. The upper bound and the ranking metrics
   (AP, R-precision) are the better comparison.
4. **Thresholds are not tuned.** Both thresholds (z > 3.5, Isolation Forest's default) were fixed in advance. Tuning on
   these labels would inflate results unless done on held-out seeds.
5. **The z-score is unstable when a zip code's prices barely vary** (§4); a MAD floor is needed.
6. **One snapshot of one dataset.**

## 6. Takeaways

- The original duplicate detectors found **under 4%** of real duplicates while flagging 116,000–134,000 valid records.
- A corrected, missing-value-tolerant duplicate detector recovered **all** planted duplicates, and found 4,815
  near-duplicate records in the real data.
- For price errors, comparing each listing with its zip code mattered more than the model: Isolation Forest's average
  precision rose from 0.08 to 0.59 with zip-relative price features, roughly matching a robust z-score (0.41) that is
  simpler to explain and has a usable threshold.
- Next, Part 3 asks what these errors cost: how much do duplicate listings that land in both train and test inflate the
  measured accuracy of a price model?
""")

nb = nbf.v4.new_notebook()
nb['cells'] = C
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                  'language_info': {'name': 'python'}}
nbf.write(nb, 'notebooks/02_detector_validation.ipynb')
print('written')
