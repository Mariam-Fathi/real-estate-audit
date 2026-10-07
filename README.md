# USA Real Estate Data Quality Audit

An audit of the [USA Real Estate Dataset](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset)
(2,226,382 realtor.com listings). The project starts by re-examining my own earlier analysis, which reported that
38.19% of listings were "suspicious".

## Part 1: Root-cause analysis ([notebook](notebooks/01_root_cause_analysis.ipynb))

| Original claim | Finding |
|---|---|
| 734,297 "placeholder dates" | All are ordinary missing values: `astype(str)` turned `NaN` into `'nan'`, which matched the placeholder list. Missingness follows status: 0% for `sold`, 100% for `ready_to_build`. |
| 57,930 "same price, different dates" groups | 99.9% are one property captured twice, once as `for_sale` and once as `sold`. The "time span" is the holding period between sales (median 8.6 years). |
| 66,960 "price manipulation" cases | The same `for_sale` + `sold` pairs, found again with a different key (99.8%). |
| A few brokers drive most flags | The top 3 flagged brokers are the 3 largest by volume, each flagged on about 8.5% of its sales. Flags track sales volume (Spearman ρ = 0.71). |
| (unreported) | `groupby` dropped 57,388 duplicated records that had a missing key value. |

**Corrected picture:** there is no evidence of fraud. 10.4% of records belong to a property that appears more than once
in the dataset. These records must be deduplicated before any property-level modelling.

## Roadmap
1. ✅ Root-cause analysis of the original flags
2. Detector validation: plant known errors and measure precision and recall across seeds
3. Downstream impact: how duplicate leakage between train and test inflates a price model's score
4. A tested Python package with data contracts and CI
5. Republish the Kaggle series

## Reproduce
```bash
pip install -r requirements.txt
python -c "import kagglehub, shutil; p = kagglehub.dataset_download('ahmedshahriarsakib/usa-real-estate-dataset'); shutil.copytree(p, 'data/raw', dirs_exist_ok=True)"
python notebooks/build_01.py
python -m nbconvert --to notebook --execute --inplace notebooks/01_root_cause_analysis.ipynb
```
The notebook asserts every number it reports, so a run that completes reproduces the results above.
