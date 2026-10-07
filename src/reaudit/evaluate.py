import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def flag_metrics(flags, labels, family, preexisting=None):
    """Precision / recall / F1 of a boolean flag Series against the positives of one error family.

    Every unplanted record counts as a negative, so flags on errors already present in the real data count
    as false positives and `precision` is a lower bound. If `preexisting` (flags raised on the unmodified
    data) is given, `precision_upper` also excludes those records from the false positives.
    """
    pos = labels.loc[labels.family == family, 'index'].unique()
    y = flags.index.isin(pos)
    tp = int((flags & y).sum())
    fp = int((flags & ~y).sum())
    fn = int((~flags & y).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    out = {'flagged': tp + fp, 'tp': tp, 'fp': fp, 'fn': fn,
           'precision': precision, 'recall': recall, 'f1': f1}
    if preexisting is not None:
        pre = int((flags & ~y & preexisting.reindex(flags.index, fill_value=False)).sum())
        out['fp_preexisting'] = pre
        out['precision_upper'] = tp / (tp + fp - pre) if tp + fp - pre else np.nan
    return out


def score_metrics(scores, labels, family):
    """Average precision (area under the precision-recall curve) and R-precision of an anomaly score."""
    pos = labels.loc[labels.family == family, 'index'].unique()
    y = scores.index.isin(pos)
    k = int(y.sum())
    top_k = scores.to_numpy().argsort()[::-1][:k]
    finite = np.nan_to_num(scores.to_numpy(), posinf=np.finfo(float).max)
    return {'average_precision': average_precision_score(y, finite),
            'r_precision': y[top_k].mean()}


def recall_by_subtype(flags, labels, family):
    lab = labels[labels.family == family]
    hit = flags.reindex(lab['index']).to_numpy()
    return lab.assign(detected=hit).groupby('subtype')['detected'].mean()
