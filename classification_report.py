import math
import os

import numpy as np
import pandas as pd

from analysis_pipeline import get_common_sample
from features import time_split
from transition_quadrant import Q_DEFAULT, quadrant, thresholds

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
MODELS = ["Naive persistence", "GLM", "Random Forest", "XGBoost"]
FSETS = ["Baseline", "Enhanced"]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _auc(y, score):
    y = np.asarray(y).astype(int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(score).rank().values
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _auprc(y, score):
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        return float("nan")
    ys = y[np.argsort(-np.asarray(score, dtype=float))]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / ys.sum())


def _binary(y, yhat):
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    tp = int(((y == 1) & (yhat == 1)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = (2 * prec * sens / (prec + sens)
          if prec and sens and np.isfinite(prec + sens) else float("nan"))
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, sensitivity=sens, specificity=spec,
                precision=prec, f1=f1, balanced_acc=(sens + spec) / 2)


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def metrics(y_true, y_hat, score):
    m = _binary(y_true, y_hat)
    _, m["sens_lo"], m["sens_hi"] = wilson(m["TP"], m["TP"] + m["FN"])
    _, m["prec_lo"], m["prec_hi"] = wilson(m["TP"], m["TP"] + m["FP"])
    m["flags"] = m["TP"] + m["FP"]
    m["auc"] = _auc(y_true, score)
    m["auprc"] = _auprc(y_true, score)
    m["prevalence"] = float(np.mean(y_true))
    m["n"] = int(len(y_true))
    return m


# --------------------------------------------------------------------------- #
# Early-warning classification (Table 4)
# --------------------------------------------------------------------------- #
def main(q=Q_DEFAULT):
    common, _ = get_common_sample()
    _, test = time_split(common)
    test = test.reset_index(drop=True)
    preds = pd.read_csv(os.path.join(OUT_DIR, "reg_preds_all.csv"))
    c_thr, p_thr = thresholds(common, q)

    keys = [(m, f) for m in MODELS for f in (["—"] if m == "Naive persistence" else FSETS)]
    wide = {}
    for target in ("y_density", "y_posrate"):
        for m, f in keys:
            d = preds[(preds.target == target) & (preds.model == m) & (preds.features == f)]
            d = d.sort_values("row")
            if len(d) != len(test) or not np.allclose(d["obs"].values, test[target].values):
                raise RuntimeError(f"Predictions for {target}/{m}/{f} do not match the test set")
            wide[(target, m, f)] = d["pred"].values

    cur_q = quadrant(test["density"], test["distribution"], c_thr, p_thr)
    nxt_q = quadrant(test["y_density"], test["y_posrate"], c_thr, p_thr)
    ctrl = cur_q == "Controlled"

    rows = []
    for m, f in keys:
        pc, pp = wide[("y_density", m, f)], wide[("y_posrate", m, f)]
        pred_q = quadrant(pc, pp, c_thr, p_thr)
        tasks = [
            ("C. Established next",
             (nxt_q == "Established").astype(int),
             (pred_q == "Established").astype(int),
             np.minimum(pd.Series(pc).rank(pct=True).values,
                        pd.Series(pp).rank(pct=True).values)),
            ("D. Leaves Controlled next (currently Controlled only)",
             (nxt_q[ctrl] != "Controlled").astype(int),
             (pred_q[ctrl] != "Controlled").astype(int),
             np.maximum(pd.Series(pc[ctrl]).rank(pct=True).values,
                        pd.Series(pp[ctrl]).rank(pct=True).values)),
        ]
        for name, y, yhat, score in tasks:
            r = metrics(y, yhat, score)
            r.update(task=name, model=m, features=f)
            rows.append(r)

    out = pd.DataFrame(rows)[
        ["task", "model", "features", "n", "prevalence", "auc", "auprc",
         "sensitivity", "sens_lo", "sens_hi", "precision", "prec_lo", "prec_hi",
         "specificity", "f1", "balanced_acc", "flags", "TP", "FP", "FN", "TN"]]
    out.to_csv(os.path.join(OUT_DIR, "classification_metrics.csv"), index=False)
    print(out[["task", "model", "features", "auc", "auprc", "sensitivity", "precision", "f1"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
