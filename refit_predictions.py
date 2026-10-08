import json
import os

import numpy as np
import pandas as pd

from analysis_pipeline import (OUT_DIR, TARGETS, TUNED_PARAMS, encode, fit_predict_glm,
                               get_common_sample, make_rf, make_xgb)
from features import feature_groups, time_split

MODELS = ["GLM", "Random Forest", "XGBoost"]
FSETS = ["Baseline", "Enhanced"]


# --------------------------------------------------------------------------- #
# Tuned hyperparameters
# --------------------------------------------------------------------------- #
def load_tuned_params():
    hp = pd.read_csv(os.path.join(OUT_DIR, "hyperparams.csv"))
    for _, r in hp.iterrows():
        TUNED_PARAMS[(r["kind"], r["target"], r["fset"])] = json.loads(r["params"])


def metrics_with_bias(y, mu):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    res = mu - y
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return dict(r2=1.0 - float((res ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan,
                rmse=float(np.sqrt((res ** 2).mean())),
                mae=float(np.abs(res).mean()),
                bias=float(res.mean()))


# --------------------------------------------------------------------------- #
# Test-set predictions with bias (Table 3)
# --------------------------------------------------------------------------- #
def main():
    load_tuned_params()
    common, tiers = get_common_sample()
    train, test = time_split(common)
    test = test.reset_index(drop=True)
    n_pred = {fs: len(feature_groups(tiers[fs])[0]) for fs in FSETS}

    perf_rows, pred_frames = [], []
    for target in ("y_density", "y_posrate"):
        ytr, yte = train[target].values, test[target].values
        cur = test[TARGETS[target]["current"]].values
        m = metrics_with_bias(yte, cur)
        m.update(target=target, model="Naive persistence", features="—", n_feat=0, n_col=0)
        perf_rows.append(m)
        pred_frames.append(pd.DataFrame({"row": np.arange(len(test)), "obs": yte, "pred": cur,
                                         "target": target, "model": "Naive persistence",
                                         "features": "—"}))
        for fset in FSETS:
            feats = tiers[fset]
            Xtr, Xte = encode(train, test, feats)
            mu = {"GLM": fit_predict_glm(train, test, target, feats),
                  "Random Forest": make_rf(target, fset).fit(Xtr, ytr).predict(Xte),
                  "XGBoost": make_xgb(target, fset).fit(Xtr, ytr).predict(Xte)}
            for mname in MODELS:
                p = np.asarray(mu[mname], dtype=float)
                r = metrics_with_bias(yte, p)
                r.update(target=target, model=mname, features=fset,
                         n_feat=n_pred[fset], n_col=len(feats))
                perf_rows.append(r)
                pred_frames.append(pd.DataFrame({"row": np.arange(len(test)), "obs": yte,
                                                 "pred": p, "target": target, "model": mname,
                                                 "features": fset}))
                print(f"  {target:<10s} {mname:<14s} {fset:<9s} R2 = {r['r2']:6.3f}  "
                      f"bias = {r['bias']:+.4f}")

    perf = pd.DataFrame(perf_rows)[["target", "model", "features", "n_feat", "n_col",
                                    "r2", "rmse", "mae", "bias"]]
    perf.to_csv(os.path.join(OUT_DIR, "perf_extended.csv"), index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "reg_preds_all.csv"), index=False)


if __name__ == "__main__":
    main()
