import glob
import hashlib
import json
import os
import shutil
import subprocess
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder

from features import (APP_ERA_START, FACILITY_ITEMS, SANITATION_ITEMS, add_lags,
                      build_features, feature_groups, load_monitoring, train_mask,
                      time_split)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
RNG = 42
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)
CAT = ["region", "business", "building"]

N_PERM_REPEATS = 30
SHADOW_N = 999
SHADOW_INNER_REPEATS = 10
SHADOW_SEED = 20260901
N_JOBS = -1

TUNE_REUSE = True
TUNE_N_SPLITS = 5
RF_N_ESTIMATORS = 500
XGB_N_ESTIMATORS = 400
RF_DEFAULT = dict(max_features="sqrt", min_samples_leaf=1)
XGB_DEFAULT = dict(max_depth=4, learning_rate=0.05, min_child_weight=1)
RF_GRID = [dict(max_features=mf, min_samples_leaf=ml)
           for mf in (0.15, 0.20, 0.25, 0.33, 0.50, 0.65, 0.80, 1.0)
           for ml in (1, 2, 5, 10, 20, 50)]
XGB_GRID = [dict(max_depth=md, learning_rate=lr, n_estimators=400, min_child_weight=mcw)
            for md in (1, 2, 3, 4, 5, 6, 7, 8)
            for lr in (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03)
            for mcw in (1, 5, 20, 50, 100, 200)]
NATURAL_FLOOR = {"min_samples_leaf": 1, "min_child_weight": 1, "max_depth": 1}
NATURAL_CEIL = {"max_features": 1.0}

GLMM_YEARMONTH_RANGE = (202504, 202603)
MAX_HORIZON = 6

BIZ_REF, REG_REF, BLD_REF = "迴轉壽司店", "臺北市", "y"
SEASON_MAP = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}

TARGETS = {
    "y_density": dict(current="density", xgb_obj="reg:tweedie"),
    "y_posrate": dict(current="distribution", xgb_obj="reg:squarederror"),
}


def save_csv(df, name):
    df.to_csv(os.path.join(OUT_DIR, name), index=False)
    return df


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
TUNED_PARAMS = {}


def glm_family(target):
    if target == "y_density":
        return sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log())
    if target == "y_posrate":
        return sm.families.Binomial()
    raise ValueError(f"Unknown target {target!r}")


def _params(kind, target, fset, default):
    return dict(TUNED_PARAMS.get((kind, target, fset), default))


def make_rf(target=None, fset="Enhanced", n_jobs=-1):
    p = dict(n_estimators=RF_N_ESTIMATORS)
    p.update(_params("rf", target, fset, RF_DEFAULT))
    return RandomForestRegressor(random_state=RNG, n_jobs=n_jobs, **p)


def make_xgb(target, fset="Enhanced", n_jobs=None):
    p = dict(n_estimators=XGB_N_ESTIMATORS, subsample=0.9, colsample_bytree=0.8)
    p.update(_params("xgb", target, fset, XGB_DEFAULT))
    return xgb.XGBRegressor(objective=TARGETS[target]["xgb_obj"], random_state=RNG,
                            n_jobs=n_jobs, **p)


def encode(train, test, feats):
    cat = [c for c in CAT if c in feats]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xtr, Xte = train[feats].copy(), test[feats].copy()
    if cat:
        Xtr[cat] = enc.fit_transform(Xtr[cat].astype(str))
        Xte[cat] = enc.transform(Xte[cat].astype(str))
    return Xtr, Xte


_GLM_REF = {"region": REG_REF, "business": BIZ_REF, "building": BLD_REF}


def _glm_levels(train, col):
    if isinstance(train[col].dtype, pd.CategoricalDtype):
        lv = [str(x) for x in train[col].cat.categories if x in set(train[col].dropna())]
    else:
        lv = sorted({str(x) for x in train[col].dropna()})
    ref = _GLM_REF.get(col)
    if ref in lv:
        lv = [ref] + [x for x in lv if x != ref]
    return lv


def encode_glm(train, test, feats):
    cats = [c for c in CAT if c in feats]
    nums = [c for c in feats if c not in cats]
    levels = {c: _glm_levels(train, c) for c in cats}

    def design(d):
        X = d[nums].astype(float).copy()
        for c in cats:
            s = pd.Categorical(d[c].astype(str), categories=levels[c])
            D = pd.get_dummies(s, prefix=c, drop_first=True, dtype=float)
            D.index = d.index
            X = pd.concat([X, D], axis=1)
        return X

    Xtr, Xte = design(train), design(test)
    return Xtr, Xte.reindex(columns=Xtr.columns, fill_value=0.0)


def fit_predict_glm(train, test, target, feats, family_key=None):
    Xtr, Xte = encode_glm(train, test, feats)
    glm = sm.GLM(train[target].values, sm.add_constant(Xtr, has_constant="add"),
                 family=glm_family(family_key or target)).fit()
    return np.asarray(glm.predict(sm.add_constant(Xte, has_constant="add")))


def reg_metrics(y, mu):
    return dict(r2=r2_score(y, mu), rmse=np.sqrt(mean_squared_error(y, mu)),
                mae=mean_absolute_error(y, mu))


# --------------------------------------------------------------------------- #
# Analytical sample
# --------------------------------------------------------------------------- #
def get_common_sample():
    df, tiers = build_features()
    num = [f for f in tiers["Enhanced"] if f not in CAT]
    common = df.dropna(subset=num).copy()

    def reorder(col, ref):
        levels = [ref] + [x for x in common[col].value_counts().index if x != ref]
        return pd.Categorical(common[col], categories=levels)

    common["business"] = reorder("business", BIZ_REF)
    common["region"] = reorder("region", REG_REF)
    common["building"] = reorder("building", BLD_REF)
    return common, tiers


# --------------------------------------------------------------------------- #
# Hyperparameter tuning (expanding-window CV within the training set)
# --------------------------------------------------------------------------- #
def _time_folds(dates, n_splits=TUNE_N_SPLITS):
    d = pd.Series(pd.to_datetime(np.asarray(dates))).reset_index(drop=True)
    qs = [j / (n_splits + 1) for j in range(1, n_splits + 2)]
    cuts = [d.quantile(q) for q in qs]
    folds = []
    for k in range(n_splits):
        lo, hi = cuts[k], cuts[k + 1]
        tr = np.where(d <= lo)[0]
        va = np.where((d > lo) & (d <= hi))[0]
        if len(tr) >= 50 and len(va) >= 20:
            folds.append((tr, va))
    return folds


def _cv_score(kind, params, target, train, feats, folds):
    scores = []
    for tr_pos, va_pos in folds:
        a, b = train.iloc[tr_pos], train.iloc[va_pos]
        Xa, Xb = encode(a, b, feats)
        if kind == "rf":
            p = dict(n_estimators=RF_N_ESTIMATORS)
            p.update(params)
            mdl = RandomForestRegressor(random_state=RNG, n_jobs=1, **p)
        else:
            p = dict(n_estimators=XGB_N_ESTIMATORS, subsample=0.9, colsample_bytree=0.8)
            p.update(params)
            mdl = xgb.XGBRegressor(objective=TARGETS[target]["xgb_obj"],
                                   random_state=RNG, n_jobs=1, **p)
        try:
            mdl.fit(Xa, a[target].values)
            scores.append(r2_score(b[target].values, mdl.predict(Xb)))
        except Exception:
            return np.full(len(folds), -np.inf)
    return np.asarray(scores, dtype=float)


def _grid_signature():
    payload = json.dumps({
        "rf": sorted(json.dumps(g, sort_keys=True) for g in RF_GRID),
        "xgb": sorted(json.dumps(g, sort_keys=True) for g in XGB_GRID),
        "n_splits": TUNE_N_SPLITS, "se_mult": 0.0,
        "rf_n": RF_N_ESTIMATORS, "xgb_n": XGB_N_ESTIMATORS,
    }, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def _grid_edges(grid, best):
    edge = []
    for k, v in best.items():
        vals = {g[k] for g in grid}
        if len(vals) == 1:
            continue
        lo, hi = min(vals), max(vals)
        if (v == lo and NATURAL_FLOOR.get(k) == v) or (v == hi and NATURAL_CEIL.get(k) == v):
            continue
        if v in (lo, hi):
            edge.append(f"{k}={v}")
    return ";".join(edge)


def tune_hyperparams(train, base_feats, full_feats):
    path = os.path.join(OUT_DIR, "hyperparams.csv")
    sig = _grid_signature()
    if TUNE_REUSE and os.path.exists(path):
        prev = pd.read_csv(path)
        if "grid_hash" in prev.columns and str(prev["grid_hash"].iloc[0]) == sig:
            for _, r in prev.iterrows():
                TUNED_PARAMS[(r["kind"], r["target"], r["fset"])] = json.loads(r["params"])
            print(f"  reusing {path}")
            return prev

    folds = _time_folds(train["put"])
    print(f"  {len(folds)} folds; grid sizes RF {len(RF_GRID)}, XGBoost {len(XGB_GRID)}")
    rows = []
    for target in ("y_density", "y_posrate"):
        for fset, feats in (("Baseline", base_feats), ("Enhanced", full_feats)):
            for kind, grid, default in (("rf", RF_GRID, RF_DEFAULT),
                                        ("xgb", XGB_GRID, XGB_DEFAULT)):
                S = np.vstack(Parallel(n_jobs=N_JOBS)(
                    delayed(_cv_score)(kind, g, target, train, feats, folds) for g in grid))
                means = S.mean(axis=1)
                ses = np.where(np.all(np.isfinite(S), axis=1),
                               S.std(axis=1, ddof=1) / np.sqrt(S.shape[1]), np.inf)
                best = int(np.argmax(means))
                params = dict(grid[best]) if np.isfinite(means[best]) else dict(default)
                TUNED_PARAMS[(kind, target, fset)] = params
                s_def = _cv_score(kind, default, target, train, feats, folds).mean()
                edge = _grid_edges(grid, grid[best])
                rows.append(dict(kind=kind, target=target, fset=fset,
                                 params=json.dumps(params), cv_r2=means[best],
                                 cv_se=ses[best], cv_r2_default=s_def, at_grid_edge=edge,
                                 n_grid=len(grid), n_folds=len(folds), grid_hash=sig))
                print(f"  {target}/{fset}/{kind}: {params}  CV R2 = {means[best]:.4f}"
                      f"  (default {s_def:.4f}){'  edge: ' + edge if edge else ''}")
    return save_csv(pd.DataFrame(rows), "hyperparams.csv")


# --------------------------------------------------------------------------- #
# Permutation importance and shadow-feature test
# --------------------------------------------------------------------------- #
def _perm_one_group(model, X, y, base, cols, n_repeats, seed):
    X = X.copy()
    rng = np.random.default_rng(seed)
    orig = X[cols].to_numpy(copy=True)
    out = np.empty(n_repeats, dtype=float)
    for r in range(n_repeats):
        idx = rng.permutation(len(X))
        X.loc[:, cols] = orig[idx]
        out[r] = base - r2_score(y, model.predict(X))
    return out


def grouped_permutation_importance(model, X, y, group_cols, n_repeats=N_PERM_REPEATS,
                                   seed=0, n_jobs=1):
    X = pd.DataFrame(X).copy()
    y = np.asarray(y)
    base = r2_score(y, model.predict(X))
    cols_list = [[c for c in g if c in X.columns] for g in group_cols]
    seeds = np.random.SeedSequence(seed).spawn(len(cols_list))
    if n_jobs == 1:
        raw = np.vstack([_perm_one_group(model, X, y, base, c, n_repeats, s)
                         for c, s in zip(cols_list, seeds)])
    else:
        raw = np.vstack(Parallel(n_jobs=n_jobs)(
            delayed(_perm_one_group)(model, X, y, base, c, n_repeats, s)
            for c, s in zip(cols_list, seeds)))
    return raw.mean(axis=1), raw.std(axis=1, ddof=1), raw


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1.0)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q


def _make_shadow(X, rng, group_cols):
    X = X.reset_index(drop=True)
    cols, shadow_groups = {}, []
    for grp in group_cols:
        grp = [c for c in grp if c in X.columns]
        if not grp:
            continue
        idx = rng.permutation(len(X))
        sh = []
        for c in grp:
            cols[f"{c}__shadow"] = X[c].values[idx]
            sh.append(f"{c}__shadow")
        shadow_groups.append(sh)
    return pd.concat([X, pd.DataFrame(cols)], axis=1), shadow_groups


def _shadow_one_replicate(b, Xtr, ytr, Xte, yte, target, group_cols):
    rng = np.random.default_rng(SHADOW_SEED + b)
    Xtr_a, sh_groups = _make_shadow(Xtr, rng, group_cols)
    Xte_a, _ = _make_shadow(Xte, rng, group_cols)
    all_groups = list(group_cols) + list(sh_groups)
    n_g = len(group_cols)
    out = {}
    for mname, mdl in (("Random Forest", make_rf(target, "Enhanced", n_jobs=1).fit(Xtr_a, ytr)),
                       ("XGBoost", make_xgb(target, "Enhanced", n_jobs=1).fit(Xtr_a, ytr))):
        m, _, _ = grouped_permutation_importance(
            mdl, Xte_a, yte, all_groups, n_repeats=SHADOW_INNER_REPEATS,
            seed=int(rng.integers(0, 2**31 - 1)))
        out[mname] = (m[:n_g], m[n_g:])
    return out


def shadow_test(Xtr, ytr, Xte, yte, target, group_cols):
    t0 = time.time()
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_shadow_one_replicate)(b, Xtr, ytr, Xte, yte, target, group_cols)
        for b in range(SHADOW_N))
    out = {}
    for mname in ("Random Forest", "XGBoost"):
        real = np.vstack([r[mname][0] for r in results])
        shad = np.vstack([r[mname][1] for r in results])
        out[mname] = dict(obs=real.mean(axis=0), null=shad,
                          win_rate=(real > shad).mean(axis=0))
    print(f"    shadow test {target}: {SHADOW_N} replicates, {time.time() - t0:.0f} s")
    return out


# --------------------------------------------------------------------------- #
# Part 1: GLMM (R/glmmTMB)
# --------------------------------------------------------------------------- #
RSCRIPT_PATH = os.environ.get("RSCRIPT_PATH")
BIZ_R = {"迴轉壽司店": "ConveyorSushi", "烘焙坊": "Bakery", "西餐": "Western", "火鍋店": "HotPot",
         "拉麵店": "Ramen", "咖啡店": "Cafe", "其他": "Other", "日式": "Japanese", "中式": "Chinese"}
REG_R = {"臺北市": "Taipei", "臺中市": "Taichung", "新北市": "NewTaipei", "南投縣": "Nantou",
         "高雄市": "Kaohsiung", "新竹市": "HsinchuCity", "桃園市": "Taoyuan", "臺南市": "Tainan",
         "Other city": "OtherCity"}


def _find_rscript():
    if RSCRIPT_PATH:
        return RSCRIPT_PATH if os.path.exists(RSCRIPT_PATH) else None
    for name in ("Rscript", "Rscript.exe"):
        if shutil.which(name):
            return shutil.which(name)
    cands = []
    for pat in (r"C:\Program Files\R\R-*\bin\Rscript.exe",
                r"C:\Program Files\R\R-*\bin\x64\Rscript.exe",
                "/usr/local/bin/Rscript", "/opt/homebrew/bin/Rscript",
                "/Library/Frameworks/R.framework/Resources/bin/Rscript"):
        cands.extend(glob.glob(pat))
    return sorted(cands)[-1] if cands else None


def fit_drivers(df):
    need = ["y_number", "y_catch_n", "y_all_n"]
    d = df.dropna(subset=need).copy()
    d = d[d["y_all_n"] > 0].copy()
    lo, hi = GLMM_YEARMONTH_RANGE
    d = d[(d["yearmonth"] >= lo) & (d["yearmonth"] <= hi)]
    if (d["yearmonth"] >= APP_ERA_START).any():
        raise RuntimeError("GLMM window overlaps the app-based audit period")
    print(f"  GLMM sample: {len(d):,} transitions, {d['cid'].nunique()} sites")

    d["season"] = pd.to_numeric(d["month"]).astype("Int64").map(SEASON_MAP).astype("object")
    d["business"] = d["business"].astype(str).str.strip().map(BIZ_R)
    d["region"] = d["region"].astype(str).str.strip().map(REG_R)
    d["building"] = d["building"].astype(str).str.strip().map({"y": "InMall", "n": "Standalone"})
    for col in ("season", "business", "region", "building"):
        if d[col].isna().any():
            raise RuntimeError(f"Unmapped category in {col}")

    items = FACILITY_ITEMS + SANITATION_ITEMS
    cols = ["cid", "y_density", "y_posrate", "y_number", "y_catch_n", "y_all_n",
            "density", "distribution", "temp", "humidity", *items,
            "season", "region", "business", "building"]
    inp = os.path.join(OUT_DIR, "glmm_input.csv")
    d[cols].to_csv(inp, index=False, encoding="utf-8")

    rscript = _find_rscript()
    if rscript is None:
        raise RuntimeError("Rscript not found; install R and glmmTMB or set RSCRIPT_PATH")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fit_glmm.R")
    done_flag = os.path.join(OUT_DIR, "glmm_done.txt")
    if os.path.exists(done_flag):
        os.remove(done_flag)
    res = subprocess.run([rscript, script, inp, OUT_DIR], capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if not os.path.exists(done_flag):
        raise RuntimeError(f"fit_glmm.R failed (exit code {res.returncode}):\n"
                           + (res.stdout or "") + (res.stderr or ""))
    print(res.stdout.strip())


# --------------------------------------------------------------------------- #
# Part 2: Baseline vs Enhanced, predictor importance
# --------------------------------------------------------------------------- #
def evaluate(train, test, target, base_feats, full_feats):
    ytr, yte = train[target].values, test[target].values
    rows, all_preds = [], []
    row_idx = np.arange(len(test))
    for fset, feats in (("Baseline", base_feats), ("Enhanced", full_feats)):
        Xtr, Xte = encode(train, test, feats)
        mu_glm = fit_predict_glm(train, test, target, feats)
        mu_rf = make_rf(target, fset).fit(Xtr, ytr).predict(Xte)
        mu_xg = make_xgb(target, fset).fit(Xtr, ytr).predict(Xte)
        n_feat = len(feature_groups(feats)[0])
        for mname, mu in (("GLM", mu_glm), ("Random Forest", mu_rf), ("XGBoost", mu_xg)):
            m = reg_metrics(yte, np.asarray(mu))
            m.update(target=target, model=mname, features=fset, n_feat=n_feat, n_col=len(feats))
            rows.append(m)
            all_preds.append(pd.DataFrame({"row": row_idx, "obs": yte, "pred": np.asarray(mu),
                                           "target": target, "model": mname,
                                           "features": fset}))
    current = test[TARGETS[target]["current"]].values
    all_preds.append(pd.DataFrame({"row": row_idx, "obs": yte, "pred": current,
                                   "target": target, "model": "Naive persistence",
                                   "features": "—"}))
    preds_all = pd.concat(all_preds, ignore_index=True)

    Xtr, Xte = encode(train, test, full_feats)
    group_names, group_cols = feature_groups(full_feats)
    xg = make_xgb(target, "Enhanced").fit(Xtr, ytr)
    rf = make_rf(target, "Enhanced").fit(Xtr, ytr)
    mu_glm_e = fit_predict_glm(train, test, target, full_feats)
    xgb_imp, xgb_sd, xgb_raw = grouped_permutation_importance(xg, Xte, yte, group_cols,
                                                              n_jobs=N_JOBS)
    rf_imp, rf_sd, rf_raw = grouped_permutation_importance(rf, Xte, yte, group_cols,
                                                           n_jobs=N_JOBS)
    imp = pd.DataFrame({"feature": group_names, "xgb_imp": xgb_imp, "xgb_sd": xgb_sd,
                        "rf_imp": rf_imp, "rf_sd": rf_sd, "target": target})
    imp["n_col"] = [len(c) for c in group_cols]

    reps = pd.DataFrame([dict(feature=f, model=mname, target=target, repeat=r,
                              delta_r2=float(raw[fi, r]))
                         for mname, raw in (("Random Forest", rf_raw), ("XGBoost", xgb_raw))
                         for fi, f in enumerate(group_names)
                         for r in range(raw.shape[1])])

    sh = shadow_test(Xtr, ytr, Xte, yte, target, group_cols)
    for mname, pfx in (("Random Forest", "rf"), ("XGBoost", "xgb")):
        obs, null = sh[mname]["obs"], sh[mname]["null"]
        p = (1.0 + (null >= obs[None, :]).sum(axis=0)) / (null.shape[0] + 1.0)
        imp[f"{pfx}_sh_imp"] = obs
        imp[f"{pfx}_sh_p"] = p
        imp[f"{pfx}_sh_q"] = bh_fdr(p)
        imp[f"{pfx}_sh_null_mean"] = null.mean(axis=0)
        imp[f"{pfx}_sh_null_sd"] = null.std(axis=0, ddof=1)
        imp[f"{pfx}_sh_null_q95"] = np.quantile(null, 0.95, axis=0)
        imp[f"{pfx}_sh_win_rate"] = sh[mname]["win_rate"]
    imp["shadow_n"] = SHADOW_N
    imp["shadow_inner_repeats"] = SHADOW_INNER_REPEATS

    pred_df = pd.concat([
        pd.DataFrame({"obs": yte, "pred": current, "target": target,
                      "model": "Naive persistence"}),
        pd.DataFrame({"obs": yte, "pred": mu_glm_e, "target": target, "model": "GLM"}),
        pd.DataFrame({"obs": yte, "pred": rf.predict(Xte), "target": target,
                      "model": "Random Forest"}),
        pd.DataFrame({"obs": yte, "pred": xg.predict(Xte), "target": target,
                      "model": "XGBoost"}),
    ], ignore_index=True)

    m_naive = reg_metrics(yte, current)
    m_naive.update(target=target, model="Naive persistence", features="—", n_feat=0, n_col=0)
    rows.append(m_naive)
    return pd.DataFrame(rows), imp, pred_df, reps, preds_all


# --------------------------------------------------------------------------- #
# Part 3: forecast horizons
# --------------------------------------------------------------------------- #
def multi_horizon(base_feats, full_feats):
    raw = load_monitoring()
    g = raw.groupby("cid", group_keys=False)
    raw = add_lags(raw, g)
    rows = []
    for target_col, tname, tkey in (("distribution", "posrate", "y_posrate"),
                                    ("density", "capture", "y_density")):
        for h in range(1, MAX_HORIZON + 1):
            d = raw.copy()
            d["y"] = g[target_col].shift(-h)
            d["step"] = g["seq"].shift(-h) - d["seq"]
            d = d[d["step"] == h].copy()
            d = d.dropna(subset=[f for f in full_feats if f not in CAT])
            m = train_mask(d)
            tr, te = d[m], d[~m]
            for mname in ("GLM", "Random Forest", "XGBoost"):
                rec = dict(target=tname, model=mname, horizon=h, weeks=h * 2, n_test=len(te))
                for fset, feats in (("Baseline", base_feats), ("Enhanced", full_feats)):
                    Xtr, Xte = encode(tr, te, feats)
                    try:
                        if mname == "GLM":
                            mu = fit_predict_glm(tr, te, "y", feats, family_key=tkey)
                        elif mname == "Random Forest":
                            mu = make_rf(tkey, fset).fit(Xtr, tr["y"].values).predict(Xte)
                        else:
                            mu = make_xgb(tkey, fset).fit(Xtr, tr["y"].values).predict(Xte)
                        rec[fset] = r2_score(te["y"], mu)
                    except Exception as e:
                        print(f"  ! {tname}/h={h}/{mname}/{fset} failed: {e}")
                        rec[fset] = float("nan")
                rows.append(rec)
    return save_csv(pd.DataFrame(rows), "horizon_decay.csv")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    common, tiers = get_common_sample()
    base_feats, full_feats = tiers["Baseline"], tiers["Enhanced"]
    train, test = time_split(common)
    print(f"Analytical sample: {len(common):,} transitions, {common['cid'].nunique()} sites; "
          f"train {len(train):,}, test {len(test):,}")

    print("[Part 0] Hyperparameter tuning")
    tune_hyperparams(train, base_feats, full_feats)

    print("[Part 1] GLMM")
    fit_drivers(common)

    print("[Part 2] Baseline vs Enhanced")
    res = [evaluate(train, test, t, base_feats, full_feats) for t in ("y_density", "y_posrate")]
    perf, imp, pred, reps, preds_all = (pd.concat(x, ignore_index=True) for x in zip(*res))
    save_csv(perf, "perf_baseline_vs_enhanced.csv")
    save_csv(imp, "perm_importance.csv")
    save_csv(pred, "reg_preds.csv")
    save_csv(preds_all, "reg_preds_all.csv")
    save_csv(reps, "perm_importance_repeats.csv")
    for t in ("y_density", "y_posrate"):
        p = perf[perf.target == t]
        print(f"  {t}: " + ", ".join(f"{r.model}/{r.features} R2 = {r.r2:.3f}"
                                      for r in p.itertuples()))

    print("[Part 3] Forecast horizons")
    multi_horizon(base_feats, full_feats)
    print(f"Done. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
