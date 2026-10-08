import os

import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
QUAD_ORDER = ["Controlled", "Focal", "Dispersed", "Established"]
Q_DEFAULT = 0.75
MIN_N_GROUP = 20


# --------------------------------------------------------------------------- #
# Risk states
# --------------------------------------------------------------------------- #
def quadrant(capture, posrate, c_thr, p_thr):
    c_hi = np.asarray(capture) > c_thr
    p_hi = np.asarray(posrate) > p_thr
    out = np.full(len(c_hi), "Controlled", dtype=object)
    out[c_hi & ~p_hi] = "Focal"
    out[~c_hi & p_hi] = "Dispersed"
    out[c_hi & p_hi] = "Established"
    return out


def thresholds(df, q=Q_DEFAULT):
    return float(df["density"].quantile(q)), float(df["distribution"].quantile(q))


# --------------------------------------------------------------------------- #
# Transitions and escalation (Fig. 2)
# --------------------------------------------------------------------------- #
def transition_quadrant(df, q=Q_DEFAULT, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    d = df.copy()
    c_thr, p_thr = thresholds(d, q)
    d["cur_q"] = quadrant(d["density"], d["distribution"], c_thr, p_thr)
    d["nxt_q"] = quadrant(d["y_density"], d["y_posrate"], c_thr, p_thr)

    thr_rows = []
    for name, col, nxt, thr in (("Capture rate", "density", "y_density", c_thr),
                                ("Positive rate", "distribution", "y_posrate", p_thr)):
        s = d[col]
        thr_rows.append(dict(metric=name, quantile=q, threshold=thr,
                             pct_above_current=float((s > thr).mean()),
                             pct_at_threshold=float((s == thr).mean()),
                             pct_above_next=float((d[nxt] > thr).mean())))
    pd.DataFrame(thr_rows).to_csv(os.path.join(out_dir, "quadrant_thresholds.csv"), index=False)

    counts = (pd.crosstab(d["cur_q"], d["nxt_q"])
              .reindex(index=QUAD_ORDER, columns=QUAD_ORDER).fillna(0).astype(int))
    counts.to_csv(os.path.join(out_dir, "transition_counts_quadrant.csv"))
    mat = counts.div(counts.sum(axis=1), axis=0)
    mat.to_csv(os.path.join(out_dir, "transition_matrix_quadrant.csv"))

    risk = (d.groupby("cur_q")
              .apply(lambda x: pd.Series({
                  "n": len(x),
                  "p_established_next": float((x["nxt_q"] == "Established").mean())}),
                  include_groups=False)
              .reindex(QUAD_ORDER).reset_index())
    risk.columns = ["state", "n", "p_established_next"]
    risk["n"] = risk["n"].astype(int)
    risk.to_csv(os.path.join(out_dir, "risk_by_current_state.csv"), index=False)

    sub = d[d["cur_q"] != "Established"].copy()
    sub["esc"] = (sub["nxt_q"] == "Established").astype(int)
    eb = sub.groupby("business", observed=True)["esc"].agg(["mean", "size"]).reset_index()
    eb.columns = ["business", "escalation_rate", "n"]
    eb = eb[eb["n"] >= MIN_N_GROUP].sort_values("escalation_rate", ascending=False)
    eb.to_csv(os.path.join(out_dir, "escalation_by_business.csv"), index=False)
    em = sub.groupby("month")["esc"].agg(["mean", "size"]).reset_index()
    em.columns = ["month", "escalation_rate", "n"]
    em.to_csv(os.path.join(out_dir, "escalation_by_month.csv"), index=False)

    pd.DataFrame([dict(n_transitions=int(len(d)), n_establishments=int(d["cid"].nunique()),
                       n_not_established=int(len(sub)),
                       overall_escalation=float(sub["esc"].mean()), quantile=q)]
                 ).to_csv(os.path.join(out_dir, "quadrant_sample.csv"), index=False)
    print(f"Thresholds: capture rate > {c_thr:.3f}, positive rate > {p_thr:.3f}")
    print(mat.round(3).to_string())


def main():
    from analysis_pipeline import get_common_sample
    common, _ = get_common_sample()
    print(f"{len(common):,} transitions, {common['cid'].nunique()} sites")
    transition_quadrant(common)


if __name__ == "__main__":
    main()
