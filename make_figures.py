import os

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from journal_style import add_row_labels, apply_style, rows_height_mm, save_journal, size

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
apply_style()
mpl.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "sans",
                     "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
                     "mathtext.cal": "sans", "mathtext.sf": "sans",
                     "mathtext.bfit": "sans:italic:bold"})

INK, SLATE, TEAL, TEALL = "#22303c", "#4a6572", "#0f7c8a", "#7fc6cf"
AMBER, RUST = "#d98c2b", "#b04a36"
GREY, GREYL = "#9aa6ad", "#d9e0e4"
CAP_C, POS_C = TEAL, RUST
PT_PANEL, PT_TITLE, PT_ROWLAB, PT_LABEL, PT_TICK, PT_ANNOT = 10.0, 9.0, 9.0, 8.5, 8.0, 8.0
HATCH_B = "///"
MODEL_DISPLAY = {"Random Forest": "Random forest"}
NAIVE = "Naive persistence"

TRANS = {
    "迴轉壽司店": "Conveyor-belt sushi", "烘焙坊": "Bakery", "西餐": "Western-style",
    "火鍋店": "Hot-pot", "拉麵店": "Ramen", "咖啡店": "Café", "其他": "Other",
    "日式": "Japanese", "中式": "Chinese",
    "臺北市": "Taipei", "臺中市": "Taichung", "新北市": "New Taipei", "高雄市": "Kaohsiung",
    "臺南市": "Tainan", "新竹市": "Hsinchu City", "桃園市": "Taoyuan", "南投縣": "Nantou",
    "Other city": "Other city",
    "y": "In-mall / department store", "n": "Standalone / street-level",
}
FEATURE_LABELS = {
    "density": "Current capture rate", "distribution": "Current positive rate",
    "business": "Business type", "region": "Region", "month": "Month",
    "temp": "Outdoor temperature", "building": "Building type", "humidity": "Relative humidity",
    "density_lag1": "Capture rate (lag 1)", "posrate_lag1": "Positive rate (lag 1)",
    "density_lag2": "Capture rate (lag 2)", "posrate_lag2": "Positive rate (lag 2)",
}


def load(name):
    return pd.read_csv(os.path.join(OUT, name))


def trans(s):
    return TRANS.get(s, s)


def despine(*axes, keep=("left", "bottom")):
    for ax in axes:
        for side in ("top", "right", "left", "bottom"):
            if side not in keep:
                ax.spines[side].set_visible(False)


def wilson_ci(k, n, z=1.96):
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return np.clip(centre - half, 0.0, 1.0), np.clip(centre + half, 0.0, 1.0)


def counts_from_rate(df, rate_col="escalation_rate", n_col="n"):
    return np.round(df[rate_col].values * df[n_col].values).astype(int)


# --------------------------------------------------------------------------- #
# Fig. 2: risk-state transitions and escalation
# --------------------------------------------------------------------------- #
QUAD_ORDER = ["Established", "Focal", "Dispersed", "Controlled"]


def fig2_transition():
    mat = pd.read_csv(os.path.join(OUT, "transition_matrix_quadrant.csv"), index_col=0)
    risk = load("risk_by_current_state.csv")
    esc_biz = load("escalation_by_business.csv")
    esc_mon = load("escalation_by_month.csv")
    mat = mat.reindex(index=QUAD_ORDER, columns=QUAD_ORDER)
    risk = risk.set_index("state").reindex(QUAD_ORDER).reset_index()
    est_col = QUAD_ORDER.index("Established")
    nz = risk[risk["state"] != "Established"]
    overall_esc = float((nz["p_established_next"] * nz["n"]).sum() / nz["n"].sum())

    fig = plt.figure(figsize=size("full", height_mm=150.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.02], height_ratios=[1.0, 0.92],
                          wspace=0.42, hspace=0.50,
                          left=0.185, right=0.985, top=0.95, bottom=0.115)

    ax0 = fig.add_subplot(gs[0, 0])
    M = mat.values
    for i in range(4):
        for j in range(4):
            v = float(M[i, j])
            if j == est_col:
                face, txt_c = plt.cm.OrRd(0.25 + 0.60 * v), ("white" if v > 0.45 else INK)
            else:
                face, txt_c = plt.cm.Greys(0.06 + 0.26 * v), INK
            ax0.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face,
                                    edgecolor="white", lw=1.2, zorder=2))
            ax0.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=PT_TICK,
                     fontweight="bold" if j == est_col else "normal", color=txt_c, zorder=3)
    ax0.set_xlim(-0.5, 3.5)
    ax0.set_ylim(3.5, -0.5)
    ax0.set_aspect("equal")
    ax0.set_xticks(range(4))
    ax0.set_yticks(range(4))
    ax0.set_xticklabels(QUAD_ORDER, fontsize=PT_TICK, rotation=30, ha="right",
                        rotation_mode="anchor")
    ax0.set_yticklabels(QUAD_ORDER, fontsize=PT_TICK)
    ax0.set_xlabel("Next period", fontsize=PT_LABEL)
    ax0.set_ylabel("Current period", fontsize=PT_LABEL)
    for sp in ax0.spines.values():
        sp.set_visible(False)
    ax0.tick_params(length=0)

    ax1 = fig.add_subplot(gs[0, 1])
    yb = np.arange(len(risk))[::-1]
    pv = risk["p_established_next"].values
    ax1.barh(yb, pv, color=TEALL, edgecolor="black", lw=0.4, zorder=3)
    ax1.set_yticks(yb)
    ax1.set_yticklabels([f"{q}\n(n = {int(n):,})" for q, n in zip(risk["state"], risk["n"])],
                        fontsize=PT_TICK)
    ax1.set_xlim(0, min(1.0, pv.max() * 1.30))
    ax1.set_xlabel("P(Established next period)", fontsize=PT_LABEL)
    base = float(pv[QUAD_ORDER.index("Controlled")])
    for k, (st, v) in enumerate(zip(risk["state"], pv)):
        lab = f"{v:.0%}" if st == "Controlled" else f"{v:.0%}  ({v / base:.0f}×)"
        ax1.text(v + 0.012, yb[k], lab, va="center", fontsize=PT_TICK, color="black")
    despine(ax1)

    eb = esc_biz.copy()
    eb["business"] = eb["business"].map(trans).fillna(eb["business"])
    eb = eb.sort_values("escalation_rate")
    ax2 = fig.add_subplot(gs[1, 0])
    y = np.arange(len(eb))
    ec_lo, ec_hi = wilson_ci(counts_from_rate(eb), eb["n"].values)
    ev = eb["escalation_rate"].values
    ax2.barh(y, ev, color=TEALL, edgecolor="black", lw=0.4, zorder=3)
    ax2.errorbar(ev, y, xerr=[ev - ec_lo, ec_hi - ev], fmt="none",
                 ecolor="black", elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=4)
    ax2.axvline(overall_esc, color="black", ls="--", lw=0.8)
    ax2.set_ylim(-0.65, len(eb) - 0.5 + 0.55)
    ax2.text(overall_esc, len(eb) - 0.42, f" overall {overall_esc:.0%}",
             fontsize=PT_TICK, color="black", va="bottom")
    ax2.set_yticks(y)
    ax2.set_yticklabels(eb["business"], fontsize=PT_TICK)
    ax2.set_xlim(0, float(ec_hi.max()) * 1.62)
    ax2.set_xlabel("Escalation rate (→ Established)", fontsize=PT_LABEL)
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    for k, v in enumerate(ev):
        ax2.text(ec_hi[k] + 0.006, y[k], f"{v:.0%}  (n = {int(eb['n'].values[k]):,})",
                 va="center", fontsize=PT_TICK, color="black", zorder=6, path_effects=halo)
    despine(ax2)

    em = esc_mon.sort_values("month")
    mnames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax3 = fig.add_subplot(gs[1, 1])
    em_lo, em_hi = wilson_ci(counts_from_rate(em), em["n"].values)
    mv = em["escalation_rate"].values
    ax3.bar(em["month"].values, mv, color=TEALL, edgecolor="black", lw=0.4, zorder=3)
    ax3.errorbar(em["month"].values, mv, yerr=[mv - em_lo, em_hi - mv], fmt="none",
                 ecolor="black", elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=4)
    ax3.axhline(overall_esc, color="black", ls="--", lw=0.8)
    ax3.set_ylim(0, float(em_hi.max()) * 1.08)
    ax3.set_xticks(range(1, 13))
    ax3.set_xticklabels(mnames, fontsize=PT_TICK, rotation=45, ha="right",
                        rotation_mode="anchor")
    ax3.set_ylabel("Escalation rate (→ Established)", fontsize=PT_LABEL)
    ax3.set_xlabel("Month", fontsize=PT_LABEL)
    despine(ax3)

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    panels = [[(ax0, "A"), (ax1, "B")], [(ax2, "C"), (ax3, "D")]]
    bbs = [[ax.get_tightbbox(rend).transformed(inv) for ax, _ in row] for row in panels]
    y_top = [max(bb.y1 for bb in row) for row in bbs]
    x_left = [max(min(bbs[r][c].x0 for r in range(2)) - 0.022, 0.004) for c in range(2)]
    for r, row in enumerate(panels):
        for c, (_, lab) in enumerate(row):
            fig.text(x_left[c], y_top[r] + 0.004, lab, fontsize=PT_PANEL, fontweight="bold",
                     color="black", va="bottom", ha="left")
    save_journal("Fig2", "transition", OUT, tight=False)


# --------------------------------------------------------------------------- #
# Fig. 3: observed vs predicted (test set)
# --------------------------------------------------------------------------- #
F3_TICK_STEP = 0.25


def fig3_persistence_vs_prediction():
    preds = load("reg_preds_all.csv")
    preds = preds[(preds["features"] == "Enhanced") | (preds["model"] == NAIVE)]
    preds = preds.reset_index(drop=True)
    models = [(NAIVE, "Persistence"), ("GLM", "GLM"),
              ("Random Forest", "Random forest"), ("XGBoost", "XGBoost")]
    targets = [("y_density", CAP_C, "Capture rate"), ("y_posrate", POS_C, "Positive rate")]

    def series(target, model):
        d = preds[(preds.target == target) & (preds.model == model)]
        return d["obs"].values, d["pred"].values

    def r2_det(target, model):
        o, p = series(target, model)
        ss_tot = float(((o - o.mean()) ** 2).sum())
        return 1.0 - float(((o - p) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    def pearson_r(target, model):
        o, p = series(target, model)
        if float(o.std()) == 0.0 or float(p.std()) == 0.0:
            return float("nan")
        return float(np.corrcoef(o, p)[0, 1])

    lims = {}
    for t, _, _ in targets:
        if t == "y_posrate":
            lims[t] = 1.0
        else:
            d = preds[preds.target == t]
            q99 = float(max(d["obs"].quantile(.99), d["pred"].quantile(.99)))
            lims[t] = float(np.ceil(q99 / F3_TICK_STEP - 1e-9) * F3_TICK_STEP)

    nrow, ncol = len(targets), len(models)
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=size("full", height_mm=rows_height_mm(nrow, 46.0, extra_mm=3.0)))
    letters = "ABCDEFGH"
    row_labels = []
    for i, (t, color, row_title) in enumerate(targets):
        for j, (mname, short_lab) in enumerate(models):
            ax = axes[i][j]
            o, p = series(t, mname)
            m = lims[t]
            ax.scatter(o, p, s=4.0, color=color, alpha=0.30, edgecolor="none", rasterized=True)
            ax.plot([0, m], [0, m], "--", color=GREY, lw=1.2)
            ax.set_xlim(0, m)
            ax.set_ylim(0, m)
            ax.set_box_aspect(1)
            ticks = np.arange(0.0, m + 1e-9, F3_TICK_STEP)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            fmt = mpl.ticker.FormatStrFormatter("%.2f")
            ax.xaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_formatter(fmt)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            if i == 0:
                ax.set_title(short_lab, fontsize=PT_TITLE, pad=5)
            if mname == NAIVE:
                ax.set_xlabel("Next period", fontsize=PT_LABEL)
                ax.set_ylabel("Current period", fontsize=PT_LABEL)
            else:
                ax.set_xlabel("Observed", fontsize=PT_LABEL)
                ax.set_ylabel("Predicted", fontsize=PT_LABEL)
            ax.text(-0.22, 1.06, letters[i * ncol + j], transform=ax.transAxes,
                    fontsize=PT_PANEL, fontweight="bold", color="black", va="bottom", ha="left")
            tr = ax.transAxes
            for txt, primary in ((f"$\\mathbfit{{R}}^\\mathbf{{2}}$ = {r2_det(t, mname):.2f}", True),
                                 (f"$r$ = {pearson_r(t, mname):.2f}", False)):
                ax.text(0.05, 0.95, txt, transform=tr, fontsize=PT_ANNOT,
                        fontweight="bold" if primary else "normal",
                        color=color, alpha=1.0 if primary else 0.85, va="top")
                tr = mpl.transforms.offset_copy(tr, fig=fig, y=-PT_ANNOT * 1.35, units="points")
            despine(ax)
        row_labels.append((axes[i][0], row_title))
    fig.tight_layout()
    add_row_labels(fig, row_labels, fontsize=PT_ROWLAB)
    save_journal("Fig3", "persistence_vs_prediction", OUT, tight=False)


# --------------------------------------------------------------------------- #
# Fig. 4: permutation importance with shadow-feature test
# --------------------------------------------------------------------------- #
TOP_N_OVERALL = 12
SIG_ALPHA = 0.05


def fig4_importance():
    imp = load("perm_importance.csv")
    targets = [("y_density", "Next capture rate"), ("y_posrate", "Next positive rate")]

    def sig_of(d, f, pfx):
        if float(d.loc[f, f"{pfx}_imp"]) <= 0:
            return ""
        q = float(d.loc[f, f"{pfx}_sh_q"])
        return "*" if np.isfinite(q) and q < SIG_ALPHA else ""

    def bars(ax, d, order):
        y = np.arange(len(order))
        h = 0.38
        rf_v = np.array([float(d.loc[f, "rf_imp"]) for f in order])
        rf_e = np.array([float(d.loc[f, "rf_sd"]) for f in order])
        xg_v = np.array([float(d.loc[f, "xgb_imp"]) for f in order])
        xg_e = np.array([float(d.loc[f, "xgb_sd"]) for f in order])
        ax.barh(y + h / 2, rf_v, h, xerr=rf_e, color=TEAL, edgecolor="black", lw=0.4,
                error_kw=dict(ecolor=SLATE, lw=0.6, capsize=1.5), label="Random forest", zorder=3)
        ax.barh(y - h / 2, xg_v, h, xerr=xg_e, color=AMBER, edgecolor="black", lw=0.4,
                hatch=HATCH_B, error_kw=dict(ecolor=SLATE, lw=0.6, capsize=1.5),
                label="XGBoost", zorder=3)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([FEATURE_LABELS.get(f, f) for f in order], fontsize=PT_TICK)
        ax.set_ylim(-0.8, len(order) - 0.2)
        lo = float(min(0.0, np.min(np.concatenate([rf_v - rf_e, xg_v - xg_e]))))
        hi = float(np.max(np.concatenate([rf_v + rf_e, xg_v + xg_e])))
        span = hi - lo if hi > lo else max(abs(hi), 1e-9)
        ax.set_xlim(lo - 0.04 * span, hi + 0.11 * span)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, prune=None))
        for k, f in enumerate(order):
            for yy, v, e, pfx, col in ((y[k] + h / 2, rf_v[k], rf_e[k], "rf", TEAL),
                                       (y[k] - h / 2, xg_v[k], xg_e[k], "xgb", AMBER)):
                mark = sig_of(d, f, pfx)
                if mark:
                    ax.text(v + e + 0.012 * span, yy, mark, va="center", ha="left",
                            fontsize=PT_ANNOT, fontweight="bold", color=col, zorder=4)
        despine(ax)

    fig, axes = plt.subplots(2, 1, squeeze=False,
                             figsize=size("full", height_mm=rows_height_mm(2, 70.0, extra_mm=4.0)))
    for i, (target, _) in enumerate(targets):
        d = imp[imp.target == target].set_index("feature")
        mean_imp = ((d["rf_imp"] + d["xgb_imp"]) / 2).sort_values()
        ax = axes[i][0]
        bars(ax, d, mean_imp.tail(TOP_N_OVERALL).index.tolist())
        ax.set_xlabel(r"Permutation importance ($\Delta R^2$)", fontsize=PT_LABEL)
        if i == 0:
            ax.legend(loc="lower right", fontsize=PT_TICK, framealpha=1.0)
        ax.text(-0.30, 1.05, "AB"[i], transform=ax.transAxes, fontsize=PT_PANEL,
                fontweight="bold", color="black", va="bottom", ha="left")
    save_journal("Fig4", "importance", OUT)


# --------------------------------------------------------------------------- #
# Fig. 5: forecast horizons
# --------------------------------------------------------------------------- #
def fig5_horizon():
    dec = load("horizon_decay.csv")
    models = ["GLM", "Random Forest", "XGBoost"]
    targets = [("capture", "Next capture rate"), ("posrate", "Next positive rate")]
    ylim = {}
    for tname, _ in targets:
        d = dec[dec.target == tname]
        lo = float(np.nanmin([d["Enhanced"].min(), d["Baseline"].min()]))
        hi = float(np.nanmax([d["Enhanced"].max(), d["Baseline"].max()]))
        pad = max(0.03, (hi - lo) * 0.18)
        ylim[tname] = (lo - pad, hi + pad * 1.35)

    nrow, ncol = len(targets), len(models)
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=size("full", height_mm=rows_height_mm(nrow, 52.0, extra_mm=11.0)))
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    for i, (tname, _) in enumerate(targets):
        span = ylim[tname][1] - ylim[tname][0]
        for j, model in enumerate(models):
            ax = axes[i][j]
            d = dec[(dec.target == tname) & (dec.model == model)].sort_values("horizon")
            x = d["weeks"].values
            for col, color, mark in (("Enhanced", RUST, "o"), ("Baseline", SLATE, "s")):
                ax.plot(x, d[col], marker=mark, color=color, lw=1.4, ms=4.5,
                        markeredgecolor="white", markeredgewidth=0.6, label=col, zorder=3)
            for xi in range(len(x)):
                e, b = d["Enhanced"].values[xi], d["Baseline"].values[xi]
                if np.isfinite(e) and np.isfinite(b):
                    off = (0.075 if abs(e - b) < span * 0.06 else 0.045) * span
                    e_off, b_off = (off, -off) if e >= b else (-off, off)
                else:
                    e_off = b_off = 0.045 * span
                if np.isfinite(e):
                    ax.text(x[xi], e + e_off, f"{e:.2f}", ha="center", va="center",
                            fontsize=PT_TICK, color=RUST, zorder=5, path_effects=halo)
                if np.isfinite(b):
                    ax.text(x[xi], b + b_off, f"{b:.2f}", ha="center", va="center",
                            fontsize=PT_TICK, color=SLATE, zorder=5, path_effects=halo)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(w)}" for w in x])
            ax.set_xlim(x.min() - 0.6, x.max() + 0.6)
            ax.set_ylim(*ylim[tname])
            if i == nrow - 1:
                ax.set_xlabel("Prediction lead time (weeks)", fontsize=PT_LABEL)
            if j == 0:
                ax.set_ylabel("Test-set $R^2$", fontsize=PT_LABEL)
            if i == 0:
                ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=PT_TITLE)
            ax.text(-0.22, 1.05, "ABCDEF"[i * ncol + j], transform=ax.transAxes,
                    fontsize=PT_PANEL, fontweight="bold", color="black", va="bottom", ha="left")
            ax.grid(axis="y", color=GREYL, lw=0.5)
            ax.set_axisbelow(True)
            despine(ax)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.945))
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=PT_TICK,
               frameon=False, handlelength=1.6, columnspacing=1.8, handletextpad=0.5)
    add_row_labels(fig, [(axes[i][0], t) for i, (_, t) in enumerate(targets)], fontsize=PT_ROWLAB)
    save_journal("Fig5", "horizon", OUT, tight=False)


def main():
    fig2_transition()
    fig3_persistence_vs_prediction()
    fig4_importance()
    fig5_horizon()
    print(f"Figures saved in {OUT}")


if __name__ == "__main__":
    main()
