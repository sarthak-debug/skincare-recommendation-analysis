"""
Week 6 - Final evaluation charts for the business report
Project: What drives customer satisfaction and recommendation in skincare?

Reads the result tables written by Weeks 3-5 (output/week3, output/week4, output/week5)
and draws the summary charts used in the final report, in black and white.

Run from the project folder (after Weeks 3-5):
    python src/week6_report_figures.py
Output: output/week6/*.png and output/week6/business_impact.csv
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

OUT = "output/week6"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

# Assumptions for the illustrative business-impact model (change to your own figures)
ASSUME = {"customers": 100_000, "contact_cost_usd": 1.0, "save_rate": 0.20, "customer_value_usd": 60.0}


def read(week, name):
    return pd.read_csv(os.path.join("output", week, name))


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=180)
    plt.close(fig)


def label_bars(ax, bars, fmt="{:.3f}", dx=0.004):
    for b in bars:
        ax.text(b.get_width() + dx, b.get_y() + b.get_height() / 2, fmt.format(b.get_width()), va="center", fontsize=8)


# ------------------------------------------------------------------ 1. model comparison
def model_comparison():
    m = read("week5", "model_comparison.csv")
    m["short"] = m["model"].str.replace(" (always recommend)", "", regex=False).str.replace(" (HistGB)", "", regex=False)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    y = np.arange(len(m))
    best = m["ROC_AUC"].idxmax()
    for ax, col, title in [(axes[0], "ROC_AUC", "ROC-AUC (ranking quality)"),
                           (axes[1], "recall_not_rec", "Share of unhappy customers caught")]:
        colors = ["black" if i == best else "lightgrey" for i in range(len(m))]
        bars = ax.barh(y, m[col], color=colors, edgecolor="black")
        label_bars(ax, bars, "{:.3f}" if col == "ROC_AUC" else "{:.0%}")
        ax.set_xlim(0, 1.08)
        ax.set_title(title, fontsize=10)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(m["short"], fontsize=8.5)
    axes[0].invert_yaxis()
    axes[0].axvline(0.5, color="grey", ls=":", lw=0.8)
    savefig(fig, "fig_model_comparison.png")


# ------------------------------------------------------------------ 2. feature journey (Week 4 ablation)
def feature_journey():
    a = read("week4", "model_ablation.csv")
    a = a[~a["feature_set"].str.startswith("Majority")]
    labels = ["Week 2 features\n(price, length,\npopularity)", "+ engineered\nfeatures\n(no text)", "+ review text\n(logistic)", "+ review text\n(gradient\nboosting)"]
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    x = np.arange(len(a))
    bars = ax.bar(x, a["ROC_AUC"], color=["white", "lightgrey", "darkgrey", "black"], edgecolor="black", width=0.6)
    for b, r in zip(bars, a["recall_not_recommended"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"AUC {b.get_height():.3f}\ncatches {r:.0%}", ha="center", fontsize=8)
    ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    ax.text(-0.45, 0.507, "guessing (0.5)", fontsize=7, color="grey", ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels(labels[:len(a)], fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("ROC-AUC on unseen products")
    ax.set_title("How each step of the project improved the model")
    savefig(fig, "fig_feature_journey.png")


# ------------------------------------------------------------------ 3. what the model does for every 1,000 reviews
def per_thousand():
    m = read("week5", "model_comparison.csv")
    best = m.iloc[m["ROC_AUC"].idxmax()]
    base_unhappy = 1 - m.loc[0, "accuracy"]           # baseline accuracy = share recommending
    unhappy = round(1000 * base_unhappy)
    caught = round(unhappy * best["recall_not_rec"])
    missed = unhappy - caught
    flagged = round(caught / best["precision_not_rec"])
    false_alarm = flagged - caught
    happy_ok = 1000 - unhappy - false_alarm
    counts = [("Unhappy, caught by the model", caught, "black", ""),
              ("Unhappy, missed", missed, "white", "xxxx"),
              ("Happy, flagged anyway (false alarm)", false_alarm, "darkgrey", ""),
              ("Happy, correctly left alone", happy_ok, "white", "")]
    cells = []
    for _, n, c, h in counts:
        cells += [(c, h)] * n
    cells = cells[:1000]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    cols = 50
    for i, (c, h) in enumerate(cells):
        r, k = divmod(i, cols)
        ax.add_patch(Rectangle((k, -r), 0.86, 0.86, facecolor=c, edgecolor="black", lw=0.25, hatch=h))
    ax.set_xlim(-0.5, cols + 0.5)
    ax.set_ylim(-20.5, 1.5)
    ax.axis("off")
    ax.legend(handles=[Patch(facecolor=c, edgecolor="black", hatch=h, label=f"{lab}: {n}") for lab, n, c, h in counts],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8, frameon=False)
    ax.set_title("What the model does for every 1,000 reviews (gradient boosting, threshold 0.5)", fontsize=10)
    savefig(fig, "fig_per_1000.png")
    pd.DataFrame([{k: v for k, v in zip(["outcome", "per_1000"], [lab, n])} for lab, n, _, _ in counts]).to_csv(
        os.path.join(OUT, "per_1000_outcomes.csv"), index=False)
    return unhappy / 1000, best


# ------------------------------------------------------------------ 4. robustness
def robustness():
    r = read("week5", "robustness_checks.csv")
    short = {"Reference": "Reference model", "Without helpful_wilson": "Without helpfulness votes",
             "Without log_loves": "Without popularity count", "No text features": "Without any review text",
             "Excluding incentivised": "Without gifted reviews", "Temporal": "Newest 20% of reviews"}
    r["label"] = r["check"].map(lambda c: next(v for k, v in short.items() if c.startswith(k)))
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    y = np.arange(len(r))
    bars = ax.barh(y, r["ROC_AUC"], color=["black"] + ["lightgrey"] * (len(r) - 1), edgecolor="black")
    label_bars(ax, bars)
    ax.set_yticks(y)
    ax.set_yticklabels(r["label"], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("Does the model hold up when conditions change?", fontsize=10)
    savefig(fig, "fig_robustness.png")


# ------------------------------------------------------------------ 5. fairness
def fairness():
    f = read("week5", "fairness_by_group.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), gridspec_kw={"width_ratios": [5, 10]})
    for ax, var, title in [(axes[0], "skin_type", "By skin type"), (axes[1], "skin_tone", "By skin tone")]:
        g = f[f["group_variable"] == var].reset_index(drop=True)
        x = np.arange(len(g))
        ax.scatter(x, g["recall_not_rec"], color="black", zorder=3)
        ax.vlines(x, 0.70, g["recall_not_rec"], color="grey", lw=0.8)
        top = g["recall_not_rec"].max()
        ax.axhspan(top - 0.10, top, color="#eeeeee", zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(g["group"].str.replace("not_reported", "not reported"), rotation=40, ha="right", fontsize=7.5)
        for xi, (v, n) in enumerate(zip(g["recall_not_rec"], g["n"])):
            ax.text(xi, v + 0.006, f"{v:.2f}", ha="center", fontsize=6.5)
        ax.set_ylim(0.70, 0.90)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("Share of unhappy customers caught")
    fig.suptitle("Fairness: the grey band is the allowed gap (0.10 below the best group)", fontsize=10)
    savefig(fig, "fig_fairness.png")


# ------------------------------------------------------------------ 6. early warning
def early_warning():
    e = read("week5", "early_warning_regression.csv")
    al = read("week5", "early_warning_alerts.csv").iloc[0]
    short = ["Predict the\naverage product", "Trust the first\n20 reviews", "Ridge\nregression", "Random\nforest", "Gradient\nboosting"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), gridspec_kw={"width_ratios": [3, 2]})
    best = e["RMSE_pp"].idxmin()
    bars = axes[0].bar(np.arange(len(e)), e["RMSE_pp"], color=["black" if i == best else "lightgrey" for i in range(len(e))], edgecolor="black")
    for b in bars:
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, f"{b.get_height():.1f}", ha="center", fontsize=8)
    axes[0].set_xticks(np.arange(len(e)))
    axes[0].set_xticklabels(short[:len(e)], fontsize=7.5)
    axes[0].set_ylabel("Average forecast error (RMSE, points)")
    axes[0].set_title("Forecasting later satisfaction: lower is better", fontsize=10)
    base = al["actually_dropped_10pp"] / al["products"]
    vals = [base, al["alert_precision"], al["alert_recall"]]
    labs = ["Base rate:\nproducts that\nreally dropped", "Alert precision:\nflagged products\nthat dropped", "Alert recall:\ndrops caught\nafter 20 reviews"]
    b2 = axes[1].bar(range(3), vals, color=["white", "black", "darkgrey"], edgecolor="black")
    for b in b2:
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{b.get_height():.0%}", ha="center", fontsize=9)
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(labs, fontsize=7.5)
    axes[1].set_ylim(0, 0.8)
    axes[1].set_title(f"Early-warning alerts ({int(al['products']):,} products)", fontsize=10)
    savefig(fig, "fig_early_warning_summary.png")


# ------------------------------------------------------------------ 7. complaint themes (negation-aware where available)
def themes():
    t = read("week4", "theme_lift.csv").set_index("theme")
    n = read("week5", "negation_theme_check.csv").rename(columns={"Unnamed: 0": "theme"})
    n["theme"] = n["theme"].str.replace("theme_", "", regex=False)
    n = n.set_index("theme")
    lift = t["lift_low_vs_high"].copy()
    for th in n.index:
        if th in lift.index and not pd.isna(n.loc[th, "lift_low_vs_high_after"]):
            lift[th] = n.loc[th, "lift_low_vs_high_after"]
    lift = lift.drop(labels=["repurchase"], errors="ignore").sort_values()
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    y = np.arange(len(lift))
    bars = ax.barh(y, lift.values, color=["black" if v >= 1.3 else "lightgrey" for v in lift.values], edgecolor="black")
    for b in bars:
        ax.text(b.get_width() * 1.04, b.get_y() + b.get_height() / 2, f"{b.get_width():.2f}x", va="center", fontsize=8)
    ax.axvline(1, color="grey", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_yticks(y)
    ax.set_yticklabels(lift.index.str.replace("_", " "), fontsize=8.5)
    ax.set_xlabel("How much more often mentioned in 1-2 star than 4-5 star reviews (log scale)")
    ax.set_title("What unhappy customers talk about", fontsize=10)
    savefig(fig, "fig_complaint_themes.png")


# ------------------------------------------------------------------ 8. business impact (illustrative)
def business_impact(unhappy_share, best):
    n, cost, save, value = ASSUME["customers"], ASSUME["contact_cost_usd"], ASSUME["save_rate"], ASSUME["customer_value_usd"]
    unhappy = n * unhappy_share
    rows = []
    flagged_model = unhappy * best["recall_not_rec"] / best["precision_not_rec"]
    plans = {
        "Model-targeted follow-up": (flagged_model, unhappy * best["recall_not_rec"]),
        "Random follow-up, same budget": (flagged_model, flagged_model * unhappy_share),
        "Contact every customer": (n, unhappy),
        "Do nothing": (0, 0),
    }
    for name, (contacted, reached) in plans.items():
        c = contacted * cost
        v = reached * save * value
        rows.append({"plan": name, "customers_contacted": round(contacted), "unhappy_reached": round(reached),
                     "cost_usd": round(c), "value_retained_usd": round(v), "net_usd": round(v - c),
                     "return_per_dollar": round(v / c, 2) if c else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "business_impact.csv"), index=False)
    breakeven = flagged_model * cost / (unhappy * best["recall_not_rec"] * value)
    with open(os.path.join(OUT, "business_impact_assumptions.txt"), "w") as f:
        f.write(f"Assumptions: {ASSUME}\nBreak-even save rate for model-targeted follow-up: {breakeven:.1%}\n")
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    d = df[df["plan"] != "Do nothing"]
    y = np.arange(len(d))
    bars = ax.barh(y, d["net_usd"] / 1000, color=["black", "lightgrey", "white"], edgecolor="black")
    for b in bars:
        ax.text(b.get_width() + 2, b.get_y() + b.get_height() / 2, f"${b.get_width():,.0f}k", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(d["plan"], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Net value per 100,000 customers (USD thousands, illustrative)")
    ax.set_title("Illustrative value of a follow-up programme", fontsize=10)
    savefig(fig, "fig_business_impact.png")


if __name__ == "__main__":
    model_comparison()
    feature_journey()
    share, best = per_thousand()
    robustness()
    fairness()
    early_warning()
    themes()
    business_impact(share, best)
    print(f"Charts saved in {OUT}/")
