"""
Week 5 - Predictive modelling and algorithm selection
Project: What drives customer satisfaction and recommendation in skincare?

Task 1 (classification): will this customer recommend the product?  (H6)
Task 2 (regression, early warning): from a product's first 20 reviews, forecast
        the recommend rate of its later reviews.

Needs: data/processed/reviews_clean.parquet (Week 3) and the Week 4 script in src/
       (its sentiment cache makes this run much faster).
Run from the project folder:
    python src/week5_modeling.py
Outputs: output/week5/ tables and black-and-white charts.  Expected time: 20-40 minutes.
"""

import os
import re
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import week4_eda_features as w4          # reuse the Week 4 feature engineering

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_recall_curve,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore")
CONFIG = {
    "out_dir": "output/week5",
    "seed": 42,
    "train_rows": 300_000,     # cap on training rows (laptop-friendly)
    "tune_rows": 80_000,       # rows used for cross-validated tuning
    "rf_rows": 150_000,        # random forest is slower, so it gets fewer rows
    "n_iter": 6,               # random-search candidates per model
    "bootstrap": 500,
    "early_n": 20,             # reviews used by the early-warning model
    "early_min_later": 30,     # product needs at least this many later reviews
}
OUT = CONFIG["out_dir"]
os.makedirs(OUT, exist_ok=True)
SEED = CONFIG["seed"]
plt.rcParams.update({"font.family": "serif", "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
T0 = time.time()


def step(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def save(df, name, index=False):
    df.to_csv(os.path.join(OUT, name), index=index)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=170)
    plt.close(fig)


# =================================================================== data and features
NEGATORS = r"(?:not|no|never|without|isn.t|wasn.t|doesn.t|didn.t|don.t|zero|nor|non)"


def negation_aware_themes(df):
    """Week 4 flagged 'not greasy' as greasy. Remove negated mentions before matching."""
    text = df["review_text"].fillna("").astype(str).str.lower()
    text = text.str.replace(r"(fragrance|scent|perfume|alcohol|oil)[\s-]*free", " ", regex=True)
    for t in ["breakout", "irritation", "dryness", "greasy", "smell", "pilling_texture", "packaging"]:
        pat = w4.THEMES[t]
        cleaned = text.str.replace(rf"\b{NEGATORS}\b(?:\W+\w+){{0,2}}?\W+(?:{pat})\w*", " ", regex=True)
        df[f"theme_{t}"] = cleaned.str.contains(pat, regex=True).astype("int8")
    comp = [f"theme_{t}" for t in ["breakout", "irritation", "dryness", "greasy", "smell", "pilling_texture", "packaging", "no_effect"]]
    df["n_complaint_themes"] = df[comp].sum(axis=1).astype("int8")
    return df


def build_data():
    df = w4.load()
    df = w4.engineer(df)                 # same features as Week 4 (sentiment from cache)
    old = df[[f"theme_{t}" for t in w4.THEMES]].mean()
    df = negation_aware_themes(df)
    new = df[[f"theme_{t}" for t in w4.THEMES]].mean()
    lo, hi = df["rating"] <= 2, df["rating"] >= 4
    lift = pd.DataFrame({"share_before": old.round(4), "share_after": new.round(4),
                         "lift_low_vs_high_after": [round(df.loc[lo, c].mean() / df.loc[hi, c].mean(), 2) if df.loc[hi, c].mean() > 0 else np.nan for c in new.index]})
    save(lift, "negation_theme_check.csv", index=True)
    df = w4.product_segments(df)
    step(f"data ready: {len(df):,} reviews")
    return df


# feature lists (Week 4 set C)
NUM = w4.NUM_BASE + w4.NUM_ENG + w4.NUM_TXT
CAT = w4.CAT_BASE + w4.CAT_ENG
BIN = w4.BIN_ENG + w4.BIN_TXT


def cols_present(df, names):
    return [c for c in names if c in df.columns or c == "brand_te"]


def make_pre(num, cat, binr, scale=True):
    num_steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="not_reported")),
                          ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100))]), cat),
        ("bin", SimpleImputer(strategy="constant", fill_value=0), binr)])


def model_zoo():
    """Candidate algorithms and the hyper-parameter ranges searched for each."""
    return {
        "Logistic regression": (LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs"),
                                {"model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}, True),
        "Decision tree": (DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
                          {"model__max_depth": [4, 6, 8, 10, 12], "model__min_samples_leaf": [50, 200, 500]}, False),
        "Random forest": (RandomForestClassifier(n_estimators=150, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
                          {"model__max_depth": [10, 16, 22], "model__min_samples_leaf": [5, 20, 50], "model__max_features": ["sqrt", 0.3]}, False),
        "Gradient boosting (HistGB)": (HistGradientBoostingClassifier(class_weight="balanced", early_stopping=True, random_state=SEED),
                                       {"model__learning_rate": [0.05, 0.08, 0.12], "model__max_leaf_nodes": [15, 31, 63],
                                        "model__l2_regularization": [0.0, 1.0], "model__max_iter": [300, 500]}, False),
    }


# =================================================================== classification
def split(df):
    d = df.dropna(subset=["is_recommended"]).copy()
    d["is_recommended"] = d["is_recommended"].astype(int)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(d, groups=d["product_id"]))
    train, test = d.iloc[tr].copy(), d.iloc[te].copy()
    if len(train) > CONFIG["train_rows"]:
        train = train.sample(CONFIG["train_rows"], random_state=SEED)
    train["brand_te"], test["brand_te"] = w4.brand_target_encoding(train, test)
    for c in CAT:
        if c in train:
            train[c], test[c] = train[c].astype(str), test[c].astype(str)
    step(f"train {len(train):,} rows / test {len(test):,} rows, {test['product_id'].nunique():,} unseen products")
    return train, test


def tune(name, est, grid, scale, train, num, cat, binr):
    """Random search with 3-fold GroupKFold (by product) on a sample of the training data."""
    sub = train.sample(min(len(train), CONFIG["tune_rows"]), random_state=SEED)
    X, y, g = sub[num + cat + binr], sub["is_recommended"], sub["product_id"]
    rows, best, best_auc = [], None, -1
    cands = list(ParameterSampler(grid, n_iter=min(CONFIG["n_iter"], np.prod([len(v) for v in grid.values()])), random_state=SEED))
    for params in cands:
        aucs, tr_aucs = [], []
        for a, b in GroupKFold(n_splits=3).split(X, y, g):
            pipe = Pipeline([("pre", make_pre(num, cat, binr, scale)), ("model", est)]).set_params(**params)
            pipe.fit(X.iloc[a], y.iloc[a])
            aucs.append(roc_auc_score(y.iloc[b], pipe.predict_proba(X.iloc[b])[:, 1]))
            tr_aucs.append(roc_auc_score(y.iloc[a], pipe.predict_proba(X.iloc[a])[:, 1]))
        rows.append({"model": name, "params": str({k.replace('model__', ''): v for k, v in params.items()}),
                     "cv_auc_mean": np.mean(aucs), "cv_auc_sd": np.std(aucs), "train_auc_mean": np.mean(tr_aucs),
                     "gap_train_minus_cv": np.mean(tr_aucs) - np.mean(aucs)})
        if np.mean(aucs) > best_auc:
            best_auc, best = np.mean(aucs), params
    step(f"  tuned {name}: CV AUC {best_auc:.3f}")
    return best, rows


def evaluate(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return {"ROC_AUC": roc_auc_score(y, prob), "PR_AUC_not_rec": average_precision_score(1 - y, 1 - prob),
            "macro_F1": f1_score(y, pred, average="macro"), "accuracy": (pred == y).mean(),
            "precision_not_rec": precision_score(y, pred, pos_label=0, zero_division=0),
            "recall_not_rec": recall_score(y, pred, pos_label=0), "brier": brier_score_loss(y, prob)}


def classification(df):
    train, test = split(df)
    num, cat, binr = cols_present(train, NUM), cols_present(train, CAT), cols_present(train, BIN)
    feats = num + cat + binr
    y_tr, y_te = train["is_recommended"], test["is_recommended"]
    cv_rows, results, fitted, probs = [], [], {}, {}
    p = y_te.mean()
    results.append({"model": "Majority baseline (always recommend)", "params": "-", "ROC_AUC": 0.5, "PR_AUC_not_rec": 1 - p,
                    "macro_F1": p / (1 + p), "accuracy": p, "precision_not_rec": 0.0, "recall_not_rec": 0.0,
                    "brier": ((p - y_te) ** 2).mean(), "train_seconds": 0})
    for name, (est, grid, scale) in model_zoo().items():
        best, rows = tune(name, est, grid, scale, train, num, cat, binr)
        cv_rows += rows
        fit_rows = train if name != "Random forest" else train.sample(min(len(train), CONFIG["rf_rows"]), random_state=SEED)
        t = time.time()
        pipe = Pipeline([("pre", make_pre(num, cat, binr, scale)), ("model", est)]).set_params(**best)
        pipe.fit(fit_rows[feats], fit_rows["is_recommended"])
        secs = time.time() - t
        prob = pipe.predict_proba(test[feats])[:, 1]
        r = evaluate(y_te, prob)
        r.update({"model": name, "params": str({k.replace('model__', ''): v for k, v in best.items()}), "train_seconds": round(secs, 1)})
        results.append(r)
        fitted[name], probs[name] = pipe, prob
        step(f"  test {name}: AUC {r['ROC_AUC']:.3f}, recall(not rec) {r['recall_not_rec']:.3f}")
    save(pd.DataFrame(cv_rows).round(4), "cv_tuning_results.csv")
    res = pd.DataFrame(results)[["model", "params", "ROC_AUC", "PR_AUC_not_rec", "macro_F1", "accuracy",
                                 "precision_not_rec", "recall_not_rec", "brier", "train_seconds"]]
    save(res.round(4), "model_comparison.csv")
    best_name = res.iloc[1:].sort_values("ROC_AUC", ascending=False)["model"].iloc[0]
    step(f"best model: {best_name}")

    curves(y_te, probs)
    bootstrap_h6(y_te, probs[best_name], probs["Logistic regression"], test)
    threshold_analysis(y_te, probs[best_name], best_name)
    calibration(y_te, probs[best_name], best_name)
    fairness(test, probs[best_name])
    importance(fitted[best_name], test, feats, best_name)
    tree_rules(train, num, cat, binr)
    learning_curve(train, test, num, cat, binr)
    depth_curve(train, test, num, cat, binr)
    robustness_checks(df, train, test, num, cat, binr)
    return best_name


def curves(y, probs):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    styles = ["-", "--", ":", "-."]
    for (name, prob), ls in zip(probs.items(), styles):
        fpr, tpr, _ = roc_curve(y, prob)
        axes[0].plot(fpr, tpr, ls, color="black", lw=1.3, label=f"{name} ({roc_auc_score(y, prob):.3f})")
        pr, rc, _ = precision_recall_curve(1 - y, 1 - prob)
        axes[1].plot(rc, pr, ls, color="black", lw=1.3, label=name)
    axes[0].plot([0, 1], [0, 1], color="grey", lw=0.7)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (AUC)")
    axes[1].axhline(1 - y.mean(), color="grey", lw=0.7, label="No-skill level")
    axes[1].set(xlabel="Recall (not recommended)", ylabel="Precision (not recommended)", title="Precision-recall: unhappy customers")
    axes[0].legend(fontsize=6.5)
    axes[1].legend(fontsize=6.5)
    savefig(fig, "fig_roc_pr_curves.png")


def bootstrap_h6(y, best, logit, test):
    rng = np.random.default_rng(SEED)
    yv, bv, lv = y.values, best, logit
    prods = test["product_id"].values
    uniq = np.unique(prods)
    idx_by_prod = pd.Series(np.arange(len(prods))).groupby(prods).apply(np.array)
    gains_base, gains_logit = [], []
    for _ in range(CONFIG["bootstrap"]):
        pick = rng.choice(uniq, len(uniq), replace=True)          # resample products (cluster bootstrap)
        idx = np.concatenate(idx_by_prod.loc[pick].values)
        if len(np.unique(yv[idx])) < 2:
            continue
        a = roc_auc_score(yv[idx], bv[idx])
        gains_base.append(a - 0.5)
        gains_logit.append(a - roc_auc_score(yv[idx], lv[idx]))
    out = pd.DataFrame({
        "comparison": ["Best model AUC minus majority baseline (0.5)", "Best model AUC minus logistic regression"],
        "mean_gain": [np.mean(gains_base), np.mean(gains_logit)],
        "ci_low_2.5": [np.percentile(gains_base, 2.5), np.percentile(gains_logit, 2.5)],
        "ci_high_97.5": [np.percentile(gains_base, 97.5), np.percentile(gains_logit, 97.5)],
        "resamples": [len(gains_base)] * 2})
    out["supports_H6_or_better"] = out["ci_low_2.5"] > 0
    save(out.round(4), "bootstrap_h6.csv")


def threshold_analysis(y, prob, name):
    rows = []
    for thr in np.round(np.arange(0.20, 0.85, 0.05), 2):
        pred = (prob >= thr).astype(int)
        flagged = (pred == 0).mean()
        rows.append({"threshold": thr, "share_flagged_unhappy": flagged,
                     "precision_not_rec": precision_score(y, pred, pos_label=0, zero_division=0),
                     "recall_not_rec": recall_score(y, pred, pos_label=0), "macro_F1": f1_score(y, pred, average="macro")})
    t = pd.DataFrame(rows)
    best = t.loc[t["macro_F1"].idxmax(), "threshold"]
    save(t.round(4), "threshold_table.csv")
    pred = (prob >= best).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    save(pd.DataFrame(cm, index=["actual not recommended", "actual recommended"],
                      columns=["predicted not recommended", "predicted recommended"]), "confusion_matrix.csv", index=True)
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Greys", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            share = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]:,}\n({share:.0%})", ha="center", va="center", fontsize=9,
                    color="white" if share > 0.6 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred. not rec.", "Pred. rec."])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Not rec.", "Rec."])
    ax.set_title(f"{name}, threshold {best:.2f}", fontsize=9)
    savefig(fig, "fig_confusion_matrix.png")


def calibration(y, prob, name):
    bins = pd.qcut(prob, 10, duplicates="drop")
    t = pd.DataFrame({"prob": prob, "y": y.values}).groupby(bins, observed=True).agg(mean_pred=("prob", "mean"), observed=("y", "mean"), n=("y", "size"))
    save(t.round(4), "calibration_table.csv", index=True)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, label="Perfect calibration")
    ax.plot(t["mean_pred"], t["observed"], marker="o", color="black", label=name)
    ax.set(xlabel="Predicted probability of recommending", ylabel="Observed share recommending", title="Calibration (10 bins)")
    ax.legend(fontsize=7)
    savefig(fig, "fig_calibration.png")


def fairness(test, prob):
    thr = 0.5
    d = test.assign(prob=prob, pred=(prob >= thr).astype(int))
    rows = []
    for col in ["skin_type", "skin_tone"]:
        for lvl, g in d.groupby(col):
            if len(g) < 500 or g["is_recommended"].nunique() < 2:
                continue
            rows.append({"group_variable": col, "group": lvl, "n": len(g), "actual_not_rec_share": 1 - g["is_recommended"].mean(),
                         "ROC_AUC": roc_auc_score(g["is_recommended"], g["prob"]),
                         "recall_not_rec": recall_score(g["is_recommended"], g["pred"], pos_label=0)})
    f = pd.DataFrame(rows)
    f["recall_gap_vs_best"] = f.groupby("group_variable")["recall_not_rec"].transform("max") - f["recall_not_rec"]
    save(f.round(4), "fairness_by_group.csv")


def importance(pipe, test, feats, name):
    sub = test.sample(min(len(test), 20_000), random_state=SEED)
    r = permutation_importance(pipe, sub[feats], sub["is_recommended"], scoring="roc_auc", n_repeats=3, random_state=SEED, n_jobs=1)
    imp = pd.DataFrame({"feature": feats, "auc_drop_mean": r.importances_mean, "auc_drop_sd": r.importances_std}).sort_values("auc_drop_mean", ascending=False)
    save(imp.round(5), "permutation_importance.csv")
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.barh(top["feature"], top["auc_drop_mean"], xerr=top["auc_drop_sd"], color="lightgrey", edgecolor="black")
    ax.set(xlabel="Drop in ROC-AUC when the feature is shuffled", title=f"Permutation importance: {name}")
    plt.setp(ax.get_yticklabels(), fontsize=8)
    savefig(fig, "fig_permutation_importance.png")


def tree_rules(train, num, cat, binr):
    """A shallow tree, written out as if-then rules a business user can read."""
    sub = train.sample(min(len(train), 100_000), random_state=SEED)
    pre = make_pre(num, cat, binr, scale=False)
    X = pre.fit_transform(sub[num + cat + binr], sub["is_recommended"])
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=500, class_weight="balanced", random_state=SEED).fit(X, sub["is_recommended"])
    names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    with open(os.path.join(OUT, "decision_tree_rules.txt"), "w") as f:
        f.write(export_text(tree, feature_names=names, show_weights=True, decimals=2))


def learning_curve(train, test, num, cat, binr):
    rows = []
    feats = num + cat + binr
    for n in [5_000, 20_000, 60_000, 150_000, len(train)]:
        n = min(n, len(train))
        s = train.sample(n, random_state=SEED)
        pipe = Pipeline([("pre", make_pre(num, cat, binr, False)),
                         ("model", HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED))]).fit(s[feats], s["is_recommended"])
        rows.append({"train_rows": n, "train_auc": roc_auc_score(s["is_recommended"], pipe.predict_proba(s[feats])[:, 1]),
                     "test_auc": roc_auc_score(test["is_recommended"], pipe.predict_proba(test[feats])[:, 1])})
    t = pd.DataFrame(rows)
    save(t.round(4), "learning_curve.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(t["train_rows"], t["train_auc"], marker="o", color="black", ls="--", label="Training data")
    ax.plot(t["train_rows"], t["test_auc"], marker="s", color="black", label="Unseen test products")
    ax.set_xscale("log")
    ax.set(xlabel="Training rows (log scale)", ylabel="ROC-AUC", title="Learning curve (gradient boosting)")
    ax.legend(fontsize=7)
    savefig(fig, "fig_learning_curve.png")


def depth_curve(train, test, num, cat, binr):
    feats = num + cat + binr
    s = train.sample(min(len(train), 150_000), random_state=SEED)
    pre = make_pre(num, cat, binr, False).fit(s[feats], s["is_recommended"])
    Xs, Xt = pre.transform(s[feats]), pre.transform(test[feats])
    rows = []
    for depth in [1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, None]:
        tr = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=SEED).fit(Xs, s["is_recommended"])
        rows.append({"max_depth": depth if depth else "unlimited", "train_auc": roc_auc_score(s["is_recommended"], tr.predict_proba(Xs)[:, 1]),
                     "test_auc": roc_auc_score(test["is_recommended"], tr.predict_proba(Xt)[:, 1])})
    t = pd.DataFrame(rows)
    save(t.round(4), "tree_depth_curve.csv")
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    x = np.arange(len(t))
    ax.plot(x, t["train_auc"], marker="o", ls="--", color="black", label="Training data")
    ax.plot(x, t["test_auc"], marker="s", color="black", label="Unseen test products")
    ax.set_xticks(x); ax.set_xticklabels(t["max_depth"].astype(str), fontsize=8)
    ax.set(xlabel="Decision tree max_depth", ylabel="ROC-AUC", title="Underfitting vs overfitting (decision tree)")
    ax.legend(fontsize=7)
    savefig(fig, "fig_tree_depth_curve.png")


def robustness_checks(df, train, test, num, cat, binr):
    """Same gradient boosting model under different conditions."""
    def fit_eval(tr, te, n, c, b, label):
        feats = n + c + b
        pipe = Pipeline([("pre", make_pre(n, c, b, False)),
                         ("model", HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED))]).fit(tr[feats], tr["is_recommended"])
        r = evaluate(te["is_recommended"], pipe.predict_proba(te[feats])[:, 1])
        r["check"] = label
        r["test_rows"] = len(te)
        return r
    rows = [fit_eval(train, test, num, cat, binr, "Reference: all features, random product split")]
    rows.append(fit_eval(train, test, [c for c in num if c != "helpful_wilson"], cat, binr,
                         "Without helpful_wilson (votes arrive after posting)"))
    rows.append(fit_eval(train, test, [c for c in num if c not in ("log_loves",)], cat, binr,
                         "Without log_loves (popularity measured at scrape time)"))
    no_txt_num = [c for c in num if c not in w4.NUM_TXT]
    no_txt_bin = [c for c in binr if c not in w4.BIN_TXT]
    rows.append(fit_eval(train, test, no_txt_num, cat, no_txt_bin, "No text features (could score a shopper before they write)"))
    te_clean = test[test["incentivised"] == 0]
    rows.append(fit_eval(train[train["incentivised"] == 0], te_clean, num, cat, binr, "Excluding incentivised reviews"))
    # temporal: train on the past, test on the most recent year
    d = df.dropna(subset=["is_recommended", "submission_time"]).copy()
    d["is_recommended"] = d["is_recommended"].astype(int)
    cut = d["submission_time"].quantile(0.8)
    past, future = d[d["submission_time"] < cut], d[d["submission_time"] >= cut]
    past = past.sample(min(len(past), CONFIG["train_rows"]), random_state=SEED).copy()
    future = future.copy()
    past["brand_te"], future["brand_te"] = w4.brand_target_encoding(past, future)
    for c in cat:
        past[c], future[c] = past[c].astype(str), future[c].astype(str)
    rows.append(fit_eval(past, future, num, cat, binr, f"Temporal: train before {cut.date()}, test after"))
    t = pd.DataFrame(rows)[["check", "test_rows", "ROC_AUC", "PR_AUC_not_rec", "macro_F1", "recall_not_rec", "brier"]]
    save(t.round(4), "robustness_checks.csv")
    step("robustness checks done")


# =================================================================== regression: early warning
def early_warning(df):
    k, m = CONFIG["early_n"], CONFIG["early_min_later"]
    d = df.dropna(subset=["submission_time"]).sort_values(["product_id", "submission_time"]).copy()
    d["pos"] = d.groupby("product_id").cumcount()
    n = d.groupby("product_id")["pos"].transform("size")
    d = d[n >= k + m]
    early, later = d[d["pos"] < k], d[d["pos"] >= k]
    first_t = early.groupby("product_id")["submission_time"].agg(["min", "max"])
    X = early.groupby("product_id").agg(
        early_mean_rating=("rating", "mean"), early_sd_rating=("rating", "std"),
        early_rec_rate=("is_recommended", "mean"), early_low_share=("low_rating", "mean"),
        early_sentiment=("sentiment", "mean"), early_incentivised=("incentivised", "mean"),
        early_complaints=("n_complaint_themes", "mean"), early_no_effect=("theme_no_effect", "mean"),
        early_breakout=("theme_breakout", "mean"), early_irritation=("theme_irritation", "mean"),
        early_len=("log_review_len", "mean"), early_first_time=("is_first_review", "mean"),
        early_reviewer_mean=("reviewer_prior_mean_rating", "mean"),
        share_oily=("skin_type", lambda s: (s == "oily").mean()), share_dry=("skin_type", lambda s: (s == "dry").mean()),
        log_price=("log_price", "first"), ppm_rel=("ppm_rel_category", "first"), exclusive=("sephora_exclusive", "first"),
        ingredient_count=("ingredient_count", "first"), category=("secondary_category", "first"), segment=("segment", "first"))
    X["days_for_first_k"] = (first_t["max"] - first_t["min"]).dt.days
    y = later.groupby("product_id")["is_recommended"].mean().reindex(X.index)
    X, y = X[y.notna()], y[y.notna()]
    num = [c for c in X.columns if c not in ("category", "segment")]
    cat = ["category", "segment"]
    X[cat] = X[cat].astype(str)

    models = {
        "Naive: later rate = early rate": None,
        "Ridge regression": Ridge(alpha=1.0),
        "Random forest regressor": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        "Gradient boosting regressor": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=SEED),
    }
    rows, preds = [], {}
    kf = GroupKFold(n_splits=5)
    for name, model in models.items():
        oof = np.zeros(len(X))
        for a, b in kf.split(X, y, groups=X.index):
            if model is None:
                oof[b] = X["early_rec_rate"].fillna(y.iloc[a].mean()).values[b]
                continue
            pre = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
                                     ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cat)])
            pipe = Pipeline([("pre", pre), ("model", model)]).fit(X.iloc[a], y.iloc[a])
            oof[b] = pipe.predict(X.iloc[b])
        oof = np.clip(oof, 0, 1)
        preds[name] = oof
        rows.append({"model": name, "RMSE_pp": np.sqrt(mean_squared_error(y, oof)) * 100, "MAE_pp": mean_absolute_error(y, oof) * 100,
                     "R2": r2_score(y, oof), "spearman": stats.spearmanr(y, oof)[0]})
    mean_pred = np.full(len(y), y.mean())
    rows.insert(0, {"model": "Mean only (predict the average product)", "RMSE_pp": np.sqrt(mean_squared_error(y, mean_pred)) * 100,
                    "MAE_pp": mean_absolute_error(y, mean_pred) * 100, "R2": 0.0, "spearman": np.nan})
    res = pd.DataFrame(rows)
    save(res.round(4), "early_warning_regression.csv")
    best = res.iloc[2:].sort_values("RMSE_pp")["model"].iloc[0]
    # flag products predicted to fall >10 points below their early rate
    p = preds[best]
    flag = (X["early_rec_rate"] - p) > 0.10
    actual_drop = (X["early_rec_rate"] - y) > 0.10
    prec = (flag & actual_drop).sum() / max(flag.sum(), 1)
    rec = (flag & actual_drop).sum() / max(actual_drop.sum(), 1)
    save(pd.DataFrame([{"products": len(X), "flagged": int(flag.sum()), "actually_dropped_10pp": int(actual_drop.sum()),
                        "alert_precision": prec, "alert_recall": rec, "model": best}]).round(4), "early_warning_alerts.csv")
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(p * 100, y * 100, s=8, facecolors="none", edgecolors="black", linewidths=0.5, alpha=0.6)
    lo = min(p.min(), y.min()) * 100
    ax.plot([lo, 100], [lo, 100], color="grey", lw=0.8)
    ax.set(xlabel="Forecast recommend rate of later reviews (%)", ylabel="Actual (%)", title=f"Early warning: {best}")
    savefig(fig, "fig_early_warning.png")
    step(f"early warning: {len(X):,} products, best {best}")


if __name__ == "__main__":
    df = build_data()
    classification(df)
    early_warning(df)
    step(f"done. Outputs in {OUT}/")
