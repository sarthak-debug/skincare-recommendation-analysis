"""
Week 2 - Data audit, preliminary EDA and baseline check
Project: What drives customer satisfaction and recommendation in skincare?
Dataset: Sephora Products and Skincare Reviews (Kaggle, nadyinky)
         https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews

How to run
1. Download the dataset from Kaggle and unzip it into a folder called data/raw/
   (product_info.csv and reviews_*.csv files).
2. pip install pandas numpy scipy scikit-learn matplotlib seaborn
3. python week2_data_audit_and_eda.py
Outputs are written to outputs/ (CSV tables and PNG charts).
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.model_selection import GroupShuffleSplit

RAW = "data/raw"
OUT = "outputs"
SEED = 42
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "figure.dpi": 120})


# ---------------------------------------------------------------- 1. Load
def load():
    products = pd.read_csv(os.path.join(RAW, "product_info.csv"))
    files = sorted(glob.glob(os.path.join(RAW, "reviews_*.csv")))
    reviews = pd.concat(
        (pd.read_csv(f, index_col=0, low_memory=False) for f in files),
        ignore_index=True,
    )
    print(f"products: {products.shape}, reviews: {reviews.shape} from {len(files)} files")
    return products, reviews


# ---------------------------------------------------------------- 2. Data quality audit
def quality_audit(df, name):
    audit = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(),
    })
    audit.to_csv(os.path.join(OUT, f"quality_audit_{name}.csv"))
    print(f"\n[{name}] duplicate rows: {df.duplicated().sum()}")
    print(audit.sort_values("missing_pct", ascending=False).head(10))
    return audit


# ---------------------------------------------------------------- 3. Cleaning
def clean(products, reviews):
    r = reviews.copy()
    r = r.drop_duplicates()
    r = r.drop_duplicates(subset=["author_id", "product_id", "review_text"])
    r["submission_time"] = pd.to_datetime(r["submission_time"], errors="coerce")
    r = r[r["rating"].between(1, 5)]
    r["review_text"] = r["review_text"].fillna("").astype(str).str.strip()
    r["review_len_words"] = r["review_text"].str.split().str.len()
    r["low_rating"] = (r["rating"] <= 2).astype(int)
    for col in ["skin_type", "skin_tone", "eye_color", "hair_color"]:
        if col in r:
            r[col] = r[col].fillna("not_reported").str.lower()
    # privacy: author_id is pseudonymised and then dropped from analysis tables
    r["reviewer_key"] = pd.util.hash_pandas_object(r["author_id"].astype(str), index=False)
    r = r.drop(columns=["author_id"])

    keep = ["product_id", "sephora_exclusive", "limited_edition", "new", "online_only",
            "loves_count", "primary_category", "secondary_category"]
    keep = [c for c in keep if c in products.columns]
    r = r.merge(products[keep], on="product_id", how="left")
    r["price_band"] = pd.qcut(r["price_usd"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    r.to_parquet(os.path.join(OUT, "reviews_clean.parquet"), index=False) if _has_parquet() else None
    print(f"\nclean reviews: {r.shape}")
    return r


def _has_parquet():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------- 4. Descriptive statistics
def describe(r):
    num = ["rating", "price_usd", "review_len_words", "helpfulness", "total_feedback_count"]
    num = [c for c in num if c in r]
    d = r[num].describe(percentiles=[.25, .5, .75, .95]).T
    d["skew"] = r[num].skew()
    d.to_csv(os.path.join(OUT, "descriptive_numeric.csv"))
    print("\n", d.round(2))
    r["is_recommended"].value_counts(normalize=True, dropna=False).to_csv(
        os.path.join(OUT, "target_balance.csv"))
    r.groupby("skin_type")["rating"].agg(["count", "mean", "median", "std"]).to_csv(
        os.path.join(OUT, "rating_by_skin_type.csv"))


# ---------------------------------------------------------------- 5. Charts (black and white)
def charts(r):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    r["rating"].value_counts().sort_index().plot.bar(ax=ax, color="white", edgecolor="black", hatch="//")
    ax.set(xlabel="Star rating", ylabel="Number of reviews", title="Distribution of ratings")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_rating_distribution.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    r.boxplot(column="rating", by="price_band", ax=ax, color="black")
    ax.set(xlabel="Price band (quartile)", ylabel="Rating", title="Rating by price band"); fig.suptitle("")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_rating_by_price.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(np.log1p(r["review_len_words"]), bins=40, color="lightgrey", edgecolor="black")
    ax.set(xlabel="log(1 + words in review)", ylabel="Reviews", title="Review length")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_review_length.png")); plt.close(fig)

    corr = r[["rating", "price_usd", "review_len_words", "helpfulness", "loves_count"]].corr("spearman")
    corr.to_csv(os.path.join(OUT, "spearman_matrix.csv"))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr, cmap="Greys", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im); ax.set_title("Spearman correlation")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_spearman_heatmap.png")); plt.close(fig)


# ---------------------------------------------------------------- 6. Hypothesis tests (alpha = 0.05)
def cramers_v(table):
    chi2 = stats.chi2_contingency(table)[0]
    n = table.values.sum()
    k = min(table.shape) - 1
    return np.sqrt(chi2 / (n * k))


def hypothesis_tests(r):
    rows = []
    # H1: price vs rating (product level to avoid inflating n)
    prod = r.groupby("product_id").agg(price=("price_usd", "first"), mean_rating=("rating", "mean"),
                                       n=("rating", "size"))
    prod = prod[prod["n"] >= 30]
    rho, p = stats.spearmanr(prod["price"], prod["mean_rating"])
    rows.append(["H1", "Spearman rho (product level, n>=30 reviews)", rho, p, f"rho = {rho:.3f}"])
    groups = [g["rating"].values for _, g in r.groupby("price_band", observed=True)]
    h, p = stats.kruskal(*groups)
    eps2 = h / (len(r) - 1)
    rows.append(["H1", "Kruskal-Wallis rating ~ price band", h, p, f"epsilon^2 = {eps2:.4f}"])

    # H2: rating by skin type
    sub = r[r["skin_type"] != "not_reported"]
    groups = [g["rating"].values for _, g in sub.groupby("skin_type")]
    h, p = stats.kruskal(*groups)
    rows.append(["H2", "Kruskal-Wallis rating ~ skin type", h, p, f"epsilon^2 = {h / (len(sub) - 1):.4f}"])

    # H3: Sephora exclusive vs recommendation
    t = pd.crosstab(r["sephora_exclusive"], r["is_recommended"])
    chi2, p, dof, _ = stats.chi2_contingency(t)
    rows.append(["H3", "Chi-square exclusive x recommended", chi2, p, f"Cramer's V = {cramers_v(t):.3f}"])

    # H5: review length, low vs high rating
    a = r.loc[r["low_rating"] == 1, "review_len_words"]
    b = r.loc[r["low_rating"] == 0, "review_len_words"]
    u, p = stats.mannwhitneyu(a, b, alternative="greater")
    rbc = 1 - 2 * u / (len(a) * len(b))
    rows.append(["H5", "Mann-Whitney U length (low > high)", u, p, f"rank-biserial r = {rbc:.3f}"])

    res = pd.DataFrame(rows, columns=["hypothesis", "test", "statistic", "p_value", "effect_size"])
    # Bonferroni correction across the family of tests
    res["p_bonferroni"] = (res["p_value"] * len(res)).clip(upper=1)
    res["decision"] = np.where(res["p_bonferroni"] < 0.05, "Reject H0", "Fail to reject H0")
    res.to_csv(os.path.join(OUT, "hypothesis_tests.csv"), index=False)
    print("\n", res)
    # H4 (sentiment vs rating) needs the sentiment score built in Week 4.


# ---------------------------------------------------------------- 7. Baseline vs simple model (H6 dry run)
def baseline(r):
    d = r.dropna(subset=["is_recommended"]).copy()
    d["is_recommended"] = d["is_recommended"].astype(int)
    feats = ["price_usd", "review_len_words", "loves_count"]
    X = pd.get_dummies(d[feats + ["skin_type", "primary_category", "secondary_category"]],
                       drop_first=True, dtype=float).fillna(0)
    X[["price_usd", "review_len_words", "loves_count"]] = np.log1p(X[["price_usd", "review_len_words", "loves_count"]])
    y = d["is_recommended"]
    # group split by product so the same product is never in both train and test (no leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(X, y, groups=d["product_id"]))
    dummy = DummyClassifier(strategy="most_frequent").fit(X.iloc[tr], y.iloc[tr])
    logit = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X.iloc[tr], y.iloc[tr])
    out = []
    for name, m in [("Baseline (always recommend)", dummy), ("Logistic regression", logit)]:
        pred = m.predict(X.iloc[te])
        prob = m.predict_proba(X.iloc[te])[:, 1]
        out.append([name, f1_score(y.iloc[te], pred, average="macro"), roc_auc_score(y.iloc[te], prob)])
        print(f"\n{name}\n", classification_report(y.iloc[te], pred, digits=3))
    pd.DataFrame(out, columns=["model", "macro_F1", "ROC_AUC"]).to_csv(
        os.path.join(OUT, "baseline_model_results.csv"), index=False)


if __name__ == "__main__":
    products, reviews = load()
    quality_audit(products, "products")
    quality_audit(reviews, "reviews")
    r = clean(products, reviews)
    describe(r)
    charts(r)
    hypothesis_tests(r)
    baseline(r)
    print(f"\nAll outputs saved in {OUT}/")