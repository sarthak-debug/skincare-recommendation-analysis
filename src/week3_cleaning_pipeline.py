"""
Week 3 - Data cleaning and preprocessing pipeline
Project: What drives customer satisfaction and recommendation in skincare?
Dataset: Sephora Products and Skincare Reviews (Kaggle, nadyinky)

How to run (from the project folder):
    pip install -r requirements.txt
    python src/week3_cleaning_pipeline.py

Input : data/raw/product_info.csv and data/raw/reviews_*.csv
Output: data/processed/reviews_clean.parquet  (stays on your computer, not GitHub)
        output/week3/cleaning_log.csv          rows before/after every step
        output/week3/validation_report.csv     pass/fail data checks
        output/week3/data_dictionary.csv       every column of the clean data
        output/week3/missing_before_after.csv  and .png
        output/week3/outlier_summary.csv
        output/week3/run_info.json             versions, checksums, settings
"""

import glob
import hashlib
import html
import json
import os
import platform
import re
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ settings
CONFIG = {
    "raw_dir": "data/raw",
    "processed_dir": "data/processed",
    "out_dir": "output/week3",
    "seed": 42,
    "min_words_for_text": 1,          # reviews with fewer words are treated as "no text"
    "drop_col_missing_pct": 80,       # product columns above this % missing are dropped
    "review_len_cap_pct": 99.5,       # winsorise review length at this percentile
    "iqr_k": 3.0,                     # price outlier fence: Q1 - k*IQR, Q3 + k*IQR (on log price)
    "min_group_size": 30,             # smallest group size reported in summaries
}
OZ_TO_ML = 29.5735

SKIN_TONE_ORDER = {  # Sephora shade labels, lightest to deepest
    "porcelain": 1, "fair": 2, "fairlight": 3, "light": 4, "lightmedium": 5,
    "medium": 6, "mediumtan": 7, "olive": 7, "tan": 8, "deep": 9, "rich": 10,
    "dark": 10, "ebony": 11,
}
INCENTIVE_PATTERN = re.compile(
    r"(?:received (?:this|it|the product)[^.]{0,40}(?:free|complimentary|gift)"
    r"|complimentary|#gifted|gifted by|in exchange for (?:my|an) (?:honest )?review"
    r"|influenster|free sample|sent (?:me|this) (?:for|to) (?:free|review))",
    flags=re.IGNORECASE,
)
INGREDIENT_FLAGS = {
    "has_fragrance": r"\b(?:fragrance|parfum)\b",
    "has_alcohol_denat": r"alcohol denat",
    "has_niacinamide": r"niacinamide",
    "has_retinoid": r"retin(?:ol|al|yl)",
    "has_hyaluronic": r"hyalur",
    "has_salicylic": r"salicylic",
    "has_vitamin_c": r"ascorbic|ascorbyl",
    "has_spf_filter": r"zinc oxide|titanium dioxide|avobenzone|homosalate|octinoxate|octocrylene",
}

LOG = []          # one row per cleaning step
os.makedirs(CONFIG["out_dir"], exist_ok=True)
os.makedirs(CONFIG["processed_dir"], exist_ok=True)
plt.rcParams.update({"font.family": "serif", "figure.dpi": 120})


def log_step(step, table, before, after, note=""):
    """Record how many rows a step changed, so the run can be audited later."""
    LOG.append({"step": step, "table": table, "rows_before": before,
                "rows_after": after, "rows_changed": before - after, "note": note})
    print(f"[{table:8}] {step:<42} {before:>9,} -> {after:>9,}  {note}")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ 1. load
def load_raw():
    p_path = os.path.join(CONFIG["raw_dir"], "product_info.csv")
    r_paths = sorted(glob.glob(os.path.join(CONFIG["raw_dir"], "reviews_*.csv")))
    if not os.path.exists(p_path) or not r_paths:
        sys.exit("Raw files not found. Put product_info.csv and reviews_*.csv in data/raw/")
    products = pd.read_csv(p_path)
    reviews = pd.concat(
        (pd.read_csv(f, index_col=0, low_memory=False, dtype={"author_id": str}) for f in r_paths),
        ignore_index=True,
    )
    checksums = {os.path.basename(f): md5(f) for f in [p_path] + r_paths}
    log_step("load", "products", len(products), len(products), f"{products.shape[1]} columns")
    log_step("load", "reviews", len(reviews), len(reviews), f"{len(r_paths)} files")
    return products, reviews, checksums


def missing_profile(df):
    return (df.isna().mean() * 100).round(2)


# ------------------------------------------------------------------ 2. standardise structure
def standardise_columns(df):
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"))
    return df


def clean_str(s):
    """Trim, lower-case and collapse spaces; turn empty strings into missing."""
    s = s.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    return s.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA})


# ------------------------------------------------------------------ 3. products
def parse_size_ml(size):
    """'1.7 oz/ 50 mL' -> 50.0 ; '1 oz' -> 29.6 ; anything else -> NaN."""
    if not isinstance(size, str):
        return np.nan
    s = size.lower().replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)\s*ml", s)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:fl\.?\s*)?oz", s)
    if m:
        return round(float(m.group(1)) * OZ_TO_ML, 1)
    m = re.search(r"(\d+(?:\.\d+)?)\s*g\b", s)       # grams treated as ~mL for creams
    if m:
        return float(m.group(1))
    return np.nan


def parse_ingredients(text):
    """The raw column is a string that looks like a Python list. Return one clean lower-case string."""
    if not isinstance(text, str):
        return pd.NA
    t = re.sub(r"[\[\]']", " ", text)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t or pd.NA


def clean_products(products):
    p = standardise_columns(products)
    n0 = len(p)
    p = p.drop_duplicates(subset=["product_id"], keep="first")
    log_step("drop duplicate product_id", "products", n0, len(p))

    for col in ["brand_name", "primary_category", "secondary_category", "tertiary_category"]:
        if col in p:
            p[col] = clean_str(p[col])
    for col in ["limited_edition", "new", "online_only", "out_of_stock", "sephora_exclusive"]:
        if col in p:
            p[col] = pd.to_numeric(p[col], errors="coerce").fillna(0).astype("int8")
    for col in ["price_usd", "value_price_usd", "sale_price_usd", "loves_count"]:
        if col in p:
            p[col] = pd.to_numeric(p[col], errors="coerce")

    n0 = len(p)
    p = p[p["price_usd"].isna() | (p["price_usd"] > 0)]
    log_step("remove price <= 0", "products", n0, len(p))

    # unit standardisation -> size in mL and price per mL
    if "size" in p:
        p["size_ml"] = p["size"].apply(parse_size_ml)
        p.loc[(p["size_ml"] <= 0) | (p["size_ml"] > 1000), "size_ml"] = np.nan
        p["price_per_ml"] = p["price_usd"] / p["size_ml"]
        log_step("parse size to mL", "products", len(p), len(p),
                 f"parsed {p['size_ml'].notna().mean():.0%} of sizes")

    # sale price: missing means 'not on sale', not unknown
    if "sale_price_usd" in p:
        p["on_sale"] = p["sale_price_usd"].notna().astype("int8")

    # ingredient features
    if "ingredients" in p:
        p["ingredients_clean"] = p["ingredients"].apply(parse_ingredients)
        p["ingredient_count"] = p["ingredients_clean"].fillna("").str.count(",") + p["ingredients_clean"].notna()
        p.loc[p["ingredients_clean"].isna(), "ingredient_count"] = np.nan
        for flag, pattern in INGREDIENT_FLAGS.items():
            p[flag] = p["ingredients_clean"].fillna("").str.contains(pattern, regex=True).astype("int8")
            p.loc[p["ingredients_clean"].isna(), flag] = -1   # -1 = ingredients unknown

    # price outliers flagged inside each category on the log scale (IQR rule)
    k = CONFIG["iqr_k"]
    logp = np.log(p["price_usd"])
    grp = p["secondary_category"].fillna("unknown") if "secondary_category" in p else pd.Series("all", index=p.index)
    q1 = logp.groupby(grp).transform(lambda s: s.quantile(.25))
    q3 = logp.groupby(grp).transform(lambda s: s.quantile(.75))
    iqr = q3 - q1
    p["price_outlier"] = ((logp < q1 - k * iqr) | (logp > q3 + k * iqr)).astype("int8")

    # drop columns that are almost empty
    miss = p.isna().mean() * 100
    drop = [c for c in miss[miss > CONFIG["drop_col_missing_pct"]].index if c not in ("sale_price_usd",)]
    p = p.drop(columns=drop + ["sale_price_usd"], errors="ignore")
    log_step("drop near-empty columns", "products", len(p), len(p), ", ".join(drop) or "none")
    return p


# ------------------------------------------------------------------ 4. reviews
def clean_text(s):
    s = s.astype("string").fillna("")
    s = s.map(html.unescape)
    s = s.str.replace(r"<[^>]+>", " ", regex=True)                       # html tags
    s = s.str.replace(r"https?://\S+|www\.\S+", " ", regex=True)          # urls
    s = s.str.replace(r"[\w.+-]+@[\w-]+\.[\w.]+", " [email] ", regex=True)  # emails (privacy)
    s = s.str.replace(r"\+?\d[\d\s\-()]{8,}\d", " [phone] ", regex=True)  # phone numbers (privacy)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def wilson_lower_bound(pos, total, z=1.96):
    """Helpfulness that is fair to reviews with few votes (95% lower bound)."""
    pos = pos.astype(float)
    total = total.astype(float)
    phat = np.where(total > 0, pos / total.replace(0, np.nan), np.nan)
    denom = 1 + z**2 / total
    centre = phat + z**2 / (2 * total)
    margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total)
    return np.where(total > 0, (centre - margin) / denom, np.nan)


def clean_reviews(reviews):
    r = standardise_columns(reviews)
    r = r.loc[:, ~r.columns.str.startswith("unnamed")]

    # types
    r["rating"] = pd.to_numeric(r["rating"], errors="coerce")
    r["is_recommended"] = pd.to_numeric(r["is_recommended"], errors="coerce")
    r["submission_time"] = pd.to_datetime(r["submission_time"], errors="coerce")
    for col in ["total_feedback_count", "total_pos_feedback_count", "total_neg_feedback_count"]:
        if col in r:
            r[col] = pd.to_numeric(r[col], errors="coerce").fillna(0).astype("int32")

    # validity rules
    n0 = len(r)
    r = r[r["rating"].between(1, 5)]
    log_step("rating within 1-5", "reviews", n0, len(r))
    n0 = len(r)
    r = r[r["product_id"].notna() & r["author_id"].notna()]
    log_step("product_id and author_id present", "reviews", n0, len(r))
    n0 = len(r)
    r = r[r["submission_time"].isna() | (r["submission_time"] <= pd.Timestamp.now())]
    log_step("no future dates", "reviews", n0, len(r))
    r.loc[~r["is_recommended"].isin([0, 1]), "is_recommended"] = np.nan

    # text cleaning (before duplicate checks so spacing differences do not hide duplicates)
    r["review_text"] = clean_text(r["review_text"])
    if "review_title" in r:
        r["review_title"] = clean_text(r["review_title"])
    r["review_len_words"] = r["review_text"].str.split().str.len().fillna(0).astype("int32")
    r["has_text"] = (r["review_len_words"] >= CONFIG["min_words_for_text"]).astype("int8")
    letters = r["review_text"].str.count(r"[A-Za-z]")
    chars = r["review_text"].str.count(r"\S").replace(0, np.nan)
    r["likely_non_english"] = ((letters / chars) < 0.5).fillna(False).astype("int8")
    log_step("clean text", "reviews", len(r), len(r),
             f"{(r['has_text'] == 0).sum():,} without text; {r['likely_non_english'].sum():,} likely non-English")

    # duplicates: 1 exact, 2 same author+product+text, 3 syndicated (same author+text on another size/variant)
    n0 = len(r)
    r = r.drop_duplicates()
    log_step("exact duplicates", "reviews", n0, len(r))
    n0 = len(r)
    r = r.drop_duplicates(subset=["author_id", "product_id", "review_text"], keep="first")
    log_step("same author + product + text", "reviews", n0, len(r))
    # Sephora copies one review onto every size/variant of a product: same author, text, brand and timestamp
    key = ["author_id", "review_text", "submission_time"] + (["brand_name"] if "brand_name" in r else [])
    syndicated = r.duplicated(subset=key, keep="first") & (r["has_text"] == 1)
    n0 = len(r)
    r = r[~syndicated]
    log_step("syndicated copies across variants", "reviews", n0, len(r))

    # incentivised reviews are flagged, not removed (used for a sensitivity test)
    r["incentivised"] = r["review_text"].str.contains(INCENTIVE_PATTERN, regex=True).fillna(False).astype("int8")
    log_step("flag incentivised", "reviews", len(r), len(r), f"{r['incentivised'].sum():,} flagged")

    # categorical profile fields: standardise, keep 'not_reported' as its own level, add indicator
    for col in ["skin_type", "skin_tone", "eye_color", "hair_color"]:
        if col in r:
            r[col] = clean_str(r[col])
            r[f"{col}_missing"] = r[col].isna().astype("int8")
            r[col] = r[col].fillna("not_reported")
    if "skin_tone" in r:
        r["skin_tone_ordinal"] = r["skin_tone"].map(SKIN_TONE_ORDER)  # NaN when not reported

    # helpfulness: structurally missing when nobody voted -> keep NaN, add vote count and Wilson score
    if {"total_pos_feedback_count", "total_feedback_count"} <= set(r.columns):
        tot = r["total_feedback_count"]
        r["helpfulness"] = np.where(tot > 0, r["total_pos_feedback_count"] / tot.replace(0, np.nan), np.nan)
        r["helpful_wilson"] = wilson_lower_bound(r["total_pos_feedback_count"], tot)
        r["has_votes"] = (tot > 0).astype("int8")

    # derived fields
    r["review_year"] = r["submission_time"].dt.year.astype("Int16")
    r["low_rating"] = (r["rating"] <= 2).astype("int8")
    r["rating_group"] = pd.cut(r["rating"], [0, 2, 3, 5], labels=["low (1-2)", "neutral (3)", "high (4-5)"])
    cap = r["review_len_words"].quantile(CONFIG["review_len_cap_pct"] / 100)
    r["review_len_capped"] = r["review_len_words"].clip(upper=cap)
    r["log_review_len"] = np.log1p(r["review_len_words"])

    # privacy: replace author_id with a salted hash, then drop it
    salt = str(CONFIG["seed"])
    r["reviewer_key"] = pd.util.hash_pandas_object(r["author_id"].astype(str) + salt, index=False).astype("uint64")
    r = r.drop(columns=["author_id"])
    return r


# ------------------------------------------------------------------ 5. join
def join(r, p):
    keep = ["product_id", "brand_name", "primary_category", "secondary_category", "tertiary_category",
            "price_usd", "size_ml", "price_per_ml", "price_outlier", "loves_count", "on_sale",
            "limited_edition", "new", "online_only", "sephora_exclusive", "ingredient_count",
            *INGREDIENT_FLAGS.keys()]
    keep = [c for c in keep if c in p.columns]
    rp = r.rename(columns={"price_usd": "price_usd_review", "brand_name": "brand_name_review",
                           "product_name": "product_name_review"})
    n0 = len(rp)
    out = rp.merge(p[keep], on="product_id", how="left", validate="many_to_one", indicator=True)
    unmatched = (out["_merge"] == "left_only").sum()
    out = out[out["_merge"] == "both"].drop(columns="_merge")
    log_step("join product master (inner)", "reviews", n0, len(out), f"{unmatched:,} reviews had no product")
    if "price_usd_review" in out:
        mismatch = (out["price_usd_review"].round(2) != out["price_usd"].round(2)).mean()
        log_step("price consistency check", "reviews", len(out), len(out),
                 f"{mismatch:.1%} reviews show a different price than product master (master kept)")
        out = out.drop(columns=["price_usd_review"])
    out = out.drop(columns=[c for c in ["brand_name_review", "product_name_review"] if c in out])
    out["log_price"] = np.log(out["price_usd"])
    out["log_loves"] = np.log1p(out["loves_count"])
    out["price_band"] = pd.qcut(out["price_usd"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    return out


# ------------------------------------------------------------------ 6. validation
def validate(df):
    checks = [
        ("rating only 1-5", df["rating"].between(1, 5).all()),
        ("is_recommended only 0, 1 or missing", df["is_recommended"].dropna().isin([0, 1]).all()),
        ("price above 0", (df["price_usd"].dropna() > 0).all()),
        ("no exact duplicate rows", not df.duplicated(subset=["reviewer_key", "product_id", "review_text"]).any()),
        ("no author_id column (privacy)", "author_id" not in df.columns),
        ("no e-mail addresses left in text", not df["review_text"].str.contains(r"[\w.+-]+@[\w-]+\.\w", regex=True).any()),
        ("helpfulness within 0-1", df["helpfulness"].dropna().between(0, 1).all() if "helpfulness" in df else True),
        ("every review matched to a product", df["primary_category"].notna().all() if "primary_category" in df else True),
        ("submission_time not in the future", (df["submission_time"].dropna() <= pd.Timestamp.now()).all()),
        ("skin_type has no blanks", df["skin_type"].notna().all() if "skin_type" in df else True),
    ]
    rep = pd.DataFrame(checks, columns=["check", "passed"])
    rep["passed"] = rep["passed"].map({True: "PASS", False: "FAIL"})
    print("\n", rep.to_string(index=False))
    return rep


def data_dictionary(df):
    rows = []
    for c in df.columns:
        s = df[c]
        example = s.dropna().iloc[0] if s.notna().any() else ""
        rows.append({"column": c, "dtype": str(s.dtype), "missing_pct": round(s.isna().mean() * 100, 2),
                     "n_unique": s.nunique(dropna=True), "example": str(example)[:60]})
    return pd.DataFrame(rows)


def outlier_summary(df):
    rows = []
    for col in ["price_usd", "price_per_ml", "review_len_words", "loves_count", "total_feedback_count"]:
        if col not in df:
            continue
        s = df[col].dropna()
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        z = (s - s.mean()) / s.std()
        rows.append({"variable": col, "median": s.median(), "q1": q1, "q3": q3,
                     "iqr_outliers_1.5": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
                     "z_above_3": int((z.abs() > 3).sum()), "max": s.max(), "skew": round(s.skew(), 2)})
    return pd.DataFrame(rows)


def missing_chart(before, after):
    cols = [c for c in before.index if c in after.index and (before[c] > 0 or after[c] > 0)]
    m = pd.DataFrame({"before": before[cols], "after": after[cols]}).sort_values("before")
    m.to_csv(os.path.join(CONFIG["out_dir"], "missing_before_after.csv"))
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(m))))
    y = np.arange(len(m))
    ax.barh(y + 0.2, m["before"], 0.4, color="white", edgecolor="black", hatch="///", label="Raw")
    ax.barh(y - 0.2, m["after"], 0.4, color="dimgrey", edgecolor="black", label="Clean")
    ax.set_yticks(y)
    ax.set_yticklabels(m.index, fontsize=8)
    ax.set_xlabel("% missing")
    ax.set_title("Missing values before and after cleaning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["out_dir"], "fig_missing_before_after.png"))
    plt.close(fig)


# ------------------------------------------------------------------ 7. model preprocessor (fit on train only)
def build_preprocessor():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = ["log_price", "log_loves", "log_review_len", "ingredient_count", "price_per_ml"]
    categorical = ["skin_type", "skin_tone", "secondary_category", "price_band"]
    binary = ["sephora_exclusive", "limited_edition", "new", "online_only", "on_sale", "incentivised"]
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                         ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="not_reported")),
                         ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50))])
    return ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, categorical),
                              ("bin", "passthrough", binary)], remainder="drop"), numeric + categorical + binary


def split_and_check(df):
    from sklearn.model_selection import GroupShuffleSplit
    d = df.dropna(subset=["is_recommended"])
    pre, cols = build_preprocessor()
    cols = [c for c in cols if c in d.columns]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=CONFIG["seed"])
    tr, te = next(gss.split(d, groups=d["product_id"]))
    train, test = d.iloc[tr], d.iloc[te]
    pre.fit(train[cols])                      # learnt on training data only (no leakage)
    Xtr, Xte = pre.transform(train[cols]), pre.transform(test[cols])
    overlap = set(train["product_id"]) & set(test["product_id"])
    pd.Series(sorted(test["product_id"].unique())).to_csv(
        os.path.join(CONFIG["out_dir"], "test_product_ids.csv"), index=False, header=["product_id"])
    note = f"train {Xtr.shape}, test {Xte.shape}, shared products {len(overlap)}"
    log_step("train/test split by product + fit preprocessor", "reviews", len(d), len(d), note)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    np.random.seed(CONFIG["seed"])
    start = datetime.now()
    products_raw, reviews_raw, checksums = load_raw()
    miss_before = pd.concat([missing_profile(standardise_columns(reviews_raw)),
                             missing_profile(standardise_columns(products_raw)).add_prefix("product.")])

    products = clean_products(products_raw)
    reviews = clean_reviews(reviews_raw)
    clean = join(reviews, products)

    miss_after = pd.concat([missing_profile(clean),
                            missing_profile(products).add_prefix("product.")])
    missing_chart(miss_before, miss_after)
    outlier_summary(clean).round(3).to_csv(os.path.join(CONFIG["out_dir"], "outlier_summary.csv"), index=False)
    validation = validate(clean)
    validation.to_csv(os.path.join(CONFIG["out_dir"], "validation_report.csv"), index=False)
    if (validation["passed"] == "FAIL").any():
        pd.DataFrame(LOG).to_csv(os.path.join(CONFIG["out_dir"], "cleaning_log.csv"), index=False)
        sys.exit("Validation failed - see output/week3/validation_report.csv. Clean data was NOT saved.")
    data_dictionary(clean).to_csv(os.path.join(CONFIG["out_dir"], "data_dictionary.csv"), index=False)
    split_and_check(clean)

    try:
        clean.to_parquet(os.path.join(CONFIG["processed_dir"], "reviews_clean.parquet"), index=False)
        products.to_parquet(os.path.join(CONFIG["processed_dir"], "products_clean.parquet"), index=False)
    except ImportError:
        clean.to_csv(os.path.join(CONFIG["processed_dir"], "reviews_clean.csv.gz"), index=False)
        products.to_csv(os.path.join(CONFIG["processed_dir"], "products_clean.csv.gz"), index=False)

    pd.DataFrame(LOG).to_csv(os.path.join(CONFIG["out_dir"], "cleaning_log.csv"), index=False)
    import sklearn
    info = {"run_started": start.isoformat(timespec="seconds"),
            "run_seconds": round((datetime.now() - start).total_seconds(), 1),
            "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
            "scikit_learn": sklearn.__version__, "config": CONFIG, "raw_file_md5": checksums,
            "rows_raw_reviews": int(len(reviews_raw)), "rows_clean_reviews": int(len(clean)),
            "columns_clean": int(clean.shape[1])}
    with open(os.path.join(CONFIG["out_dir"], "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nDone: {len(reviews_raw):,} raw reviews -> {len(clean):,} clean rows, {clean.shape[1]} columns")
    print(f"Outputs in {CONFIG['out_dir']}/  |  clean data in {CONFIG['processed_dir']}/")