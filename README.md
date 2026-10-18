# Skincare Customer Satisfaction and Recommendation Analysis

Data Science with Python Analyst Internship (Beauty and Wellness), 2026
Author: Sarthak Sameer Gharat

## Project overview

What drives customer satisfaction and recommendation in skincare, and can we predict which customers will be unhappy and which new products will disappoint?

This project cleans and analyses about 1 million Sephora skincare reviews, engineers 36 features describing customer behaviour, and builds two models:

1. **Classification:** will this customer recommend the product?
2. **Early warning (regression):** from a new product's first 20 reviews, forecast the satisfaction of later buyers.

## Key results

| Result | Value |
|---|---|
| Clean reviews analysed | 971,500 (from 1,094,411 raw; 122,353 duplicate copies removed) |
| Best model | Gradient boosting |
| ROC-AUC on products never seen in training | 0.897 (baseline 0.500) |
| Unhappy customers caught | 82% |
| Gain over baseline (95% bootstrap CI) | +0.396 (0.391 to 0.402) |
| Fairness: largest recall gap across skin types / tones | 0.038 / 0.051 (limit 0.10) |
| Early warning: forecast error vs trusting early reviews | 9.0 vs 13.0 points (RMSE) |
| Early-warning alert precision | 56% (base rate 27%) |

## Main findings

- Review tone, "it did nothing" complaints and reviewers' own rating habits drive satisfaction far more than price, brand or skin type.
- Price per mL has no relationship with satisfaction.
- A product's first 10 reviews are 0.16 stars higher than later ones; 27% of products lose 10+ points of recommend rate after launch.
- Self tanners, sunscreen and wellness products underperform; customers who do not state their skin type are up to 14 points less satisfied.

## Dataset

- **Primary:** [Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews) (Kaggle)

The raw data is **not** stored in this repository because of its size and to protect reviewer privacy. Download it from Kaggle and place the CSV files in `data/raw/`.

## Repository structure

```
Report/      weekly Word reports (Weeks 1-6)
src/         Python scripts for each week
output/      charts and result tables (aggregated, no personal data)
data/        local only, not uploaded (see .gitignore)
```

## How to run

```
pip install -r requirements.txt
pip install vaderSentiment
python src/week3_cleaning_pipeline.py     # cleaning, 10 validation checks
python src/week4_eda_features.py          # EDA and feature engineering
python src/week5_modeling.py              # models, validation, early warning
python src/week6_report_figures.py        # final report charts
```

## Progress

| Week | Task | Status |
|------|------|--------|
| 1 | Problem definition, research questions and hypotheses | Completed |
| 2 | Data sourcing strategy, data audit and pilot analysis | Completed |
| 3 | Data cleaning and preprocessing pipeline | Completed |
| 4 | Exploratory data analysis and feature engineering | Completed |
| 5 | Predictive modelling and algorithm selection | Completed |
| 6 | Model evaluation, reporting and business impact | Completed |

## Tools

Python, pandas, NumPy, SciPy, scikit-learn, Matplotlib, vaderSentiment, pyarrow
