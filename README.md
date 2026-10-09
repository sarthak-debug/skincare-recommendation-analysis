# Skincare Customer Satisfaction and Recommendation Analysis

Data Science with Python Analyst Internship (Beauty and Wellness), 2026
Author: Sarthak Sameer Gharat

## Project overview

This project studies what drives customer satisfaction and product recommendation in skincare, and whether we can predict if a customer will recommend a product.

**Main question:** Using product, customer and review information, can we predict whether a customer will recommend a skincare product, and which factors matter most?

## Dataset

- **Primary:** [Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews) (Kaggle). About 8,000 products and 1.09 million reviews.
- **Validation:** [Amazon Reviews 2023, Beauty](https://amazon-reviews-2023.github.io/)

The raw data is **not** stored in this repository because of its size and to protect reviewer privacy. To run the code, download the dataset from Kaggle and place the CSV files in `data/raw/`.

## Repository structure

```
report/     weekly Word reports
src/         Python scripts
outputs/     charts and result tables (aggregated, no personal data)
data/        local only, not uploaded (see .gitignore)
```

## How to run

```
pip install -r requirements.txt
python src/week2_data_audit_and_eda.py
```

Results are saved in `outputs/`.

## Progress

| Week | Task | Status |
|------|------|--------|
| 1 | Problem definition, research questions and hypotheses | Completed |
| 2 | Data sourcing strategy, data audit and pilot analysis | Completed |
| 3 | Data cleaning, EDA and hypothesis testing | Planned |
| 4 | Sentiment analysis and topic modelling | Planned |
| 5 | Machine learning models (Logistic Regression, Random Forest, XGBoost) | Planned |
| 6 | Dashboard and final report | Planned |

## Key pilot findings (Week 2)

- About 83% of reviews recommend the product, so the classes are unbalanced.
- Higher-priced products have slightly higher average ratings (Spearman rho = 0.165).
- Skin type, Sephora exclusivity and price band have statistically significant but negligible effects.
- A model without text features (AUC 0.549) only slightly beats the baseline (AUC 0.50), so review text is expected to be the key signal.

## Tools

Python, pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn
