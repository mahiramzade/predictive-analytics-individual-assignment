# Predictive Analytics: Crime and Property Sale Prices in Seattle

Predicting residential property sale prices in Seattle using property characteristics and spatially-joined crime statistics. The analysis compares Ridge, Random Forest, XGBoost, and MLP models.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/mahiramzade/predictive-analytics-individual-assignment.git

# 2. Create and activate a virtual environment (Python 3.12)
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download data files (from Kaggle, links are below)
#    Place both CSVs in the project root:
#      - kingco_sales.csv          (~138 MB)
#      - SPD_Crime_Data__2008-Present.csv  (~307 MB)

# 5. Run the notebook
jupyter notebook main.ipynb
```

## Headless Execution

To execute the notebook without a browser (e.g. on a server or for CI):

```bash
jupyter nbconvert --to notebook --execute main.ipynb --inplace
```

Expected runtime: 5–15 minutes depending on hardware (the cKDTree spatial join is the bottleneck).

## Project Structure

```
/
├── main.ipynb                 # Full analysis notebook
├── report.md                  # 2000-word written report + appendix
├── requirements.txt           # Pinned Python dependencies
├── README.md                  # This file
└── images/
    └── predictive_analytics/  # Generated plots (300 DPI PNGs)
        ├── attribute_histogram_plots.png
        ├── correlation_heatmap_numeric.png
        ├── correlation_heatmap_crime_sale_price.png
        ├── sale_price_by_crime_decile.png
        ├── sale_price_histogram.png
        ├── actual_vs_predicted.png
        ├── residual_analysis.png
        ├── feature_importance.png
        ├── model_comparison_validation.png
        ├── model_comparison_final.png
        └── error_by_price_segment.png
```

## Notebook Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Obtain a Dataset and Frame the Predictive Problem | Research question, datasets, success metrics, constraints, agent tooling plan |
| 2 | Explore the Data to Gain Insights | Column profiling, outlier detection, validation checks, visualisations, correlation analysis |
| 3 | Prepare the Data | Feature selection, train/val/test split, preprocessing pipeline |
| 4 | Explore Different Models and Shortlist the Best Ones | Ridge, Random Forest, XGBoost, MLP training and comparison |
| 5 | Fine-Tune and Evaluate | XGBoost hyperparameter tuning, test evaluation, error analysis |
| 6 | Present the Final Solution | Model selection, model card, limitations, next steps |

## Data

Both datasets are publicly available on Kaggle under CC0 license.

| Dataset | File | Size | Source |
|---------|------|------|--------|
| Property Sales | `kingco_sales.csv` | ~138 MB | [Kaggle](https://www.kaggle.com/datasets/andykrause/kingcountysales) |
| Crime Reports | `SPD_Crime_Data__2008-Present.csv` | ~307 MB | [Kaggle](https://www.kaggle.com/datasets/migrantworkerdatahub/seattle-police-department-crime-dataset) |

## Requirements

- Python 3.12
- Dependencies listed in `requirements.txt` (install with `pip install -r requirements.txt`)
- ~2 GB RAM (crime file is loaded in chunks to manage memory)

## Reproducibility

- All random operations use `random_state=42`
- Year window and success metric thresholds are computed dynamically from the data
- No hardcoded dollar amounts or date ranges
- Preprocessing is fit on training data only (no leakage)
