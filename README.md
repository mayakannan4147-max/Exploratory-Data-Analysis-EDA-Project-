# Titanic Exploratory Data Analysis

This project explores passenger survival patterns using descriptive statistics and visualizations. It includes a reproducible Python analysis and a generated Markdown report.

## Files

- `data/titanic.csv` — source dataset from [seaborn-data](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv).
- `analyze_titanic.py` — analysis and report-generation script.
- `reports/eda_report.md` — findings, statistical summaries, and chart links.
- `reports/figures/` — generated visualizations.

## Run

Python 3.10 or later is recommended.

```powershell
python -m pip install -r requirements.txt
python analyze_titanic.py
```

The script reads `data/titanic.csv` and recreates the report and figures. It validates that the expected columns exist and reports missing or empty data errors directly.

## Scope and interpretation

The report compares survival rates by sex, passenger class, and age band, summarizes numeric features, and displays Pearson correlations. These are exploratory associations, not causal claims. Age values are not imputed; age comparisons use passengers with recorded ages.
