from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "titanic.csv"
REPORT_DIR = PROJECT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
REQUIRED_COLUMNS = {"survived", "pclass", "sex", "age", "fare"}


def rate_table(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        data.groupby(column, observed=True)["survived"]
        .agg(survival_rate="mean", passengers="count")
        .sort_values("survival_rate", ascending=False)
    )


def save_figures(data: pd.DataFrame, correlations: pd.Series) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=data, x="sex", y="survived", errorbar=None, ax=ax)
    ax.set(title="Survival rate by sex", xlabel="Sex", ylabel="Survival rate")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "survival_by_sex.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=data, x="pclass", y="survived", errorbar=None, ax=ax)
    ax.set(
        title="Survival rate by passenger class",
        xlabel="Passenger class (1 = first, 3 = third)",
        ylabel="Survival rate",
    )
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "survival_by_class.png", dpi=150)
    plt.close(fig)

    age_data = data.dropna(subset=["age"]).copy()
    age_data["age_group"] = pd.cut(
        age_data["age"],
        bins=[0, 12, 18, 35, 60, float("inf")],
        labels=["Child", "Teen", "Young adult", "Adult", "Older adult"],
        right=False,
    )
    age_data = age_data.dropna(subset=["age_group"])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=age_data,
        x="age_group",
        y="survived",
        order=["Child", "Teen", "Young adult", "Adult", "Older adult"],
        errorbar=None,
        ax=ax,
    )
    ax.set(title="Survival rate by age group", xlabel="Age group", ylabel="Survival rate")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "survival_by_age_group.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=data, x="fare", hue="survived", bins=35, element="step", ax=ax)
    ax.set(title="Fare distribution by survival outcome", xlabel="Fare", ylabel="Passengers")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fare_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        correlations.to_frame("Correlation with survival"),
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set(title="Numeric feature correlations with survival", xlabel="", ylabel="Feature")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "survival_correlations.png", dpi=150)
    plt.close(fig)


def build_report(data: pd.DataFrame, correlations: pd.Series) -> str:
    missing = data.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    by_sex = rate_table(data, "sex")
    by_class = rate_table(data, "pclass").sort_index()

    age_data = data.dropna(subset=["age"]).copy()
    age_data["age_group"] = pd.cut(
        age_data["age"],
        bins=[0, 12, 18, 35, 60, float("inf")],
        labels=["Child", "Teen", "Young adult", "Adult", "Older adult"],
        right=False,
    )
    by_age = rate_table(age_data.dropna(subset=["age_group"]), "age_group")

    highest_corr = correlations.abs().idxmax()
    highest_corr_value = correlations[highest_corr]
    female_rate = by_sex.loc["female", "survival_rate"]
    male_rate = by_sex.loc["male", "survival_rate"]
    first_class_rate = by_class.loc[1, "survival_rate"]
    third_class_rate = by_class.loc[3, "survival_rate"]

    missing_lines = [
        f"| {column} | {count:,} | {count / len(data):.1%} |"
        for column, count in missing.items()
    ]
    if not missing_lines:
        missing_lines = ["| None | 0 | 0.0% |"]

    sex_lines = [
        f"| {sex} | {int(row['passengers']):,} | {row['survival_rate']:.1%} |"
        for sex, row in by_sex.iterrows()
    ]
    class_lines = [
        f"| {int(class_id)} | {int(row['passengers']):,} | {row['survival_rate']:.1%} |"
        for class_id, row in by_class.iterrows()
    ]
    age_lines = [
        f"| {age_group} | {int(row['passengers']):,} | {row['survival_rate']:.1%} |"
        for age_group, row in by_age.iterrows()
    ]
    corr_lines = [
        f"| {feature} | {value:+.3f} |"
        for feature, value in correlations.sort_values(key=abs, ascending=False).items()
    ]
    numeric_summary = data[["age", "fare", "sibsp", "parch"]].describe().round(2)

    return f"""# Titanic Passenger EDA

## Dataset and method

- **Observations:** {len(data):,} passengers; **variables:** {data.shape[1]}.
- **Outcome:** `survived` (1 = survived, 0 = did not survive).
- **Overall survival rate:** {data['survived'].mean():.1%}.
- **Source:** [seaborn-data Titanic CSV](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv).
- Rates below are descriptive comparisons, not evidence that a feature caused an outcome.

## Statistical summary

| Measure | Age | Fare | Siblings/spouses aboard | Parents/children aboard |
|---|---:|---:|---:|---:|
| Count | {int(numeric_summary.loc['count', 'age']):,} | {int(numeric_summary.loc['count', 'fare']):,} | {int(numeric_summary.loc['count', 'sibsp']):,} | {int(numeric_summary.loc['count', 'parch']):,} |
| Mean | {numeric_summary.loc['mean', 'age']:.2f} | {numeric_summary.loc['mean', 'fare']:.2f} | {numeric_summary.loc['mean', 'sibsp']:.2f} | {numeric_summary.loc['mean', 'parch']:.2f} |
| Median | {numeric_summary.loc['50%', 'age']:.2f} | {numeric_summary.loc['50%', 'fare']:.2f} | {numeric_summary.loc['50%', 'sibsp']:.2f} | {numeric_summary.loc['50%', 'parch']:.2f} |
| Standard deviation | {numeric_summary.loc['std', 'age']:.2f} | {numeric_summary.loc['std', 'fare']:.2f} | {numeric_summary.loc['std', 'sibsp']:.2f} | {numeric_summary.loc['std', 'parch']:.2f} |

### Missing values

| Column | Missing | Share of rows |
|---|---:|---:|
{chr(10).join(missing_lines)}

Age is missing for a subset of passengers, so age-group comparisons use only rows with a recorded age. Missing values are not imputed for this descriptive analysis.

## Survival comparisons

### By sex

| Sex | Passengers | Survival rate |
|---|---:|---:|
{chr(10).join(sex_lines)}

Women survived at {female_rate:.1%}, compared with {male_rate:.1%} for men in this dataset.

### By passenger class

| Class | Passengers | Survival rate |
|---|---:|---:|
{chr(10).join(class_lines)}

The first-class survival rate was {first_class_rate:.1%}; the third-class rate was {third_class_rate:.1%}.

### By age group

| Age group | Passengers with age recorded | Survival rate |
|---|---:|---:|
{chr(10).join(age_lines)}

Age bands are: child (0–11), teen (12–17), young adult (18–34), adult (35–59), and older adult (60+).

## Correlation with survival

Pearson correlations use complete pairs only; sex is encoded as female = 1 and male = 0. They capture linear association and do not adjust for confounding.

| Feature | Correlation |
|---|---:|
{chr(10).join(corr_lines)}

The strongest absolute numeric correlation among these encoded/numeric features is **{highest_corr}** ({highest_corr_value:+.3f}). This is an association, not a causal effect.

## Visualizations

![Survival by sex](figures/survival_by_sex.png)

![Survival by passenger class](figures/survival_by_class.png)

![Survival by age group](figures/survival_by_age_group.png)

![Fare distribution by survival](figures/fare_distribution.png)

![Numeric correlations with survival](figures/survival_correlations.png)

## Takeaways and limitations

1. Survival differed substantially by sex and passenger class in this sample.
2. Fare and passenger class are related descriptors, so their individual associations should not be read as independent effects.
3. Age-based estimates cover fewer passengers because age is missing; the smaller age groups may also be less stable.
4. This is a historical passenger manifest sample, not a randomized experiment or a representative population. These summaries cannot establish why individual passengers survived.
"""


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("Dataset contains no rows.")

    correlation_data = data.assign(
        sex_encoded=data["sex"].map({"female": 1, "male": 0})
    )
    correlation_features = ["pclass", "age", "fare", "sibsp", "parch", "sex_encoded"]
    correlations = (
        correlation_data[["survived", *correlation_features]]
        .corr(numeric_only=True)["survived"]
        .drop("survived")
        .dropna()
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_figures(data, correlations)
    report = build_report(data, correlations)
    report_path = REPORT_DIR / "eda_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Analyzed {len(data):,} rows.")
    print(f"Report: {report_path}")
    print(f"Figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
