"""Summary statistics for the exploratory analysis: churn rates and associations (D38)."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats.contingency import association

from churn.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET


def churn_rate_by(df: pd.DataFrame, column: str | pd.Series) -> pd.DataFrame:
    """Churn rate and number of customers per level of `column` (name or series), highest first."""
    levels = df[column] if isinstance(column, str) else column
    return (
        df[TARGET]
        .eq("Yes")
        .groupby(levels, observed=True)
        .agg(churn_rate="mean", customers="size")
        .sort_values("churn_rate", ascending=False)
    )


def service_mix(df: pd.DataFrame) -> pd.Series:
    """Phone and internet service combined, e.g. 'Phone only', 'DSL only', 'Phone + DSL' (D63)."""
    phone, internet = df["PhoneService"].eq("Yes"), df["InternetService"]
    online = internet.ne("No")
    mix = np.select(
        [phone & online, phone, online],
        ["Phone + " + internet, "Phone only", internet + " only"],
        default="Neither",
    )
    return pd.Series(mix, index=df.index, name="Services")


def observed_vs_expected(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Customers and churners per level, and the churners if every level churned at the overall
    rate: the gap between observed and expected counts is what the chi² test measures (D65)."""
    churned = df[TARGET].eq("Yes")
    table = churned.groupby(df[column], observed=True).agg(customers="size", churned="sum")
    table["churned_if_unrelated"] = table["customers"] * churned.mean()
    return table


def quantile_bins(values: pd.Series, q: int = 5) -> pd.Series:
    """Bin a numeric series into (at most) `q` quantile bins with readable labels like '6–20'."""
    edges = np.unique(values.quantile(np.linspace(0, 1, q + 1)))
    labels = [f"{lo:.0f}–{hi:.0f}" for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    return pd.cut(values, edges, labels=labels, include_lowest=True)


def _discretised(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric columns as quintile bins, so every pair can be compared with Cramér's V."""
    return df.assign(**{col: quantile_bins(df[col]) for col in NUMERIC_COLUMNS})


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V: association strength of two categorical series (0 = none, 1 = perfect)."""
    return float(association(pd.crosstab(x, y).to_numpy(), method="cramer"))


def association_with_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Cramér's V, chi² statistic and p-value of every feature vs churn, plus its riskiest level."""
    data = _discretised(df)
    rows = []
    for col in CATEGORICAL_COLUMNS + NUMERIC_COLUMNS:
        # D38: the plain chi² of the notebook's formula and of Cramér's V; scipy's default would
        # apply Yates' correction to the yes/no features.
        test = chi2_contingency(pd.crosstab(data[col], data[TARGET]), correction=False)
        rates = churn_rate_by(data, col)
        rows.append(
            {
                "feature": col,
                "cramers_v": cramers_v(data[col], data[TARGET]),
                "chi2": test.statistic,
                "p_value": test.pvalue,
                "riskiest_level": rates.index[0],
                "its_churn_rate": rates["churn_rate"].iloc[0],
            }
        )
    return pd.DataFrame(rows).set_index("feature").sort_values("cramers_v", ascending=False)


def association_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Cramér's V between churn and all features, ordered by association with churn."""
    data = _discretised(df)
    order = [TARGET, *association_with_churn(df).index]
    matrix = pd.DataFrame(np.eye(len(order)), index=order, columns=order)
    for a, b in combinations(order, 2):
        matrix.loc[a, b] = matrix.loc[b, a] = cramers_v(data[a], data[b])
    return matrix


def format_p_value(p: float) -> str:
    """P-value with two significant digits; tiny values underflow, so they are shown as a bound."""
    return "< 1e-300" if p < 1e-300 else f"{p:.2g}"


def churn_by_bill_change(features: pd.DataFrame, y: pd.Series, margin: float = 2.0) -> pd.DataFrame:
    """Churn rate by how the current bill compares with the average so far (D20, D65).

    Increases and decreases beyond ± `margin` USD stay apart, to show that both go with churn.
    Only customers with a billing history (tenure > 1): for the others, the average is the bill.
    """
    history = features["tenure"] > 1
    gap = (features["MonthlyCharges"] - features["AvgMonthlyCharges"])[history]
    labels = [f"up > USD {margin:g}", f"within ± USD {margin:g}", f"down > USD {margin:g}"]
    change = np.select([gap > margin, gap < -margin], [labels[0], labels[2]], default=labels[1])
    change = pd.Series(change, index=gap.index, name="bill_vs_average")
    return y[history].groupby(change).agg(churn_rate="mean", customers="size").loc[labels]


def charge_correlations(features: pd.DataFrame) -> pd.Series:
    """Pearson r behind D20: the average bill vs the current one, the total vs tenure × bill."""
    tenure, monthly = features["tenure"], features["MonthlyCharges"]
    return pd.Series(
        {
            "AvgMonthlyCharges vs MonthlyCharges": features["AvgMonthlyCharges"].corr(monthly),
            "TotalCharges vs tenure × MonthlyCharges": features["TotalCharges"].corr(
                tenure * monthly
            ),
        },
        name="pearson_r",
    )
