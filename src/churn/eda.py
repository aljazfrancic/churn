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
    """Cramér's V and chi² p-value of every feature vs churn, plus its riskiest level."""
    data = _discretised(df)
    rows = []
    for col in CATEGORICAL_COLUMNS + NUMERIC_COLUMNS:
        table = pd.crosstab(data[col], data[TARGET])
        rates = churn_rate_by(data, col)
        rows.append(
            {
                "feature": col,
                "cramers_v": cramers_v(data[col], data[TARGET]),
                "p_value": chi2_contingency(table).pvalue,
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
    """Compact p-value for tables; tiny values underflow, so they are shown as a bound."""
    return "< 1e-300" if p < 1e-300 else f"{p:.0e}"
