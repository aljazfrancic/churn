"""Plotting helpers: one shared look, churn highlighted in blue, everything else in gray (D34).

Colours come from a validated, colour-blind-safe reference palette. Emphasis form: churned
customers / churn rates are the accent blue, retained customers the de-emphasis gray, magnitudes a
single-hue blue ramp (darker = more churn). Figures are static PNGs on an opaque light surface.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from churn.config import NUMERIC_COLUMNS, TARGET
from churn.eda import churn_rate_by, quantile_bins

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CHURN = "#2a78d6"  # accent
STAYED = "#898781"  # de-emphasis gray
BLUE_RAMP = [  # steps 100 → 700
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]  # fmt: skip
SEQUENTIAL = LinearSegmentedColormap.from_list("churn_blues", BLUE_RAMP)
# Contract length is ordered → one-hue ordinal ramp (steps 650 / 450 / 250), darkest = riskiest.
CONTRACT_COLORS = {"Month-to-month": "#104281", "One year": "#2a78d6", "Two year": "#86b6ef"}

SHORT_LABELS = {
    "Bank transfer (automatic)": "Bank transfer (auto)",
    "Credit card (automatic)": "Credit card (auto)",
}
NUMERIC_LABELS = {
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly charges ($)",
    "TotalCharges": "Total charges ($)",
}
NUMERIC_BINS = {
    "tenure": np.arange(0, 76, 3),
    "MonthlyCharges": np.arange(15, 125, 5),
    "TotalCharges": np.arange(0, 9001, 300),
}
MIN_CUSTOMERS = 30  # rates over fewer customers are too noisy to show


def set_theme() -> None:
    """Apply the shared look: opaque light surface, hairline grid, recessive axes."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 100,
            "font.size": 10,
            "text.color": INK,
            "figure.titlesize": 13,
            "figure.titleweight": "bold",
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.labelsize": 9.5,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def _suptitle(fig: Figure, title: str) -> None:
    fig.suptitle(title, x=0.01, ha="left")


def plot_target_distribution(df: pd.DataFrame, title: str) -> Figure:
    """Share of churned vs retained customers as one split bar (a meter, not a pie)."""
    counts = df[TARGET].value_counts()
    shares = counts / counts.sum()
    fig, ax = plt.subplots(figsize=(11, 1.3))
    left = 0.0
    for name, level, color, text_color in [
        ("Churned", "Yes", CHURN, "white"),
        ("Stayed", "No", to_rgba(STAYED, 0.35), INK),
    ]:
        ax.barh(
            0, shares[level], left=left, height=0.6, color=color, edgecolor=SURFACE, linewidth=2
        )
        label = f"{name}  {counts[level]:,} ({shares[level]:.1%})"
        ax.text(left + shares[level] / 2, 0, label, ha="center", va="center", color=text_color)
        left += shares[level]
    ax.set_xlim(0, 1)
    ax.set_axis_off()
    ax.set_title(title, fontsize=13, pad=10)
    fig.subplots_adjust(left=0, right=1)  # full width, like the other figures
    return fig


def _rate_panel(ax: Axes, rates: pd.DataFrame, overall: float, slots: int, title: str) -> None:
    """Churn-rate bars, first level on top, with direct values and the overall rate (dashed).

    `rates` has the columns churn_rate and customers; `slots` is the level count of the largest
    panel, so panels with fewer levels are centred and all bars keep one thickness.
    """
    y = np.arange(len(rates))[::-1] + (slots - len(rates)) / 2
    ax.barh(y, rates["churn_rate"], height=0.6, color=CHURN, edgecolor=SURFACE, linewidth=1)
    labels = [f"{SHORT_LABELS.get(lvl, lvl)} ({n:,})" for lvl, n in rates["customers"].items()]
    ax.set_yticks(y, labels, fontsize=8.5)
    for yi, rate in zip(y, rates["churn_rate"], strict=True):
        # Surface-coloured box so the dashed reference line breaks behind the value.
        ax.text(
            rate + 0.012,
            yi,
            f"{rate:.0%}",
            va="center",
            fontsize=8.5,
            color=INK_SECONDARY,
            bbox={"boxstyle": "square,pad=0.1", "facecolor": SURFACE, "edgecolor": "none"},
        )
    ax.axvline(overall, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.set(xlim=(0, 0.62), ylim=(-0.6, slots - 0.4), xticks=[], title=title)
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)


def plot_churn_rate_by_category(
    df: pd.DataFrame, columns: list[str], title: str, ncols: int = 3
) -> Figure:
    """Small multiples: churn rate per level of each categorical feature, strongest first.

    Every bar carries its value, so panels need no x-axis; the dashed line is the overall rate.
    """
    overall = df[TARGET].eq("Yes").mean()
    rates = {col: churn_rate_by(df, col) for col in columns}  # highest rate first
    spread = {col: r["churn_rate"].max() - r["churn_rate"].min() for col, r in rates.items()}
    ordered = sorted(columns, key=spread.get, reverse=True)
    max_levels = max(len(r) for r in rates.values())
    nrows = -(-len(columns) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 1.6 * nrows))
    for ax, col in zip(axes.flat, ordered, strict=False):
        _rate_panel(ax, rates[col], overall, max_levels, col)

    for ax in axes.flat[len(ordered) :]:
        ax.set_axis_off()
    axes.flat[-1].text(
        0.05,
        0.5,
        f"Bar = churn rate of a level\n(customers in brackets)\n\n"
        f"Dashed line = overall\nchurn rate ({overall:.1%})\n\nPanels sorted by spread",
        transform=axes.flat[-1].transAxes,
        va="center",
        color=INK_SECONDARY,
        fontsize=9,
    )
    _suptitle(fig, title)
    fig.tight_layout()
    return fig


def plot_new_features(features: pd.DataFrame, y: pd.Series, title: str) -> Figure:
    """Churn rate per level of each engineered feature, in natural order (D20).

    Continuous features are binned for display. NumAddons is shown for internet customers (the
    others have nothing to count), AbsChargeDelta for customers with a billing history.
    """
    internet = features["InternetService"] != "No"
    history = features["tenure"] > 1  # tenure 0–1: the average equals the current bill
    delta = pd.cut(
        features.loc[history, "AbsChargeDelta"],
        [0, 1, 2, 5, np.inf],
        right=False,
        labels=["< $1", "$1–2", "$2–5", "≥ $5"],
    )
    panels = {
        "TenureGroup (months)": features["TenureGroup"],
        "NumAddons (internet only)": features.loc[internet, "NumAddons"],
        "AbsChargeDelta (tenure > 1)": delta,
        "AvgMonthlyCharges ($)": quantile_bins(features["AvgMonthlyCharges"]),
    }
    overall = y.mean()
    slots = max(levels.nunique() for levels in panels.values())

    fig, axes = plt.subplots(1, len(panels), figsize=(11, 2.8))
    for ax, (name, levels) in zip(axes, panels.items(), strict=True):
        # Select the panel's customers first, so integer levels don't turn into floats.
        rates = (
            y.loc[levels.index]
            .groupby(levels, observed=True)
            .agg(churn_rate="mean", customers="size")
        )
        _rate_panel(ax, rates, overall, slots, name)
    note = f"Customers in brackets · dashed line = overall churn rate ({overall:.1%})"
    fig.text(0.01, -0.02, note, color=MUTED, fontsize=8.5)
    _suptitle(fig, title)
    fig.tight_layout()
    return fig


def plot_numeric_distributions(df: pd.DataFrame, title: str) -> Figure:
    """Distribution of each numeric feature for churned vs retained customers (% of each group)."""
    fig, axes = plt.subplots(1, len(NUMERIC_COLUMNS), figsize=(11, 3.4))
    for ax, col in zip(axes, NUMERIC_COLUMNS, strict=True):
        edges = NUMERIC_BINS[col]
        shares = {}
        for level, color in [("No", STAYED), ("Yes", CHURN)]:
            values = df.loc[df[TARGET] == level, col]
            shares[level] = np.histogram(values, bins=edges)[0] / len(values)
            ax.stairs(shares[level], edges, fill=True, color=color, alpha=0.1, linewidth=0)
            ax.stairs(shares[level], edges, color=color, linewidth=2)
        if ax is axes[0]:
            # Direct labels above the bin where each group most exceeds the other (free space).
            for name, own, other in [("Churned", "Yes", "No"), ("Stayed", "No", "Yes")]:
                i = (shares[own] - shares[other]).argmax()
                right_half = i > len(edges) / 2
                x, ha = (edges[i + 1], "right") if right_half else (edges[i], "left")
                ax.text(x, shares[own][i] + 0.004, name, ha=ha, va="bottom", color=INK_SECONDARY)
            ax.margins(y=0.12)
        ax.set_title(NUMERIC_LABELS[col])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="x", visible=False)
        ax.set_xlim(edges[0], edges[-1])
    axes[0].set_ylabel("Share of group")
    handles = [Line2D([], [], color=c, linewidth=2) for c in (CHURN, STAYED)]
    fig.legend(handles, ["Churned", "Stayed"], loc="upper right", ncols=2)
    _suptitle(fig, title)
    fig.tight_layout()
    return fig


def plot_churn_heatmap(df: pd.DataFrame, title: str) -> Figure:
    """Churn rate on a tenure × monthly-charges grid (cells with < 30 customers left blank)."""
    tenure = pd.cut(
        df["tenure"], [-1, 6, 12, 24, 48, 72], labels=["0–6", "7–12", "13–24", "25–48", "49–72"]
    )
    charges = pd.cut(
        df["MonthlyCharges"],
        [0, 30, 50, 70, 90, np.inf],
        labels=["< $30", "$30–50", "$50–70", "$70–90", "≥ $90"],
        right=False,
    )
    stats = df[TARGET].eq("Yes").groupby([tenure, charges], observed=False).agg(["mean", "size"])
    rate = stats["mean"].where(stats["size"] >= MIN_CUSTOMERS).unstack()

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sns.heatmap(
        rate,
        cmap=SEQUENTIAL,
        vmin=0,
        vmax=0.8,
        annot=True,
        fmt=".0%",
        linewidths=2,
        linecolor=SURFACE,
        cbar_kws={"format": PercentFormatter(1, decimals=0), "label": "Churn rate"},
        ax=ax,
    )
    ax.set(xlabel="Monthly charges", ylabel="Tenure (months)")
    ax.tick_params(axis="y", rotation=0)
    ax.grid(False)
    _suptitle(fig, title)
    if rate.isna().any().any():
        note = f"Blank cells: fewer than {MIN_CUSTOMERS} customers."
        fig.text(0.01, -0.02, note, color=MUTED, fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_churn_by_tenure_and_contract(df: pd.DataFrame, title: str) -> Figure:
    """Churn rate over tenure (6-month bins) for each contract type."""
    edges = [-1, *range(6, 73, 6)]  # tenure 0 joins the first bin
    fig, ax = plt.subplots(figsize=(11, 4))
    for contract, color in CONTRACT_COLORS.items():
        subset = df[df["Contract"] == contract]
        stats = churn_rate_by(subset, pd.cut(subset["tenure"], edges)).sort_index()
        stats = stats[stats["customers"] >= MIN_CUSTOMERS]
        x = [interval.right - 3 for interval in stats.index]  # bin centre
        ax.plot(
            x,
            stats["churn_rate"],
            color=color,
            marker="o",
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=contract,
        )
        # Direct label at the line end (outside the plot area, so it never covers data).
        ax.text(x[-1] + 2, stats["churn_rate"].iloc[-1], contract, va="center", clip_on=False)
    ax.set(xlabel="Tenure (months)", ylabel="Churn rate", xlim=(0, 72), ylim=(0, None))
    ax.set_xticks(range(0, 73, 12))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="x", visible=False)
    ax.legend(title="Contract", loc="upper right", alignment="left")
    _suptitle(fig, title)
    fig.tight_layout()
    return fig


def plot_association_matrix(matrix: pd.DataFrame, title: str) -> Figure:
    """Lower triangle of a Cramér's V matrix; values labelled for churn and strong pairs only."""
    m = matrix.iloc[1:, :-1]  # first row / last column would be empty in a lower triangle
    mask = np.triu(np.ones(m.shape, dtype=bool), k=1)
    labels = m.map(lambda v: f"{v:.2f}")
    strong = (m >= 0.3).to_numpy(copy=True)
    strong[:, 0] = True  # the churn column is the story, label all of it
    labels = labels.where(strong, "")

    fig, ax = plt.subplots(figsize=(11, 9.5))
    sns.heatmap(
        m,
        mask=mask,
        cmap=SEQUENTIAL,
        vmin=0,
        vmax=1,
        annot=labels,
        fmt="",
        annot_kws={"fontsize": 7.5},
        linewidths=1,
        linecolor=SURFACE,
        square=True,
        cbar_kws={"shrink": 0.6, "label": "Cramér's V"},
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=90)
    ax.grid(False)
    _suptitle(fig, title)
    fig.tight_layout()
    return fig
