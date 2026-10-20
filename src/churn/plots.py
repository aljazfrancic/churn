"""Plotting helpers: one shared look, churn highlighted in blue, everything else in gray (D34).

Colours come from a validated, colour-blind-safe reference palette. Emphasis form: churned
customers / churn rates are the accent blue, retained customers the de-emphasis gray, magnitudes a
single-hue blue ramp (darker = more churn); in model plots the selected model takes the accent,
and threshold metrics the palette's first three categorical hues. Figures are static PNGs on an
opaque light surface.
"""

from collections.abc import Collection

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MaxNLocator, NullLocator, PercentFormatter
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from churn.config import NUMERIC_COLUMNS, TARGET
from churn.eda import churn_rate_by, quantile_bins
from churn.evaluation import SIMPLEST_FIRST
from churn.explain import ShapValues

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
SELECTED_MODEL, OTHER_MODEL = CHURN, STAYED  # model plots: emphasis on the chosen model
# Metric lines: categorical slots 1–3 (blue, orange, aqua), validated for every pair; aqua is
# below 3:1 contrast on the surface, so every line also gets a direct label.
METRIC_COLORS = {"f2": "#2a78d6", "recall": "#eb6834", "precision": "#1baf7a"}
# Input values in SHAP plots (low → high): the palette's diverging pair with a gray midpoint.
INPUT_VALUE = LinearSegmentedColormap.from_list("input_value", ["#2a78d6", MUTED, "#e34948"])

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


def _fold_panel(
    ax: Axes, folds: pd.DataFrame, highlight: Collection[str], labels: dict, title: str
) -> None:
    """Box + dots of the fold scores, one row per column of `folds`, first column on top.

    Columns in `highlight` take the accent. Whiskers span all folds, as every fold is drawn
    anyway; `labels` sit right of the panel.
    """
    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(folds))  # same figure every run
    transform = ax.get_yaxis_transform()  # x in axes fraction, y in data
    for y, name in zip(range(folds.shape[1] - 1, -1, -1), folds.columns, strict=True):
        color = SELECTED_MODEL if name in highlight else OTHER_MODEL
        text_color = SELECTED_MODEL if name in highlight else INK_SECONDARY
        ax.text(
            1.03, y, labels[name], transform=transform, va="center", fontsize=9, color=text_color
        )
        values = folds[name].dropna()  # empty for the reference of a difference panel
        if values.empty:
            continue
        line = {"color": color, "linewidth": 1.2}
        ax.boxplot(
            values,
            positions=[y],
            orientation="horizontal",
            widths=0.5,
            whis=(0, 100),
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            boxprops={**line, "facecolor": to_rgba(color, 0.12), "edgecolor": color},
            medianprops={"color": color, "linewidth": 2},
            whiskerprops=line,
            capprops=line,
        )
        ax.scatter(values, y + jitter[: len(values)], s=14, color=color, alpha=0.7, linewidth=0)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)


def plot_model_comparison(scores: pd.DataFrame, selected: str, title: str) -> Figure:
    """Fold scores per model: PR-AUC, its difference to `selected` on the same fold, Brier (D41).

    Rows are sorted by mean PR-AUC, best on top. Only models the selection rule considers are
    drawn: the baseline's PR-AUC of 0.265 would squash the axes.
    """
    scores = scores[scores["model"].isin(SIMPLEST_FIRST)]
    pr_auc, brier = (
        scores.pivot(index="fold", columns="model", values=m) for m in ["pr_auc", "brier"]
    )
    order = pr_auc.mean().sort_values(ascending=False).index
    pr_auc, brier = pr_auc[order], brier[order]
    gain = pr_auc.sub(pr_auc[selected], axis=0).drop(columns=selected).reindex(columns=order)
    wins = (gain > 0).sum()
    n_folds = len(gain)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    means = {name: f"{value:.3f}" for name, value in pr_auc.mean().items()}
    _fold_panel(axes[0], pr_auc, {selected}, means, "PR-AUC (higher = better)")
    counts = {name: f"better in {wins[name]}/{n_folds}" for name in order}
    counts[selected] = "reference"
    _fold_panel(axes[1], gain, {selected}, counts, f"PR-AUC vs {selected}")
    axes[1].axvline(0, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    means = {name: f"{value:.3f}" for name, value in brier.mean().items()}
    _fold_panel(axes[2], brier, {selected}, means, "Brier score (lower = better)")
    for ax, metric in zip(axes, ["PR-AUC", "PR-AUC difference", "Brier score"], strict=True):
        ax.set_xlabel(metric)

    axes[0].set_yticks(range(len(order) - 1, -1, -1), order)
    axes[0].set_ylim(-0.6, len(order) - 0.4)
    for label in axes[0].get_yticklabels():
        if label.get_text() == selected:
            label.set(color=SELECTED_MODEL, fontweight="bold")
    note = (
        f"Dots: {n_folds} folds · box: middle half, line: median · numbers: mean PR-AUC and "
        f"Brier score · centre: PR-AUC minus {selected} on the same fold"
    )
    fig.text(0.01, -0.02, note, color=MUTED, fontsize=8.5)
    _suptitle(fig, title)
    fig.tight_layout()
    return fig


def _change_label(changes: pd.Series) -> str:
    """Mean change (typographic minus) and the folds where it rose; 'identical' if none moved."""
    if (changes == 0).all():
        return "identical"
    mean = round(changes.mean(), 3) + 0.0  # + 0.0 turns -0.0 into 0.0
    return f"{mean:+.3f} ({(changes > 0).sum()}/{len(changes)})".replace("-", "\N{MINUS SIGN}")


def plot_ablation(effects: pd.DataFrame, highlight: str, title: str) -> Figure:
    """Same-fold PR-AUC change of each preprocessing step, one panel per model (D44).

    `effects` comes from churn.ablation.step_effects. Rows are the steps in the order they are
    added; every panel has its own x-scale, as the unscaled SVM is far behind every other variant.
    """
    models = effects.columns.unique(level=0)
    steps = effects[models[0]].columns
    fig, axes = plt.subplots(1, len(models), figsize=(11, 3.4), sharey=True)
    for ax, model in zip(axes, models, strict=True):
        changes = effects[model]
        labels = {step: _change_label(changes[step]) for step in steps}
        rows, name = (steps, f"{model} (final)") if model == highlight else ([], model)
        _fold_panel(ax, changes, rows, labels, name)
        ax.axvline(0, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        # Narrow panels: few, round ticks so their labels never collide.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        ax.set_xlabel("PR-AUC change")

    axes[0].set_yticks(range(len(steps) - 1, -1, -1), steps)
    axes[0].set_ylim(-0.6, len(steps) - 0.4)
    note = (
        f"Dots: {len(effects)} folds · box: middle half, line: median · "
        "numbers: mean PR-AUC change (folds where it rose) · each panel has its own scale"
    )
    fig.text(0.01, -0.02, note, color=MUTED, fontsize=8.5)
    _suptitle(fig, title)
    fig.tight_layout()
    return fig


def _threshold_marker(
    ax: Axes, x: float, y: float, label: str, chosen: bool, offset: tuple[int, int]
) -> None:
    """Where a threshold sits on a curve: filled for the chosen one, hollow for the others."""
    face, edge = (SELECTED_MODEL, SURFACE) if chosen else (SURFACE, SELECTED_MODEL)
    ax.plot(x, y, "o", markersize=8, markerfacecolor=face, markeredgecolor=edge, mew=2, zorder=4)
    ha = "left" if offset[0] > 0 else "right"
    va = "bottom" if offset[1] > 0 else "top"
    ax.annotate(label, (x, y), xytext=offset, textcoords="offset points", ha=ha, va=va, fontsize=9)


def _diagonal_label(ax: Axes, x: float, text: str) -> None:
    """Muted label just below the dashed diagonal, rotated with it whatever the panel's shape."""
    ax.text(
        x,
        x - 0.03,
        text,
        rotation=45,
        transform_rotates_text=True,
        rotation_mode="anchor",
        ha="center",
        va="top",
        color=MUTED,
        fontsize=9,
    )


def plot_holdout_curves(
    y_true: pd.Series, proba: pd.DataFrame, selected: str, thresholds: dict[str, float], title: str
) -> Figure:
    """ROC and precision–recall curves on the test set (D33, D47).

    `selected` takes the accent, the other models the gray; markers show where each threshold
    sits on the selected curve (the last one filled: the chosen threshold). The dummy is left
    out, as its curves are the no-skill lines.
    """
    y = np.asarray(y_true)
    others = [name for name in proba.columns if name in SIMPLEST_FIRST and name != selected]
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(11, 4.6))
    for name in [*others, selected]:  # the selected curve on top
        style = (
            {"color": SELECTED_MODEL, "linewidth": 2, "zorder": 3}
            if name == selected
            else {"color": OTHER_MODEL, "linewidth": 1, "alpha": 0.6}
        )
        fpr, tpr, _ = roc_curve(y, proba[name])
        roc_ax.plot(fpr, tpr, **style)
        precision, recall, _ = precision_recall_curve(y, proba[name])
        pr_ax.plot(recall, precision, drawstyle="steps-post", **style)

    dashed = {"color": MUTED, "linewidth": 1, "linestyle": (0, (3, 3))}
    roc_ax.plot([0, 1], [0, 1], **dashed)
    _diagonal_label(roc_ax, 0.75, "no skill")
    pr_ax.axhline(y.mean(), **dashed)
    pr_ax.text(0.02, y.mean() - 0.02, f"no skill ({y.mean():.1%} churn)", va="top", color=MUTED)

    p = proba[selected].to_numpy()
    for i, (label, threshold) in enumerate(thresholds.items()):
        flagged = p >= threshold
        recall, fpr, precision = flagged[y == 1].mean(), flagged[y == 0].mean(), y[flagged].mean()
        chosen = i == len(thresholds) - 1
        _threshold_marker(roc_ax, fpr, recall, label, chosen, offset=(8, -6))
        _threshold_marker(pr_ax, recall, precision, label, chosen, offset=(8, 6))

    roc_ax.set(
        title="ROC curve",
        xlabel="False-positive rate (loyal customers flagged)",
        ylabel="Recall (churners flagged)",
    )
    pr_ax.set(
        title="Precision–recall curve",
        xlabel="Recall (churners flagged)",
        ylabel="Precision (flagged customers who churn)",
    )
    for ax in (roc_ax, pr_ax):
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    handles = [
        Line2D([], [], color=SELECTED_MODEL, linewidth=2),
        Line2D([], [], color=OTHER_MODEL, linewidth=1, alpha=0.6),
    ]
    roc_ax.legend(handles, [f"{selected} (final)", "other models"], loc="lower right")
    note = "Markers: customers with a churn probability at or above the threshold are flagged"
    fig.text(0.01, -0.02, note, color=MUTED, fontsize=8.5)
    _suptitle(fig, title)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(
    y_true: pd.Series, proba: np.ndarray, thresholds: dict[str, float], title: str
) -> Figure:
    """Test-set confusion matrix per threshold: counts and % of the actual class (D30, D47).

    Rows and columns as in the metrics table of section 4.1: churners first, flagged first.
    """
    churns, p = np.asarray(y_true, dtype=bool), np.asarray(proba)
    meaning = [["churners reached", "churners missed"], ["offers wasted", "left alone"]]
    rows = [f"Churns ({churns.sum():,})", f"Stays ({(~churns).sum():,})"]
    fig, axes = plt.subplots(1, len(thresholds), figsize=(11, 3.2))
    for ax, (label, threshold) in zip(axes, thresholds.items(), strict=True):
        flagged = p >= threshold
        counts = np.array(
            [
                [np.sum(churns & flagged), np.sum(churns & ~flagged)],
                [np.sum(~churns & flagged), np.sum(~churns & ~flagged)],
            ]
        )
        shares = counts / counts.sum(axis=1, keepdims=True)
        cells = [
            [f"{counts[i, j]:,} ({shares[i, j]:.0%})\n{meaning[i][j]}" for j in range(2)]
            for i in range(2)
        ]
        sns.heatmap(
            shares,
            cmap=SEQUENTIAL,
            vmin=0,
            vmax=1,
            annot=cells,
            fmt="",
            annot_kws={"fontsize": 9.5},
            cbar=False,
            linewidths=2,
            linecolor=SURFACE,
            xticklabels=["Flagged (offer)", "Not flagged"],
            yticklabels=rows,
            ax=ax,
        )
        ax.tick_params(axis="y", rotation=0)
        ax.set_title(f"Threshold {label}")
        ax.grid(False)
    note = "Count (share of the actual class): each row adds up to 100%"
    fig.text(0.01, -0.02, note, color=MUTED, fontsize=8.5)
    _suptitle(fig, title)
    fig.tight_layout()
    return fig


def _label_line(
    ax: Axes, lines: pd.DataFrame, column: str, text: str, avoid: Collection[float]
) -> None:
    """Direct label where the line is farthest from the other lines, inside the frame and clear
    of the vertical lines at `avoid`."""
    inside = lines.loc[0.05:0.8]
    clear = np.abs(np.subtract.outer(inside.index.to_numpy(), list(avoid))).min(axis=1) > 0.04
    inside = inside[clear & inside[column].between(0.08, 0.9)]
    gaps = inside.drop(columns=column).sub(inside[column], axis=0)
    distance = gaps.abs().min(axis=1)
    x = distance.idxmax()
    nearest = gaps.loc[x].abs().idxmin()
    above = gaps.loc[x, nearest] < 0  # the nearest line runs below: put the label above
    ax.annotate(
        text,
        (x, inside.loc[x, column]),
        xytext=(0, 6 if above else -6),
        textcoords="offset points",
        ha="center",
        va="bottom" if above else "top",
        color=INK_SECONDARY,
        fontsize=9,
        bbox={"boxstyle": "square,pad=0.1", "facecolor": SURFACE, "edgecolor": "none"},
    )


def plot_threshold_metrics(
    curves: pd.DataFrame, thresholds: dict[str, float], n_customers: int, title: str
) -> Figure:
    """Precision, recall, F2 and the share of customers flagged at every threshold (D33).

    `curves` comes from churn.threshold.metrics_by_threshold. Precision is hidden where fewer
    than 30 customers are flagged; vertical lines mark `thresholds` (the last one: chosen).
    """
    curves = curves.copy()
    flagged = curves["share_flagged"] * n_customers
    curves["precision"] = curves["precision"].where(flagged >= MIN_CUSTOMERS)
    lines = {
        "precision": ("Precision", METRIC_COLORS["precision"], "-"),
        "recall": ("Recall", METRIC_COLORS["recall"], "-"),
        "f2": ("F2", METRIC_COLORS["f2"], "-"),
        "share_flagged": ("Customers flagged", INK_SECONDARY, (0, (4, 3))),
    }
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for column, (label, color, style) in lines.items():
        ax.plot(curves.index, curves[column], color=color, linestyle=style, label=label)
    for column, (label, *_) in lines.items():
        _label_line(ax, curves[list(lines)], column, label, thresholds.values())
    for i, (label, threshold) in enumerate(thresholds.items()):
        chosen = i == len(thresholds) - 1
        ax.axvline(
            threshold,
            color=INK_SECONDARY if chosen else MUTED,
            linewidth=1,
            linestyle="-" if chosen else (0, (3, 3)),
        )
        ax.text(
            threshold,
            1.01,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK_SECONDARY,
        )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Threshold (churn probability from which a customer is flagged)",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(loc="upper right", ncols=4, bbox_to_anchor=(1, 1.16))
    note = (
        f"Test set ({n_customers:,} customers) · precision hidden where fewer than "
        f"{MIN_CUSTOMERS} customers are flagged"
    )
    fig.text(0.01, -0.02, note, color=MUTED, fontsize=8.5)
    _suptitle(fig, title)
    fig.tight_layout()
    return fig


def plot_calibration_and_gains(
    y_true: pd.Series, proba: np.ndarray, thresholds: dict[str, float], title: str
) -> Figure:
    """Calibration curve (10 equal-size bins) and cumulative gains on the test set (D33).

    Gains: customers contacted in order of predicted risk; markers show the share contacted and
    the share of churners reached at each threshold (the last one filled: the chosen threshold).
    """
    y, p = np.asarray(y_true), np.asarray(proba)
    fig, (cal_ax, gain_ax) = plt.subplots(1, 2, figsize=(11, 4.4))
    dashed = {"color": MUTED, "linewidth": 1, "linestyle": (0, (3, 3))}

    observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
    cal_ax.plot([0, 1], [0, 1], **dashed)
    _diagonal_label(cal_ax, 0.85, "perfect calibration")
    cal_ax.plot(
        predicted,
        observed,
        color=SELECTED_MODEL,
        marker="o",
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
    )
    cal_ax.set(
        title="Calibration",
        xlabel="Predicted churn risk (mean of the bin)",
        ylabel="Observed churn rate",
    )

    order = np.argsort(-p, kind="stable")
    contacted = np.arange(len(y) + 1) / len(y)
    reached = np.concatenate([[0], np.cumsum(y[order]) / y.sum()])
    gain_ax.plot([0, 1], [0, 1], **dashed)
    _diagonal_label(gain_ax, 0.75, "random order")
    gain_ax.plot([0, y.mean(), 1], [0, 1, 1], color=MUTED, linewidth=1)
    gain_ax.text(y.mean() + 0.02, 0.97, "perfect ranking", color=MUTED, va="top", fontsize=9)
    gain_ax.plot(contacted, reached, color=SELECTED_MODEL)
    for i, (label, threshold) in enumerate(thresholds.items()):
        flagged = p >= threshold
        chosen = i == len(thresholds) - 1
        _threshold_marker(
            gain_ax, flagged.mean(), flagged[y == 1].mean(), label, chosen, offset=(8, -6)
        )
    gain_ax.set(
        title="Cumulative gains",
        xlabel="Customers contacted, highest risk first",
        ylabel="Churners reached",
    )
    for ax in (cal_ax, gain_ax):
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    note = f"Calibration: 10 bins of about {len(y) // 10} test customers each, by predicted risk"
    fig.text(0.01, -0.02, note, color=MUTED, fontsize=8.5)
    _suptitle(fig, title)
    fig.tight_layout()
    return fig


def _input_label(name: str, feature: str) -> str:
    """`Contract_Two year` → `Contract = Two year`; numeric inputs keep their name."""
    return name if name == feature else f"{feature} = {name[len(feature) + 1 :]}"


def _swarm_offsets(values: np.ndarray, width: float = 0.38, bins: int = 40) -> np.ndarray:
    """Vertical offsets in proportion to the local density of the values (a beeswarm look)."""
    counts, edges = np.histogram(values, bins=bins)
    which = np.clip(np.digitize(values, edges[1:-1]), 0, bins - 1)
    spread = width * counts[which] / counts.max()
    return np.random.default_rng(0).uniform(-1, 1, len(values)) * spread  # same figure every run


def plot_shap_summary(shap_values: ShapValues, title: str, top: int = 15) -> Figure:
    """Mean |SHAP| per original feature, and how the strongest inputs push the risk (D49).

    Left: the one-hot columns of a feature summed. Right: one dot per customer and input,
    coloured by the input's value (percentile for numeric inputs, no/yes for one-hot columns).
    """
    by_feature = shap_values.by_feature().abs().mean().sort_values(ascending=False)
    strongest = shap_values.values.abs().mean().sort_values(ascending=False).index[:top]
    fig, (bar_ax, swarm_ax) = plt.subplots(
        1, 2, figsize=(11, 6.4), gridspec_kw={"width_ratios": [1, 1.5]}
    )

    y = np.arange(len(by_feature))[::-1]
    bar_ax.barh(y, by_feature, height=0.6, color=CHURN)
    for yi, value in zip(y, by_feature, strict=True):
        bar_ax.text(
            value + 0.008, yi, f"{value:.2f}", va="center", fontsize=8.5, color=INK_SECONDARY
        )
    bar_ax.set_yticks(y, by_feature.index, fontsize=8.5)
    bar_ax.set(
        title="Per feature", xlabel="Mean |SHAP value| (log-odds)", ylim=(-0.6, len(y) - 0.4)
    )
    bar_ax.grid(axis="y", visible=False)

    for row, name in zip(range(top - 1, -1, -1), strongest, strict=True):
        values, inputs = shap_values.values[name], shap_values.inputs[name]
        binary = inputs.nunique() <= 2
        shade = (
            (inputs - inputs.min()) / (inputs.max() - inputs.min())
            if binary
            else inputs.rank(pct=True)
        )
        offsets = _swarm_offsets(values.to_numpy())
        style = {"s": 5, "alpha": 0.8, "linewidth": 0, "rasterized": True}
        swarm_ax.scatter(values, row + offsets, c=shade, cmap=INPUT_VALUE, vmin=0, vmax=1, **style)
    labels = [_input_label(name, shap_values.groups[name]) for name in strongest]
    swarm_ax.set_yticks(range(top - 1, -1, -1), labels, fontsize=8.5)
    swarm_ax.axvline(0, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    swarm_ax.set(
        title=f"Strongest {top} model inputs",
        xlabel="SHAP value (log-odds; > 0 raises the churn risk)",
        ylim=(-0.6, top - 0.4),
    )
    swarm_ax.grid(axis="y", visible=False)
    bar = fig.colorbar(ScalarMappable(cmap=INPUT_VALUE), ax=swarm_ax, shrink=0.4, aspect=15)
    bar.set_ticks([0, 1], labels=["low / no", "high / yes"])
    bar.set_label("Input value", color=INK_SECONDARY)
    bar.outline.set_visible(False)
    note = (
        f"{len(shap_values.values):,} test customers · SHAP value: how far an input moves a "
        "customer's log-odds from those of a customer with every input at its training mean"
    )
    fig.text(0.01, -0.02, note, color=MUTED, fontsize=8.5)
    _suptitle(fig, title)
    fig.tight_layout()
    return fig


def _format_value(value: object) -> str:
    """Feature values for labels: amounts with two decimals, counts and levels as they are."""
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.2f}"
    return str(int(value)) if isinstance(value, (int, float, np.integer)) else str(value)


def _log_odds(probability: float) -> float:
    return float(np.log(probability / (1 - probability)))


def _risk(log_odds: float) -> float:
    return float(1 / (1 + np.exp(-log_odds)))


def plot_shap_waterfall(
    shap_values: ShapValues, customer: object, threshold: float, title: str, top: int = 8
) -> Figure:
    """How each feature moves one customer's log-odds away from the average customer (D49).

    Blue bars raise the churn risk, gray ones lower it; features beyond the `top` strongest are
    summed in one bar. The dashed line is the decision threshold.
    """
    contributions = shap_values.by_feature().loc[customer]
    order = contributions.abs().sort_values(ascending=False).index
    features = shap_values.features.loc[customer]
    steps = {
        f"{name} = {_format_value(features[name])}": contributions[name] for name in order[:top]
    }
    steps[f"{len(order) - top} other features"] = contributions[order[top:]].sum()
    start = shap_values.base_value
    end = start + contributions.sum()

    fig, ax = plt.subplots(figsize=(11, 4.8))
    rows = len(steps) + 2  # start, the steps, end
    y = np.arange(rows)[::-1]
    left = start
    for yi, value in zip(y[1:-1], steps.values(), strict=True):
        color = CHURN if value > 0 else STAYED
        ax.barh(yi, value, left=left, height=0.6, color=color, edgecolor=SURFACE, linewidth=1)
        text = f"{value:+.2f}".replace("-", "\N{MINUS SIGN}")
        x, ha = (left + value + 0.04, "left") if value > 0 else (left + value - 0.04, "right")
        ax.text(x, yi, text, va="center", ha=ha, fontsize=8.5, color=INK_SECONDARY)
        ax.plot([left + value] * 2, [yi - 0.3, yi - 0.7], color=AXIS, linewidth=1)  # connector
        left += value
    path = start + np.cumsum([0, *steps.values()])
    low, high = min(path.min(), _log_odds(threshold)), max(path.max(), _log_odds(threshold))
    pad = 0.12 * (high - low)  # room for the labels at both ends
    for yi, value, side in [(y[0], start, -1), (y[-1], end, 1)]:
        ax.plot(value, yi, "o", markersize=9, color=INK, markeredgecolor=SURFACE, mew=2)
        ha = "right" if side < 0 else "left"
        label = f"risk {_risk(value):.0%}"
        ax.text(value + side * pad / 4, yi, label, ha=ha, va="center", fontsize=9, color=INK)
    ax.axvline(_log_odds(threshold), color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(
        _log_odds(threshold), rows - 0.45, f" threshold {threshold:.2f}", color=MUTED, fontsize=8.5
    )
    ax.set_yticks(y, ["Average customer", *steps, "This customer"], fontsize=9)
    ax.set(xlabel="Log-odds of churning", xlim=(low - pad, high + pad), ylim=(-0.7, rows - 0.2))
    ax.grid(axis="y", visible=False)
    note = (
        "Bars: SHAP values per feature (one-hot columns summed) · average customer: every input at "
        "its training mean · risk = 1 / (1 + e^−log-odds)"
    )
    fig.text(0.01, -0.02, note, color=MUTED, fontsize=8.5)
    _suptitle(fig, title)
    fig.tight_layout()
    return fig


def plot_odds_ratios(ratios: pd.DataFrame, title: str) -> Figure:
    """Odds ratios with 95 % bootstrap intervals, the feature with the strongest effect first.

    `ratios` comes from churn.explain.odds_ratios. Blue: the interval excludes 1 (a clear
    effect); gray: it includes 1, so the input may have no effect at all (D50).
    """
    strength = np.log(ratios["odds_ratio"]).abs().groupby(level="feature", sort=False).max()
    ordered = ratios.loc[strength.sort_values(ascending=False).index]  # levels stay together
    features = ordered.index.get_level_values("feature")
    gaps = np.cumsum(np.r_[0, features[1:] != features[:-1]]) * 0.5  # space between features
    y = -(np.arange(len(ordered)) + gaps)
    clear = (ordered["ci_low"] > 1) | (ordered["ci_high"] < 1)
    colors = np.where(clear, CHURN, STAYED)

    fig, ax = plt.subplots(figsize=(11, 0.24 * len(ordered) + 1.8))
    ax.hlines(y, ordered["ci_low"], ordered["ci_high"], colors=colors, linewidth=2)
    ax.scatter(
        ordered["odds_ratio"], y, c=colors, s=45, zorder=3, edgecolors=SURFACE, linewidths=1.5
    )
    ax.axvline(1, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    transform = ax.get_yaxis_transform()
    for yi, (_, row) in zip(y, ordered.iterrows(), strict=True):
        text = f"{row['odds_ratio']:.2f} ({row['ci_low']:.2f}–{row['ci_high']:.2f})"
        ax.text(1.02, yi, text, transform=transform, va="center", fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(
        y, [f"{feature}: {comparison}" for feature, comparison in ordered.index], fontsize=8
    )
    ax.set_xscale("log")
    ticks = [
        t
        for t in (0.125, 0.25, 0.5, 1, 2, 4)
        if ordered["ci_low"].min() / 1.5 <= t <= ordered["ci_high"].max() * 1.5
    ]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:g}×" for t in ticks])
    ax.set(
        xlabel="Odds ratio (log scale; < 1 lowers, > 1 raises the odds of churning)",
        ylim=(y.min() - 0.7, 0.7),
    )
    ax.grid(axis="y", visible=False)
    note = (
        "Other inputs held fixed · numeric inputs: per +1 SD after Yeo-Johnson · levels: vs the "
        "most common level · lines: 95 % bootstrap intervals · blue: interval excludes 1"
    )
    fig.text(0.01, -0.02, note, color=MUTED, fontsize=8.5)
    _suptitle(fig, title)
    fig.tight_layout()
    return fig
