"""
Publication-quality plotting utilities.
统一的论文级绘图风格设置模块。

All figures follow these specifications:
- Font: Times New Roman
- DPI: 300
- Cell Metabolism biomedical palette (soft, cool, clean)
- Consistent font sizes and line widths
- Grid: light gray dashed
- Remove top/right spines where appropriate

Style reference: "16-h fasting optimizes cancer immunotherapy in mice and humans"
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =============================================================================
# Cell Metabolism Biomedical Palette
# =============================================================================
# Soft, cool, clean colors - reference: Cell Metabolism journal style
# Avoid high-saturation colors, rainbow/jet maps, use white background
BIOMED_PALETTE = {
    'deep_blue': '#4F587D',      # Deep slate blue (primary, Six_XGBoost, No Anxiety)
    'mauve': '#C68DC0',          # Muted mauve (Anxiety, Ratio_LASSO)
    'pale_cyan': '#C2E0EE',      # Pale cyan (Integrated_XGBoost, needs border for visibility)
    'muted_purple': '#776B97',   # Muted purple (Six_RF)
    'lavender': '#DBC8ED',       # Light lavender (ABIS_LR, needs border)
    'dark_gray': '#333333',      # Dark gray (text, borders)
    'mid_gray': '#888888',       # Mid gray (CRP_only, baseline models)
    'light_gray': '#D9D9D9',     # Light gray (backgrounds)
}

# Group colors for Anxiety vs Non-Anxiety comparison
GROUP_COLORS = {
    'No_Anxiety': BIOMED_PALETTE['deep_blue'],  # #4F587D - Deep slate blue
    'Anxiety': BIOMED_PALETTE['mauve'],          # #C68DC0 - Muted mauve
}

# Model-specific colors for consistent visualization across all steps
MODEL_COLORS = {
    "Six_XGBoost": BIOMED_PALETTE['deep_blue'],          # #4F587D - Deep slate blue (best model)
    "Six_RF": BIOMED_PALETTE['muted_purple'],            # #776B97 - Muted purple
    "Six_LASSO": BIOMED_PALETTE['light_gray'],           # #D9D9D9 - Light gray
    "Integrated_XGBoost": BIOMED_PALETTE['pale_cyan'],   # #C2E0EE - Pale cyan
    "Integrated_RF": BIOMED_PALETTE['mid_gray'],         # #888888 - Mid gray
    "Integrated_LASSO": BIOMED_PALETTE['light_gray'],    # #D9D9D9 - Light gray
    "Ratio_XGBoost": BIOMED_PALETTE['pale_cyan'],        # #C2E0EE - Pale cyan
    "Ratio_RF": BIOMED_PALETTE['mid_gray'],              # #888888 - Mid gray
    "Ratio_LASSO": BIOMED_PALETTE['mauve'],              # #C68DC0 - Muted mauve
    "ABIS_LR": BIOMED_PALETTE['lavender'],               # #DBC8ED - Light lavender
    "Best_Single": BIOMED_PALETTE['mid_gray'],           # #888888 - Mid gray
    "CRP_only": BIOMED_PALETTE['mid_gray'],              # #888888 - Mid gray
}

# Alternative single colors (backup)
PALETTE_BLUE = BIOMED_PALETTE['deep_blue']
PALETTE_PURPLE = BIOMED_PALETTE['muted_purple']
PALETTE_MAUVE = BIOMED_PALETTE['mauve']
PALETTE_CYAN = BIOMED_PALETTE['pale_cyan']
PALETTE_GRAY = BIOMED_PALETTE['mid_gray']

# Default color for single-series plots
DEFAULT_COLOR = PALETTE_BLUE

# Old Okabe-Ito palette (backup only, NOT recommended for this paper)
OKABE_ITO = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion
    "#CC79A7",  # Reddish Purple
    "#000000",  # Black
]

# =============================================================================
# Font and Style Settings
# =============================================================================
FONT_FAMILY = "Times New Roman"
BASE_FONT_SIZE = 9
AXIS_LABEL_SIZE = 10
TITLE_SIZE = 11
LEGEND_SIZE = 8
TICK_LABEL_SIZE = 8
LINE_WIDTH = 1.5
GRID_ALPHA = 0.25
DPI = 300


def apply_publication_style():
    """
    Apply global publication-quality matplotlib style.
    Call this at the beginning of each plotting script.
    """
    # Cell Metabolism color cycle
    color_cycle = [
        BIOMED_PALETTE['deep_blue'],
        BIOMED_PALETTE['muted_purple'],
        BIOMED_PALETTE['mauve'],
        BIOMED_PALETTE['pale_cyan'],
        BIOMED_PALETTE['lavender'],
        BIOMED_PALETTE['mid_gray'],
        BIOMED_PALETTE['light_gray'],
    ]

    plt.rcParams.update(
        {
            # Font settings
            "font.family": FONT_FAMILY,
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            # Line and border settings
            "axes.linewidth": 0.8,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": 6,
            "patch.linewidth": 0.5,
            # Grid settings (light gray)
            "axes.grid": True,
            "grid.alpha": GRID_ALPHA,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "grid.color": BIOMED_PALETTE['light_gray'],  # Light gray grid
            # Legend settings
            "legend.frameon": False,
            "legend.borderpad": 0.3,
            # Figure settings
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # Tick settings
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.direction": "out",
            "ytick.direction": "out",
            # Color cycle (Cell Metabolism biomedical palette)
            "axes.prop_cycle": plt.cycler(color=color_cycle),
        }
    )


def get_colorblind_palette():
    """
    Return the Cell Metabolism biomedical palette (soft, cool, clean).
    """
    return [
        BIOMED_PALETTE['deep_blue'],
        BIOMED_PALETTE['muted_purple'],
        BIOMED_PALETTE['mauve'],
        BIOMED_PALETTE['pale_cyan'],
        BIOMED_PALETTE['lavender'],
        BIOMED_PALETTE['mid_gray'],
        BIOMED_PALETTE['light_gray'],
    ]


def get_biomed_palette():
    """
    Return the full BIOMED_PALETTE dictionary.
    """
    return BIOMED_PALETTE


def get_group_colors():
    """
    Return GROUP_COLORS for Anxiety vs Non-Anxiety comparison.
    """
    return GROUP_COLORS


def get_model_color(model_name):
    """
    Return the standardized color for a given model name.
    """
    return MODEL_COLORS.get(model_name, DEFAULT_COLOR)


def get_line_kwargs(model_name, linewidth=LINE_WIDTH):
    """
    Get matplotlib line kwargs for a model, handling pale colors with borders.

    Pale colors (pale_cyan, lavender) need darker borders for visibility.
    """
    color = get_model_color(model_name)

    # Pale colors need border/edge enhancement
    if model_name in ['Integrated_XGBoost', 'Integrated_RF', 'Ratio_XGBoost', 'Ratio_RF', 'ABIS_LR']:
        return {
            'color': color,
            'linewidth': linewidth + 0.5,  # Thicker line for pale colors
            'edgecolor': BIOMED_PALETTE['dark_gray'],
            'edgewidth': 0.8,
        }
    else:
        return {
            'color': color,
            'linewidth': linewidth,
        }


def save_figure(fig, output_path, dpi=DPI):
    """
    Save figure in publication-quality PNG format.

    Args:
        fig: matplotlib Figure object
        output_path: Path or str for output file
        dpi: Resolution (default 300)
    """
    output_path = Path(output_path)
    fig.savefig(
        output_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close(fig)


def create_figure(figsize=(3.5, 2.5)):
    """
    Create a publication-quality figure with standard settings.

    Args:
        figsize: tuple (width, height) in inches
                 Default 3.5" is single column for most journals

    Returns:
        fig, ax: matplotlib Figure and Axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def remove_top_right_spines(ax):
    """
    Remove top and right spines from axes for cleaner appearance.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_light_grid(ax):
    """
    Add light gray dashed grid to axes.
    """
    ax.grid(True, alpha=GRID_ALPHA, linestyle="--", linewidth=0.5, color="gray")


def style_axis(ax, xlabel=None, ylabel=None, title=None):
    """
    Apply standard styling to axis labels and title.

    Args:
        ax: matplotlib Axes object
        xlabel: str, x-axis label (with units)
        ylabel: str, y-axis label (with units)
        title: str, figure title
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE, fontfamily=FONT_FAMILY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE, fontfamily=FONT_FAMILY)
    if title:
        ax.set_title(
            title, fontsize=TITLE_SIZE, fontweight="bold", fontfamily=FONT_FAMILY
        )

    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)


def add_legend(ax, loc="best", frameon=False):
    """
    Add legend with standard formatting.

    Args:
        ax: matplotlib Axes object
        loc: legend location
        frameon: whether to show legend frame
    """
    legend = ax.legend(loc=loc, fontsize=LEGEND_SIZE, frameon=frameon)
    return legend


# =============================================================================
# ROC Curve Utilities
# =============================================================================
def plot_roc_curve(ax, fpr, tpr, auc, label, color=None, linewidth=LINE_WIDTH):
    """
    Plot a single ROC curve with AUC annotation.

    Args:
        ax: matplotlib Axes object
        fpr: false positive rate array
        tpr: true positive rate array
        auc: AUC value
        label: model name for legend
        color: line color (default from palette)
        linewidth: line width
    """
    if color is None:
        color = get_model_color(label)

    ax.plot(
        fpr, tpr, color=color, linewidth=linewidth, label=f"{label} (AUC={auc:.3f})"
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label="Random (AUC=0.500)",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("True Positive Rate", fontsize=AXIS_LABEL_SIZE)


# =============================================================================
# PR Curve Utilities
# =============================================================================
def plot_pr_curve(
    ax, recall, precision, auc_pr, label, color=None, linewidth=LINE_WIDTH
):
    """
    Plot a single Precision-Recall curve with AUC annotation.

    Args:
        ax: matplotlib Axes object
        recall: recall array
        precision: precision array
        auc_pr: PR-AUC value
        label: model name for legend
        color: line color (default from palette)
        linewidth: line width
    """
    if color is None:
        color = get_model_color(label)

    ax.plot(
        recall,
        precision,
        color=color,
        linewidth=linewidth,
        label=f"{label} (PR-AUC={auc_pr:.3f})",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Precision", fontsize=AXIS_LABEL_SIZE)


# =============================================================================
# Bar Plot Utilities
# =============================================================================
def plot_bar_horizontal(
    ax, values, labels, title=None, xlabel=None, color=DEFAULT_COLOR, show_values=True
):
    """
    Create a horizontal bar plot with publication styling.

    Args:
        ax: matplotlib Axes object
        values: array of bar heights
        labels: array of bar labels
        title: plot title
        xlabel: x-axis label
        color: bar color
        show_values: whether to show value labels on bars
    """
    # Sort by value (ascending for horizontal bars, top will be highest)
    sorted_idx = np.argsort(values)
    values_sorted = values[sorted_idx]
    labels_sorted = labels[sorted_idx]

    y_pos = range(len(values_sorted))
    ax.barh(y_pos, values_sorted, color=color, edgecolor="black", linewidth=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_sorted, fontsize=TICK_LABEL_SIZE)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE)
    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")

    remove_top_right_spines(ax)

    if show_values:
        for i, v in enumerate(values_sorted):
            ax.text(
                v + 0.01 * max(values_sorted),
                i,
                f"{v:.3f}",
                va="center",
                fontsize=TICK_LABEL_SIZE - 1,
            )


def plot_bar_with_error(
    ax, values, errors, labels, title=None, xlabel=None, color=DEFAULT_COLOR
):
    """
    Create a horizontal bar plot with error bars (95% CI).

    Args:
        ax: matplotlib Axes object
        values: array of bar heights
        errors: array of error bar heights
        labels: array of bar labels
        title: plot title
        xlabel: x-axis label
        color: bar color
    """
    sorted_idx = np.argsort(values)
    values_sorted = values[sorted_idx]
    errors_sorted = errors[sorted_idx]
    labels_sorted = labels[sorted_idx]

    y_pos = range(len(values_sorted))
    ax.barh(
        y_pos,
        values_sorted,
        xerr=errors_sorted,
        color=color,
        edgecolor="black",
        linewidth=0.3,
        error_kw={"lw": 0.8, "capsize": 2},
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_sorted, fontsize=TICK_LABEL_SIZE)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE)
    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")

    remove_top_right_spines(ax)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")


# =============================================================================
# Heatmap Utilities
# =============================================================================
def plot_heatmap(
    ax,
    data,
    xticklabels,
    yticklabels,
    title=None,
    cmap="viridis",
    cbar_label=None,
    annot=False,
):
    """
    Create a heatmap with publication styling.

    Args:
        ax: matplotlib Axes object
        data: 2D array for heatmap
        xticklabels: labels for x-axis
        yticklabels: labels for y-axis
        title: plot title
        cmap: colormap (viridis, Blues, RdBu_r, etc.)
        cbar_label: label for colorbar
        annot: whether to annotate cells with values
    """

    im = ax.imshow(data, cmap=cmap, aspect="auto")

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    if cbar_label:
        cbar.ax.set_ylabel(cbar_label, fontsize=AXIS_LABEL_SIZE)

    # Tick labels
    ax.set_xticks(range(len(xticklabels)))
    ax.set_yticks(range(len(yticklabels)))
    ax.set_xticklabels(xticklabels, fontsize=TICK_LABEL_SIZE, rotation=45, ha="right")
    ax.set_yticklabels(yticklabels, fontsize=TICK_LABEL_SIZE)

    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")

    if annot:
        for i in range(len(yticklabels)):
            for j in range(len(xticklabels)):
                val = data[i, j]
                text_color = "white" if val > np.median(data) else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=TICK_LABEL_SIZE - 1,
                    color=text_color,
                )


# =============================================================================
# Box Plot Utilities
# =============================================================================
def plot_boxplot(
    ax, data, labels, title=None, ylabel=None, colors=None, show_points=True
):
    """
    Create a box plot with optional individual points.

    Args:
        ax: matplotlib Axes object
        data: list of arrays for each box
        labels: labels for each box
        title: plot title
        ylabel: y-axis label
        colors: list of colors for boxes
        show_points: whether to show individual data points
    """
    if colors is None:
        colors = OKABE_ITO[: len(data)]

    bp = ax.boxplot(
        data, tick_labels=labels, patch_artist=True, widths=0.5, showfliers=True
    )

    # Set box colors
    for i, (box, color) in enumerate(zip(bp["boxes"], colors)):
        box.set_facecolor(color)
        box.set_alpha(0.7)

    # Median line styling
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    if show_points:
        for i, d in enumerate(data):
            jitter = np.random.normal(i + 1, 0.04, size=len(d))
            ax.scatter(jitter, d, alpha=0.3, s=10, color="black", zorder=3)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")

    remove_top_right_spines(ax)


# =============================================================================
# Calibration Curve Utilities
# =============================================================================
def plot_calibration_curve(
    ax, mean_predicted, fraction_of_positives, label=None, color=None, n_bins=None
):
    """
    Plot calibration curve with perfect calibration reference line.

    Args:
        ax: matplotlib Axes object
        mean_predicted: mean predicted probability per bin
        fraction_of_positives: fraction of positives per bin
        label: model name for legend
        color: line color
        n_bins: number of bins (for annotation)
    """
    if color is None:
        color = DEFAULT_COLOR

    # Perfect calibration line
    ax.plot(
        [0, 1],
        [0, 1],
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label="Perfect Calibration",
    )

    # Model calibration curve
    ax.plot(
        mean_predicted,
        fraction_of_positives,
        marker="o",
        linewidth=LINE_WIDTH,
        color=color,
        label=label if label else "Model",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Mean Predicted Probability", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Fraction of Positives", fontsize=AXIS_LABEL_SIZE)

    add_light_grid(ax)
    remove_top_right_spines(ax)


# =============================================================================
# DCA Curve Utilities
# =============================================================================
def plot_dca_curve(
    ax,
    thresholds,
    net_benefits,
    model_names,
    colors=None,
    treat_all=True,
    treat_none=True,
):
    """
    Plot Decision Curve Analysis curves.

    Args:
        ax: matplotlib Axes object
        thresholds: threshold probabilities array
        net_benefits: dict of {model_name: net_benefit_array}
        model_names: list of model names to plot
        colors: dict of {model_name: color}
        treat_all: whether to show "Treat All" line
        treat_none: whether to show "Treat None" line
    """
    if colors is None:
        colors = {name: get_model_color(name) for name in model_names}

    # Treat All line
    if treat_all:
        prevalence = net_benefits.get("prevalence", 0.5)
        treat_all_nb = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
        ax.plot(
            thresholds,
            treat_all_nb,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            label="Treat All",
        )

    # Treat None line (always 0)
    if treat_none:
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, label="Treat None")

    # Model curves
    for name in model_names:
        if name in net_benefits:
            ax.plot(
                thresholds,
                net_benefits[name],
                color=colors.get(name, DEFAULT_COLOR),
                linewidth=LINE_WIDTH,
                label=name,
            )

    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel("Threshold Probability", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Net Benefit", fontsize=AXIS_LABEL_SIZE)

    add_light_grid(ax)
    remove_top_right_spines(ax)


# =============================================================================
# Multi-panel Figure Utilities
# =============================================================================
def create_multi_panel_figure(
    n_rows, n_cols, figsize_per_panel=(3.5, 2.5), hspace=0.3, wspace=0.3
):
    """
    Create a multi-panel figure with consistent styling.

    Args:
        n_rows: number of rows
        n_cols: number of columns
        figsize_per_panel: size per panel
        hspace: height space between panels
        wspace: width space between panels

    Returns:
        fig: matplotlib Figure
        axes: array of Axes objects
    """
    figsize = (figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    plt.subplots_adjust(hspace=hspace, wspace=wspace)

    return fig, axes


def add_panel_label(ax, label, loc=(-0.15, 1.05)):
    """
    Add bold panel label (A, B, C, etc.) to subplot.

    Args:
        ax: matplotlib Axes object
        label: str, panel label (e.g., "A")
        loc: tuple, location relative to axes
    """
    ax.text(
        loc[0],
        loc[1],
        label,
        transform=ax.transAxes,
        fontsize=TITLE_SIZE,
        fontweight="bold",
        va="top",
    )


# =============================================================================
# Initialize style on import
# =============================================================================
# Apply style when module is imported
apply_publication_style()
