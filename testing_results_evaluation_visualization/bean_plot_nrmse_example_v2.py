
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ==============================
# Global style settings
# ==============================
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

# ==============================
# User settings
# ==============================
BEFORE_FILE = r"/mnt/data/0bd579d2-5f09-49b0-aea1-5b821cc354eb.xlsx"
AFTER_FILE = r"/mnt/data/0bd579d2-5f09-49b0-aea1-5b821cc354eb.xlsx"   # Replace with your "after feature enhancement" file
METRIC_NAME = "NRMSE"   # Change to "CWC" or "IS" when needed
OUTPUT_DIR = r"/mnt/data/bean_plot_output"
OUTPUT_NAME = f"bean_chart_{METRIC_NAME.lower()}"

PALETTE_VIOLIN = {
    "Before": "#F9C4C4",
    "After": "#BCE0F4",
}
PALETTE_SCATTER = {
    "Before": "#E87C7C",
    "After": "#6AB5DE",
}

# ==============================
# Helper functions
# ==============================
def get_metric_column_range(raw_df: pd.DataFrame, metric_name: str):
    """
    Find the start and end columns for a target metric block.
    The first row stores metric names, and the second row stores algorithm names.
    """
    metric_row = raw_df.iloc[0, :]
    metric_positions = [i for i, value in enumerate(metric_row) if str(value).strip() == metric_name]
    if not metric_positions:
        raise ValueError(f"Metric '{metric_name}' was not found in the first header row.")

    start_col = metric_positions[0]
    end_col = raw_df.shape[1] - 1

    for col in range(start_col + 1, raw_df.shape[1]):
        cell_value = raw_df.iat[0, col]
        if pd.notna(cell_value) and str(cell_value).strip() != "":
            end_col = col - 1
            break

    return start_col, end_col


def extract_metric_long_table(file_path: str, metric_name: str, condition_name: str):
    """
    Read one Excel file and convert the selected metric block to a long-format table.
    Expected file format:
    - Row 0: metric names
    - Row 1: algorithm names
    - Row 2: blank row
    - Remaining rows: station-level records
    """
    raw_df = pd.read_excel(file_path, header=None)

    start_col, end_col = get_metric_column_range(raw_df, metric_name)
    algorithm_names = raw_df.iloc[1, start_col:end_col + 1].tolist()

    station_mask = raw_df.iloc[:, 0].astype(str).str.startswith("PV_")
    data_df = raw_df.loc[station_mask, [0] + list(range(start_col, end_col + 1))].copy()
    data_df.columns = ["Station"] + algorithm_names

    long_df = data_df.melt(
        id_vars="Station",
        var_name="Algorithm",
        value_name="Value"
    )
    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
    long_df["Condition"] = condition_name
    long_df = long_df.dropna(subset=["Value"]).reset_index(drop=True)

    return long_df, algorithm_names


def get_significance_label(p_value: float):
    """Convert p-value to significance symbols."""
    if pd.isna(p_value):
        return "ns"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def add_summary_lines_and_text(ax, plot_df, algorithm_order, condition_order):
    """
    Draw the white IQR lines, mean points, and mean value labels on each side.
    """
    stats_df = (
        plot_df.groupby(["Algorithm", "Condition"])["Value"]
        .agg(
            mean="mean",
            q1=lambda x: np.percentile(x, 25),
            q3=lambda x: np.percentile(x, 75),
        )
        .reset_index()
    )

    offset_map = {"Before": -0.12, "After": 0.12}
    text_offset_map = {"Before": -0.05, "After": 0.05}
    cap_width = 0.03

    for i, algorithm in enumerate(algorithm_order):
        for condition in condition_order:
            row = stats_df[
                (stats_df["Algorithm"] == algorithm) &
                (stats_df["Condition"] == condition)
            ].iloc[0]

            mean_val = row["mean"]
            q1_val = row["q1"]
            q3_val = row["q3"]

            x_pos = i + offset_map[condition]

            ax.plot([x_pos, x_pos], [q1_val, q3_val], color="white", lw=2.4, zorder=3)
            ax.plot([x_pos - cap_width, x_pos + cap_width], [q1_val, q1_val], color="white", lw=2.4, zorder=3)
            ax.plot([x_pos - cap_width, x_pos + cap_width], [q3_val, q3_val], color="white", lw=2.4, zorder=3)
            ax.plot(x_pos, mean_val, marker="o", color="white", markersize=5.5, zorder=4)

            ha = "right" if condition == "Before" else "left"
            ax.text(
                x_pos + text_offset_map[condition],
                mean_val,
                f"{mean_val:.3f}",
                ha=ha,
                va="center",
                fontsize=10,
                fontweight="bold",
                color="black",
                zorder=5
            )


def add_significance_labels(ax, plot_df, algorithm_order):
    """
    Perform paired t-tests between before/after values for the same stations
    and place significance labels above each algorithm.
    """
    y_range = plot_df["Value"].max() - plot_df["Value"].min()
    line_height = 0.02 * y_range
    text_gap = 0.015 * y_range

    for i, algorithm in enumerate(algorithm_order):
        before_df = plot_df[
            (plot_df["Algorithm"] == algorithm) &
            (plot_df["Condition"] == "Before")
        ][["Station", "Value"]].rename(columns={"Value": "Before"})

        after_df = plot_df[
            (plot_df["Algorithm"] == algorithm) &
            (plot_df["Condition"] == "After")
        ][["Station", "Value"]].rename(columns={"Value": "After"})

        paired_df = pd.merge(before_df, after_df, on="Station", how="inner")

        if len(paired_df) >= 2:
            _, p_value = stats.ttest_rel(paired_df["Before"], paired_df["After"], nan_policy="omit")
        else:
            p_value = np.nan

        sig_label = get_significance_label(p_value)
        group_max = max(before_df["Before"].max(), after_df["After"].max())
        y_pos = group_max + 0.05 * y_range

        x_left = i - 0.12
        x_right = i + 0.12

        ax.plot(
            [x_left, x_left, x_right, x_right],
            [y_pos - line_height, y_pos, y_pos, y_pos - line_height],
            color="black",
            lw=1.4,
            zorder=6
        )
        ax.text(
            i,
            y_pos + text_gap,
            sig_label,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="black",
            zorder=7
        )


def plot_bean_chart(plot_df: pd.DataFrame, metric_name: str, output_dir: str, output_name: str):
    """
    Plot the split bean chart for one metric.
    """
    os.makedirs(output_dir, exist_ok=True)

    algorithm_order = plot_df["Algorithm"].drop_duplicates().tolist()
    condition_order = ["Before", "After"]

    fig, ax = plt.subplots(figsize=(18, 7), dpi=300)

    sns.violinplot(
        data=plot_df,
        x="Algorithm",
        y="Value",
        hue="Condition",
        hue_order=condition_order,
        split=True,
        inner=None,
        palette=PALETTE_VIOLIN,
        linewidth=0,
        cut=0,
        saturation=1,
        ax=ax
    )

    sns.stripplot(
        data=plot_df,
        x="Algorithm",
        y="Value",
        hue="Condition",
        dodge=True,
        palette=PALETTE_SCATTER,
        alpha=1.0,
        size=5.5,
        jitter=0.16,
        zorder=2,
        order=algorithm_order,
        hue_order=condition_order,
        edgecolor="white",
        linewidth=1.0,
        ax=ax
    )

    add_summary_lines_and_text(ax, plot_df, algorithm_order, condition_order)
    add_significance_labels(ax, plot_df, algorithm_order)

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles[:2],
        labels[:2],
        title="",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False
    )
    plt.setp(legend.get_texts(), fontweight="bold", fontsize=11)

    y_min = plot_df["Value"].min()
    y_max = plot_df["Value"].max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.08 * y_range, y_max + 0.20 * y_range)

    ax.set_title(f"{metric_name} Comparison Before and After Feature Enhancement", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("")
    ax.set_ylabel(metric_name, fontsize=13, fontweight="bold")

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=4,
        width=1.8,
        labelsize=11
    )

    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha("right")
        tick.set_fontweight("bold")

    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color("black")

    ax.grid(False)
    plt.tight_layout()

    png_path = os.path.join(output_dir, f"{output_name}.png")
    pdf_path = os.path.join(output_dir, f"{output_name}.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()


# ==============================
# Main execution
# ==============================
if __name__ == "__main__":
    before_long, algorithm_order_before = extract_metric_long_table(
        file_path=BEFORE_FILE,
        metric_name=METRIC_NAME,
        condition_name="Before"
    )

    after_long, algorithm_order_after = extract_metric_long_table(
        file_path=AFTER_FILE,
        metric_name=METRIC_NAME,
        condition_name="After"
    )

    if algorithm_order_before != algorithm_order_after:
        raise ValueError("The algorithm order in the two files is inconsistent.")

    plot_df = pd.concat([before_long, after_long], ignore_index=True)

    plot_df["Algorithm"] = pd.Categorical(
        plot_df["Algorithm"],
        categories=algorithm_order_before,
        ordered=True
    )

    plot_df["Condition"] = pd.Categorical(
        plot_df["Condition"],
        categories=["Before", "After"],
        ordered=True
    )

    plot_df = plot_df.sort_values(["Algorithm", "Condition", "Station"]).reset_index(drop=True)

    plot_bean_chart(
        plot_df=plot_df,
        metric_name=METRIC_NAME,
        output_dir=OUTPUT_DIR,
        output_name=OUTPUT_NAME
    )
