# Plot, inside the results folders, how accuracy, f1, and mcc
# intervals above a threshold grow or shrink as parameters change.

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THRESHOLD = 0.85

# -------------------------
# Experiment parameters
# -------------------------
TASK = "TRAJECTORY"  # "MNIST", "TRAJECTORY"
OUTPUT_FEATURES = "trace"  # "statistics", "trace"
PARAM_NAME = "current_amplitude"  # "beta", "membrane_threshold", "current_amplitude"
NUM_WEIGHT_STEPS = 71  # e.g. 71, 101
DATE = "2025_11_27"

RESULTS_DIR = f"results/results_{TASK}_{OUTPUT_FEATURES}_{PARAM_NAME}_{DATE}"
CSV_NAME = os.path.join(
    RESULTS_DIR,
    f"experiment_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.csv",
)

# Horizontal offset to separate the three metric types on the x-axis
METRIC_X_OFFSET = 0.01


def compute_intervals(
    input_df: pd.DataFrame,
    metric_columns: Dict[str, str],
) -> pd.DataFrame:
    """Compute weight intervals where each metric stays above THRESHOLD * max."""
    rows = []

    # Ensure numeric types
    df = input_df.copy()
    df["param_value"] = df["param_value"].astype(float)
    df["weight"] = df["weight"].astype(float)

    for param_value, group_df in df.groupby("param_value"):
        sorted_group = group_df.sort_values("weight")

        weights = sorted_group["weight"].values

        for metric_name, column_name in metric_columns.items():
            metric_values = sorted_group[column_name].values.astype(float)

            if len(metric_values) == 0:
                continue

            max_value = np.nanmax(metric_values)
            if np.isnan(max_value):
                continue

            threshold_value = THRESHOLD * max_value
            above_threshold_mask = metric_values >= threshold_value

            # Pathological case: no value reaches the threshold
            if not np.any(above_threshold_mask):
                continue

            start_weight = weights[above_threshold_mask][0]
            end_weight = weights[above_threshold_mask][-1]
            width = end_weight - start_weight

            rows.append(
                {
                    "param_value": param_value,
                    "metric": metric_name,
                    "start_weight": start_weight,
                    "end_weight": end_weight,
                    "width": width,
                }
            )

    return pd.DataFrame(rows)


def plot_intervals(
    intervals_df: pd.DataFrame,
    model_label: str,
    output_png_path: str,
) -> None:
    """Plot interval widths vs param value for each metric."""
    if intervals_df.empty:
        print(f"No data for model {model_label}, plot not created.")
        return

    plt.figure(figsize=(8, 5))

    metrics_order = ["accuracy", "f1", "mcc"]
    markers = {
        "accuracy": "o",
        "f1": "s",
        "mcc": "^",
    }
    colors = {
        "accuracy": "tab:blue",
        "f1": "tab:orange",
        "mcc": "tab:green",
    }

    # Horizontal offset for the three metrics: -offset, 0, +offset
    offsets = {
        "accuracy": -METRIC_X_OFFSET,
        "f1": 0.0,
        "mcc": METRIC_X_OFFSET,
    }

    for metric in metrics_order:
        metric_df = intervals_df[intervals_df["metric"] == metric]
        if metric_df.empty:
            continue

        param_values_with_offset = metric_df["param_value"].values + offsets[metric]
        interval_widths = metric_df["width"].values

        plt.scatter(
            param_values_with_offset,
            interval_widths,
            label=metric,
            marker=markers[metric],
            color=colors[metric],
        )

    plt.xlabel(PARAM_NAME)
    plt.ylabel("interval width (weight)")
    plt.title(
        f"Width of intervals above threshold ({THRESHOLD} * max) - {model_label}"
    )
    plt.legend(title="Metric")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    plt.close()


def main() -> None:
    """Run interval computation and plotting pipeline."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Read input CSV
    input_df = pd.read_csv(CSV_NAME)

    # Metric columns for RF and SLP
    rf_metrics = {
        "accuracy": "accuracy_rf",
        "f1": "f1_rf",
        "mcc": "mcc_rf",
    }
    slp_metrics = {
        "accuracy": "accuracy_slp",
        "f1": "f1_slp",
        "mcc": "mcc_slp",
    }

    # Intervals for RF
    intervals_rf = compute_intervals(input_df, rf_metrics)
    rf_csv_output = os.path.join(
        RESULTS_DIR,
        f"intervals_rf_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.csv",
    )
    intervals_rf.to_csv(rf_csv_output, index=False)

    # Intervals for SLP
    intervals_slp = compute_intervals(input_df, slp_metrics)
    slp_csv_output = os.path.join(
        RESULTS_DIR,
        f"intervals_slp_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.csv",
    )
    intervals_slp.to_csv(slp_csv_output, index=False)

    # Plots
    rf_png_output = os.path.join(
        RESULTS_DIR,
        f"intervals_rf_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.png",
    )
    slp_png_output = os.path.join(
        RESULTS_DIR,
        f"intervals_slp_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.png",
    )

    plot_intervals(intervals_rf, "RF", rf_png_output)
    plot_intervals(intervals_slp, "SLP", slp_png_output)

    print("Done.")
    print(f"CSV RF:  {rf_csv_output}")
    print(f"CSV SLP: {slp_csv_output}")
    print(f"PNG RF:  {rf_png_output}")
    print(f"PNG SLP: {slp_png_output}")


if __name__ == "__main__":
    main()
