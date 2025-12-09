"""
Post-processing for SNN experiments.

1) Computes, saves, and plots interval widths (in weight space) where
   accuracy/F1/MCC stay above THRESHOLD * max for each param value (RF + SLP).

2) Plots metric vs weight (with std band) and spike count vs weight.

All plots are saved to files; nothing is shown on screen.

Global configuration is loaded from settings/config_plot.yaml.
"""

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# -------------------------
# Globals filled from YAML
# -------------------------

THRESHOLD: float
TASK: str
OUTPUT_FEATURES: str
PARAM_NAME: str
NUM_WEIGHT_STEPS: int
DATE: str

RESULTS_DIR: str
CSV_NAME: str
METADATA_YAML_NAME: str  # experiment_metadata.yaml path


# Column names for RF and SLP metrics
RF_METRIC_COLUMNS: Dict[str, str] = {
    "accuracy": "accuracy_rf",
    "f1": "f1_rf",
    "mcc": "mcc_rf",
}
SLP_METRIC_COLUMNS: Dict[str, str] = {
    "accuracy": "accuracy_slp",
    "f1": "f1_slp",
    "mcc": "mcc_slp",
}

# Horizontal offset to separate the three metric types on the x-axis
METRIC_X_OFFSET = 0.01

CONFIG_PLOT_PATH = "settings/config_plot.yaml"


# -------------------------
# Config loading
# -------------------------

def load_plot_config(path: str = CONFIG_PLOT_PATH) -> Dict:
    """Load plotting configuration from YAML and set globals."""
    global THRESHOLD, TASK, OUTPUT_FEATURES, PARAM_NAME
    global NUM_WEIGHT_STEPS, DATE
    global RESULTS_DIR, CSV_NAME, METADATA_YAML_NAME

    if not os.path.exists(path):
        raise FileNotFoundError(f"Plot config not found: {path}")

    with open(path, "r") as file:
        cfg = yaml.safe_load(file)

    THRESHOLD = float(cfg["THRESHOLD"])
    TASK = cfg["TASK"]
    OUTPUT_FEATURES = cfg["OUTPUT_FEATURES"]
    PARAM_NAME = cfg["PARAM_NAME"]
    NUM_WEIGHT_STEPS = int(cfg["NUM_WEIGHT_STEPS"])
    DATE = cfg["DATE"]

    RESULTS_DIR = f"results/results_{TASK}_{OUTPUT_FEATURES}_{PARAM_NAME}_{DATE}"
    CSV_NAME = os.path.join(
        RESULTS_DIR,
        f"experiment_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.csv",
    )
    METADATA_YAML_NAME = os.path.join(RESULTS_DIR, "experiment_metadata.yaml")

    return cfg


# -------------------------
# Utilities
# -------------------------

def load_metadata(yaml_path: str) -> Dict:
    """Load experiment metadata from a YAML file."""
    with open(yaml_path, "r") as file:
        metadata = yaml.safe_load(file)
    return metadata


# -------------------------
# Interval computation
# -------------------------

def compute_intervals(
    input_df: pd.DataFrame,
    metric_columns: Dict[str, str],
) -> pd.DataFrame:
    """Compute weight intervals where each metric stays above THRESHOLD * max."""
    rows = []

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
    print(f"Saved interval plot: {output_png_path}")


# -------------------------
# Metric / spike plots
# -------------------------

def plot_metric_model(
    results_df: pd.DataFrame,
    metadata: Dict,
    metric_col: str,
    std_col: Optional[str],
    model_name: str,
    ylabel: str,
    filename: str,
) -> None:
    """Plot metric vs weight with optional std band for each parameter value."""
    param_values = metadata["tested_parameter"]["values"]

    plt.figure()

    for param_value in param_values:
        parameter_df = results_df[
            results_df["param_value"] == float(param_value)
        ].copy()
        parameter_df = parameter_df.sort_values(by="weight")

        line, = plt.plot(
            parameter_df["weight"],
            parameter_df[metric_col],
            marker="o",
            label=f"{PARAM_NAME}={param_value}",
        )

        if std_col and std_col in parameter_df.columns:
            lower = parameter_df[metric_col] - parameter_df[std_col]
            upper = parameter_df[metric_col] + parameter_df[std_col]
            plt.fill_between(
                parameter_df["weight"],
                lower,
                upper,
                color=line.get_color(),
                alpha=0.2,
            )

        max_metric = parameter_df[metric_col].max()
        metric_threshold = THRESHOLD * max_metric
        eligible = parameter_df[parameter_df[metric_col] >= metric_threshold]
        if not eligible.empty:
            weight_min = eligible["weight"].min()
            weight_max = eligible["weight"].max()
            plt.hlines(
                y=metric_threshold,
                xmin=weight_min,
                xmax=weight_max,
                colors="black",
                linestyles="dashed",
            )

    plt.xlabel("Mean synaptic weight")
    plt.ylabel(ylabel)
    plt.title(
        f"{model_name} for {PARAM_NAME}:\n{ylabel} vs weight"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved metric plot: {plot_path}")


def plot_spike_count(results_df: pd.DataFrame, metadata: Dict) -> None:
    """Plot mean spike count vs weight for each parameter value."""
    param_values = metadata["tested_parameter"]["values"]

    plt.figure()

    for param_value in param_values:
        parameter_df = results_df[
            results_df["param_value"] == float(param_value)
        ].copy()
        parameter_df = parameter_df.sort_values(by="weight")

        plt.plot(
            parameter_df["weight"],
            parameter_df["spike_count"],
            marker="o",
            label=f"{PARAM_NAME}={param_value}",
        )

    plt.xlabel("Mean synaptic weight")
    plt.ylabel("Mean spike count")
    plt.title(f"Spike count vs weight")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "plot_spike_count.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved spike-count plot: {plot_path}")


# -------------------------
# Main
# -------------------------

def main() -> None:
    """Run all post-processing: intervals + metric and spike plots."""
    load_plot_config()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results_df = pd.read_csv(CSV_NAME)
    metadata = load_metadata(METADATA_YAML_NAME)

    # ---- Interval computation (RF + SLP) ----
    intervals_rf = compute_intervals(results_df, RF_METRIC_COLUMNS)
    rf_csv_output = os.path.join(
        RESULTS_DIR,
        f"intervals_rf_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.csv",
    )
    intervals_rf.to_csv(rf_csv_output, index=False)

    intervals_slp = compute_intervals(results_df, SLP_METRIC_COLUMNS)
    slp_csv_output = os.path.join(
        RESULTS_DIR,
        f"intervals_slp_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.csv",
    )
    intervals_slp.to_csv(slp_csv_output, index=False)

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

    print(f"Saved RF interval CSV:  {rf_csv_output}")
    print(f"Saved SLP interval CSV: {slp_csv_output}")

    # ---- Metric vs weight plots + spike count ----
    plots = [
        (
            RF_METRIC_COLUMNS["accuracy"],
            "std_accuracy_rf",
            "Random Forest",
            "Mean accuracy",
            "plot_accuracy_rf.png",
        ),
        (
            SLP_METRIC_COLUMNS["accuracy"],
            "std_accuracy_slp",
            "Single-layer perceptron",
            "Mean accuracy",
            "plot_accuracy_slp.png",
        ),
        (
            RF_METRIC_COLUMNS["f1"],
            "std_f1_rf",
            "Random Forest",
            "Mean F1",
            "plot_f1_rf.png",
        ),
        (
            SLP_METRIC_COLUMNS["f1"],
            "std_f1_slp",
            "Single-layer perceptron",
            "Mean F1",
            "plot_f1_slp.png",
        ),
        (
            RF_METRIC_COLUMNS["mcc"],
            "std_mcc_rf",
            "Random Forest",
            "Mean MCC",
            "plot_mcc_rf.png",
        ),
        (
            SLP_METRIC_COLUMNS["mcc"],
            "std_mcc_slp",
            "Single-layer perceptron",
            "Mean MCC",
            "plot_mcc_slp.png",
        ),
    ]

    for metric_col, std_col, model_name, ylabel, filename in plots:
        std_col_used = std_col if std_col in results_df.columns else None
        plot_metric_model(
            results_df=results_df,
            metadata=metadata,
            metric_col=metric_col,
            std_col=std_col_used,
            model_name=model_name,
            ylabel=ylabel,
            filename=filename,
        )

    plot_spike_count(results_df, metadata)

    print("All plots and interval files saved.")


if __name__ == "__main__":
    main()
