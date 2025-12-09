import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import yaml

THRESHOLD = 0.8

TASK = "MNIST"  # possible values: "MNIST", "TRAJECTORY"
OUTPUT_FEATURES = "trace"  # possible values: "statistics", "trace"
PARAM_NAME = "beta"  # possible values: "beta", "membrane_threshold", "current_amplitude"
NUM_WEIGHT_STEPS = 101
DATE = "2025_11_25"

RESULTS_DIR = f"results/results_{TASK}_{OUTPUT_FEATURES}_{PARAM_NAME}_{DATE}"
CSV_NAME = os.path.join(RESULTS_DIR, f"experiment_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.csv")
YAML_NAME = os.path.join(RESULTS_DIR, "experiment_metadata.yaml")


def load_metadata(yaml_path: str) -> Dict:
    """Load experiment metadata from a YAML file."""
    with open(yaml_path, "r") as file:
        metadata = yaml.safe_load(file)
    return metadata


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

    mean_current = metadata["experiment"]["mean_I"]
    membrane_threshold = metadata["global_parameters"]["membrane_threshold"]
    refractory_period = metadata["global_parameters"]["refractory_period"]
    small_world_graph_k = metadata["global_parameters"]["small_world_graph_k"]
    num_neurons = metadata["global_parameters"]["num_neurons"]

    critical_weight = (
        membrane_threshold
        - 2 * (mean_current / num_neurons) * refractory_period
    ) / (small_world_graph_k / 2)

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

        # Shaded area with std (if available)
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

        # Segment above relative threshold (THRESHOLD * max for this param)
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

    # Critical weight line
    plt.axvline(
        x=critical_weight,
        color="red",
        linestyle="--",
        label="critical weight",
    )

    plt.xlabel("Mean synaptic weight")
    plt.ylabel(ylabel)
    plt.title(
        f"{model_name}: {ylabel} vs weight for different {PARAM_NAME} values"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(plot_path, dpi=150)
    print(f"Saved {plot_path}")


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
    plt.title(f"Spike count vs weight for different {PARAM_NAME} values")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "plot_spike_count.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved {plot_path}")


def main() -> None:
    """Run metric and spike-count plotting from CSV and metadata."""
    results_df = pd.read_csv(CSV_NAME)
    metadata = load_metadata(YAML_NAME)

    # (metric_col, std_col, model_name, ylabel, filename)
    plots = [
        (
            "accuracy_rf",
            "std_accuracy_rf",
            "Random Forest",
            "Mean CV accuracy",
            "plot_accuracy_rf.png",
        ),
        (
            "accuracy_slp",
            "std_accuracy_slp",
            "Single-layer perceptron",
            "Mean CV accuracy",
            "plot_accuracy_slp.png",
        ),
        (
            "f1_rf",
            "std_f1_rf",
            "Random Forest",
            "Mean CV F1",
            "plot_f1_rf.png",
        ),
        (
            "f1_slp",
            "std_f1_slp",
            "Single-layer perceptron",
            "Mean CV F1",
            "plot_f1_slp.png",
        ),
        (
            "mcc_rf",
            "std_mcc_rf",
            "Random Forest",
            "Mean CV MCC",
            "plot_mcc_rf.png",
        ),
        (
            "mcc_slp",
            "std_mcc_slp",
            "Single-layer perceptron",
            "Mean CV MCC",
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

    plt.show()


if __name__ == "__main__":
    main()
