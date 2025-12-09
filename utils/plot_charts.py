import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt

THRESHOLD = 0.8

TASK = "MNIST"  # possible values: "MNIST", "TRAJECTORY"
OUTPUT_FEATURES = "trace"  # possible values: "statistics", "trace"
PARAM_NAME = "beta"  # possible value: "beta", "membrane_threshold", "current_amplitude"
NUM_WEIGHT_STEPS = 101
DATE = "2025_11_25"

RESULTS_DIR = f"results/results_{TASK}_{OUTPUT_FEATURES}_{PARAM_NAME}_{DATE}"
CSV_NAME = os.path.join(RESULTS_DIR, f"experiment_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.csv")
YAML_NAME = os.path.join(RESULTS_DIR, "experiment_metadata.yaml")


def load_metadata(yaml_path: str):
    with open(yaml_path, "r") as file:
        metadata = yaml.safe_load(file)
    return metadata


def plot_metric_model(
    results_df: pd.DataFrame,
    metadata: dict,
    metric_col: str,
    std_col: str | None,
    model_name: str,
    ylabel: str,
    filename: str,
):
    """
    Disegna METRICA (es. accuracy/F1) vs peso con area (std) per tutti i valori del parametro testato.
    metric_col/std_col sono i nomi delle colonne del CSV, es. 'accuracy_rf', 'std_accuracy_rf', 'f1_slp', 'std_f1_slp'.
    Se std_col non è presente nel dataframe o è None, l'area non viene disegnata.
    """
    param_values = metadata["tested_parameter"]["values"]
    accuracy_threshold = metadata["global_parameters"]["accuracy_threshold"]
    I = metadata["experiment"]["mean_I"]
    membrane_threshold = metadata["global_parameters"]["membrane_threshold"]
    refractory_period = metadata["global_parameters"]["refractory_period"]
    small_world_graph_k = metadata["global_parameters"]["small_world_graph_k"]
    num_neurons = metadata["global_parameters"]["num_neurons"]
    accuracy_threshold = THRESHOLD
    # stessa formula che avevi tu
    w_critical = (membrane_threshold - 2 * (I / num_neurons) * refractory_period) / (small_world_graph_k / 2)

    plt.figure()

    for value in param_values:
        parameter_df = results_df[results_df["param_value"] == float(value)].copy()
        parameter_df = parameter_df.sort_values(by="weight")

        line, = plt.plot(
            parameter_df["weight"],
            parameter_df[metric_col],
            marker="o",
            label=f"{PARAM_NAME}={value}",
        )

        # shaded area con la std (se presente)
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

        # segmento sopra la soglia relativa (rispetto al max di questo parametro)
        max_metric = parameter_df[metric_col].max()
        threshold = accuracy_threshold * max_metric
        eligible = parameter_df[parameter_df[metric_col] >= threshold]
        if not eligible.empty:
            w1 = eligible["weight"].min()
            w2 = eligible["weight"].max()
            plt.hlines(
                y=threshold,
                xmin=w1,
                xmax=w2,
                colors="black",
                linestyles="dashed",
            )

    # linea del peso critico
    plt.axvline(
        x=w_critical,
        color="red",
        linestyle="--",
        label="critical weight",
    )

    plt.xlabel("Mean synaptic weight")
    plt.ylabel(ylabel)
    plt.title(f"{model_name}: {ylabel} vs weight for different {PARAM_NAME} values")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(plot_path, dpi=150)
    print(f"saved {plot_path}")


def plot_spike_count(results_df: pd.DataFrame, metadata: dict):
    param_values = metadata["tested_parameter"]["values"]

    plt.figure()

    for value in param_values:
        parameter_df = results_df[results_df["param_value"] == float(value)].copy()
        parameter_df = parameter_df.sort_values(by="weight")

        plt.plot(
            parameter_df["weight"],
            parameter_df["spike_count"],
            marker="o",
            label=f"{PARAM_NAME}={value}",
        )

    plt.xlabel("Mean synaptic weight")
    plt.ylabel("Mean spike count")
    plt.title(f"Spike count vs weight for different {PARAM_NAME} values")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "plot_spike_count.png")
    plt.savefig(plot_path, dpi=150)
    print(f"saved {plot_path}")


def main():
    # Il CSV ora include: param_value, weight, accuracy_*, f1_*, mcc_*, spike_count (+ std_*)
    results_df = pd.read_csv(CSV_NAME)
    metadata = load_metadata(YAML_NAME)

    # elenco (metrica_col, std_col, modello, ylabel, filename)
    plots = [
        ("accuracy_rf", "std_accuracy_rf", "Random Forest", "Mean CV accuracy", "plot_accuracy_rf.png"),
        ("accuracy_slp", "std_accuracy_slp", "Single-layer perceptron", "Mean CV accuracy", "plot_accuracy_slp.png"),
        ("f1_rf", "std_f1_rf", "Random Forest", "Mean CV F1", "plot_f1_rf.png"),
        ("f1_slp", "std_f1_slp", "Single-layer perceptron", "Mean CV F1", "plot_f1_slp.png"),
        # ✅ nuovi grafici MCC
        ("mcc_rf", "std_mcc_rf", "Random Forest", "Mean CV MCC", "plot_mcc_rf.png"),
        ("mcc_slp", "std_mcc_slp", "Single-layer perceptron", "Mean CV MCC", "plot_mcc_slp.png"),
    ]

    for metric_col, std_col, model_name, ylabel, filename in plots:
        # se la colonna std non esiste nel CSV, passiamola comunque:
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

    # opzionale: spike count
    plot_spike_count(results_df, metadata)

    plt.show()



if __name__ == "__main__":
    main()
