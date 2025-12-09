import os
from datetime import date

import numpy as np
import pandas as pd
import yaml

from LSM.model import SimulationParams
from functions.simulates import simulate
from functions.cross_validations import cross_validation_rf, cross_validation_slp

# the following global variables will be loaded using function load_config
MIN_INTERVAL = 1 / 100
MAX_INTERVAL = 2.2
WEIGHT_VARATION_COEFFICIENT = None
RELOAD = None
TASK = None
CV_NUM_SPLITS = None
NUM_NEURONS = None
MEMBRANE_THRESHOLD = None
REFRACTORY_PERIOD = None
NUM_OUTPUT_NEURONS = None
LEAK_COEFFICIENT = None
CURRENT_AMPLITUDE = None
PRESYNAPTIC_DEGREE = None
SMALL_WORLD_GRAPH_P = None
SMALL_WORLD_GRAPH_K = None
TRACE_TAU = None
NUM_WEIGHT_STEPS = None
PARAM_NAME = None
PARAMETER_VALUES = None
DATASET_PATH = None
RESULTS_DIR_STATISTICS = None
RESULTS_DIR_TRACE = None
CSV_NAME_STATISTICS = None
CSV_NAME_TRACE = None
STATISTIC_SET = None
MEMBRANE_RESET = None


def load_config(path: str = "settings/config.yaml"):
    global TASK, CV_NUM_SPLITS
    global NUM_NEURONS, MEMBRANE_THRESHOLD, REFRACTORY_PERIOD, NUM_OUTPUT_NEURONS
    global LEAK_COEFFICIENT, CURRENT_AMPLITUDE, PRESYNAPTIC_DEGREE
    global SMALL_WORLD_GRAPH_P, SMALL_WORLD_GRAPH_K, TRACE_TAU
    global NUM_WEIGHT_STEPS, PARAM_NAME, PARAMETER_VALUES
    global DATASET_PATH, CSV_NAME_STATISTICS, CSV_NAME_TRACE
    global RESULTS_DIR_STATISTICS, RESULTS_DIR_TRACE
    global STATISTIC_SET, MEMBRANE_RESET, WEIGHT_VARATION_COEFFICIENT, RELOAD

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # derived value
    presynaptic_degree = cfg["PRESYNAPTIC_DEGREE"]
    num_neurons = cfg["NUM_NEURONS"]
    cfg["SMALL_WORLD_GRAPH_K"] = int(presynaptic_degree * num_neurons * 2)

    # assign to globals
    TASK = cfg["TASK"]
    CV_NUM_SPLITS = cfg["CV_NUM_SPLITS"]
    NUM_NEURONS = cfg["NUM_NEURONS"]
    MEMBRANE_THRESHOLD = cfg["MEMBRANE_THRESHOLD"]
    REFRACTORY_PERIOD = cfg["REFRACTORY_PERIOD"]
    NUM_OUTPUT_NEURONS = cfg["NUM_OUTPUT_NEURONS"]
    LEAK_COEFFICIENT = cfg["LEAK_COEFFICIENT"]
    CURRENT_AMPLITUDE = cfg["CURRENT_AMPLITUDE"]
    PRESYNAPTIC_DEGREE = cfg["PRESYNAPTIC_DEGREE"]
    SMALL_WORLD_GRAPH_P = cfg["SMALL_WORLD_GRAPH_P"]
    SMALL_WORLD_GRAPH_K = cfg["SMALL_WORLD_GRAPH_K"]
    TRACE_TAU = cfg["TRACE_TAU"]
    NUM_WEIGHT_STEPS = cfg["NUM_WEIGHT_STEPS"]
    PARAM_NAME = cfg["PARAM_NAME"]
    PARAMETER_VALUES = cfg["PARAMETER_VALUES"]
    MEMBRANE_RESET = cfg["MEMBRANE_RESET"]
    WEIGHT_VARATION_COEFFICIENT = cfg["WEIGHT_VARATION_COEFFICIENT"]
    RELOAD = cfg["RELOAD"]

    if MEMBRANE_RESET:
        print("membrane reset true")
    else:
        print("membrane reset false")

    if TASK == "MNIST":
        DATASET_PATH = "dati/mnist_rate_encoded.npz"
        STATISTIC_SET = 1
    elif TASK == "TRAJECTORY":
        DATASET_PATH = "dati/trajectory_spike_encoded.npz"
        STATISTIC_SET = 2
    else:
        print("Selected unknown task.")
        exit()

    today_str = date.today().strftime("%Y_%m_%d")
    RESULTS_DIR_STATISTICS = f"results/results_{TASK}_statistics_{PARAM_NAME}_{today_str}"
    RESULTS_DIR_TRACE = f"results/results_{TASK}_trace_{PARAM_NAME}_{today_str}"
    CSV_NAME_STATISTICS = os.path.join(
        RESULTS_DIR_STATISTICS,
        f"experiment_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.csv",
    )
    CSV_NAME_TRACE = os.path.join(
        RESULTS_DIR_TRACE,
        f"experiment_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.csv",
    )

    return cfg


def load_dataset(filename: str):
    """Return (data, labels) from a .npz file."""
    npz_data = np.load(filename)
    return npz_data["data"], npz_data["labels"]


def _to_builtin(obj):
    """Recursively convert numpy/pandas objects to Python built-ins."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_to_builtin(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {_to_builtin(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_builtin(x) for x in obj]
    return obj  # strings, bool, None, etc.


def save_experiment_metadata(
    results_dir: str,
    parameter_name: str,
    parameter_values: list[float],
    output_features: str,
    mean_I,
) -> None:
    """Save experiment configuration and metadata as YAML."""
    metadata = {
        "experiment": {
            "task": TASK,
            "output_features": output_features,
            "mean_I": mean_I,
        },
        "tested_interval": {
            "min_interval": MIN_INTERVAL,
            "max_interval": MAX_INTERVAL,
            "num_weight_steps": NUM_WEIGHT_STEPS,
        },
        "global_parameters": {
            "num_neurons": NUM_NEURONS,
            "membrane_threshold": MEMBRANE_THRESHOLD,
            "refractory_period": REFRACTORY_PERIOD,
            "num_output_neurons": NUM_OUTPUT_NEURONS,
            "leak_coefficient": LEAK_COEFFICIENT,
            "current_amplitude": CURRENT_AMPLITUDE,
            "presynaptic_degree": PRESYNAPTIC_DEGREE,
            "small_world_graph_p": SMALL_WORLD_GRAPH_P,
            "small_world_graph_k": SMALL_WORLD_GRAPH_K,
            "trace_tau": TRACE_TAU,
            "weight_variation_coefficinet": WEIGHT_VARATION_COEFFICIENT,
            "cv_num_splits": CV_NUM_SPLITS,
            "membrane_reset": MEMBRANE_RESET,
            "reload": RELOAD,
        },
        "tested_parameter": {
            "name": parameter_name,
            "values": parameter_values,
        },
    }

    metadata = _to_builtin(metadata)

    yaml_path = os.path.join(results_dir, "experiment_metadata.yaml")
    with open(yaml_path, "w") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)


def compute_mean_I(inputs: np.ndarray):
    """Estimate the average input current injected into the network."""
    avg_input_current = np.sum(inputs) / (inputs.shape[0] * inputs.shape[2])
    return avg_input_current


def compute_critical_weight(
    inputs: np.ndarray,
    parameter_name: str,
    parameter_value: float,
):
    """Estimate the average input current and the critical synaptic weight."""
    beta = PRESYNAPTIC_DEGREE
    current_amplitude = CURRENT_AMPLITUDE
    membrane_threshold = MEMBRANE_THRESHOLD

    if parameter_name == "beta":
        beta = parameter_value
    elif parameter_name == "current_amplitude":
        current_amplitude = parameter_value
    elif parameter_name == "membrane_threshold":
        membrane_threshold = parameter_value

    avg_input_current = (
        np.sum(inputs)
        / (NUM_NEURONS * inputs.shape[0] * inputs.shape[2])
    )

    critical_weight = (
        membrane_threshold
        - 2 * avg_input_current * current_amplitude * (REFRACTORY_PERIOD + 1)
    ) / (beta * NUM_NEURONS)

    return avg_input_current, critical_weight


def test_parameter_values(
    data,
    labels,
    param_name: str,
    param_values: list[float],
):
    """Test different parameter values over a range of mean weights."""
    sim_params = SimulationParams(
        num_neurons=NUM_NEURONS,
        mean_weight=0.0,
        weight_variance=WEIGHT_VARATION_COEFFICIENT,
        current_amplitude=CURRENT_AMPLITUDE,
        num_output_neurons=NUM_OUTPUT_NEURONS,
        is_random_uniform=False,
        membrane_threshold=MEMBRANE_THRESHOLD,
        leak_coefficient=LEAK_COEFFICIENT,
        refractory_period=REFRACTORY_PERIOD,
        small_world_graph_p=SMALL_WORLD_GRAPH_P,
        small_world_graph_k=SMALL_WORLD_GRAPH_K,
        input_spike_times=np.zeros(
            (data.shape[1], data.shape[2]),
            dtype=np.uint8,
        ),
    )

    all_results_statistics: list[dict] = []
    all_results_trace: list[dict] = []

    for param_value in param_values:
        print(f"\n### Testing {param_name} = {param_value} ###")

        # set the parameter we are testing
        if param_name == "beta":
            sim_params.small_world_graph_k = int(param_value * NUM_NEURONS * 2)
        elif param_name == "current_amplitude":
            sim_params.current_amplitude = param_value
        elif param_name == "membrane_threshold":
            sim_params.membrane_threshold = param_value
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        _, critical_weight = compute_critical_weight(
            data,
            param_name,
            param_value,
        )

        weight_values = np.linspace(
            critical_weight * MIN_INTERVAL,
            critical_weight * MAX_INTERVAL,
            NUM_WEIGHT_STEPS,
        )

        cnt = 0
        first_flag = True
        for weight in weight_values:
            cnt += 1
            print(
                f"\n--- mean_weight = {weight:.6f} --- "
                f"execution {cnt}/{NUM_WEIGHT_STEPS} "
                f"with {param_name} = {param_value}"
            )
            sim_params.mean_weight = weight

            (
                statistics_dataset,
                trace_dataset,
                spike_count,
            ) = simulate(
                data=data,
                labels=labels,
                parameters=sim_params,
                trace_tau=TRACE_TAU,
                statistic_set=STATISTIC_SET,
                is_first=first_flag,
                reload=RELOAD,
                membrane_reset=MEMBRANE_RESET,
            )

            first_flag = False

            (
                mean_accuracy_rf_trace,
                std_accuracy_rf_trace,
                mean_f1_rf_trace,
                std_f1_rf_trace,
                mean_mcc_rf_trace,
                std_mcc_rf_trace,
            ) = cross_validation_rf(
                trace_dataset,
                n_splits=CV_NUM_SPLITS,
            )
            print(
                "Trace RF - mean accuracy: ",
                mean_accuracy_rf_trace,
                " std accuracy: ",
                std_accuracy_rf_trace,
            )

            (
                mean_accuracy_slp_trace,
                std_accuracy_slp_trace,
                mean_f1_slp_trace,
                std_f1_slp_trace,
                mean_mcc_slp_trace,
                std_mcc_slp_trace,
            ) = cross_validation_slp(
                trace_dataset,
                n_splits=CV_NUM_SPLITS,
            )
            print(
                "Trace SLP - mean accuracy: ",
                mean_accuracy_slp_trace,
                " std accuracy: ",
                std_accuracy_slp_trace,
            )

            all_results_trace.append(
                {
                    "param_value": float(param_value),
                    "weight": float(weight),
                    "accuracy_rf": float(mean_accuracy_rf_trace),
                    "std_accuracy_rf": float(std_accuracy_rf_trace),
                    "accuracy_slp": float(mean_accuracy_slp_trace),
                    "std_accuracy_slp": float(std_accuracy_slp_trace),
                    "f1_rf": float(mean_f1_rf_trace),
                    "std_f1_rf": float(std_f1_rf_trace),
                    "f1_slp": float(mean_f1_slp_trace),
                    "std_f1_slp": float(std_f1_slp_trace),
                    "mcc_rf": float(mean_mcc_rf_trace),
                    "std_mcc_rf": float(std_mcc_rf_trace),
                    "mcc_slp": float(mean_mcc_slp_trace),
                    "std_mcc_slp": float(std_mcc_slp_trace),
                    "spike_count": float(spike_count),
                }
            )

            (
                mean_accuracy_rf_statistics,
                std_accuracy_rf_statistics,
                mean_f1_rf_statistics,
                std_f1_rf_statistics,
                mean_mcc_rf_statistics,
                std_mcc_rf_statistics,
            ) = cross_validation_rf(
                statistics_dataset,
                n_splits=CV_NUM_SPLITS,
            )
            print(
                "Statistics RF - mean accuracy: ",
                mean_accuracy_rf_statistics,
                " std accuracy: ",
                std_accuracy_rf_statistics,
            )

            (
                mean_accuracy_slp_statistics,
                std_accuracy_slp_statistics,
                mean_f1_slp_statistics,
                std_f1_slp_statistics,
                mean_mcc_slp_statistics,
                std_mcc_slp_statistics,
            ) = cross_validation_slp(
                statistics_dataset,
                n_splits=CV_NUM_SPLITS,
            )
            print(
                "Statistics SLP - mean accuracy: ",
                mean_accuracy_slp_statistics,
                " std accuracy: ",
                std_accuracy_slp_statistics,
            )

            all_results_statistics.append(
                {
                    "param_value": float(param_value),
                    "weight": float(weight),
                    "accuracy_rf": float(mean_accuracy_rf_statistics),
                    "std_accuracy_rf": float(std_accuracy_rf_statistics),
                    "accuracy_slp": float(mean_accuracy_slp_statistics),
                    "std_accuracy_slp": float(std_accuracy_slp_statistics),
                    "f1_rf": float(mean_f1_rf_statistics),
                    "std_f1_rf": float(std_f1_rf_statistics),
                    "f1_slp": float(mean_f1_slp_statistics),
                    "std_f1_slp": float(std_f1_slp_statistics),
                    "mcc_rf": float(mean_mcc_rf_statistics),
                    "std_mcc_rf": float(std_mcc_rf_statistics),
                    "mcc_slp": float(mean_mcc_slp_statistics),
                    "std_mcc_slp": float(std_mcc_slp_statistics),
                    "spike_count": float(spike_count),
                }
            )

    return all_results_statistics, all_results_trace


def main() -> None:
    """Load config, run sweeps, save results and metadata."""
    load_config()

    os.makedirs("dati", exist_ok=True)

    data, labels = load_dataset(DATASET_PATH)
    print(f"Loaded data: {data.shape}, labels: {labels.shape}")
    mean_I = compute_mean_I(data)

    os.makedirs(RESULTS_DIR_STATISTICS, exist_ok=True)
    os.makedirs(RESULTS_DIR_TRACE, exist_ok=True)

    (
        all_results_statistics,
        all_results_trace,
    ) = test_parameter_values(
        data,
        labels,
        PARAM_NAME,
        PARAMETER_VALUES,
    )

    results_df_statistics = pd.DataFrame(all_results_statistics)
    results_df_statistics.to_csv(CSV_NAME_STATISTICS, index=False)
    print(f"Saved results to {CSV_NAME_STATISTICS}")

    # save metadata for statistics features
    save_experiment_metadata(
        results_dir=RESULTS_DIR_STATISTICS,
        parameter_name=PARAM_NAME,
        parameter_values=PARAMETER_VALUES,
        output_features="statistics",
        mean_I=mean_I,
    )

    results_df_trace = pd.DataFrame(all_results_trace)
    results_df_trace.to_csv(CSV_NAME_TRACE, index=False)
    print(f"Saved results to {CSV_NAME_TRACE}")

    # save metadata for trace features
    save_experiment_metadata(
        results_dir=RESULTS_DIR_TRACE,
        parameter_name=PARAM_NAME,
        parameter_values=PARAMETER_VALUES,
        output_features="trace",
        mean_I=mean_I,
    )


if __name__ == "__main__":
    main()
