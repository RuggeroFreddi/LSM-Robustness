from typing import Tuple

import numpy as np
import pandas as pd

from LSM.model import Reservoir

TOPOLOGY_PATH = "data/topology.npz"
MEMBRANE_POTENTIALS_PATH = "data/membrane_potentials.npy"


def simulate(
    data: np.ndarray,
    labels: np.ndarray,
    parameters,
    trace_tau: int,
    statistic_set: int = 1,
    reload: bool = False,
    is_first: bool = False,
    membrane_reset: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Simulate SNN and build statistics/trace feature tables."""
    if statistic_set == 1:
        num_features = 4
    elif statistic_set == 2:
        num_features = 5
    else:
        raise ValueError("statistic_set must be 1 or 2.")

    num_trace_features = parameters.num_output_neurons * num_features
    start_index = parameters.num_neurons - num_trace_features - 100
    end_index = start_index + num_trace_features
    kept_indices = list(range(start_index, end_index))

    snn = Reservoir(parameters)
    if reload:
        if is_first:
            snn.save_membrane_potentials(MEMBRANE_POTENTIALS_PATH)
            snn.save_topology(TOPOLOGY_PATH)
            snn.reset_synaptic_weights(
                parameters.mean_weight,
                parameters.weight_variance,
            )
        else:
            snn.load_membrane_potentials(MEMBRANE_POTENTIALS_PATH)
            snn.load_topology(TOPOLOGY_PATH)
            snn.reset_synaptic_weights(
                parameters.mean_weight,
                parameters.weight_variance,
            )

    initial_membrane_potentials = snn.get_membrane_potentials()

    rows_statistics = []
    rows_trace = []
    avg_spike_count = 0.0
    num_output_neurons = None

    for i in range(len(data)):
        if i % 100 == 0:
            print(f"Processed {i} of {len(data)} samples")

        sample = data[i]
        label = labels[i]

        snn.set_input_spike_times(sample)
        if membrane_reset:
            snn.set_membrane_potentials(initial_membrane_potentials)
        snn.simulate(trace_tau=trace_tau, reset_trace=True)

        avg_spike_count += snn.tot_spikes
        trace = np.asarray(snn.get_trace()).reshape(-1)

        selected_trace_values = trace[kept_indices].tolist()
        row_trace = selected_trace_values + [label]
        rows_trace.append(row_trace)

        if statistic_set == 1:
            spike_counts = snn.get_spike_counts()
            spike_variances = snn.get_spike_variances()
            first_spike_times = snn.get_first_spike_times()
            mean_spike_times = snn.get_mean_spike_times()

            if num_output_neurons is None:
                num_output_neurons = len(spike_counts)

            sample_features = np.stack(
                (
                    spike_counts,
                    spike_variances,
                    first_spike_times,
                    mean_spike_times,
                ),
                axis=1,
            )
        elif statistic_set == 2:
            mean_spike_times = snn.get_mean_spike_times()
            first_spike_times = snn.get_first_spike_times()
            last_spike_times = snn.get_last_spike_times()
            mean_isi_per_neuron = snn.get_mean_isi_per_neuron()
            isi_variance_per_neuron = snn.get_isi_variance_per_neuron()

            if num_output_neurons is None:
                num_output_neurons = len(mean_spike_times)

            sample_features = np.stack(
                (
                    mean_spike_times,
                    first_spike_times,
                    last_spike_times,
                    mean_isi_per_neuron,
                    isi_variance_per_neuron,
                ),
                axis=1,
            )

        row_statistics = sample_features.flatten().tolist() + [label]
        rows_statistics.append(row_statistics)

    if statistic_set == 1:
        metrics = [
            "spike_count",
            "spike_variance",
            "first_spike_time",
            "mean_spike_time",
        ]
    else:
        metrics = [
            "mean_spike_times",
            "first_spike_times",
            "last_spike_times",
            "mean_isi_per_neuron",
            "isi_variance_per_neuron",
        ]

    column_names_statistics = [
        f"neuron_{i}_{metric}"
        for i in range(num_output_neurons)
        for metric in metrics
    ]
    column_names_statistics.append("label")

    df_statistics = pd.DataFrame(
        rows_statistics,
        columns=column_names_statistics,
    )

    trace_columns = [f"neuron_{idx}_trace" for idx in kept_indices]
    trace_column_names = trace_columns + ["label"]

    df_trace = pd.DataFrame(rows_trace, columns=trace_column_names)

    mean_spike_count = avg_spike_count / len(data)
    return df_statistics, df_trace, mean_spike_count
