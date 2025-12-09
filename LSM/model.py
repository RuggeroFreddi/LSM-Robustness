"""Spiking Neural Network (SNN) model."""

from dataclasses import dataclass, field
from functools import wraps
import os
from typing import Optional, List
import warnings

import networkx as nx  
import numpy as np
from scipy.ndimage import label
from scipy.sparse import (
    csr_matrix,
    issparse,
    load_npz,
    save_npz,
    spmatrix,
    random as sparse_random,  
)


DEFAULT_MATRIX_PATH = "dati/snn_matrices.npz"
DEFAULT_POTENTIALS_PATH = "dati/membrane_potentials.npy"
DEFAULT_OUTPUT_NEURONS_PATH = "dati/output_neurons.npy"


@dataclass
class SimulationParams:
    """Parameters for running the Spiking Neural Network (SNN) simulation."""

    membrane_threshold: float
    leak_coefficient: float
    refractory_period: int
    small_world_graph_p: Optional[float] = None
    small_world_graph_k: Optional[float] = None
    connection_prob: Optional[float] = None
    is_random_uniform: Optional[bool] = None
    mean_weight: Optional[float] = None
    num_neurons: Optional[int] = None
    num_output_neurons: Optional[int] = None
    output_neurons: Optional[np.ndarray] = None
    membrane_potentials: Optional[np.ndarray] = None
    duration: Optional[int] = None
    current_amplitude: Optional[float] = None
    weight_variance: Optional[float] = None
    input_spike_times: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.uint8)
    )
    adjacency_matrix: Optional[spmatrix] = None

    def __post_init__(self):
        # === Basic type checks ===
        if self.connection_prob is not None:
            if not isinstance(self.connection_prob, (int, float)):
                raise TypeError("'connection_prob' must be a float or int.")
            if not (0 <= self.connection_prob <= 1):
                raise ValueError("'connection_prob' must be between 0 and 1.")

        if self.is_random_uniform is not None and not isinstance(
            self.is_random_uniform, bool
        ):
            raise TypeError("'is_random_uniform' must be a boolean if provided.")

        if self.num_neurons is not None:
            if not isinstance(self.num_neurons, int):
                raise TypeError("'num_neurons' must be an integer if provided.")
            if self.num_neurons <= 0:
                raise ValueError("'num_neurons' must be positive.")

        if self.num_output_neurons is not None:
            if not isinstance(self.num_output_neurons, int):
                raise TypeError("'num_output_neurons' must be an integer if provided.")
            if self.num_output_neurons <= 0:
                raise ValueError("'num_output_neurons' must be positive.")

        if self.output_neurons is not None:
            if not isinstance(self.output_neurons, np.ndarray):
                raise TypeError("'output_neurons' must be a NumPy array if provided.")
            if self.output_neurons.ndim != 1:
                raise ValueError("'output_neurons' must be a 1D NumPy array.")
            if not np.issubdtype(self.output_neurons.dtype, np.integer):
                raise TypeError("'output_neurons' must contain integers.")

        if not isinstance(self.membrane_threshold, (int, float)):
            raise TypeError("'membrane_threshold' must be a float or int.")

        if not isinstance(self.leak_coefficient, (int, float)):
            raise TypeError("'leak_coefficient' must be a float or int.")
        if not (0 <= self.leak_coefficient < 1):
            raise ValueError("'leak_coefficient' must be in the range [0, 1).")

        if not isinstance(self.refractory_period, int):
            raise TypeError("'refractory_period' must be an integer.")
        if self.refractory_period <= 0:
            raise ValueError("'refractory_period' must be positive.")

        if self.duration is not None:
            if not isinstance(self.duration, int):
                raise TypeError("'duration' must be an integer if provided.")
            if self.duration <= 0:
                raise ValueError("'duration' must be positive.")

        if self.current_amplitude is not None:
            if not isinstance(self.current_amplitude, (int, float)):
                raise TypeError(
                    "'current_amplitude' must be a float or int if provided."
                )

        if self.small_world_graph_p is not None:
            if not isinstance(self.small_world_graph_p, (int, float)):
                raise TypeError("'small_world_graph_p' must be a float.")
            if not (0 <= self.small_world_graph_p <= 1):
                raise ValueError("'small_world_graph_p' must be between 0 and 1.")

        if self.small_world_graph_k is not None:
            if not isinstance(self.small_world_graph_k, int):
                raise TypeError("'small_world_graph_k' must be an int.")
            if self.small_world_graph_k < 2 or self.small_world_graph_k % 2 != 0:
                raise ValueError(
                    "'small_world_graph_k' must be an even integer ≥ 2."
                )

        if self.weight_variance is not None:
            if not isinstance(self.weight_variance, (int, float)):
                raise TypeError("'weight_variance' must be a float or int if provided.")
            if self.weight_variance < 0:
                raise ValueError("'weight_variance' must be non-negative.")

        if not isinstance(self.input_spike_times, np.ndarray):
            raise TypeError("'input_spike_times' must be a NumPy array.")
        if self.input_spike_times.ndim != 2:
            raise ValueError("'input_spike_times' must be a 2D NumPy array.")
        if not np.issubdtype(self.input_spike_times.dtype, np.integer):
            raise TypeError("'input_spike_times' must contain integers.")

        if self.adjacency_matrix is not None:
            if not issparse(self.adjacency_matrix):
                raise TypeError("'adjacency_matrix' must be a SciPy sparse matrix.")
            if self.adjacency_matrix.shape[0] != self.adjacency_matrix.shape[1]:
                raise ValueError("'adjacency_matrix' must be square.")
            if not np.issubdtype(self.adjacency_matrix.data.dtype, np.floating):
                raise TypeError("'adjacency_matrix' must contain float weights.")
            if (
                self.adjacency_matrix.shape[0]
                < self.input_spike_times.shape[0]
            ):
                raise ValueError(
                    "'adjacency_matrix' must accommodate all input neurons."
                )

        if self.membrane_potentials is not None:
            if not isinstance(self.membrane_potentials, np.ndarray):
                raise TypeError(
                    "'membrane_potentials' must be a NumPy array if provided."
                )
            if self.membrane_potentials.ndim != 1:
                raise ValueError("'membrane_potentials' must be a 1D NumPy array.")
            if not np.issubdtype(self.membrane_potentials.dtype, np.floating):
                raise TypeError("'membrane_potentials' must contain float values.")
            if self.membrane_threshold > 0:
                if (
                    np.any(self.membrane_potentials < 0)
                    or np.any(self.membrane_potentials > self.membrane_threshold)
                ):
                    raise ValueError(
                        "'membrane_potentials' must be in the range "
                        "[0, membrane_threshold]."
                    )
            elif self.membrane_threshold < 0:
                if (
                    np.any(self.membrane_potentials > 0)
                    or np.any(self.membrane_potentials < self.membrane_threshold)
                ):
                    raise ValueError(
                        "'membrane_potentials' must be in the range "
                        "[membrane_threshold, 0]."
                    )

        # === Mutual exclusivity checks ===
        if self.num_neurons is not None and self.adjacency_matrix is not None:
            raise ValueError(
                "Provide either 'num_neurons' OR 'adjacency_matrix', not both."
            )
        if self.num_neurons is None and self.adjacency_matrix is None:
            raise ValueError(
                "You must provide one of 'num_neurons' or 'adjacency_matrix'."
            )

        if self.num_neurons is not None and self.mean_weight is None:
            raise ValueError(
                "When 'num_neurons' is provided, 'mean_weight' must also be provided."
            )

        if self.adjacency_matrix is not None:
            if self.mean_weight is not None:
                raise ValueError(
                    "Do not provide 'mean_weight' when using 'adjacency_matrix'."
                )
            if self.weight_variance is not None:
                raise ValueError(
                    "Do not provide 'weight_variance' when using 'adjacency_matrix'."
                )

        # === Output neuron configuration ===
        if (self.num_output_neurons is None) == (self.output_neurons is None):
            raise ValueError("Exactly one of 'num_output_neurons' or 'output_neurons' must be provided.")

        if self.output_neurons is not None:
            n = self.num_neurons if self.num_neurons is not None else (
                self.adjacency_matrix.shape[0] if self.adjacency_matrix is not None else None
            )
            if n is not None:
                if np.any(self.output_neurons < 0) or np.any(self.output_neurons >= n):
                    raise ValueError("'output_neurons' contains invalid indices (out of bounds).")

        # === Connection type validation ===
        if self.num_neurons is not None:
            if self.is_random_uniform is True:
                if (
                    self.small_world_graph_k is not None
                    or self.small_world_graph_p is not None
                ):
                    raise ValueError(
                        "'small_world_graph_k' and 'small_world_graph_p' must not be "
                        "provided when 'is_random_uniform' is True."
                    )
                if self.connection_prob is None:
                    raise ValueError(
                        "'connection_prob' must be provided when 'is_random_uniform' is True."
                    )
            else:
                if self.connection_prob is not None:
                    raise ValueError(
                        "'connection_prob' must not be provided when 'is_random_uniform' is False."
                    )
                if (
                    self.small_world_graph_k is None
                    or self.small_world_graph_p is None
                ):
                    raise ValueError(
                        "'small_world_graph_k' and 'small_world_graph_p' must be provided "
                        "when 'is_random_uniform' is False."
                    )
            

def require_simulation_run(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.spike_matrix_output is None:
            raise ValueError("Simulation not yet run.")
        return method(self, *args, **kwargs)
    return wrapper


class Reservoir:
    """Implements a Spiking Neural Network (SNN) with customizable topology and simulation features."""

    def __init__(self, simulation_params: SimulationParams) -> None:
        self.simulation_params = simulation_params

        # === Configuration from simulation parameters ===
        self.current_amplitude = (
            simulation_params.membrane_threshold
            if simulation_params.current_amplitude is None
            else simulation_params.current_amplitude
        )

        self.num_input_neurons: int = simulation_params.input_spike_times.shape[0]
        self.membrane_threshold: float = simulation_params.membrane_threshold
        self.refractory_period: int = simulation_params.refractory_period
        self.leak_coefficient: float = simulation_params.leak_coefficient

        self.simulation_duration: int = (
            simulation_params.input_spike_times.shape[1]
            if simulation_params.duration is None
            else simulation_params.duration
        )
        self.input_spike_times = simulation_params.input_spike_times

        if simulation_params.num_neurons is not None:
            self.num_neurons = simulation_params.num_neurons
            self.weights_mean = simulation_params.mean_weight
            self.weights_variance = (
                simulation_params.weight_variance
                if simulation_params.weight_variance is not None
                else 0.1 
            )

            if simulation_params.is_random_uniform is True:
                self._generate_synaptic_weights_random_uniform(
                    simulation_params.connection_prob
                )
            else:
                self._generate_synaptic_weights_small_world(
                    simulation_params.small_world_graph_p,
                    simulation_params.small_world_graph_k,
                )

        elif simulation_params.adjacency_matrix is not None:
            W = simulation_params.adjacency_matrix.tocsr(copy=True)
            W.sort_indices()
            W.data = W.data.astype(np.float32, copy=False)
            self.synaptic_weights = W
            self.num_neurons = W.shape[0]
            in_degrees = W.getnnz(axis=0)
            self.mean_in_degree = float(in_degrees.mean())
            if W.nnz == 0:
                raise ValueError("Provided adjacency matrix has no non-zero weights.")
            self.weights_mean = float(W.data.mean())
            self.weights_variance = float(W.data.var())


        else:
            raise ValueError(
                "You must provide either 'num_neurons' or 'adjacency_matrix'."
            )

        if simulation_params.num_output_neurons is not None:
            self.num_output_neurons: int = simulation_params.num_output_neurons
            hidden_indices = np.arange(self.num_input_neurons, self.num_neurons)
            if len(hidden_indices) < self.num_output_neurons:
                raise ValueError("Not enough hidden neurons to select output neurons.")
            self.output_neurons = np.random.choice(
                hidden_indices,
                size=self.num_output_neurons,
                replace=False,
            )
        else:
            self.output_neurons = simulation_params.output_neurons
            self.num_output_neurons = len(simulation_params.output_neurons)

        self.time_step: int = 1

        # === Internal state ===
        self.tot_spikes: int = 0
        if simulation_params.membrane_potentials is None:
            thr = float(self.membrane_threshold)
            lo, hi = (0.0, thr) if thr >= 0.0 else (thr, 0.0)
            self.membrane_potentials = np.random.uniform(lo, hi, self.num_neurons).astype(np.float32, copy=False)
        else:
            if len(simulation_params.membrane_potentials) != self.num_neurons:
                raise ValueError("Length of 'membrane_potentials' must be equal to 'num_neurons'.")
            self.membrane_potentials = simulation_params.membrane_potentials.astype(np.float32, copy=True)


        self.membrane_potentials_init = self.membrane_potentials.copy()
        self.spike_matrix: Optional[np.ndarray] = None
        self.spike_matrix_output: Optional[np.ndarray] = None
        self.refractory_timer: np.ndarray = np.zeros(self.num_neurons, dtype=np.int32)
        self.trace = np.zeros(self.num_neurons, dtype=np.float32)

        self.leak_cv = 0.5
        self._init_leak_vector()

    def _init_leak_vector(self) -> None:
        """Initialize per-neuron leak and leak factor based on num_neurons and leak_coefficient."""
        m = float(self.leak_coefficient)
        if m > 0.0:
            s = m * self.leak_cv
            sigma2 = np.log(1.0 + (s * s) / (m * m))
            sigma = np.sqrt(sigma2)
            mu = np.log(m) - 0.5 * sigma2
            self.leak_vector = np.random.lognormal(
                mean=mu, sigma=sigma, size=self.num_neurons
            ).astype(np.float32, copy=False)
        else:
            self.leak_vector = np.full(
                self.num_neurons, m, dtype=np.float32
            )
        self.leak_factor_vector = 1.0 - self.leak_vector

    def _generate_synaptic_weights_small_world(
        self,
        small_world_graph_p: float = 0.1,
        small_world_graph_k: int = 10,
    ) -> None:
        """Generate synaptic weights using a small-world graph without dense NxN allocation."""
        small_world_graph = nx.watts_strogatz_graph(
            n=self.num_neurons,
            k=small_world_graph_k,
            p=small_world_graph_p,
            seed=None,
        )

        edges = np.asarray(small_world_graph.edges(), dtype=np.int32)
        if edges.size == 0:
            self.synaptic_weights = csr_matrix((self.num_neurons, self.num_neurons), dtype=np.float32)
            self.mean_in_degree = 0.0
            return

        flip = np.random.rand(edges.shape[0]) < 0.5
        rows = np.where(flip, edges[:, 0], edges[:, 1])
        cols = np.where(flip, edges[:, 1], edges[:, 0])

        weights = np.random.normal(
            loc=self.weights_mean,
            scale=abs(self.weights_mean) * self.weights_variance,
            size=rows.shape[0],
        ).astype(np.float32, copy=False)

        self.synaptic_weights = csr_matrix(
            (weights, (rows, cols)), shape=(self.num_neurons, self.num_neurons)
        )
        self.synaptic_weights.setdiag(0.0)  
        self.synaptic_weights.eliminate_zeros() 
        self.synaptic_weights.sort_indices()
        self.synaptic_weights.data = self.synaptic_weights.data.astype(np.float32, copy=False)

        in_degrees = self.synaptic_weights.getnnz(axis=0)
        self.mean_in_degree = in_degrees.mean()

    def _generate_synaptic_weights_random_uniform(self, connection_prob: float = 0.2) -> None:
        """Generate sparse synaptic weights with Bernoulli connections and normal weights."""
        N = self.num_neurons
        rng = np.random.default_rng()
        def data_rvs(k):
            return rng.normal(
                loc=self.weights_mean,
                scale=abs(self.weights_mean) * self.weights_variance,
                size=k,
            ).astype(np.float32, copy=False)

        W = sparse_random(
            N, N, density=float(connection_prob),
            data_rvs=data_rvs, format="csr", dtype=np.float32
        )
        W.setdiag(0.0)
        W.eliminate_zeros()
        W.sort_indices()
        self.synaptic_weights = W

        in_degrees = W.getnnz(axis=0)
        self.mean_in_degree = in_degrees.mean()

    @property
    def avg_in_degree(self):
        return self.synaptic_weights.getnnz(axis=0).mean()

    def simulate(self, trace_tau=10, reset_trace=False) -> np.ndarray:
        """Run the optimized simulation of the SNN."""
        if trace_tau <= 0:
            raise ValueError("'trace_tau' must be > 0.")
        if reset_trace:
            self.trace[:] = 0.0
        if self.input_spike_times.shape[0] > self.num_neurons:
            warnings.warn(
                "Number of input spike rows exceeds 'num_neurons'. "
                "Simulation may behave unexpectedly.",
                category=UserWarning
            )

        if np.any(self.output_neurons >= self.num_neurons) or np.any(self.output_neurons < 0):
            warnings.warn(
                "Provided output neuron indices are out of bounds. "
                "Simulation may behave unexpectedly.",
                category=UserWarning
            )

        if len(self.membrane_potentials) != self.num_neurons:
            warnings.warn(
                "Length of 'membrane_potentials' is not equal to 'num_neurons'. "
                "Simulation may behave unexpectedly.",
                category=UserWarning
            )

        if self.synaptic_weights.data.size == 0:
            warnings.warn(
                "Provided adjacency matrix has no non-zero weights. "
                "Simulation may behave unexpectedly.",
                category=UserWarning
            )

        T = self.simulation_duration
        N = self.num_neurons
        Nin = self.num_input_neurons
        inputs = self.input_spike_times
        inputs_T = inputs.shape[1] if inputs.size else 0
        mem = self.membrane_potentials
        refr = self.refractory_timer
        out_idx = self.output_neurons
        curr_amp = np.float32(self.current_amplitude)

        if mem.dtype != np.float32:
            mem = mem.astype(np.float32, copy=False)
            self.membrane_potentials = mem
        if refr.dtype not in (np.int32, np.int16):
            refr = refr.astype(np.int32, copy=False)
            self.refractory_timer = refr

        W = self.synaptic_weights.tocsr()
        indptr, indices, data = W.indptr, W.indices, W.data

        decrease = float(np.exp(-self.time_step / trace_tau))

        self.tot_spikes = 0
        self.spike_matrix = np.zeros((T, N), dtype=np.int8)
        spikes_out = self.spike_matrix  # alias

        for t in range(T):
            refr -= self.time_step
            np.clip(refr, 0, None, out=refr)
            if t < inputs_T:
                spikes_t = inputs[:, t] 
                mem[:Nin] += curr_amp * spikes_t  

            spiking_mask = (mem >= self.membrane_threshold) & (refr == 0)
            spikes_out[t, :] = spiking_mask

            spk_idx = np.flatnonzero(spiking_mask)
            self.tot_spikes += spk_idx.size

            self.trace *= decrease
            self.trace[spk_idx] += 1.0

            mem[spiking_mask] = 0.0
            refr[spiking_mask] = self.refractory_period + 1

            mem *= self.leak_factor_vector

            for j in spk_idx:
                start, end = indptr[j], indptr[j + 1]
                if start != end:
                    cols = indices[start:end]
                    mem[cols] += data[start:end]

        self.spike_matrix_output = self.spike_matrix[:, out_idx]
        return self.spike_matrix_output

    def get_trace(self) -> np.ndarray:
        """Return neuron trace."""
        return self.trace.copy()

    @require_simulation_run
    def get_spike_time_lists_output(self) -> List[List[int]]:
        """Return a list of lists with spike times for each output neuron."""
        return [
            list(np.where(self.spike_matrix_output[:, i] == 1)[0])
            for i in range(self.num_output_neurons)
        ]


    @require_simulation_run
    def get_spike_counts(self) -> np.ndarray:
        """Total spike count per output neuron."""
        return np.sum(self.spike_matrix_output, axis=0)


    @require_simulation_run
    def get_spike_variances(self) -> np.ndarray:
        """Variance of spike sequences per output neuron."""
        return np.var(self.spike_matrix_output, axis=0)


    @require_simulation_run
    def get_first_spike_times(self) -> np.ndarray:
        """First spike time per output neuron."""
        has_spike = np.any(self.spike_matrix_output == 1, axis=0)
        first_spike_times = np.argmax(self.spike_matrix_output == 1, axis=0)
        return np.where(has_spike, first_spike_times, -1)


    @require_simulation_run
    def get_mean_spike_times(self) -> np.ndarray:
        """Mean spike time per output neuron."""
        spike_counts = self.get_spike_counts()
        times = np.arange(self.spike_matrix_output.shape[0])[:, None]
        weighted_times = self.spike_matrix_output * times
        sum_times = np.sum(weighted_times, axis=0)

        mean_spike_times = np.full_like(spike_counts, fill_value=-1.0, dtype=float)
        nonzero_mask = spike_counts > 0
        mean_spike_times[nonzero_mask] = (
            sum_times[nonzero_mask] / spike_counts[nonzero_mask]
        )
        return mean_spike_times

    @require_simulation_run
    def get_last_spike_times(self) -> np.ndarray:
        """Last spike time per output neuron."""
        has_spike = np.any(self.spike_matrix_output == 1, axis=0)
        last_spike_times = (
            self.spike_matrix_output.shape[0] - 1
            - np.argmax(self.spike_matrix_output[::-1] == 1, axis=0)
        )
        return np.where(has_spike, last_spike_times, -1)

    @require_simulation_run
    def get_mean_isi_per_neuron(self) -> np.ndarray:
        """Mean ISI (inter-spike interval) per output neuron."""
        mean_isis = np.full(self.num_output_neurons, -1.0, dtype=float)
        for i in range(self.num_output_neurons):
            spike_times = np.where(self.spike_matrix_output[:, i] == 1)[0]
            if len(spike_times) > 1:
                isis = np.diff(spike_times)
                mean_isis[i] = np.mean(isis)
        return mean_isis

    @require_simulation_run
    def get_isi_variance_per_neuron(self) -> np.ndarray:
        """ISI variance per output neuron."""
        isi_vars = np.full(self.num_output_neurons, -1.0, dtype=float)
        for i in range(self.num_output_neurons):
            spike_times = np.where(self.spike_matrix_output[:, i] == 1)[0]
            if len(spike_times) > 1:
                isis = np.diff(spike_times)
                isi_vars[i] = np.var(isis)
        return isi_vars

    @require_simulation_run
    def get_burst_counts(self) -> np.ndarray:
        """Return the number of spike bursts per output neuron."""
        burst_counts = np.zeros(self.num_output_neurons, dtype=int)
        for i in range(self.num_output_neurons):
            spike_train = self.spike_matrix_output[:, i]
            _, num_bursts = label(spike_train)
            burst_counts[i] = num_bursts
        return burst_counts
    
    @require_simulation_run
    def extract_features_from_spikes(self) -> dict:
        """Extract all key features from output neurons as a dictionary."""
        return {
            "spike_counts": self.get_spike_counts(),
            "spike_variances": self.get_spike_variances(),
            "mean_spike_times": self.get_mean_spike_times(),
            "first_spike_times": self.get_first_spike_times(),
            "last_spike_times": self.get_last_spike_times(),
            "mean_isi": self.get_mean_isi_per_neuron(),
            "isi_variances": self.get_isi_variance_per_neuron(),
            "burst_counts": self.get_burst_counts(),  
        }
        
    def save_topology(self, filename: str = DEFAULT_MATRIX_PATH) -> None:
        """Save synaptic weights matrix to .npz file."""
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        save_npz(filename, self.synaptic_weights)

    def load_topology(self, filename: str = DEFAULT_MATRIX_PATH) -> None:
        """Load synaptic weights matrix from a .npz file."""
        weight_matrix = load_npz(filename).tocsr()
        weight_matrix.sort_indices()
        self.synaptic_weights = weight_matrix

        self.num_neurons = self.synaptic_weights.shape[0]
        in_degrees = self.synaptic_weights.getnnz(axis=0)
        self.mean_in_degree = in_degrees.mean()

        weights_array = self.synaptic_weights.data
        self.weights_mean = float(np.mean(weights_array))
        self.weights_variance = float(np.var(weights_array))
        self.synaptic_weights.data = self.synaptic_weights.data.astype(np.float32, copy=False)
        self._init_leak_vector()


    def set_topology(self, topology) -> None:
        """Set synaptic weights matrix."""
        weight_matrix = topology.tocsr(copy=True) if issparse(topology) else csr_matrix(topology)
        weight_matrix.sort_indices()
        self.synaptic_weights = weight_matrix

        self.num_neurons = weight_matrix.shape[0]
        in_degrees = weight_matrix.getnnz(axis=0)
        self.mean_in_degree = in_degrees.mean()

        weights_array = weight_matrix.data
        self.weights_mean = float(np.mean(weights_array))
        self.weights_variance = float(np.var(weights_array))
        self.synaptic_weights.data = self.synaptic_weights.data.astype(np.float32, copy=False)
        self._init_leak_vector()

    def get_topology(self) -> csr_matrix:
        """Returns synaptic weights matrix."""
        return self.synaptic_weights.copy()

    def save_membrane_potentials(
        self,
        filename: str = DEFAULT_POTENTIALS_PATH
    ) -> None:
        """Save membrane potentials to .npy file."""
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        np.save(filename, self.membrane_potentials)

    def load_membrane_potentials(
        self,
        filename: str = DEFAULT_POTENTIALS_PATH
    ) -> None:
        """Load membrane potentials from .npy file."""
        self.membrane_potentials = np.load(filename)


    def set_membrane_potentials(self, membrane_potentials: np.ndarray) -> None:
        """Set membrane potentials from provided array."""
        self.membrane_potentials = membrane_potentials.copy()
        
    def get_membrane_potentials(self) -> np.ndarray:
        """Returns membrane potentials."""
        return self.membrane_potentials.copy()


    def save_output_neurons(self, filename: str = DEFAULT_OUTPUT_NEURONS_PATH) -> None:
        """Save selected output neuron indices to a .npy file."""
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        np.save(filename, self.output_neurons)

    def load_output_neurons(self, filename: str = DEFAULT_OUTPUT_NEURONS_PATH) -> None:
        """Load the list of indices identifying the selected output neurons."""
        arr = np.load(filename)
        arr = np.asarray(arr)

        if arr.ndim != 1:
            raise ValueError("'output_neurons' must be a 1D array.")
        if not np.issubdtype(arr.dtype, np.integer):
            if np.all(np.isfinite(arr)) and np.all(np.mod(arr, 1) == 0):
                arr = arr.astype(np.int64)
            else:
                raise TypeError("'output_neurons' must contain integer indices.")

        n = getattr(self, "num_neurons", None)
        if n is None and hasattr(self, "synaptic_weights"):
            n = self.synaptic_weights.shape[0]
        if n is not None:
            if np.any(arr < 0) or np.any(arr >= n):
                raise ValueError("'output_neurons' are out of bounds.")

        self.output_neurons = arr
        self.num_output_neurons = int(arr.size)

    def set_output_neurons(self, indices: np.ndarray) -> None:
        """Sets the list of indices identifying the selected output neurons."""
        arr = np.asarray(indices)

        if arr.ndim != 1:
            raise ValueError("'output_neurons' must be a 1D array.")
        if not np.issubdtype(arr.dtype, np.integer):
            if np.all(np.isfinite(arr)) and np.all(np.mod(arr, 1) == 0):
                arr = arr.astype(np.int64)
            else:
                raise TypeError("'output_neurons' must contain integer indices.")

        n = getattr(self, "num_neurons", None)
        if n is None and hasattr(self, "synaptic_weights"):
            n = self.synaptic_weights.shape[0]
        if n is not None:
            if np.any(arr < 0) or np.any(arr >= n):
                raise ValueError("'output_neurons' are out of bounds.")

        self.output_neurons = arr
        self.num_output_neurons = int(arr.size)

    def get_output_neurons(self) -> np.ndarray:
        """Returns the list of indices identifying the selected output neurons."""
        return self.output_neurons

    def set_input_spike_times(self, input_spike_times: np.ndarray) -> None:
        """Set input spike times and adjust simulation duration."""
        self.input_spike_times = input_spike_times
        self.num_input_neurons = input_spike_times.shape[0]
        self.simulation_duration = (
            self.simulation_params.duration
            if self.simulation_params.duration is not None
            else input_spike_times.shape[1]
        )

    def calculate_mean_isi(self) -> float:
        """Compute mean inter-spike interval (ISI)."""
        spike_times_list = self.get_spike_time_lists_output()
        total_intervals = []
        for spike_times in spike_times_list:
            if len(spike_times) > 1:
                intervals = np.diff(spike_times)
                total_intervals.extend(intervals)
        if total_intervals:
            return float(np.mean(total_intervals))
        return float(self.simulation_duration)
    
    def rescale_synaptic_weights_to_mean(self, target_mean: float) -> None:
        """Scale non-zero synaptic weights so their mean equals target_mean."""
        if not hasattr(self, "synaptic_weights"):
            raise ValueError("Synaptic weights not initialized.")
        if not np.isfinite(target_mean):
            raise ValueError("'target_mean' must be finite.")

        data = self.synaptic_weights.data
        if data.size == 0:
            raise ValueError("Synaptic weights matrix has no non-zero entries.")

        current_mean = float(np.mean(data))
        if current_mean == 0.0:
            if target_mean == 0.0:
                self.weights_mean = 0.0
                self.weights_variance = float(np.var(data))
                return
            raise ValueError("Cannot rescale from zero current mean to a non-zero target.")

        scale = float(target_mean) / current_mean
        self.synaptic_weights.data *= scale 
        weights_array = self.synaptic_weights.data
        self.weights_mean = float(np.mean(weights_array))
        self.weights_variance = float(np.var(weights_array))

    def reset_synaptic_weights(self, mean: float, std: Optional[float] = None) -> None:
        """Reset synaptic weights with new normal distribution."""
        if not hasattr(self, "synaptic_weights"):
            raise ValueError("Synaptic weights not initialized.")

        if std is None:
            std = 0.1

        old_weight_matrix = self.synaptic_weights.tocsr()
        rows, cols = old_weight_matrix.nonzero()
        new_weights = np.random.normal(loc=mean, scale=std * mean, size=rows.size)

        new_weight_matrix = csr_matrix((new_weights, (rows, cols)), shape=old_weight_matrix.shape)
        new_weight_matrix.sort_indices()               
        self.synaptic_weights = new_weight_matrix
        self.synaptic_weights.data = self.synaptic_weights.data.astype(np.float32, copy=False)

        self.weights_mean = mean
        self.weights_variance = (std * mean) ** 2

    def reset(self) -> None:
        """
        Reset internal simulation state: membrane potentials, spike matrices, refractory timers.
        Does not reset the synaptic weights or topology.
        """
        self.tot_spikes = 0
        self.spike_matrix = None
        self.spike_matrix_output = None
        self.refractory_timer = np.zeros(self.num_neurons, dtype=np.int32)
        self.membrane_potentials = self.membrane_potentials_init.copy()
        self.trace[:] = 0.0 

    def get_network_parameters(self) -> dict:
        """Return key parameters of the spiking neural network."""
        return {
            "num_neurons": self.num_neurons,
            "num_input_neurons": self.num_input_neurons,
            "num_output_neurons": self.num_output_neurons,
            "output_neurons": (
                self.output_neurons.tolist()
                if hasattr(self.output_neurons, "tolist")
                else self.output_neurons
            ),
            "membrane_threshold": self.membrane_threshold,
            "refractory_period": self.refractory_period,
            "leak_coefficient": self.leak_coefficient,
            "simulation_duration": self.simulation_duration,
            "weights_mean": self.weights_mean,
            "weights_variance": self.weights_variance,
            "mean_in_degree": self.mean_in_degree,
            "time_step": self.time_step,
            "current_amplitude": self.current_amplitude,
        }


def load_output_neurons(filename: str = DEFAULT_OUTPUT_NEURONS_PATH) -> np.ndarray:
    """Load selected output neuron indices from a .npy file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' not found.")
    return np.load(filename)


def load_membrane_potentials(filename: str = DEFAULT_POTENTIALS_PATH) -> np.ndarray:
    """Load membrane potentials from a .npy file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' not found.")
    return np.load(filename)


def load_topology(filename: str = DEFAULT_MATRIX_PATH) -> csr_matrix:
    """Load synaptic weights matrix from a .npz file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' not found.")
    return load_npz(filename)