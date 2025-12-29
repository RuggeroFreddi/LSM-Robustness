# Robustness Analysis of Liquid State Machines

The purpose of the project is to study the robustness of Liquid State Machine (LSM) performance with respect to variations of the mean synaptic weight, under different dynamical and architectural conditions.

The repository provides a complete pipeline for:
- spike-based dataset generation,
- configurable experimental sweeps,
- quantitative robustness analysis and visualization.

In this work, robustness is quantified as the range of mean synaptic weights for which a given performance metric stays above a fixed fraction (THRESHOLD) of its maximum value.

---

## Workflow Summary

1. Dataset Generation: generate spike-train datasets from raw data.
2. Experiment Execution: run LSM experiments driven by a YAML configuration file.
3. Post-processing and Visualization: post-process results to obtain metrics, robustness intervals, and plots.

All steps are reproducible and configuration-driven (see Configuration Files).

---

## Main Scripts

### 1. Dataset Generation

- **`generate_MNIST_dataset.py`**  
  Generates a spike-train version of the MNIST dataset and saves it in the `data/` directory.  
  The original MNIST dataset is automatically downloaded and stored in `data/MNIST/` if not already present.  
  Output format: `(spiketrains, labels)`.

- **`generate_trajectory_dataset.py`**  
  Generates spike-train datasets from trajectory-based data and saves them in the `data/` directory.  
  Output format: `(spiketrains, labels)`.

These datasets are used as inputs to the LSM.

---

### 2. Experiment Execution

- **`experiment.py`**  
  Runs the main experimental pipeline.
  - Reads all parameters from `settings/config.yaml`.
  - Performs a sweep over a selected control parameter.
  - Executes cross-validation for each parameter value.
  - Automatically creates an output directory in `results/`.

For each experiment, the script produces:
- `experiment_metadata.yaml`: metadata and configuration of the experiment.
- `experiment_<parameter>_<num_steps>.csv`: aggregated results containing:
  - parameter value,
  - mean synaptic weight,
  - mean and standard deviation of accuracy, F1-macro, and MCC.

---

### 3. Post-processing and Visualization

- **`post_processing.py`**  
  Processes and visualizes experimental results.
  - Reads options from `settings/config_plot.yaml`.
  - Selects an experiment directory.
  - Generates:
    - metric vs. mean synaptic weight plots (accuracy, F1-macro, MCC). Separate plots for Random Forest and Single-Layer Perceptron readouts,
    - spike count vs. mean synaptic weight plots,
    - robustness interval plots (robustness width vs. swept parameter).
  - Exports two CSV summary files (one per readout) summarizing robustness results.

---

## Configuration Files

The experimental pipeline is controlled by two separate YAML configuration files, both located in the `settings/` directory.

### Experiment Configuration (`config.yaml`)

This file defines how experiments are executed and how the LSM is instantiated.

It specifies:

- **Task and data**
  - `TASK`: dataset to use (`MNIST` or `TRAJECTORY`).
  - `CV_NUM_SPLITS`: number of folds for cross-validation.

- **LSM architecture and neuron dynamics**
  - Number of neurons, membrane threshold, refractory period, leak coefficient.
  - Input current amplitude and neuron trace time constant.

- **Network topology**
  - Presynaptic degree controlling connection density.
  - Small-world rewiring probability.

- **Synaptic weight sweep**
  - `NUM_WEIGHT_STEPS`: number of mean synaptic weight values tested.
  - `WEIGHT_VARATION_COEFFICIENT`: controls variability around the mean weight.
  - Performance metrics are evaluated across this range for each experiment.

- **Parameter sweep**
  - `PARAM_NAME`: model parameter to sweep (e.g. membrane threshold).
  - `PARAMETER_VALUES`: list of values considered for the swept parameter.

For each value of the swept parameter, performance robustness is evaluated with respect to variations of the mean synaptic weight.

---

### Post-processing Configuration (`config_plot.yaml`)

This file controls how experimental results are selected, analyzed, and visualized.

It specifies:

- **Experiment selection**
  - `TASK`, `OUTPUT_FEATURES`, `PARAM_NAME`, and `DATE` identify the results directory to be analyzed.
  - `NUM_WEIGHT_STEPS` must match the value used during experiment execution.

- **Robustness definition**
  - `THRESHOLD`: relative threshold (fraction of the maximum metric) used to define robustness intervals.

Based on these settings, post-processing generates performance plots, robustness interval summaries, and aggregated CSV files.

---

## Directory Structure

```text
.
├── data/
│   ├── MNIST/
│   ├── mnist_spikes/
│   └── trajectory_spikes/
│
├── settings/
│   ├── config.yaml
│   └── config_plot.yaml
│
├── results/
│   └── results_<task>_<feature>_<parameter>_<date>/
│       ├── experiment_metadata.yaml
│       ├── experiment_<parameter>_<num_steps>.csv
│       └── plots/
│
├── LSM/
│   └── model.py
│
├── functions/
│   ├── cross_validation.py
│   └── lsm_forward.py
│
├── generate_MNIST_dataset.py
├── generate_trajectory_dataset.py
├── experiment.py
├── post_processing.py
├── requirements.txt
└── .gitignore
```

---
## Installation
```bash
pip install -r requirements.txt
```
A Python 3 environment is required.

## Typical Usage
**Generate datasets**
```bash
python generate_MNIST_dataset.py
python generate_trajectory_dataset.py
```

**Run experiments**
```bash
python experiment.py
```

**Post-process results**
Edit settings/config_plot.yaml to select the experiment and plotting options.
```bash
python post_processing.py
```

## Notes

- All experiments are driven by YAML configuration files to ensure reproducibility.
- Output directories are created automatically if missing.
- The codebase is intended for systematic robustness analyses rather than single-run evaluations.