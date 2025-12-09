# dentro le cartelle dei results fa il plot per vedere che accuracy, f1, mcc aumentano o diminuiscono

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

THRESHOLD = 0.85
# -------------------------
# Parametri esperimento
# -------------------------
TASK = "TRAJECTORY"  # "MNIST", "TRAJECTORY"
OUTPUT_FEATURES = "trace"  # "statistics", "trace" 
PARAM_NAME = "current_amplitude"  # "beta", "membrane_threshold", "current_amplitude"
NUM_WEIGHT_STEPS = 71 #71 , 101
DATE = "2025_11_27"

RESULTS_DIR = f"results/results_{TASK}_{OUTPUT_FEATURES}_{PARAM_NAME}_{DATE}"
CSV_NAME = os.path.join(RESULTS_DIR, f"experiment_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.csv")

# Offset orizzontale per separare i tre tipi di metriche lungo l'asse x
Y_OFFSET = 0.01  # puoi modificare questo valore a piacere

def compute_intervals(df: pd.DataFrame, metric_cols: dict) -> pd.DataFrame:
    """
    df: DataFrame con colonne 'param_value', 'weight' e le colonne metriche.
    metric_cols: dict {nome_metric: nome_colonna_df}
    Ritorna DataFrame con colonne: param_value, metric, start_weight, end_weight, width.
    """
    rows = []

    # Assicuriamoci che i tipi siano corretti
    df = df.copy()
    df["param_value"] = df["param_value"].astype(float)
    df["weight"] = df["weight"].astype(float)

    for param_val, g in df.groupby("param_value"):
        g_sorted = g.sort_values("weight")

        weights = g_sorted["weight"].values

        for metric_name, col_name in metric_cols.items():
            values = g_sorted[col_name].values.astype(float)

            if len(values) == 0:
                continue

            max_val = np.nanmax(values)
            if np.isnan(max_val):
                continue

            threshold = THRESHOLD * max_val
            mask = values >= threshold

            if not np.any(mask):
                # Caso patologico: nessun valore supera la soglia (compreso il massimo)
                continue

            start_weight = weights[mask][0]
            end_weight = weights[mask][-1]
            width = end_weight - start_weight

            rows.append(
                {
                    "param_value": param_val,
                    "metric": metric_name,
                    "start_weight": start_weight,
                    "end_weight": end_weight,
                    "width": width,
                }
            )

    return pd.DataFrame(rows)


def plot_intervals(intervals_df: pd.DataFrame, model_label: str, out_png_path: str):
    """
    intervals_df: DataFrame con colonne param_value, metric, width
    model_label: stringa "RF" o "SLP" (solo per titolo)
    out_png_path: path del file PNG di output
    """
    if intervals_df.empty:
        print(f"Nessun dato per il modello {model_label}, grafico non creato.")
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

    # offset per le tre metriche: -Y_OFFSET, 0, +Y_OFFSET
    offsets = {
        "accuracy": -Y_OFFSET,
        "f1": 0.0,
        "mcc": Y_OFFSET,
    }

    for metric in metrics_order:
        sub = intervals_df[intervals_df["metric"] == metric]
        if sub.empty:
            continue

        # x = param_value sfalsato
        x = sub["param_value"].values + offsets[metric]
        y = sub["width"].values

        plt.scatter(
            x,
            y,
            label=metric,
            marker=markers[metric],
            color=colors[metric],
        )

    plt.xlabel(f"{PARAM_NAME}")
    plt.ylabel("ampiezza intervallo (weight)")
    plt.title(f"Ampiezza intervalli sopra soglia (0.8*max) - {model_label}")
    plt.legend(title="Metrica")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=300)
    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Lettura CSV di input
    df = pd.read_csv(CSV_NAME)

    # Definizione colonne metriche per RF e SLP
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

    # Calcolo intervalli per RF
    intervals_rf = compute_intervals(df, rf_metrics)
    rf_csv_out = os.path.join(
        RESULTS_DIR, f"intervals_rf_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.csv"
    )
    intervals_rf.to_csv(rf_csv_out, index=False)

    # Calcolo intervalli per SLP
    intervals_slp = compute_intervals(df, slp_metrics)
    slp_csv_out = os.path.join(
        RESULTS_DIR, f"intervals_slp_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.csv"
    )
    intervals_slp.to_csv(slp_csv_out, index=False)

    # Grafici
    rf_png_out = os.path.join(
        RESULTS_DIR, f"intervals_rf_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.png"
    )
    slp_png_out = os.path.join(
        RESULTS_DIR, f"intervals_slp_{PARAM_NAME}_{NUM_WEIGHT_STEPS}.png"
    )

    plot_intervals(intervals_rf, "RF", rf_png_out)
    plot_intervals(intervals_slp, "SLP", slp_png_out)

    print("Fatto.")
    print(f"CSV RF:  {rf_csv_out}")
    print(f"CSV SLP: {slp_csv_out}")
    print(f"PNG RF:  {rf_png_out}")
    print(f"PNG SLP: {slp_png_out}")


if __name__ == "__main__":
    main()
