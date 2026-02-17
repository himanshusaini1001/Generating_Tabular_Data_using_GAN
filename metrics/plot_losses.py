"""
Plot D/G losses over epochs from train_metrics.csv.

Run (after parse_logs.py):
  python metrics/plot_losses.py
"""

import os
import csv
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_METRICS = os.path.join(BASE_DIR, "static", "metrics")
TRAIN_CSV = os.path.join(STATIC_METRICS, "train_metrics.csv")

os.makedirs(STATIC_METRICS, exist_ok=True)


def load_rows():
    rows = []
    if not os.path.exists(TRAIN_CSV):
        return rows
    with open(TRAIN_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            try:
                rows.append(
                    {
                        "run_id": int(r["run_id"]),
                        "epoch": int(r["epoch"]),
                        "d_loss": float(r["d_loss"]),
                        "g_loss": float(r["g_loss"]),
                    }
                )
            except Exception:
                continue
    return rows


def plot():
    rows = load_rows()
    if not rows:
        print("No training metrics found.")
        return

    runs = {}
    for r in rows:
        runs.setdefault(r["run_id"], {"epoch": [], "d": [], "g": []})
        runs[r["run_id"]]["epoch"].append(r["epoch"])
        runs[r["run_id"]]["d"].append(r["d_loss"])
        runs[r["run_id"]]["g"].append(r["g_loss"])

    plt.figure(figsize=(9, 5))
    for run_id, data in runs.items():
        plt.plot(data["epoch"], data["d"], label=f"Run {run_id} D", linestyle="-", marker="o")
        plt.plot(data["epoch"], data["g"], label=f"Run {run_id} G", linestyle="--", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("D/G Loss per Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = os.path.join(STATIC_METRICS, "losses.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot()

