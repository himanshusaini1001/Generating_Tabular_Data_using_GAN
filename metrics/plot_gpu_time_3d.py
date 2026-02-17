"""
Create a 3D scatter of Epoch vs GPU Memory vs Time per Epoch.

Run (after parse_logs.py):
  python metrics/plot_gpu_time_3d.py
"""

import os
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
                gpu = float(r["gpu_alloc_gb"]) if r["gpu_alloc_gb"] else None
                t = float(r["time_sec"]) if r["time_sec"] else None
                if gpu is None or t is None:
                    continue
                rows.append(
                    {
                        "run_id": int(r["run_id"]),
                        "epoch": int(r["epoch"]),
                        "gpu_alloc": gpu,
                        "time_sec": t,
                    }
                )
            except Exception:
                continue
    return rows


def plot():
    rows = load_rows()
    if not rows:
        print("No GPU/time metrics found.")
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    runs = {}
    for r in rows:
        runs.setdefault(r["run_id"], {"epoch": [], "gpu": [], "time": []})
        runs[r["run_id"]]["epoch"].append(r["epoch"])
        runs[r["run_id"]]["gpu"].append(r["gpu_alloc"])
        runs[r["run_id"]]["time"].append(r["time_sec"])

    markers = ["o", "^", "s", "D", "P", "X"]
    colors = ["#4caf50", "#3498db", "#ff6b6b", "#f39c12", "#9b59b6", "#2ecc71"]
    for idx, (run_id, data) in enumerate(runs.items()):
        m = markers[idx % len(markers)]
        c = colors[idx % len(colors)]
        ax.scatter(data["epoch"], data["gpu"], data["time"], marker=m, color=c, label=f"Run {run_id}", alpha=0.85, s=30)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("GPU Alloc (GB)")
    ax.set_zlabel("Time per Epoch (s)")
    ax.set_title("3D GPU vs Time per Epoch")
    ax.legend(loc="upper right")

    out_path = os.path.join(STATIC_METRICS, "gpu_time_3d.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot()

