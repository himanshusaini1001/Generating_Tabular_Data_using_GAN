"""
Create a 3D scatter of D/G losses vs epoch.

Run (after parse_logs.py):
  python metrics/plot_losses_3d.py
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

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot each run with different marker/color
    runs = {}
    for r in rows:
        runs.setdefault(r["run_id"], {"epoch": [], "d": [], "g": []})
        runs[r["run_id"]]["epoch"].append(r["epoch"])
        runs[r["run_id"]]["d"].append(r["d_loss"])
        runs[r["run_id"]]["g"].append(r["g_loss"])

    markers = ["o", "^", "s", "D", "P", "X"]
    colors = ["#ff6b6b", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#e74c3c"]
    for idx, (run_id, data) in enumerate(runs.items()):
        m = markers[idx % len(markers)]
        c = colors[idx % len(colors)]
        ax.scatter(data["epoch"], data["d"], data["g"], marker=m, color=c, label=f"Run {run_id}", alpha=0.85, s=30)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("D_loss")
    ax.set_zlabel("G_loss")
    ax.set_title("3D Loss Landscape (Epoch vs D/G)")
    ax.legend(loc="upper right")

    out_path = os.path.join(STATIC_METRICS, "losses_3d.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot()

