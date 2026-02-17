"""
Plot generation elapsed times from gen_metrics.csv.

Run (after parse_logs.py):
  python metrics/plot_gen_times.py
"""

import os
import csv
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_METRICS = os.path.join(BASE_DIR, "static", "metrics")
GEN_CSV = os.path.join(STATIC_METRICS, "gen_metrics.csv")

os.makedirs(STATIC_METRICS, exist_ok=True)


def load_rows():
    rows = []
    if not os.path.exists(GEN_CSV):
        return rows
    with open(GEN_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            try:
                rows.append(
                    {
                        "run_id": int(r["run_id"]),
                        "elapsed_sec": float(r["elapsed_sec"]),
                        "checkpoint": r.get("checkpoint", ""),
                    }
                )
            except Exception:
                continue
    return rows


def plot():
    rows = load_rows()
    if not rows:
        print("No generation metrics found.")
        return

    # Bar per run_id
    plt.figure(figsize=(8, 4.5))
    xs = [f"Run {r['run_id']}" for r in rows]
    ys = [r["elapsed_sec"] for r in rows]
    plt.bar(xs, ys, color="#4caf50", alpha=0.8)
    plt.ylabel("Elapsed (s)")
    plt.title("Generation Time per Run")
    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.3)
    out_path = os.path.join(STATIC_METRICS, "gen_times.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot()

