"""
Plot epoch time and system utilization from train_metrics.csv.

Run (after parse_logs.py):
  python metrics/plot_time_cpu.py
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
                        "time_sec": float(r["time_sec"]) if r["time_sec"] else None,
                        "cpu_pct": float(r["cpu_pct"]) if r["cpu_pct"] else None,
                        "ram_pct": float(r["ram_pct"]) if r["ram_pct"] else None,
                        "gpu_alloc": float(r["gpu_alloc_gb"]) if r["gpu_alloc_gb"] else None,
                        "gpu_resv": float(r["gpu_resv_gb"]) if r["gpu_resv_gb"] else None,
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
        runs.setdefault(r["run_id"], []).append(r)

    # Time per epoch (stacked area by run)
    plt.figure(figsize=(9, 5))
    for run_id, data in runs.items():
        xs = [d["epoch"] for d in data]
        ys = [d["time_sec"] or 0 for d in data]
        plt.fill_between(xs, ys, alpha=0.25, label=f"Run {run_id}")
        plt.plot(xs, ys, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Time per Epoch (s)")
    plt.title("Epoch Duration")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_time = os.path.join(STATIC_METRICS, "epoch_time.png")
    plt.tight_layout()
    plt.savefig(out_time, dpi=150)
    plt.close()

    # CPU / RAM lines
    plt.figure(figsize=(9, 5))
    for run_id, data in runs.items():
        xs = [d["epoch"] for d in data]
        cpu = [d["cpu_pct"] for d in data if d["cpu_pct"] is not None]
        ram = [d["ram_pct"] for d in data if d["ram_pct"] is not None]
        xs_cpu = [d["epoch"] for d in data if d["cpu_pct"] is not None]
        xs_ram = [d["epoch"] for d in data if d["ram_pct"] is not None]
        if xs_cpu:
            plt.plot(xs_cpu, cpu, label=f"CPU run {run_id}", linestyle="-", marker="o")
        if xs_ram:
            plt.plot(xs_ram, ram, label=f"RAM run {run_id}", linestyle="--", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Utilization (%)")
    plt.title("CPU / RAM Utilization")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_util = os.path.join(STATIC_METRICS, "cpu_ram.png")
    plt.tight_layout()
    plt.savefig(out_util, dpi=150)
    plt.close()

    # GPU memory bars (allocated)
    plt.figure(figsize=(9, 5))
    for run_id, data in runs.items():
        xs = [d["epoch"] for d in data if d["gpu_alloc"] is not None]
        ys = [d["gpu_alloc"] for d in data if d["gpu_alloc"] is not None]
        if xs:
            plt.bar(xs, ys, alpha=0.6, label=f"GPU alloc run {run_id}")
    plt.xlabel("Epoch")
    plt.ylabel("GPU Alloc (GB)")
    plt.title("GPU Memory Allocation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_gpu = os.path.join(STATIC_METRICS, "gpu_alloc.png")
    plt.tight_layout()
    plt.savefig(out_gpu, dpi=150)
    plt.close()

    print(f"Saved {out_time}, {out_util}, {out_gpu}")


if __name__ == "__main__":
    plot()

