"""
Parse training and generation logs into CSVs for downstream plotting.

Inputs:
  - logs/train_log.txt
  - logs/gen_log.txt

Outputs (written under static/metrics/):
  - train_metrics.csv
      columns: run_id, epoch, d_loss, g_loss, time_sec, cpu_pct, ram_pct, gpu_alloc_gb, gpu_resv_gb, timestamp
  - gen_metrics.csv
      columns: run_id, started_at, finished_at, elapsed_sec, checkpoint, n, seq_len, device

Run:
  python metrics/parse_logs.py
"""

import csv
import os
import re
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
STATIC_METRICS = os.path.join(BASE_DIR, "static", "metrics")

TRAIN_LOG = os.path.join(LOG_DIR, "train_log.txt")
GEN_LOG = os.path.join(LOG_DIR, "gen_log.txt")

os.makedirs(STATIC_METRICS, exist_ok=True)

epoch_re = re.compile(
    r"^\[(?P<ts>.+?)\]\s*Epoch\s+(?P<epoch>\d+)/(?P<total>\d+)\s*\|\s*D_loss=(?P<d>[-\d\.eE]+)\s*\|\s*G_loss=(?P<g>[-\d\.eE]+)\s*\|?\s*(?:Time=(?P<time>[\d\.]+)s)?(?:\s*\|\s*(?P<stats>.*))?$"
)
cpu_re = re.compile(r"CPU=(?P<val>[\d\.]+)%")
ram_re = re.compile(r"RAM=(?P<val>[\d\.]+)%")
gpu_alloc_re = re.compile(r"GPU alloc=(?P<val>[\d\.]+)GB")
gpu_resv_re = re.compile(r"GPU resv=(?P<val>[\d\.]+)GB")


def parse_train():
    rows = []
    run_id = 0
    if not os.path.exists(TRAIN_LOG):
        return rows
    with open(TRAIN_LOG, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("TRAIN started"):
                run_id += 1
                continue
            m = epoch_re.match(line)
            if not m:
                continue
            ts = m.group("ts")
            try:
                timestamp = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            except Exception:
                timestamp = ts
            stats_text = m.group("stats") or ""
            cpu = _extract(cpu_re, stats_text)
            ram = _extract(ram_re, stats_text)
            ga = _extract(gpu_alloc_re, stats_text)
            gr = _extract(gpu_resv_re, stats_text)
            rows.append(
                {
                    "run_id": run_id,
                    "epoch": int(m.group("epoch")),
                    "d_loss": float(m.group("d")),
                    "g_loss": float(m.group("g")),
                    "time_sec": float(m.group("time")) if m.group("time") else None,
                    "cpu_pct": cpu,
                    "ram_pct": ram,
                    "gpu_alloc_gb": ga,
                    "gpu_resv_gb": gr,
                    "timestamp": timestamp,
                }
            )
    return rows


def parse_gen():
    rows = []
    run_id = 0
    if not os.path.exists(GEN_LOG):
        return rows
    start_re = re.compile(r"^GEN started at (?P<ts>.+)$")
    finish_re = re.compile(r"^Generation finished at (?P<ts>.+)\s*\|\s*Elapsed time:\s*(?P<elapsed>[\d\.]+)")
    ckpt_re = re.compile(r"^  checkpoint:\s*(?P<ckpt>.+)$")
    n_re = re.compile(r"^  n:\s*(?P<n>\d+)$")
    seq_re = re.compile(r"^  seq_len:\s*(?P<seq>\d+)$")
    dev_re = re.compile(r"^  device:\s*(?P<dev>.+)$")

    cur = {}
    with open(GEN_LOG, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("GEN started"):
                run_id += 1
                m = start_re.match(line)
                cur = {"run_id": run_id, "started_at": m.group("ts") if m else None}
            elif line.startswith("Generated"):
                continue
            elif line.startswith("Generation finished"):
                m = finish_re.match(line)
                if m:
                    cur["finished_at"] = m.group("ts")
                    cur["elapsed_sec"] = float(m.group("elapsed"))
                rows.append(cur)
                cur = {}
            else:
                for regex, key in (
                    (ckpt_re, "checkpoint"),
                    (n_re, "n"),
                    (seq_re, "seq_len"),
                    (dev_re, "device"),
                ):
                    m = regex.match(line)
                    if m:
                        if m.groupdict():
                            # Use the first (only) named group
                            cur[key] = next(iter(m.groupdict().values()))
                        else:
                            cur[key] = m.group(1)
    return rows


def _extract(regex, text):
    m = regex.search(text)
    return float(m.group("val")) if m else None


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    train_rows = parse_train()
    gen_rows = parse_gen()

    if train_rows:
        write_csv(
            os.path.join(STATIC_METRICS, "train_metrics.csv"),
            train_rows,
            [
                "run_id",
                "epoch",
                "d_loss",
                "g_loss",
                "time_sec",
                "cpu_pct",
                "ram_pct",
                "gpu_alloc_gb",
                "gpu_resv_gb",
                "timestamp",
            ],
        )
        print(f"Wrote {len(train_rows)} train rows")
    else:
        print("No train rows found")

    if gen_rows:
        write_csv(
            os.path.join(STATIC_METRICS, "gen_metrics.csv"),
            gen_rows,
            ["run_id", "started_at", "finished_at", "elapsed_sec", "checkpoint", "n", "seq_len", "device"],
        )
        print(f"Wrote {len(gen_rows)} gen rows")
    else:
        print("No gen rows found")


if __name__ == "__main__":
    main()

