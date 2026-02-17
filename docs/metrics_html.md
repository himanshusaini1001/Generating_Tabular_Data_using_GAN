# Comparison Metrics API (`/metrics_json`)

## Purpose
Serve model comparison metrics as JSON for frontends (metrics/comparison pages).

## Source
- Backed by `comparison_results_gpu.csv` (written by `compare_models.py`).
- If CSV missing, recomputes metrics using sample data and saves CSV.

## Shape
Per model: GC_error, kmer_JS, Uniqueness, EditDist, MotifScore, GC_content, Precision, Recall, Accuracy, F1.

## Rationale
Lightweight API to populate charts/tables without recomputing heavy evaluations on each page load.

