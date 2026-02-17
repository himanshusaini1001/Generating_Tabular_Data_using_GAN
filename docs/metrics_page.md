# Metrics Page (`templates/metrics.html`)

## Purpose
Static/interactive page to show key evaluation metrics for generated sequences and model performance snapshots.

## Contents
- GC/content visualizations, charts fed by `/metrics_json` (similar data shape as comparison page).
- Navigation hooks to other pages (generator, comparison).

## Data
- `/metrics_json` endpoint provides model metrics (GC_error, kmer_JS, Uniqueness, EditDist, MotifScore, GC_content, Precision, Recall).

## Rationale
- Gives a quick one-page summary without the full comparison dashboard complexity.

