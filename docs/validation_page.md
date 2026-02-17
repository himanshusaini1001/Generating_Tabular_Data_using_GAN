# Validation Page (`templates/validation.html`)

## Purpose
Shows sanity checks comparing synthetic sequences to real training windows.

## Metrics (computed server-side in `app.py`)
- GC mean diff vs real.
- Base frequency JS divergence.
- 2-mer JS divergence.
- Length stats (min/max/mean/n50) for real vs synth.
- Status: PASS/WARN based on thresholds.

## UI
- Displays metrics table/values and pass/warn flags; intended for quick health-check of generated data realism.

## Rationale
- Lightweight validation to ensure synthetic windows resemble training distribution before downstream use.

