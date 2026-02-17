# Verification Page (`templates/verification.html`)

## Purpose
Basic checks to ensure generated sequences are clean and within expected bounds.

## Metrics (server-side)
- Count of sequences, length min/max/mean.
- GC mean/std.
- Invalid base fraction.
- PASS/WARN flags for length, GC range, invalid characters.

## UI
- Presents these checks for a quick “is my synthetic data sane?” gate before other tasks.

## Rationale
- Early warning system to catch obvious data issues (length drift, invalid characters, GC outliers) without deep analysis.

