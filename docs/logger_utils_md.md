# Logging Utilities (`logger_utils.py`)

## Purpose
Lightweight logger for training/generation with optional system stats and loss plotting.

## Features
- Timestamped logs to console and `logs/{mode}_log.txt`.
- Epoch logging (D/G losses, epoch time) with CPU/RAM and optional GPU mem (if available).
- Free-form `log_message` with stats appended.
- Loss plotting to `logs/loss_plot.png` for training runs.
- Sequence saver helper for generation outputs.

## Rationale
- Centralized, minimal dependency (psutil optional) logger that feeds downstream CSV parsing and runtime dashboards.

