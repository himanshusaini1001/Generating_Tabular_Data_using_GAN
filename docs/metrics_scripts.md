# Metrics Scripts (`metrics/` folder)

## parse_logs.py
Parses `logs/train_log.txt` and `logs/gen_log.txt` into CSVs:
- `static/metrics/train_metrics.csv`: run_id, epoch, d_loss, g_loss, time_sec, cpu_pct, ram_pct, gpu_alloc_gb, gpu_resv_gb, timestamp
- `static/metrics/gen_metrics.csv`: run_id, started_at, finished_at, elapsed_sec, checkpoint, n, seq_len, device
Rationale: structured data for plotting/runtime dashboards.

## plot_losses.py
Plots D/G losses per epoch per run into `static/metrics/losses.png` (line/marker per run).

## plot_time_cpu.py
Generates:
- `epoch_time.png` (epoch duration area/line)
- `cpu_ram.png` (CPU/RAM utilization lines)
- `gpu_alloc.png` (GPU mem bars)

## plot_gen_times.py
Bar chart of generation elapsed per run → `gen_times.png`.

## plot_losses_3d.py
3D scatter of epoch vs D_loss vs G_loss → `losses_3d.png`.

## plot_gpu_time_3d.py
3D scatter of epoch vs GPU alloc vs time per epoch → `gpu_time_3d.png`.

## Rationale
Separates parsing from visualization; keeps PNGs consumable by the comparison page (runtime & 3D tabs).

