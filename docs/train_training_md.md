# Training Module (`train/training.py`)

## Purpose
Implements StackedGAN training loop, models, optimizers, early stopping, schedulers, and bagging hooks (when used by `main.py`).

## Highlights
- `StackedGAN` architecture with generator/discriminator/cond encoder; supports GC penalty and early stopping.
- `train_gan` (in `main.py`) coordinates batching, validation split, learning-rate schedulers, history tracking, and checkpoint save.
- Bagging support via `train/bagging.py` (ensemble training and saving).

## Data Flow
- Consumes tokenized sequences (CharTokenizer) and trains on CUDA if available.
- Outputs checkpoints under `checkpoints/` (or `checkpoints/bagging/` for ensembles).

## Rationale
- Centralized GAN training logic reused by CLI, comparison pipeline, and app generation flows.

