# Model Comparison Pipeline (`compare_models.py`)

## Purpose
Train/evaluate multiple GAN variants (WGAN, CTGAN, CGAN, CramerGAN, DraGAN, StackedGAN) against real data and emit comparison metrics.

## Workflow
1) Load dataset via `load_dataset` (from `main`).
2) Train each model variant (simple LSTM-based baselines and StackedGAN).
3) Generate synthetic samples per model.
4) Compute metrics vs real: GC_error, kmer_JS, Uniqueness, EditDist, MotifScore, GC_content, Precision, Recall.
5) Save results to CSV (`comparison_results_gpu.csv`) and return dict for UI/JSON.

## Metrics Helpers
- GC content, k-mer distribution + JS divergence, uniqueness ratio, average edit distance, motif score.
- Precision/Recall (custom): precision = realistic fraction (valid bases + GC within tolerance), recall = exp(-JS divergence).

## Rationale
- Provides a consistent benchmark across multiple GAN variants; outputs drive the comparison page and `/metrics_json`.

