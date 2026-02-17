# Application Server (`app.py`)

## Role
Flask app wiring routes, authentication, model loading, and API endpoints for generation, comparison, validation, and visual data services.

## Key Responsibilities
- **Auth**: Login/register/logout with username or +91 phone; password hashing; flash messaging.
- **Model Load**: Loads single StackedGAN or bagging ensemble checkpoints on startup (CUDA target).
- **Generation**: `/generate` POST returns sequences, GC content, FASTA.
- **Comparison**: Serves comparison page, `/metrics_json`, heatmap data, GC distribution, embedding PCA, correlation heatmaps.
- **Treatment Demo**: `/analyze_treatment` mock DNA edits per disease.
- **Validation/Verification**: Computes summary metrics over synthetic vs real slices.

## Data Paths
- `CSV_PATH` for comparison results, `GENERATED_DATA_PATH` & `SAMPLE_DATA_PATH` for sequence datasets.
- Logs: user activity, train/gen logs, loss plot.

## Rationale
- Central orchestration layer exposing both UI templates and JSON endpoints used by frontend pages and visualizations.

