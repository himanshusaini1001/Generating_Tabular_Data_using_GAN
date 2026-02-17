# Treatment Page (`templates/treatment.html`)

## Purpose
Demo flow to propose disease-specific DNA treatment edits with basic mock logic.

## Behavior
- Accepts disease selection, DNA sequence (optional), intensity.
- If sequence missing, generates a random 30-mer.
- Applies mock edits (positions & target mutations per disease), returns modified sequence and summary.

## Data / Logic
- Disease DB in `app.py` route: mutation positions, target bases, improvement ranges, descriptions.
- Endpoints: `/analyze_treatment` (POST, JSON) responds with modified sequence, changes list, effectiveness, and summary.

## UI
- Form for disease/intensity/sequence; shows results (original vs modified) and summary stats.

## Rationale
- Illustrative only; not a clinical tool. Shows how sequence transformations might be presented end-to-end.

