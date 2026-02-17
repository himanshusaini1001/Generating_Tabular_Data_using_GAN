# Frontend Generator Logic (`static/script.js`)

## Overview
Client-side controller for the generator page: calls `/generate`, renders composition charts, computes a quick quality score, visualizes a 3D helix, and lets users download FASTA/CSV.

## DOM Bindings
- Buttons: `generate_btn`, `download_fasta`, `download_csv`
- Outputs: `output` (text area), `gc_label`, `gc_progress`
- Charts: `baseChart` (overall synthetic composition), `seqChart` (per-sequence composition), `comparisonChart` (synthetic vs mock original)
- Selectors: `sequence_dropdown` to pick a generated sequence
- Quality: `quality_score`
- 3D Canvas: `dna3d` (Three.js helix)

## State
- `latestFasta`, `latestSequences`: cached generation results.
- Chart handles: `baseChart`, `seqChart`, `comparisonChart` (destroyed/recreated to avoid Chart.js leaks).

## Functions
### `updateChart(a,t,g,c)`
- Purpose: Overall synthetic composition (% A/T/G/C) to spot bias.
- Why destroy first: avoid stacked canvases/memory leaks in Chart.js.

### `updateSeqChart(sequence)`
- Purpose: Per-sequence composition to catch outliers vs overall mix.

### `updateComparisonChart(syntheticData)`
- Purpose: Compare synthetic to a mock “original” baseline (placeholder until wired to real training stats).
- Why mock: lets UI work before server provides real composition; easy swap later.

### `calculateQualityScore(syntheticData)`
- Heuristic: sums absolute diffs vs mock original, maps to 0–100 (lower diff → higher score). UX hint, not a scientific metric.

### 3D DNA (`initDNA3D`, `animateDNA`, `drawDNA`)
- `initDNA3D`: Sets up Three.js scene/camera/lights/renderer and a `helixGroup`. Clears old canvas to avoid duplicates.
- `animateDNA`: Render loop with gentle Y-rotation.
- `drawDNA(sequence)`: Clears old meshes; for each base, adds two colored cylinders (base + complement) on opposite helix sides; disposes old geometry/material to avoid GPU leaks.

### Generate flow (`generateBtn.onclick`)
1) POST `/generate` with requested count.
2) Receive `{fasta, sequences, gc_content}`.
3) Populate dropdown and text with GC annotations.
4) Aggregate composition → `updateChart`; compute quality.
5) Render first sequence in 3D (`drawDNA`) and per-sequence chart (`updateSeqChart`).
6) Reveal hidden UI panels.

### Dropdown handler
- On change, re-render selected sequence in 3D and per-sequence chart.

### Downloads
- FASTA: Blob from `latestFasta` → `generated_sequences.fasta`.
- CSV: ID/Sequence pairs from `latestSequences`.
- Why client-side: avoids extra endpoints; uses already-fetched data.

### Init
- `DOMContentLoaded` → `initDNA3D()` to prep the scene before interaction.

## Design Choices & Rationale
- Destroy/recreate charts: prevents Chart.js leaks/overlays.
- Mock “original” distro: keeps UI functional pre-backend wiring.
- Simple 3D helix: low GPU load, clear visual cue.
- Client GC check: instant feedback; server remains authority.
- Blob downloads: fast UX, no extra APIs.

## Extension Ideas
- Bind `comparisonChart` to real training composition from backend.
- Add per-base tooltips; confidence bands if available.
- Add loaders/error toasts around `/generate`.
- FPS throttle/pause 3D on tab blur for low-power devices.


