# Generator Page (`templates/generator.html`)

## Purpose
UI to generate synthetic DNA sequences via the backend `/generate` endpoint, visualize composition, preview a 3D helix, and export results.

## Flow
1) User enters number of sequences and submits.
2) Frontend POSTs to `/generate`.
3) Response: sequences, fasta string, GC content.
4) Page renders charts, quality info, per-sequence dropdown, and 3D helix.

## Key Elements
- Inputs: number of sequences.
- Buttons: Generate, Download FASTA, Download CSV.
- Outputs: text block with sequences & GC annotation, GC meter, quality score.
- Charts: overall composition, per-sequence composition, comparison vs mock original.
- 3D: DNA helix rendered with Three.js via `static/script.js`.

## Data / APIs
- `/generate` (POST) — returns generated sequences and fasta; uses StackedGAN/bagging depending on server.

## Rationale
- Quick single-form UX for generation.
- Immediate visual feedback (charts + GC gauge + helix).
- Client-side downloads avoid extra server roundtrips.

