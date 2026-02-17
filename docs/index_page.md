# Index Page Overview (`templates/index.html`)

## Purpose
The landing page (“Genomic Discovery Hub”) showcases the Stacked SeqGAN platform, highlights navigation to generator/metrics/comparison/treatment, and provides a 3D DNA hero plus feature callouts. It uses inline CSS for the hero layout and pulls Chart.js and Three.js for interactive bits (the DNA helix lives on `script.js`).

## Key Sections
- **Top Nav**: Links to Home, Generator, Metrics, Comparison, Treatment plus theme toggle and auth links.
- **Hero / Stats**: Gradient background, headline, call-to-action buttons, and a Three.js DNA canvas.
- **Feature Cards**: Cards summarizing DNA generation, metrics, validation, and treatment flows.
- **Metrics Preview**: Uses Chart.js (via `script.js`) to show compositional charts.
- **Footer / CTAs**: Links to comparison/treatment/register/login to drive engagement.

## Notable Assets
- **Chart.js CDN**: For lightweight chart previews.
- **Three.js CDN**: Powers the rotating DNA helix.
- **Fonts / Styling**: Space Grotesk and a dark theme palette defined in `:root`.

## Data Flow
- Static content; dynamic bits (charts, 3D DNA) are wired via `static/script.js`.
- No server calls from the index page itself; generation happens on the generator page.

## Rationale
- **Dark theme + gradients** to match a “lab” feel and emphasize neon DNA coloring.
- **Inline CSS** keeps the page self-contained for faster initial load.
- **CDN scripts** reduce bundle size and allow quick iteration without bundling.


