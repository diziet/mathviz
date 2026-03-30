# Static Demo Site

MathViz can build a self-contained static demo site that showcases generators
in an interactive 3D gallery. The output is a directory of HTML, JS, geometry
files (GLB/PLY), thumbnails, and a `manifest.json` — ready to deploy to
Cloudflare Pages or any static hosting provider.

## Building the demo

```bash
mathviz export-demo
```

This runs the pipeline for a curated set of ~15 visually impressive generators,
exports geometry and thumbnails, and assembles everything into `dist/`.

### Options

| Flag | Default | Description |
|---|---|---|
| `--generators` | curated list of 15 | Comma-separated generator names, or `all` |
| `--output` | `dist/` | Output directory |
| `--profile` | `preview` | Sampling profile (`preview` for fast builds, `production` for high quality) |
| `--verbose` | off | Enable debug logging |
| `--quiet` | off | Suppress non-error output |

### Examples

Build with specific generators:

```bash
mathviz export-demo --generators lorenz,gyroid,mandelbulb
```

Build all generators at production quality:

```bash
mathviz export-demo --generators all --profile production --output build/
```

## Previewing locally

After building, preview the site with any static file server:

```bash
cd dist && python -m http.server
```

Then open `http://localhost:8000` in your browser.

## Deploying to Cloudflare Pages

Deploy the built directory with Wrangler:

```bash
npx wrangler pages deploy dist/
```

This uploads the entire `dist/` directory as a Cloudflare Pages deployment.
You can also connect a Git repository for automatic deployments — see the
[Cloudflare Pages docs](https://developers.cloudflare.com/pages/).

## Customizing the generator list

The default generator list is a curated selection of visually interesting forms.
To customize it, pass a comma-separated list of generator names:

```bash
mathviz export-demo --generators lorenz,rossler,gyroid,torus_knot
```

To see all available generators:

```bash
mathviz list
```

To build with every generator (slow — processes all 89+):

```bash
mathviz export-demo --generators all
```

## Output structure

```
dist/
├── index.html          # Main gallery page
├── manifest.json       # Generator metadata array
├── buildbanner.js      # Build info banner
├── favicon.ico
├── demo-*.js           # Gallery JavaScript modules
└── data/
    ├── lorenz/
    │   ├── mesh.glb        # 3D mesh (GLB format)
    │   ├── cloud.ply       # Point cloud (binary PLY)
    │   └── thumbnail.png   # Preview thumbnail
    ├── gyroid/
    │   ├── mesh.glb
    │   ├── cloud.ply
    │   └── thumbnail.png
    └── ...
```

## Alternative: standalone script

The same build logic is available as a standalone script for use outside the
CLI (e.g. in CI):

```bash
python scripts/build_demo.py --generators lorenz,gyroid --output build/
```
