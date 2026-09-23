# CLI Reference

The `mathviz` command is the MathViz command-line interface. Every command
accepts `--help`, which prints its usage and options.

## generate

Runs a generator through the full pipeline (Generate → Represent → Transform →
Sample → Validate → Export).

```bash
mathviz generate <generator_name> [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--param` | key=value | | Generator parameter (repeatable) |
| `--seed` | int | 42 | Random seed for deterministic output |
| `--output` | path | | Output file path (triggers export) |
| `--format` | string | | Export format override |
| `--config` | path | | Per-object TOML config file |
| `--profile` | string | | Sampling profile name (preview, production, custom) |
| `--width` | float | 100.0 | Container width in mm |
| `--height` | float | 100.0 | Container height in mm |
| `--depth` | float | 100.0 | Container depth in mm |
| `--dry-run` | flag | | Show what would happen without running |
| `--report` | path | | Write JSON report to file |
| `--json` | flag | | Output as JSON |
| `--verbose` | flag | | Enable debug logging |
| `--quiet` | flag | | Suppress non-error output |

Examples:

```bash
# Basic generation with export
mathviz generate lorenz --output lorenz.ply

# Custom parameters and seed
mathviz generate lorenz --param sigma=12 --param rho=30 --seed 7 --output lorenz.ply

# Production quality with sampling profile
mathviz generate gyroid --profile production --output gyroid.ply

# Custom container dimensions
mathviz generate torus --width 120 --height 120 --depth 60 --output torus.ply

# Dry run to preview what would happen
mathviz generate mandelbulb --output mandelbulb.ply --dry-run

# JSON output for scripting
mathviz generate lorenz --output lorenz.ply --json

# Per-object config file
mathviz generate lorenz --config block_config.toml --output lorenz.ply
```

## list

Lists all available generators.

```bash
mathviz list [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--json` | flag | | Output as JSON |

Examples:

```bash
mathviz list
mathviz list --json
```

## info

Shows a generator's details and parameter schema.

```bash
mathviz info <generator_name> [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--json` | flag | | Output as JSON |

Examples:

```bash
mathviz info lorenz
mathviz info gyroid --json
```

## benchmark

Runs generators through the pipeline and writes an HTML report of per-stage
timings.

```bash
mathviz benchmark [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--generators` | string | all non-data-driven | Comma-separated list of generators to benchmark |
| `--workers` | int | CPU count | Number of parallel workers |
| `--output` | path | benchmark_report.html | Output HTML report file path |
| `--runs` | int | 3 | Number of runs per generator for averaging |
| `--verbose` | flag | | Enable debug logging |
| `--quiet` | flag | | Suppress non-error output |

Examples:

```bash
# Benchmark all generators
mathviz benchmark

# Benchmark specific generators
mathviz benchmark --generators lorenz,torus,gyroid

# Single run with custom output
mathviz benchmark --runs 1 --output results.html

# Limit parallelism
mathviz benchmark --workers 2
```

## export-demo

Builds a self-contained static demo site from generator pipeline output. The
site contains GLB meshes, PLY point clouds, PNG thumbnails, and an interactive
gallery viewer.

```bash
mathviz export-demo [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--generators` | string | curated list of 15 | Comma-separated generator names, or 'all' |
| `--output` | path | dist/ | Output directory for the demo site |
| `--profile` | string | preview | Sampling profile name (e.g. preview, production) |
| `--verbose` | flag | | Enable debug logging |
| `--quiet` | flag | | Suppress non-error output |

Examples:

```bash
# Build demo with default curated generators
mathviz export-demo

# Build demo with specific generators
mathviz export-demo --generators lorenz,gyroid,mandelbulb

# Custom output directory and production quality
mathviz export-demo --output build/ --profile production

# Build all generators
mathviz export-demo --generators all
```

See [demo.md](demo.md) for deployment and customization.

## cache clear

Removes all cached generation results from the disk cache. The cache stores
serialized geometry files (GLB, PLY), so an identical request does not run the
pipeline again.

```bash
mathviz cache clear [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--verbose` | flag | | Enable debug logging |
| `--quiet` | flag | | Suppress non-error output |

Example:

```bash
mathviz cache clear
```

## validate

Generates and validates without exporting. Runs the pipeline through the
Validate stage and reports each check as pass or fail.

```bash
mathviz validate <generator_name> [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--param` | key=value | | Generator parameter (repeatable) |
| `--seed` | int | 42 | Random seed |
| `--config` | path | | Per-object TOML config file |
| `--profile` | string | | Sampling profile name |
| `--width` | float | 100.0 | Container width in mm |
| `--height` | float | 100.0 | Container height in mm |
| `--depth` | float | 100.0 | Container depth in mm |
| `--json` | flag | | Output as JSON |
| `--verbose` | flag | | Enable debug logging |
| `--quiet` | flag | | Suppress non-error output |

Examples:

```bash
mathviz validate lorenz
mathviz validate gyroid --json
```

## preview

Starts the interactive 3D preview server and opens a browser window with a
Three.js viewer served by FastAPI.

```bash
mathviz preview <target> [OPTIONS]
```

The target is a generator name or the path of a file to preview.

| Option | Type | Default | Description |
|---|---|---|---|
| `--param` | key=value | | Generator parameter (repeatable) |
| `--seed` | int | 42 | Random seed |
| `--port` | int | 8000 | Server port |
| `--no-open` | flag | | Don't open browser automatically |
| `--verbose` | flag | | Enable debug logging |
| `--quiet` | flag | | Suppress non-error output |

Examples:

```bash
# Preview a generator
mathviz preview lorenz

# Preview with custom parameters
mathviz preview lorenz --param sigma=12 --seed 7

# Preview an exported file
mathviz preview output.ply

# Custom port, no auto-open
mathviz preview gyroid --port 9000 --no-open
```

## render

Renders a generator to a high-resolution PNG image. Requires the `[render]`
optional dependency (PyVista).

```bash
mathviz render <generator_name> --output <path.png> [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--output`, `-o` | path | (required) | Output PNG file path |
| `--param` | key=value | | Generator parameter (repeatable) |
| `--seed` | int | 42 | Random seed |
| `--width` | int | 1920 | Image width in pixels |
| `--height` | int | 1080 | Image height in pixels |
| `--verbose` | flag | | Enable debug logging |
| `--quiet` | flag | | Suppress non-error output |

Examples:

```bash
mathviz render lorenz -o lorenz.png
mathviz render gyroid -o gyroid.png --width 3840 --height 2160
```

## render-2d

Renders a 2D projection of a generator to PNG. Requires the `[render]` optional
dependency (PyVista).

```bash
mathviz render-2d <generator_name> --output <path.png> [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--output`, `-o` | path | (required) | Output PNG file path |
| `--view` | string | top | Projection view: top, front, side, angle |
| `--param` | key=value | | Generator parameter (repeatable) |
| `--seed` | int | 42 | Random seed |
| `--width` | int | 1920 | Image width in pixels |
| `--height` | int | 1080 | Image height in pixels |
| `--verbose` | flag | | Enable debug logging |
| `--quiet` | flag | | Suppress non-error output |

Examples:

```bash
mathviz render-2d lorenz -o lorenz_top.png --view top
mathviz render-2d gyroid -o gyroid_front.png --view front
mathviz render-2d mandelbulb -o mandelbulb_angle.png --view angle
```

## render-all

Renders every generator in several views, in parallel. The output directory
has one subdirectory per generator. Requires the `[render]` optional dependency
(PyVista).

```bash
mathviz render-all [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--output-dir` | path | renders/ | Base output directory |
| `--views` | string | top,front,side,angle | Comma-separated views to render |
| `--workers` | int | CPU count | Number of parallel workers |
| `--generators` | string | all non-data-driven | Comma-separated list of generators |
| `--style` | string | <!-- fact:render-all-style-default -->vertex<!-- /fact --> | Render style: <!-- fact:render-styles -->shaded, wireframe, vertex<!-- /fact --> |
| `--width` | int | 1920 | Image width in pixels |
| `--height` | int | 1080 | Image height in pixels |
| `--verbose` | flag | | Enable debug logging |
| `--quiet` | flag | | Suppress non-error output |

Examples:

```bash
# Render all generators with default views
mathviz render-all

# Render specific generators
mathviz render-all --generators lorenz,torus,gyroid

# Single view, sequential processing
mathviz render-all --views top --workers 1

# Custom output directory and style
mathviz render-all --output-dir gallery/ --style shaded
```

## render-thumbnail

Generates a WebP thumbnail for one generator, or for all generators with
`--all`. Thumbnails are cached at
`~/.mathviz/thumbnails/<view_mode>/<name>.webp`. The preview server runs this
command in a subprocess, because VTK crashes on macOS when it renders outside
the main thread.

```bash
mathviz render-thumbnail [NAME] [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--view-mode` | string | <!-- fact:thumbnail-view-mode-default -->vertex<!-- /fact --> | View mode: <!-- fact:thumbnail-view-modes -->vertex, shaded, wireframe<!-- /fact -->. `points` exits with code 2 (checked 2026-09-23). |
| `--all` | flag | | Generate thumbnails for all generators |
| `--verbose` | flag | | Enable debug logging |
| `--quiet` | flag | | Suppress non-error output |

Examples:

```bash
# Generate thumbnail for a single generator
mathviz render-thumbnail torus

# Pre-generate all thumbnails
mathviz render-thumbnail --all

# Generate shaded-style thumbnails
mathviz render-thumbnail --all --view-mode shaded
```

## convert

Converts geometry between file formats: STL, OBJ, PLY, XYZ, and PCD.

```bash
mathviz convert <input_path> <output_path> [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--auto-sample` | flag | | Auto-sample mesh to point cloud if converting from mesh to cloud format |
| `--format` | string | | Output format override (inferred from extension by default) |
| `--num-points` | int | | Point count for auto-sample |
| `--verbose` | flag | | Enable debug logging |
| `--quiet` | flag | | Suppress non-error output |

Examples:

```bash
# Convert STL to OBJ
mathviz convert model.stl model.obj

# Convert mesh to point cloud with auto-sampling
mathviz convert model.stl cloud.ply --auto-sample --num-points 100000

# Explicit format override
mathviz convert input.dat output.dat --format ply
```

## sample

Samples a mesh into a point cloud.

```bash
mathviz sample <input_path> <output_path> [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--method` | string | uniform_surface | Sampling method: uniform_surface, random_surface, volume_fill |
| `--num-points` | int | | Target point count |
| `--density` | float | | Points per mm² |
| `--seed` | int | 42 | RNG seed |
| `--format` | string | | Output format override |
| `--verbose` | flag | | Enable debug logging |
| `--quiet` | flag | | Suppress non-error output |

Examples:

```bash
# Sample with uniform distribution
mathviz sample model.stl cloud.ply --num-points 50000

# Sample with density target
mathviz sample model.obj cloud.ply --density 8.0

# Volume fill sampling
mathviz sample model.stl filled.ply --method volume_fill --num-points 100000
```

## transform

Fits geometry within a container's bounding box.

```bash
mathviz transform <input_path> <output_path> [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--width` | float | 100.0 | Container width in mm |
| `--height` | float | 100.0 | Container height in mm |
| `--depth` | float | 100.0 | Container depth in mm |
| `--format` | string | | Output format override |
| `--verbose` | flag | | Enable debug logging |
| `--quiet` | flag | | Suppress non-error output |

Examples:

```bash
# Fit to default container (100x100x100mm)
mathviz transform model.stl fitted.stl

# Fit to custom container
mathviz transform model.stl fitted.stl --width 120 --height 120 --depth 60
```

## schema

Writes JSON Schema files for all configuration models: Container,
PlacementPolicy, SamplerConfig, RepresentationConfig and EngravingProfile, plus a
parameter schema for each generator that defines one.

```bash
mathviz schema <output_dir> [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--verbose` | flag | | Enable debug logging |
| `--quiet` | flag | | Suppress non-error output |

Examples:

```bash
mathviz schema schemas/
```

Output structure:

```
schemas/
├── Container.json
├── PlacementPolicy.json
├── SamplerConfig.json
├── RepresentationConfig.json
├── EngravingProfile.json
└── generators/
    ├── lorenz.json
    ├── gyroid.json
    └── ...
```

## grid

The grid commands manage the installation grid manifest. Every grid subcommand
reads or writes a TOML manifest file, `grid.toml` by default.

See [grid.md](grid.md) for details.

### grid init

Creates a new grid manifest.

```bash
mathviz grid init <rows> <cols> [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--path` | path | grid.toml | Manifest file path |
| `--json` | flag | | Output as JSON |

### grid show

Prints the grid as an ASCII table or as JSON.

```bash
mathviz grid show [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--path` | path | grid.toml | Manifest file path |
| `--json` | flag | | Output as JSON |

### grid assign

Assigns a generator preset to a grid position.

```bash
mathviz grid assign <row> <col> <preset> [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--config` | string | | Per-block config path |
| `--path` | path | grid.toml | Manifest file path |
| `--json` | flag | | Output as JSON |

### grid status

Shows or sets the status of a grid block.

```bash
mathviz grid status <row> <col> [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--set` | string | | Set status: assigned, exported, error, empty |
| `--path` | path | grid.toml | Manifest file path |
| `--json` | flag | | Output as JSON |

### grid neighbors

~~Show the 8 surrounding blocks for a position.~~ Shows up to 8 surrounding
blocks for a position, and fewer at an edge or corner (checked 2026-09-23).

```bash
mathviz grid neighbors <row> <col> [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--path` | path | grid.toml | Manifest file path |
| `--json` | flag | | Output as JSON |

### grid summary

Shows the number of blocks in each status.

```bash
mathviz grid summary [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--path` | path | grid.toml | Manifest file path |
| `--json` | flag | | Output as JSON |

### grid export-all

Runs each block that has a preset and status `assigned` or `error` through the
pipeline, one block at a time.

```bash
mathviz grid export-all [OPTIONS]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `--path` | path | grid.toml | Manifest file path |
| `--output-dir` | path | export | Output directory |
| `--format` | string | | Export format |
| `--json` | flag | | Output as JSON |

Examples:

```bash
# Create a 10x10 grid
mathviz grid init 10 10

# Assign generators to positions
mathviz grid assign 0 0 lorenz
mathviz grid assign 0 1 gyroid
mathviz grid assign 1 0 mandelbulb

# View the grid
mathviz grid show

# Export all assigned blocks
mathviz grid export-all --output-dir output/
```
