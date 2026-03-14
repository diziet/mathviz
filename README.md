# MathViz

MathViz is a pipeline for programmatically generating 3D mathematical forms —
strange attractors, fractals, knots, minimal surfaces, and more — and preparing
them for subsurface laser engraving in crystal glass blocks. The output is a
wall-mounted installation of glass blocks arranged in a grid, each containing a
unique form rendered as a monochrome point cloud of micro-fractures.

## Features

- **48 generators** across 12 categories (attractors, fractals, knots, parametric surfaces, and more)
- **Linear pipeline**: Generate → Represent → Transform → Sample → Validate → Export
- **9 representation strategies** for controlling how forms appear when engraved
- **Deterministic output**: every form is reproducible from its seed
- **Grid manifest** for managing multi-block installations
- **Interactive 3D preview** via browser (Three.js + FastAPI)
- **High-resolution rendering** to PNG (3D and 2D projections)
- **Configurable containers**: glass block dimensions, margins, and placement policies
- **Sampling profiles**: preview (fast) and production (high-quality)
- **Format conversion**: STL, OBJ, PLY, XYZ, PCD

## Installation

Requires Python 3.11+.

```bash
pip install .
```

### Optional extras

```bash
# High-resolution 3D/2D rendering (requires PyVista)
pip install ".[render]"

# Development tools (pytest, ruff, httpx)
pip install ".[dev]"
```

## Quickstart

Generate a Lorenz attractor and export to PLY:

```bash
mathviz generate lorenz --output lorenz.ply
```

Preview interactively in the browser:

```bash
mathviz preview lorenz
```

List all available generators:

```bash
mathviz list
```

Get detailed info about a generator:

```bash
mathviz info lorenz
```

Override parameters and seed:

```bash
mathviz generate lorenz --param sigma=12 --param rho=30 --seed 7 --output lorenz.ply
```

Use a sampling profile for production quality:

```bash
mathviz generate gyroid --profile production --output gyroid.ply
```

## Generators

| Category | Generators | Description |
|---|---|---|
| attractors | `lorenz`, `rossler`, `chen`, `aizawa`, `thomas`, `halvorsen`, `double_pendulum` | Strange attractor trajectories |
| curves | `cardioid`, `fibonacci_spiral`, `lissajous_curve`, `logarithmic_spiral`, `parabolic_envelope` | Mathematical curves extended to 3D |
| data_driven | `building_extrude`, `heightmap`, `soundwave` | Forms derived from external data files |
| fractals | `fractal_slice`, `julia3d`, `mandelbrot_heightmap`, `mandelbulb` | 3D fractals and fractal heightmaps |
| geometry | `generic_parametric`, `voronoi_3d` | User-defined parametric surfaces and Voronoi |
| implicit | `gyroid`, `schwarz_d`, `schwarz_p`, `genus2_surface` | Triply periodic minimal surfaces via marching cubes |
| knots | `figure_eight_knot`, `lissajous_knot`, `seven_crossing_knots`, `torus_knot` | Mathematical knot curves |
| number_theory | `digit_encoding`, `prime_gaps`, `sacks_spiral`, `ulam_spiral` | Number-theoretic visualizations |
| parametric | `boy_surface`, `costa_surface`, `enneper_surface`, `klein_bottle`, `lissajous_surface`, `mobius_strip`, `spherical_harmonics`, `superellipsoid`, `torus` | Parametric surface meshes |
| physics | `kepler_orbit`, `nbody`, `planetary_positions` | Physics simulations |
| procedural | `noise_surface`, `reaction_diffusion`, `terrain` | Procedurally generated surfaces |

See [docs/generators.md](docs/generators.md) for full parameter tables and examples.

## CLI Commands

| Command | Description |
|---|---|
| `mathviz generate` | Run a generator through the full pipeline |
| `mathviz list` | List all available generators |
| `mathviz info` | Show generator details and parameter schema |
| `mathviz validate` | Generate and validate without exporting |
| `mathviz preview` | Start interactive 3D preview server |
| `mathviz render` | High-resolution 3D PNG rendering |
| `mathviz render-2d` | 2D projection rendering |
| `mathviz convert` | Convert geometry between formats |
| `mathviz sample` | Sample a mesh into a point cloud |
| `mathviz transform` | Fit geometry within a container |
| `mathviz schema` | Generate JSON Schema files from config models |
| `mathviz grid init` | Create a new grid manifest |
| `mathviz grid show` | Display the grid |
| `mathviz grid assign` | Assign a preset to a grid position |
| `mathviz grid status` | Show or update block status |
| `mathviz grid neighbors` | Show surrounding blocks |
| `mathviz grid summary` | Show counts by status |
| `mathviz grid export-all` | Batch export all assigned blocks |

See [docs/cli.md](docs/cli.md) for full flag reference and examples.

## Documentation

- [Generators](docs/generators.md) — all generator categories with parameter tables and examples
- [Pipeline](docs/pipeline.md) — the Generate → Represent → Transform → Sample → Validate → Export pipeline
- [CLI Reference](docs/cli.md) — every CLI command with flags, options, and examples
- [Configuration](docs/configuration.md) — config file format, precedence rules, sampling profiles
- [Representation Strategies](docs/representation.md) — how raw geometry is realized for engraving
- [Rendering](docs/rendering.md) — `render` and `render-2d` commands, optional dependencies
- [Grid Layout](docs/grid.md) — grid manifest format and grid CLI
- [Python API](docs/api.md) — using MathViz as a Python library

## Testing

```bash
pip install ".[dev]"
pytest
```

## License

See LICENSE file for details.
