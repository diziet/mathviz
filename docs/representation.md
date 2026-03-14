# Representation Strategies

Representation strategies control how raw mathematical geometry is realized for
laser engraving in glass. The representation layer separates "what the math
produces" from "how it looks engraved," keeping generators free of fabrication
concerns.

Each generator has a default representation strategy. You can override it via
configuration.

## Strategies

### surface_shell

Renders geometry as a hollow surface shell. The outermost surface of the mesh
becomes a thin layer of engraving points that trace the shape's boundary.

Best for: closed surfaces, manifolds, parametric surfaces.

**Config options:**
- `shell_thickness` — thickness of the shell layer

### tube

Wraps curve-like geometry (attractors, knots, spirals) in a tubular mesh. The
curve is thickened into a cylindrical surface that can be sampled for engraving.

Best for: attractor trajectories, knot curves, spirals.

**Config options:**
- `tube_radius` — radius of the tube cross-section
- `tube_sides` — number of polygon sides in the tube cross-section (default: 16)

### raw_point_cloud

Uses the generator's point cloud output directly without mesh conversion. The
raw points become engraving positions as-is.

Best for: point-based generators (number theory, digit encoding), generators
that natively produce cloud data.

### volume_fill

Fills the interior volume of a mesh with evenly distributed points. Creates a
solid-looking engraving rather than a surface shell.

Best for: simple convex shapes, fractals where interior structure matters.

**Config options:**
- `volume_density` — density of interior points

### sparse_shell

Similar to surface_shell but with reduced density, creating a sparser, more
transparent appearance in the glass.

Best for: surfaces where a lighter, more ethereal look is desired.

**Config options:**
- `surface_density` — density of surface points (lower = sparser)

### slice_stack

Slices the geometry into a series of cross-sectional planes stacked along an
axis. Each slice becomes a 2D contour of engraving points.

Best for: revealing internal structure, CT-scan-like visualization.

**Config options:**
- `slice_count` — number of cross-sectional slices
- `slice_axis` — axis to slice along: `"x"`, `"y"`, or `"z"` (default: `"z"`)

### wireframe

Extracts mesh edges and renders them as thin lines. Only the structural edges
of the geometry are engraved, creating a skeletal appearance.

Best for: geometric forms where edge structure is the focus (Voronoi, polyhedra).

**Config options:**
- `wireframe_thickness` — line thickness for wireframe edges

### weighted_cloud

Like raw_point_cloud but with a density weighting function that varies point
density across the shape. Creates gradient effects and emphasis regions.

Best for: highlighting specific features, artistic density variation.

**Config options:**
- `density_weight_function` — mathematical expression for density weighting

### heightmap_relief

Interprets geometry as a heightmap and renders it as a relief surface. The z
values of a 2D grid become engraving depths.

Best for: terrain, heightmap-based generators, Mandelbrot visualizations.

## Configuration

Set the representation strategy in a TOML config file:

```toml
[representation]
type = "tube"
tube_radius = 0.15
tube_sides = 24
```

Or in a per-object config passed via `--config`:

```bash
mathviz generate lorenz --config attractor_config.toml --output lorenz.ply
```

Example config for slice_stack:

```toml
[representation]
type = "slice_stack"
slice_count = 50
slice_axis = "z"
```

## Default Strategies

When no representation config is provided, MathViz selects a default strategy
based on the generator. Curve-like generators (attractors, knots, spirals)
default to `tube`. Surface generators default to `surface_shell`. Point-based
generators default to `raw_point_cloud`.
