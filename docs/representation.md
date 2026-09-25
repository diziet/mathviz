# Representation Strategies

A representation strategy controls how a generator's raw geometry becomes the
geometry that is laser-engraved in glass. The representation layer keeps what
the math produces separate from how it looks engraved, so generators contain no
fabrication logic.

Each generator has a default representation strategy. Configuration can
override it.

## Strategies

### surface_shell

Represents the geometry as a hollow surface shell. The outermost surface of the
mesh becomes a thin layer of engraving points along the shape's boundary.

Best for: closed surfaces, manifolds, parametric surfaces.

**Config options:**
- `shell_thickness` — ~~thickness of the shell layer~~ No pipeline code reads
  this option, so it has no effect (checked 2026-09-23).

### tube

Thickens curve geometry (attractors, knots, spirals) into a tubular mesh. The
tube's cylindrical surface can then be sampled for engraving.

Best for: attractor trajectories, knot curves, spirals.

**Config options:**
- `tube_radius` — radius of the tube cross-section
- `tube_sides` — number of polygon sides in the tube cross-section (default: 16)

### raw_point_cloud

Uses the generator's points directly, without mesh conversion. Each raw point
becomes an engraving position unchanged.

Best for: point-based generators (number theory, digit encoding) and other
generators that produce point data.

### volume_fill

Fills the interior volume of a mesh with evenly distributed points. The
engraving looks solid instead of showing only a surface shell.

Best for: simple convex shapes, fractals where interior structure matters.

**Config options:**
- `volume_density` — density of interior points

### sparse_shell

Like surface_shell, but with a lower point density. The engraving looks sparser
and more transparent in the glass.

The sampled points depend on the seed (`--seed`, default 42). The same mesh and seed give the
same points, and a different seed gives different points. This also applies to generators whose
mesh does not depend on the seed, such as julia3d and mandelbulb. A point-cloud input passes
through unchanged.

Best for: surfaces that should look lighter and less dense.

**Config options:**
- `surface_density` — density of surface points (lower = sparser)

### slice_stack

Cuts the geometry with a series of parallel cross-sectional planes along one
axis. Each slice becomes a 2D contour of engraving points.

Best for: showing internal structure as a stack of slices, like a CT scan.

**Config options:**
- `slice_count` — number of cross-sectional slices
- `slice_axis` — axis to slice along: `"x"`, `"y"`, or `"z"` (default: `"z"`)

### wireframe

Extracts the mesh edges and thickens each one into a thin tube. Only the edges
of the geometry are engraved.

Best for: geometric forms whose edges are the subject (Voronoi, polyhedra).

**Config options:**
- `wireframe_thickness` — line thickness for wireframe edges

### weighted_cloud

~~Like raw_point_cloud but with a density weighting function that varies point
density across the shape. Creates gradient effects and emphasis regions.~~ The
`weighted_cloud` handler passes the generator's point cloud through unchanged
and keeps its per-point intensities (checked 2026-09-23).

Best for: emphasizing specific features, deliberate variation in density.

**Config options:**
- `density_weight_function` — ~~mathematical expression for density weighting~~
  No pipeline code reads this option, so it has no effect (checked 2026-09-23).

### heightmap_relief

Treats the geometry as a heightmap and builds a relief surface from it. The z
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

Or in a per-object config passed with `--config`:

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

The `[representation]` section applies to `mathviz generate`, `mathviz validate`, `mathviz render`,
`mathviz render-2d` and `mathviz grid export-all`. `mathviz render` and `mathviz render-2d` read
it only from `mathviz.toml`, because they have no `--config` option. A key in a per-object config
overrides the same key in `mathviz.toml`.

The section replaces the generator's default strategy, including its default tube radius. A
section with `type = "tube"` therefore needs `tube_radius`.

MathViz validates the section as a `RepresentationConfig` before it runs the generator. A missing
or unknown `type`, or an invalid value such as `tube_sides = 0`, stops the command with exit code
2 and an error that names the field. `mathviz grid export-all` marks that block as `error` and
exports the other blocks.

## Default Strategies

Without a representation config, MathViz uses the generator's default strategy.
Curve generators (attractors, knots, spirals) default to `tube`. Surface
generators default to `surface_shell`. Point-based generators default to
`raw_point_cloud`.

The default tube radius of `borromean_rings` is its `ring_thickness` parameter, and the default
tube radius of `chain_links` is its `link_thickness` parameter.
