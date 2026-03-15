# Generators

MathViz includes 78 generators across 12 categories. Each generator produces a
deterministic 3D mathematical form from a seed and a set of parameters.

## Attractors

Strange attractor trajectories computed by integrating dynamical systems.

### lorenz

Lorenz strange attractor trajectory.

| Parameter | Default | Description |
|---|---|---|
| `sigma` | 10.0 | Prandtl number |
| `rho` | 28.0 | Rayleigh number |
| `beta` | 2.667 | Geometric factor |
| `transient_steps` | 1000 | Initial transient steps to discard |

Aliases: `lorenz_attractor`

```bash
mathviz generate lorenz --output lorenz.ply
mathviz generate lorenz --param sigma=12 --param rho=30 --output lorenz.ply
```

### rossler

Rössler strange attractor trajectory.

| Parameter | Default | Description |
|---|---|---|
| `a` | 0.2 | System parameter a |
| `b` | 0.2 | System parameter b |
| `c` | 5.7 | System parameter c |
| `transient_steps` | 1000 | Initial transient steps to discard |

Aliases: `rossler_attractor`

```bash
mathviz generate rossler --output rossler.ply
```

### chen

Chen double-scroll strange attractor trajectory.

| Parameter | Default | Description |
|---|---|---|
| `a` | 35.0 | System parameter a |
| `b` | 3.0 | System parameter b |
| `c` | 28.0 | System parameter c |
| `transient_steps` | 1000 | Initial transient steps to discard |

Aliases: `chen_attractor`

```bash
mathviz generate chen --output chen.ply
```

### aizawa

Aizawa torus-like strange attractor trajectory.

| Parameter | Default | Description |
|---|---|---|
| `a` | 0.95 | System parameter a |
| `b` | 0.7 | System parameter b |
| `c` | 0.6 | System parameter c |
| `d` | 3.5 | System parameter d |
| `e` | 0.25 | System parameter e |
| `f` | 0.1 | System parameter f |
| `transient_steps` | 1000 | Initial transient steps to discard |

Aliases: `aizawa_attractor`

```bash
mathviz generate aizawa --output aizawa.ply
```

### thomas

Thomas cyclically symmetric strange attractor trajectory.

| Parameter | Default | Description |
|---|---|---|
| `b` | 0.208186 | Dissipation constant |
| `transient_steps` | 1000 | Initial transient steps to discard |

Aliases: `thomas_attractor`

```bash
mathviz generate thomas --output thomas.ply
```

### halvorsen

Halvorsen three-winged strange attractor trajectory.

| Parameter | Default | Description |
|---|---|---|
| `a` | 1.89 | System parameter a |
| `transient_steps` | 1000 | Initial transient steps to discard |

Aliases: `halvorsen_attractor`

```bash
mathviz generate halvorsen --output halvorsen.ply
```

### double_pendulum

Double pendulum chaotic trajectory (4D phase space projected to 3D).

| Parameter | Default | Description |
|---|---|---|
| `mass` | 1.0 | Pendulum mass |
| `length` | 1.0 | Pendulum length |
| `gravity` | 9.81 | Gravitational acceleration |
| `theta1` | 2.5 | Initial angle of first pendulum (radians) |
| `theta2` | 2.0 | Initial angle of second pendulum (radians) |
| `omega1` | 0.0 | Initial angular velocity of first pendulum |
| `omega2` | 0.0 | Initial angular velocity of second pendulum |
| `transient_steps` | 500 | Initial transient steps to discard |

Aliases: `double_pendulum_attractor`

```bash
mathviz generate double_pendulum --param theta1=3.0 --output pendulum.ply
```

### clifford

Clifford 2D iterated-map attractor (point cloud). Extended to 3D via scaled
iteration count as z-coordinate.

| Parameter | Default | Description |
|---|---|---|
| `a` | -1.4 | Map parameter a |
| `b` | 1.6 | Map parameter b |
| `c` | 1.0 | Map parameter c |
| `d` | 0.7 | Map parameter d |

Resolution: `num_points` (default: 500000)

Aliases: `clifford_attractor`

Output varies with seed (random initial condition).

Recommended representation: SPARSE_SHELL

```bash
mathviz generate clifford --output clifford.ply
mathviz generate clifford --param a=-1.7 --param b=1.3 --seed 7 --output clifford.ply
```

### dequan_li

Dequan Li multi-scroll chaotic attractor trajectory.

| Parameter | Default | Description |
|---|---|---|
| `a` | 40.0 | System parameter a |
| `c` | 1.833 | System parameter c |
| `d` | 0.16 | System parameter d |
| `e` | 0.65 | System parameter e |
| `f` | 20.0 | System parameter f |
| `k` | 55.0 | System parameter k |
| `transient_steps` | 1000 | Initial transient steps to discard |

Resolution: `integration_steps` (default: 100000)

Aliases: `dequan_li_attractor`

Output varies with seed (initial condition perturbation).

Recommended representation: TUBE

```bash
mathviz generate dequan_li --output dequan_li.ply
```

### sprott

Sprott minimal chaotic flows (multiple variants). Selectable via the `system`
parameter. Each variant is a 3D ODE with very few terms producing visually
distinct attractors.

| Parameter | Default | Description |
|---|---|---|
| `system` | `"sprott_a"` | Variant to use (sprott_a, sprott_b, sprott_g, sprott_n, sprott_s) |
| `transient_steps` | 1000 | Initial transient steps to discard |

Resolution: `integration_steps` (default: 100000)

Aliases: `sprott_attractor`

Output varies with seed (initial condition perturbation).

Recommended representation: TUBE

**Parameter presets:**

| Preset | `system` | Description |
|---|---|---|
| Sprott A | `sprott_a` | Simplest quadratic flow with chaos |
| Sprott B | `sprott_b` | Minimal chaotic jerk system |
| Sprott G | `sprott_g` | Chaotic flow with one quadratic nonlinearity |
| Sprott N | `sprott_n` | Chaotic flow with quadratic z term |
| Sprott S | `sprott_s` | Chaotic flow with quadratic z term (variant) |

```bash
mathviz generate sprott --output sprott_a.ply
mathviz generate sprott --param system=sprott_b --output sprott_b.ply
mathviz generate sprott --param system=sprott_g --seed 7 --output sprott_g.ply
```

## Curves

Mathematical curves extended to 3D.

### cardioid

Heart-shaped cardioid curve extended to 3D.

| Parameter | Default | Description |
|---|---|---|
| `radius` | 1.0 | Cardioid radius |
| `height` | 0.3 | Height extension in z |

```bash
mathviz generate cardioid --output cardioid.ply
```

### fibonacci_spiral

Golden-ratio spiral with exponential radius growth.

| Parameter | Default | Description |
|---|---|---|
| `turns` | 4.0 | Number of spiral turns |
| `height` | 0.5 | Height extension in z |
| `scale` | 1.0 | Overall scale factor |

Aliases: `golden_spiral`

```bash
mathviz generate fibonacci_spiral --param turns=6 --output spiral.ply
```

### lissajous_curve

3D Lissajous curve with configurable frequencies and phases.

| Parameter | Default | Description |
|---|---|---|
| `nx` | 3 | X frequency |
| `ny` | 2 | Y frequency |
| `nz` | 1 | Z frequency |
| `phase_x` | 0.0 | X phase offset |
| `phase_y` | 0.5 | Y phase offset |
| `phase_z` | 0.0 | Z phase offset |
| `scale` | 1.0 | Overall scale factor |

```bash
mathviz generate lissajous_curve --param nx=5 --param ny=4 --output lissajous.ply
```

### logarithmic_spiral

Logarithmic spiral with exponential radius growth.

| Parameter | Default | Description |
|---|---|---|
| `growth_rate` | 0.15 | Exponential growth rate |
| `turns` | 3.0 | Number of turns |
| `height` | 1.0 | Height extension in z |
| `scale` | 1.0 | Overall scale factor |

```bash
mathviz generate logarithmic_spiral --output logspiral.ply
```

## Surfaces

Ruled surfaces and envelopes.

### parabolic_envelope

Ruled surface from a family of lines forming a parabolic envelope.

| Parameter | Default | Description |
|---|---|---|
| `line_count` | 32 | Number of lines in the family |
| `scale` | 1.0 | Overall scale factor |
| `height` | 0.5 | Height extension in z |

```bash
mathviz generate parabolic_envelope --param line_count=64 --output envelope.ply
```

## Data-Driven

Forms derived from external data files.

### building_extrude

Extrude GeoJSON polygons into 3D building meshes.

| Parameter | Default | Description |
|---|---|---|
| `input_file` | (required) | Path to GeoJSON file |
| `default_height` | 1.0 | Default extrusion height |
| `height_property` | `"height"` | GeoJSON property for building height |

```bash
mathviz generate building_extrude --param input_file=buildings.geojson --output buildings.ply
```

### heightmap

Heightmap relief from image or GeoTIFF file.

| Parameter | Default | Description |
|---|---|---|
| `input_file` | (required) | Path to image or GeoTIFF file |
| `height_scale` | 1.0 | Vertical scale multiplier |
| `downsample` | 1 | Downsample factor (1 = full resolution) |

Aliases: `heightmap_image`

```bash
mathviz generate heightmap --param input_file=terrain.png --output terrain.ply
```

### soundwave

3D waveform visualization from WAV audio file.

| Parameter | Default | Description |
|---|---|---|
| `input_file` | (required) | Path to WAV audio file |
| `amplitude_scale` | 1.0 | Vertical scale for amplitude |
| `length` | 2.0 | Length of the waveform in world units |

Aliases: `audio_waveform`

```bash
mathviz generate soundwave --param input_file=audio.wav --output waveform.ply
```

## Fractals

3D fractals and fractal-derived heightmaps.

### mandelbulb

Mandelbulb 3D fractal with numba-JIT escape-time kernel.

| Parameter | Default | Description |
|---|---|---|
| `power` | 8.0 | Mandelbulb power exponent |
| `max_iterations` | 10 | Maximum escape-time iterations |
| `extent` | 1.5 | Spatial extent of the sampling volume |

```bash
mathviz generate mandelbulb --param power=6 --output mandelbulb.ply
```

### julia3d

3D Julia set fractal with numba-JIT escape-time kernel.

| Parameter | Default | Description |
|---|---|---|
| `power` | 8.0 | Power exponent |
| `max_iterations` | 10 | Maximum escape-time iterations |
| `extent` | 1.5 | Spatial extent of the sampling volume |
| `c_re` | -0.2 | Real part of the Julia constant |
| `c_im` | 0.6 | Imaginary part of the Julia constant |
| `c_z` | 0.2 | Z component of the Julia constant |

Aliases: `julia_3d`

```bash
mathviz generate julia3d --param c_re=-0.4 --param c_im=0.8 --output julia.ply
```

### quaternion_julia

Quaternion Julia set — 4D fractal sliced to 3D via marching cubes.

| Parameter | Default | Description |
|---|---|---|
| `c_real` | -0.2 | Real part of the quaternion constant |
| `c_i` | 0.8 | i-component of the quaternion constant |
| `c_j` | 0.0 | j-component of the quaternion constant |
| `c_k` | 0.0 | k-component of the quaternion constant |
| `max_iter` | 10 | Maximum escape-time iterations |
| `escape_radius` | 2.0 | Escape radius for divergence test |
| `extent` | 1.5 | Spatial extent of the sampling volume |
| `slice_w` | 0.0 | Fixed w-component for the 4D→3D slice |

Aliases: `qjulia`

```bash
mathviz generate quaternion_julia --param c_real=-0.2 --param c_i=0.8 --seed 42 --output qjulia.ply
```

### fractal_slice

2D cross-section through a Mandelbulb for heightmap relief.

| Parameter | Default | Description |
|---|---|---|
| `power` | 8.0 | Mandelbulb power exponent |
| `max_iterations` | 20 | Maximum escape-time iterations |
| `extent` | 1.5 | Spatial extent |
| `slice_axis` | `"z"` | Axis to slice through (x, y, or z) |
| `slice_position` | 0.0 | Position along the slice axis |

Aliases: `fractal_cross_section`

```bash
mathviz generate fractal_slice --param slice_axis=y --output slice.ply
```

### menger_sponge

Menger sponge fractal via recursive cube subdivision. Removes the center of each
face plus the center cube at every level, keeping 20 of 27 sub-cubes.

| Parameter | Default | Description |
|---|---|---|
| `level` | 3 | Recursion depth (0–4) |
| `size` | 1.0 | Side length of the bounding cube |

Aliases: `menger`

```bash
mathviz generate menger_sponge --param level=2 --output menger.ply
```

### sierpinski_tetrahedron

Sierpinski tetrahedron (tetrix) fractal via recursive corner subdivision. The 3D
analogue of the Sierpinski triangle — at each level, replaces each tetrahedron
with 4 half-scale copies at the corners.

| Parameter | Default | Description |
|---|---|---|
| `level` | 5 | Recursion depth (0–8) |
| `size` | 1.0 | Edge length of the initial tetrahedron |

Aliases: `tetrix`, `sierpinski_tetrix`

```bash
mathviz generate sierpinski_tetrahedron --param level=4 --output tetrix.ply
```

### apollonian_3d

3D Apollonian gasket — recursive sphere packing where each gap between tangent
spheres is filled with the largest fitting sphere.

| Parameter | Default | Description |
|---|---|---|
| `max_depth` | 5 | Recursion depth (0–8) |
| `min_radius` | 0.01 | Minimum sphere radius before pruning |
| `icosphere_subdivisions` | 1 | Subdivision level for each sphere mesh |

Aliases: `apollonian_gasket_3d`

```bash
mathviz generate apollonian_3d --param max_depth=4 --seed 42 --output apollonian.ply
```

### burning_ship

Burning Ship escape-time heightmap for 3D relief engraving.

| Parameter | Default | Description |
|---|---|---|
| `center_x` | -0.4 | Real center of the view |
| `center_y` | -0.6 | Imaginary center of the view |
| `zoom` | 3.0 | Zoom level |
| `max_iterations` | 256 | Maximum iteration count |
| `height_scale` | 0.3 | Vertical scale multiplier |

```bash
mathviz generate burning_ship --param zoom=5 --param center_x=-0.4 --output burning_ship.ply
```

### mandelbrot_heightmap

Mandelbrot escape-time heightmap for 3D relief engraving.

| Parameter | Default | Description |
|---|---|---|
| `center_real` | -0.5 | Real center of the view |
| `center_imag` | 0.0 | Imaginary center of the view |
| `zoom` | 1.0 | Zoom level |
| `max_iterations` | 256 | Maximum iteration count |
| `height_scale` | 1.0 | Vertical scale multiplier |
| `smoothing` | true | Enable smooth coloring |

```bash
mathviz generate mandelbrot_heightmap --param zoom=10 --param center_real=-0.75 --output mandelbrot.ply
```

### ifs_fractal

IFS fractal generator (Barnsley fern, maple leaf, spiral, custom affine transforms).

| Parameter | Default | Description |
|---|---|---|
| `preset` | barnsley_fern | IFS preset (barnsley_fern, maple_leaf, spiral, custom) |
| `dimensions` | 3d | Output mode (3d, 2d_extruded) |

The `custom` preset requires additional parameters: `matrices` (list of 2×2 or
3×3 affine matrices), `offsets` (list of translation vectors matching matrix
dimension), and `probabilities` (list of non-negative floats summing to 1.0).
Note: 3×3 custom matrices are incompatible with `dimensions=2d_extruded`.

Aliases: `ifs`, `barnsley_fern`

```bash
mathviz generate ifs_fractal --output ifs.ply
mathviz generate ifs_fractal --param preset=maple_leaf --output maple.ply
mathviz generate ifs_fractal --param preset=spiral --param dimensions=2d_extruded --output spiral.ply
```

### koch_3d

Koch snowflake curve extruded or revolved into 3D. Generates the classic Koch
snowflake at a given recursion level, then produces 3D geometry via extrusion
along the z-axis or revolution around the y-axis.

| Parameter | Default | Description |
|---|---|---|
| `level` | 4 | Recursion depth (0–6) |
| `mode` | extrude | 3D method: `extrude` or `revolve` |
| `height` | 0.3 | Extrusion height (extrude mode only) |

Aliases: `koch_snowflake_3d`

```bash
mathviz generate koch_3d --param level=4 --param mode=extrude --seed 42 --output koch.ply
```

## Geometry

User-defined parametric surfaces and spatial constructions.

### generic_parametric

Parametric surface from user-supplied f(u,v) expressions.

| Parameter | Default | Description |
|---|---|---|
| `x_expr` | `"(1 + 0.4 * cos(v)) * cos(u)"` | X coordinate expression |
| `y_expr` | `"(1 + 0.4 * cos(v)) * sin(u)"` | Y coordinate expression |
| `z_expr` | `"0.4 * sin(v)"` | Z coordinate expression |
| `u_range` | `[0, 2*pi]` | Parameter u range |
| `v_range` | `[0, 2*pi]` | Parameter v range |
| `wrap_u` | true | Wrap u parameter for seamless join |
| `wrap_v` | true | Wrap v parameter for seamless join |

```bash
mathviz generate generic_parametric \
  --param x_expr="cos(u)*sin(v)" \
  --param y_expr="sin(u)*sin(v)" \
  --param z_expr="cos(v)" \
  --output sphere.ply
```

### voronoi_3d

3D Voronoi cell boundaries as wireframe edges.

| Parameter | Default | Description |
|---|---|---|
| `num_points` | 20 | Number of seed points |
| `scale` | 1.0 | Overall scale factor |

```bash
mathviz generate voronoi_3d --param num_points=50 --seed 42 --output voronoi.ply
```

### voronoi_sphere

Voronoi tessellation on a sphere surface with geodesic cells. Distributes seed
points via perturbed Fibonacci spiral and computes spherical Voronoi diagram.
Supports ridge curves, cell face meshes, or both.

| Parameter | Default | Description |
|---|---|---|
| `num_cells` | 64 | Number of Voronoi cells |
| `radius` | 1.0 | Sphere radius |
| `edge_width` | 0.05 | Width of ridge edges |
| `edge_height` | 0.1 | Height of ridges above sphere |
| `cell_style` | `"ridges_only"` | Style: `ridges_only`, `cells_only`, or `both` |

Resolution: `arc_resolution` (default: 16)

Output varies with seed (Fibonacci sphere perturbation).

Recommended representation: TUBE

```bash
mathviz generate voronoi_sphere --output vsphere.ply
mathviz generate voronoi_sphere --param num_cells=128 --param cell_style=both --output vsphere.ply
```

## Implicit Surfaces

Triply periodic minimal surfaces extracted via marching cubes.

### gyroid

Triply periodic gyroid minimal surface.

| Parameter | Default | Description |
|---|---|---|
| `cell_size` | 1.0 | Unit cell size |
| `periods` | 2 | Number of repeating periods |

```bash
mathviz generate gyroid --param periods=3 --output gyroid.ply
```

### schwarz_d

Triply periodic Schwarz D (Diamond) minimal surface.

| Parameter | Default | Description |
|---|---|---|
| `cell_size` | 1.0 | Unit cell size |
| `periods` | 2 | Number of repeating periods |

```bash
mathviz generate schwarz_d --output schwarz_d.ply
```

### schwarz_p

Triply periodic Schwarz P (Primitive) minimal surface.

| Parameter | Default | Description |
|---|---|---|
| `cell_size` | 1.0 | Unit cell size |
| `periods` | 2 | Number of repeating periods |

```bash
mathviz generate schwarz_p --output schwarz_p.ply
```

### genus2_surface

Genus-2 surface via smooth blending of two overlapping tori.

| Parameter | Default | Description |
|---|---|---|
| `separation` | 1.15 | Distance between torus centers |
| `major_radius` | 1.0 | Major radius of each torus |
| `tube_radius` | 0.4 | Tube radius of each torus |
| `blend_sharpness` | 15.0 | Sharpness of the blending function |

```bash
mathviz generate genus2_surface --param separation=1.3 --output genus2.ply
```

## Knots

Mathematical knot curves rendered as tube meshes.

### torus_knot

Torus knot curve with configurable (p, q) winding numbers.

| Parameter | Default | Description |
|---|---|---|
| `p` | 2 | Longitudinal winding number |
| `q` | 3 | Meridional winding number |
| `R` | 1.0 | Major radius |
| `r` | 0.4 | Minor radius |

Aliases: `trefoil`, `cinquefoil`

```bash
mathviz generate torus_knot --output trefoil.ply
mathviz generate torus_knot --param p=2 --param q=5 --output cinquefoil.ply
```

### figure_eight_knot

Figure-eight knot with crossing number 4.

| Parameter | Default | Description |
|---|---|---|
| `scale` | 1.0 | Overall scale factor |

```bash
mathviz generate figure_eight_knot --output fig8.ply
```

### lissajous_knot

Lissajous knot with configurable frequencies and phases.

| Parameter | Default | Description |
|---|---|---|
| `nx` | 2 | X frequency |
| `ny` | 3 | Y frequency |
| `nz` | 5 | Z frequency |
| `phase_x` | 0.0 | X phase offset |
| `phase_y` | 0.7 | Y phase offset |
| `phase_z` | 0.2 | Z phase offset |
| `scale` | 1.0 | Overall scale factor |

```bash
mathviz generate lissajous_knot --param nx=3 --param ny=4 --output lknot.ply
```

### seven_crossing_knots

Knots with crossing number 7, selectable by knot_index.

| Parameter | Default | Description |
|---|---|---|
| `knot_index` | 1 | Index of the 7-crossing knot variant |
| `scale` | 1.0 | Overall scale factor |

```bash
mathviz generate seven_crossing_knots --param knot_index=3 --output knot7.ply
```

### trefoil_on_torus

(2,3) torus knot rendered alongside its host torus surface, showing how the
knot sits on the torus. Produces both a knot curve (TUBE) and a torus mesh
(WIREFRAME).

| Parameter | Default | Description |
|---|---|---|
| `torus_R` | 1.0 | Major radius of the torus |
| `torus_r` | 0.4 | Minor (tube) radius of the torus |

Resolution: `curve_points` (default: 1024), `torus_resolution` (default: 32)

Recommended representation: TUBE (knot) + WIREFRAME (torus)

```bash
mathviz generate trefoil_on_torus --output trefoil_torus.ply
mathviz generate trefoil_on_torus --param torus_R=1.5 --output trefoil_torus.ply
```

### pretzel_knot

Pretzel knot with p left-hand and q right-hand twists, forming a closed curve
with p+q lobes.

| Parameter | Default | Description |
|---|---|---|
| `p` | 2 | Number of left-hand twists |
| `q` | 3 | Number of right-hand twists |

Resolution: `curve_points` (default: 1024)

Recommended representation: TUBE

```bash
mathviz generate pretzel_knot --output pretzel.ply
mathviz generate pretzel_knot --param p=3 --param q=5 --output pretzel35.ply
```

### cinquefoil_knot

Cinquefoil (2,5) torus knot — a five-lobed star knot rendered as a standalone
generator.

No additional parameters.

Resolution: `curve_points` (default: 1024)

Recommended representation: TUBE

```bash
mathviz generate cinquefoil_knot --output cinquefoil.ply
```

### borromean_rings

Three mutually linked rings — removing any one frees the other two. Each ring
lies in an orthogonal plane with slight deformation to create linking.

| Parameter | Default | Description |
|---|---|---|
| `ring_radius` | 1.0 | Radius of each ring |
| `ring_thickness` | 0.08 | Tube thickness for representation |

Resolution: `curve_points` (default: 512)

Recommended representation: TUBE

```bash
mathviz generate borromean_rings --output borromean.ply
mathviz generate borromean_rings --param ring_radius=1.5 --output borromean.ply
```

### chain_links

Chain of interlocking torus links with alternating orientation (xy/xz planes).

| Parameter | Default | Description |
|---|---|---|
| `num_links` | 5 | Number of chain links |
| `link_radius` | 0.5 | Radius of each link |
| `link_thickness` | 0.1 | Tube thickness for representation |

Resolution: `curve_points` (default: 256)

Recommended representation: TUBE

```bash
mathviz generate chain_links --output chain.ply
mathviz generate chain_links --param num_links=8 --param link_radius=0.3 --output chain8.ply
```

## Number Theory

Visualizations derived from number-theoretic sequences.

### ulam_spiral

Ulam spiral with primes elevated in z.

| Parameter | Default | Description |
|---|---|---|
| `prime_height` | 1.0 | Z elevation for prime numbers |
| `spacing` | 0.1 | Distance between spiral positions |

Aliases: `ulam`

```bash
mathviz generate ulam_spiral --output ulam.ply
```

### sacks_spiral

Sacks spiral — primes on an Archimedean spiral.

| Parameter | Default | Description |
|---|---|---|
| `prime_height` | 1.0 | Z elevation for prime numbers |
| `scale` | 0.1 | Spiral scale factor |

Aliases: `sacks`

```bash
mathviz generate sacks_spiral --output sacks.ply
```

### prime_gaps

Prime gaps visualized as a 3D ribbon.

| Parameter | Default | Description |
|---|---|---|
| `x_spacing` | 0.1 | Horizontal spacing |
| `y_scale` | 0.1 | Gap height scale |
| `ribbon_width` | 0.5 | Width of the ribbon |

```bash
mathviz generate prime_gaps --output gaps.ply
```

### digit_encoding

Digits of mathematical constants as 3D point heights.

| Parameter | Default | Description |
|---|---|---|
| `constant` | `"pi"` | Mathematical constant to encode |
| `height_scale` | 0.1 | Vertical scale per digit |
| `spacing` | 0.1 | Horizontal spacing between digits |

Aliases: `digits`

```bash
mathviz generate digit_encoding --param constant=e --output digits.ply
```

## Parametric Surfaces

Parametric surface meshes from mathematical formulas.

### torus

Parametric torus surface with configurable radii.

| Parameter | Default | Description |
|---|---|---|
| `major_radius` | 1.0 | Distance from center to tube center |
| `minor_radius` | 0.4 | Tube radius |

```bash
mathviz generate torus --param major_radius=1.5 --output torus.ply
```

### klein_bottle

Klein bottle immersion with self-intersection.

| Parameter | Default | Description |
|---|---|---|
| `scale` | 1.0 | Overall scale factor |

```bash
mathviz generate klein_bottle --output klein.ply
```

### mobius_strip

Möbius strip with configurable radius and width.

| Parameter | Default | Description |
|---|---|---|
| `radius` | 1.0 | Strip radius |
| `half_width` | 0.4 | Half-width of the strip |

```bash
mathviz generate mobius_strip --output mobius.ply
```

### enneper_surface

Enneper minimal surface with configurable range and order.

| Parameter | Default | Description |
|---|---|---|
| `range` | 2.0 | Parameter range |
| `order` | 1 | Surface order |

```bash
mathviz generate enneper_surface --param range=3.0 --output enneper.ply
```

### boy_surface

Boy surface (RP² immersion) with triple self-intersection.

| Parameter | Default | Description |
|---|---|---|
| `scale` | 1.0 | Overall scale factor |

```bash
mathviz generate boy_surface --output boy.ply
```

### bour_surface

Bour's minimal surface — interpolates between a helicoid and a catenoid.
Parameterized by order n, with n=2 giving the classic helicoid-catenoid interpolation.

| Parameter | Default | Description |
|---|---|---|
| `n` | 2 | Surface order (controls shape) |
| `r_max` | 1.0 | Maximum radial extent |

Aliases: `bour`

```bash
mathviz generate bour_surface --param n=3 --output bour.ply
```

### dna_helix

DNA double helix with twin helices offset by 180 degrees, connected by base pair
rungs at regular intervals.

| Parameter | Default | Description |
|---|---|---|
| `turns` | 3 | Number of helix turns |
| `radius` | 1.0 | Helix radius |
| `rise_per_turn` | 3.4 | Vertical rise per full turn |
| `base_pairs_per_turn` | 10 | Number of base pair rungs per turn |

Aliases: `dna`, `double_helix`

```bash
mathviz generate dna_helix --param turns=5 --seed 42 --output dna.ply
```

### hopf_fibration

Hopf fibration — circles in S³ projected to R³ via stereographic projection,
forming nested tori of linked rings.

| Parameter | Default | Description |
|---|---|---|
| `num_fibers` | 32 | Number of fiber curves per base circle |
| `num_circles` | 5 | Number of latitude circles on S² |
| `projection_point` | [0.0, 0.0, 0.0, 2.0] | Stereographic projection offset in S³ |
| `fiber_points` | 256 | Points per fiber curve (resolution) |

Aliases: `hopf`

```bash
mathviz generate hopf_fibration --param num_fibers=48 --param num_circles=7 --seed 42 --output hopf.ply
```

### seifert_surface

Seifert surface — orientable surface bounded by a knot. Uses the Milnor fiber
parameterization for torus knots (trefoil) and spanning surface construction for
the figure-eight knot.

| Parameter | Default | Description |
|---|---|---|
| `knot_type` | `trefoil` | Knot type: `trefoil` or `figure_eight` |
| `theta` | 0.0 | Milnor fiber angle (trefoil) / phase offset (figure-eight) |

Aliases: `seifert`

```bash
mathviz generate seifert_surface --param knot_type=trefoil --output seifert.ply
```

### roman_surface

Roman (Steiner) surface — self-intersecting non-orientable surface with tetrahedral symmetry.

| Parameter | Default | Description |
|---|---|---|
| `scale` | 1.0 | Overall scale factor |

Aliases: `steiner_surface`

```bash
mathviz generate roman_surface --output roman.ply
```

### calabi_yau

Calabi-Yau manifold cross-section — crystalline flower-like string theory shape.

| Parameter | Default | Description |
|---|---|---|
| `n` | 5 | Exponent in z1^n + z2^n = 1 |
| `alpha` | π/4 | Projection angle from C² to R³ |

```bash
mathviz generate calabi_yau --param n=6 --output calabi_yau.ply
```

### costa_surface

Costa minimal surface via Weierstrass-Enneper representation.

| Parameter | Default | Description |
|---|---|---|
| `scale` | 0.1 | Overall scale factor |

```bash
mathviz generate costa_surface --output costa.ply
```

### cross_cap

Cross-cap — non-orientable immersion of the real projective plane in R³.

| Parameter | Default | Description |
|---|---|---|
| `scale` | 1.0 | Overall scale factor |

Aliases: `crosscap`

```bash
mathviz generate cross_cap --output crosscap.ply
```

### dini_surface

Dini's surface — twisted pseudospherical surface resembling a seashell or spiral horn.
Has constant negative Gaussian curvature.

| Parameter | Default | Description |
|---|---|---|
| `a` | 1.0 | Scale factor |
| `b` | 0.2 | Twist rate (helical pitch) |
| `turns` | 2 | Number of helical turns |

Aliases: `dini`

```bash
mathviz generate dini_surface --param turns=3 --output dini.ply
```

### dupin_cyclide

Dupin cyclide — inversive geometry shape generalizing torus, cylinder, and cone.
Parameterized via Möbius transformation of a torus.

| Parameter | Default | Description |
|---|---|---|
| `a` | 1.0 | Primary shape parameter |
| `b` | 0.8 | Tube cross-section parameter |
| `c` | 0.5 | Inversion parameter (must be < a) |
| `d` | 0.6 | Offset parameter |

Aliases: `cyclide`

```bash
mathviz generate dupin_cyclide --param d=1.2 --output cyclide.ply
```

### spherical_harmonics

Sphere modulated by spherical harmonics basis functions.

| Parameter | Default | Description |
|---|---|---|
| `l` | 0 | Degree of the harmonic |
| `m` | 0 | Order of the harmonic |
| `base_radius` | 1.0 | Base sphere radius |
| `amplitude` | 0.3 | Modulation amplitude |

```bash
mathviz generate spherical_harmonics --param l=3 --param m=2 --output harmonics.ply
```

### superellipsoid

Superellipsoid with configurable exponents and radii.

| Parameter | Default | Description |
|---|---|---|
| `a1` | 1.0 | X-axis radius |
| `a2` | 1.0 | Y-axis radius |
| `a3` | 1.0 | Z-axis radius |
| `e1` | 1.0 | Latitude exponent |
| `e2` | 1.0 | Longitude exponent |

```bash
mathviz generate superellipsoid --param e1=0.5 --param e2=0.5 --output super.ply
```

### lissajous_surface

Tubular surface around a Lissajous knot curve.

| Parameter | Default | Description |
|---|---|---|
| `nx` | 2 | X frequency |
| `ny` | 3 | Y frequency |
| `nz` | 5 | Z frequency |
| `phase_x` | 0.0 | X phase offset |
| `phase_y` | 0.0 | Y phase offset |
| `phase_z` | 0.0 | Z phase offset |
| `tube_radius` | 0.1 | Tube radius |

```bash
mathviz generate lissajous_surface --param tube_radius=0.2 --output lsurf.ply
```

## Physics

Physics simulations and orbital mechanics.

### kepler_orbit

Elliptical orbit from classical orbital elements.

| Parameter | Default | Description |
|---|---|---|
| `semi_major_axis` | 1.0 | Semi-major axis of the orbit |
| `eccentricity` | 0.5 | Orbital eccentricity (0 = circle, <1 = ellipse) |
| `inclination` | 0.0 | Orbital inclination (radians) |

```bash
mathviz generate kepler_orbit --param eccentricity=0.3 --output orbit.ply
```

### nbody

Gravitational N-body simulation with seed-based initial conditions.

| Parameter | Default | Description |
|---|---|---|
| `num_bodies` | 3 | Number of gravitating bodies |
| `time_span` | 10.0 | Simulation time span |

Aliases: `n_body`

```bash
mathviz generate nbody --param num_bodies=5 --seed 7 --output nbody.ply
```

### planetary_positions

Solar system orbits and planet positions at a given epoch.

| Parameter | Default | Description |
|---|---|---|
| `epoch_jd` | 2451545.0 | Julian date of the epoch (J2000.0 default) |

Aliases: `solar_system`

```bash
mathviz generate planetary_positions --output solar.ply
```

### electron_orbital

Hydrogen atom electron orbital probability density isosurface. Computes |ψ(r,θ,φ)|²
for hydrogen wavefunctions using radial functions and spherical harmonics. Different
(n, l, m) quantum numbers produce iconic orbital shapes: s-orbitals (spheres),
p-orbitals (dumbbells), d-orbitals (cloverleaf), and higher.

| Parameter | Default | Description |
|---|---|---|
| `n` | 3 | Principal quantum number (≥ 1) |
| `l` | 2 | Angular momentum quantum number (0 ≤ l < n) |
| `m` | 0 | Magnetic quantum number (−l ≤ m ≤ l) |
| `iso_level` | 0.01 | Isosurface threshold for probability density |
| `voxel_resolution` | 128 | Voxels per axis (N³ cost) |

Aliases: `hydrogen_orbital`

```bash
mathviz generate electron_orbital --param n=2 --param l=1 --param m=0 --output orbital.ply
```

### magnetic_field

3D magnetic field lines for dipole and quadrupole configurations. Field lines
are computed by RK4 integration of the magnetic field vector from seed points
distributed on a ring around the source. Each field line becomes a tube curve.

| Parameter | Default | Description |
|---|---|---|
| `field_type` | `dipole` | Field configuration (`dipole` or `quadrupole`) |
| `num_lines` | 24 | Number of field lines |
| `spread` | 0.3 | Seed point distribution radius |

Resolution: `line_points` (default 500) — integration steps per field line.

Aliases: `mag_field`

```bash
mathviz generate magnetic_field --param field_type=quadrupole --param num_lines=32 --output field.ply
```

### gravitational_lensing

Warped coordinate grid showing spacetime curvature around a point mass.
Grid lines are deflected using the Schwarzschild deflection formula and
extruded to 3D using the deflection magnitude as z-displacement.

| Parameter | Default | Description |
|---|---|---|
| `mass` | 1.0 | Point mass strength |
| `grid_lines` | 20 | Number of grid lines per axis |
| `grid_extent` | 5.0 | Spatial extent of the grid |

Resolution: `grid_points` (default 200) — sample points per grid line.

Aliases: `grav_lens`, `spacetime_grid`

```bash
mathviz generate gravitational_lensing --param mass=2.0 --param grid_lines=30 --output lensing.ply
```

### wave_interference

3D standing wave interference pattern from multiple point sources with
isosurface extraction. Computes the superposition of spherical waves
sin(k·r − ωt) / r on a voxel grid and extracts an isosurface via marching
cubes.

| Parameter | Default | Description |
|---|---|---|
| `num_sources` | 3 | Number of point sources (1–20) |
| `wavelength` | 0.5 | Wavelength of each spherical wave |
| `source_spacing` | 1.0 | Spacing between sources on z=0 plane |
| `iso_level` | 0.5 | Isosurface threshold (0–1, normalized field) |
| `time` | 0.0 | Time parameter for wave animation |

Resolution: `voxel_resolution` (default 128) — voxels per axis (N³ cost).

Aliases: `wave_pattern`, `interference`

```bash
mathviz generate wave_interference --param num_sources=5 --param wavelength=0.3 --seed 42 --output wave.ply
```

## Procedural

Procedurally generated surfaces from noise and simulation.

### noise_surface

Seed-controlled simplex noise heightmap surface.

| Parameter | Default | Description |
|---|---|---|
| `frequency` | 4.0 | Noise frequency |
| `height_scale` | 1.0 | Vertical scale multiplier |

Aliases: `simplex_surface`

```bash
mathviz generate noise_surface --param frequency=8.0 --seed 42 --output noise.ply
```

### reaction_diffusion

Gray-Scott reaction-diffusion pattern as heightmap.

| Parameter | Default | Description |
|---|---|---|
| `feed_rate` | 0.035 | Feed rate (F) |
| `kill_rate` | 0.065 | Kill rate (k) |
| `diffusion_u` | 0.16 | Diffusion rate of U |
| `diffusion_v` | 0.08 | Diffusion rate of V |
| `timesteps` | 5000 | Number of simulation steps |
| `dt` | 1.0 | Time step size |
| `height_scale` | 1.0 | Vertical scale multiplier |

Aliases: `gray_scott`

**Parameter presets:**

| Preset | `feed_rate` | `kill_rate` | Description |
|---|---|---|---|
| Default (spots) | 0.035 | 0.065 | Isolated spot patterns |
| Stripes | 0.055 | 0.062 | Stripe / labyrinthine patterns |
| Maze | 0.029 | 0.057 | Dense maze-like patterns |

```bash
mathviz generate reaction_diffusion --output rd.ply
mathviz generate reaction_diffusion --param feed_rate=0.055 --param kill_rate=0.062 --output rd_stripes.ply
mathviz generate reaction_diffusion --param feed_rate=0.029 --param kill_rate=0.057 --output rd_maze.ply
```

### lsystem

L-system fractal trees, bushes, ferns, and space-filling curves. Produces 3D
branching structures from Lindenmayer system grammars with named presets and
configurable parameters for organic variation.

| Parameter | Default | Description |
|---|---|---|
| `preset` | `"tree"` | Named preset (tree, bush, fern, hilbert3d, sierpinski) |
| `iterations` | 5 | Number of rewriting iterations (1–10) |
| `angle` | 25.0 | Branch angle in degrees |
| `length_scale` | 1.0 | Initial segment length |
| `length_decay` | 0.7 | Length multiplier per generation (0–1] |
| `thickness_decay` | 0.6 | Thickness multiplier per generation (0–1] |
| `jitter` | 5.0 | Random angle jitter in degrees |

Output varies with seed (angle jitter randomization).

Recommended representation: TUBE

**Parameter presets:**

| Preset | Angle | Iterations | Description |
|---|---|---|---|
| tree | 25.0 | 5 | Simple branching tree |
| bush | 22.5 | 4 | Dense bush with many branches |
| fern | 25.0 | 6 | Fern-like branching pattern |
| hilbert3d | 90.0 | 2 | 3D Hilbert space-filling curve |
| sierpinski | 120.0 | 6 | Sierpinski triangle as continuous path |

```bash
mathviz generate lsystem --output tree.ply
mathviz generate lsystem --param preset=bush --param iterations=5 --output bush.ply
mathviz generate lsystem --param preset=fern --param angle=30 --seed 7 --output fern.ply
mathviz generate lsystem --param preset=hilbert3d --param iterations=3 --output hilbert.ply
```

### rd_surface

Gray-Scott reaction-diffusion on curved surface meshes. Runs the Gray-Scott
model on a surface mesh (torus, sphere, or Klein bottle) using the mesh
Laplacian for diffusion. Displaces vertices along normals proportional to the
V concentration to produce Turing-pattern geometry.

| Parameter | Default | Description |
|---|---|---|
| `base_surface` | `"torus"` | Base mesh: `torus`, `sphere`, or `klein_bottle` |
| `feed_rate` | 0.055 | Feed rate (F) |
| `kill_rate` | 0.062 | Kill rate (k) |
| `diffusion_u` | 0.16 | Diffusion rate of U |
| `diffusion_v` | 0.08 | Diffusion rate of V |
| `iterations` | 5000 | Number of simulation steps (100–50000) |
| `displacement_scale` | 0.1 | Vertex displacement along normals |

Resolution: `grid_resolution` (default: 128)

Aliases: `reaction_diffusion_surface`

Output varies with seed (random V-concentration patches).

Recommended representation: SURFACE_SHELL

**Parameter presets:**

| Preset | `feed_rate` | `kill_rate` | Description |
|---|---|---|---|
| Spots | 0.035 | 0.065 | Isolated spot patterns |
| Stripes | 0.055 | 0.062 | Stripe / labyrinthine patterns |
| Maze | 0.029 | 0.057 | Dense maze-like patterns |

```bash
mathviz generate rd_surface --output rd_torus.ply
mathviz generate rd_surface --param base_surface=sphere --param feed_rate=0.035 --param kill_rate=0.065 --output rd_spots.ply
mathviz generate rd_surface --param base_surface=klein_bottle --param feed_rate=0.029 --param kill_rate=0.057 --output rd_maze.ply
```

### terrain

Multi-octave simplex noise terrain heightmap.

| Parameter | Default | Description |
|---|---|---|
| `octaves` | 6 | Number of noise octaves |
| `persistence` | 0.5 | Amplitude persistence per octave |
| `lacunarity` | 2.0 | Frequency multiplier per octave |
| `base_frequency` | 3.0 | Base noise frequency |
| `height_scale` | 1.0 | Vertical scale multiplier |

Aliases: `terrain_heightmap`

```bash
mathviz generate terrain --param octaves=8 --seed 42 --output terrain.ply
```
