# Generators

MathViz includes 48 generators across 12 categories. Each generator produces a
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

### costa_surface

Costa minimal surface via Weierstrass-Enneper representation.

| Parameter | Default | Description |
|---|---|---|
| `scale` | 0.1 | Overall scale factor |

```bash
mathviz generate costa_surface --output costa.ply
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

```bash
mathviz generate reaction_diffusion --param feed_rate=0.04 --param kill_rate=0.06 --output rd.ply
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
