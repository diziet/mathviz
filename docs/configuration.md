# Configuration

MathViz merges configuration values from several layers. A value in a higher
layer overrides the same value in a lower layer.

## Precedence

From lowest to highest priority:

1. **Built-in defaults** — hardcoded in the source
2. **Project config** — `mathviz.toml` in the working directory
3. **Per-object config** — TOML file passed via `--config`
4. **Sampling profile** — merged into per-object config layer
5. **CLI flags** — `--seed`, `--width`, `--height`, `--depth`, `--param`

Nested sections such as `[container]` are deep-merged: a layer overrides only
the keys it sets.

## Project Config (mathviz.toml)

A `mathviz.toml` file in the current working directory sets project-wide
defaults. A command that loads its configuration reads this file if it exists.

```toml
[container]
width_mm = 120.0
height_mm = 120.0
depth_mm = 50.0
margin_x_mm = 5.0
margin_y_mm = 5.0
margin_z_mm = 5.0

[placement]
anchor = "center"
viewing_axis = "+z"
preserve_aspect_ratio = true
depth_bias = 1.0

[sampling]
method = "uniform_surface"
num_points = 500000
seed = 42
```

## Per-Object Config

`--config` takes a TOML file with per-object overrides:

```bash
mathviz generate lorenz --config block_config.toml --output lorenz.ply
```

The per-object config has the same format as the project config. Its values
override the project config.

## Sampling Profiles

Named sampling profiles are TOML files in the package's `profiles/` directory.
`--profile` selects one by name:

```bash
mathviz generate gyroid --profile production --output gyroid.ply
```

### Built-in profiles

#### preview

Fast iteration with a low point budget.

```toml
[sampling]
method = "random_surface"
num_points = 10000
```

#### production

High-quality output for laser engraving.

```toml
[sampling]
method = "uniform_surface"
density = 8.0
```

#### custom

A template for user-defined settings.

```toml
[sampling]
method = "uniform_surface"
num_points = 500000
```

## Config Models

### Container

Defines the glass block dimensions and margins.

| Field | Type | Default | Description |
|---|---|---|---|
| `width_mm` | float | 100.0 | Block width in millimeters |
| `height_mm` | float | 100.0 | Block height in millimeters |
| `depth_mm` | float | 100.0 | Block depth in millimeters |
| `margin_x_mm` | float | 5.0 | Horizontal margin |
| `margin_y_mm` | float | 5.0 | Vertical margin |
| `margin_z_mm` | float | 5.0 | Depth margin |

### PlacementPolicy

Controls how geometry is positioned within the container.

| Field | Type | Default | Description |
|---|---|---|---|
| `anchor` | string | `"center"` | Anchor point: center, front, back, top, bottom, left, right |
| `viewing_axis` | string | `"+z"` | Viewing axis: +z, -z, +x, -x, +y, -y |
| `preserve_aspect_ratio` | bool | true | Maintain proportions during scaling |
| `depth_bias` | float | 1.0 | Z-axis scale factor (>1 = deeper, <1 = flatter) |
| `offset_mm` | tuple | (0, 0, 0) | Translation offset in mm |
| `scale_override` | float | null | Manual scale factor (bypasses auto-fit) |
| `rotation_degrees` | tuple | (0, 0, 0) | Rotation around each axis in degrees |

### SamplerConfig

Controls mesh-to-point-cloud sampling.

| Field | Type | Default | Description |
|---|---|---|---|
| `method` | string | `"uniform_surface"` | Sampling method: uniform_surface, random_surface, volume_fill |
| `density` | float | null | Points per mm² (mutually exclusive with num_points) |
| `num_points` | int | null | Target point count (mutually exclusive with density) |
| `seed` | int | 42 | RNG seed for sampling |
| `resample` | bool | false | Force resampling even if cloud exists |

### RepresentationConfig

Controls how raw geometry is represented for engraving.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | string | (required) | Representation strategy (see [representation.md](representation.md)) |
| `tube_radius` | float | null | Tube radius for tube strategy |
| `tube_sides` | int | 16 | Number of tube polygon sides |
| `shell_thickness` | float | null | ~~Shell thickness for surface_shell~~ No pipeline code reads this field (checked 2026-09-23) |
| `volume_density` | float | null | Density for volume_fill |
| `slice_count` | int | null | Number of slices for slice_stack |
| `slice_axis` | string | `"z"` | Axis for slicing: x, y, z |
| `wireframe_thickness` | float | null | Line thickness for wireframe |
| `surface_density` | float | null | Density for sparse_shell |
| `density_weight_function` | string | null | ~~Weight function expression for weighted_cloud~~ No pipeline code reads this field (checked 2026-09-23) |

### EngravingProfile

Controls engraving-specific validation and optimization.

| Field | Type | Default | Description |
|---|---|---|---|
| `point_budget` | int | 2,000,000 | Maximum number of engraving points |
| `min_point_spacing_mm` | float | 0.05 | Minimum distance between points |
| `max_point_spacing_mm` | float | 2.0 | Maximum distance between points |
| `occlusion_mode` | string | `"none"` | Occlusion mode: none, shell_fade, radial_gradient, custom |
| `occlusion_shell_layers` | int | 3 | Number of shell layers for occlusion |
| `occlusion_density_falloff` | float | 0.5 | Density falloff factor (0-1) |
| `depth_compensation` | bool | false | Enable depth-based density compensation |
| `depth_compensation_factor` | float | 1.5 | Depth compensation multiplier |

## Environment Variables

MathViz reads these environment variables:

| Variable | Default | Description |
|---|---|---|
| `MATHVIZ_SNAPSHOTS_DIR` | `~/.mathviz/snapshots` | Directory for saved preview snapshots (geometry, metadata, thumbnails) |
| `MATHVIZ_GENERATION_TIMEOUT` | `300` | Maximum generation time in seconds for the preview server. Values ≤ 0 are ignored with a warning. |
| `MATHVIZ_THUMBNAILS_DIR` | `~/.mathviz/thumbnails` | Directory for cached generator thumbnails, stored as `<view_mode>/<name>.webp` |
| `PYVISTA_OFF_SCREEN` | (unset) | Set to `true` for headless rendering without a display (see [rendering.md](rendering.md)) |

~~Environment variables are checked at startup.~~ MathViz reads
`MATHVIZ_SNAPSHOTS_DIR` for each snapshot operation, and
`MATHVIZ_GENERATION_TIMEOUT` for each generation or batch that has no
per-request timeout. Neither is read once at startup (checked 2026-09-23). An
invalid `MATHVIZ_GENERATION_TIMEOUT` logs a warning, and the default applies.

```bash
# Example: custom snapshot directory and longer timeout
export MATHVIZ_SNAPSHOTS_DIR=/mnt/data/mathviz-snapshots
export MATHVIZ_GENERATION_TIMEOUT=600
```

## JSON Schema Generation

`mathviz schema` writes JSON Schema files for all config models:

```bash
mathviz schema schemas/
```

It writes one schema file each for Container, PlacementPolicy, SamplerConfig,
RepresentationConfig and EngravingProfile, plus a parameter schema for each
generator that defines one. Editors can use these files to autocomplete and
validate TOML config files.
