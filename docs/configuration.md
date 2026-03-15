# Configuration

MathViz uses a layered configuration system. Configuration values are merged
from multiple sources with a clear precedence order.

## Precedence

From lowest to highest priority:

1. **Built-in defaults** — hardcoded in the source
2. **Project config** — `mathviz.toml` in the working directory
3. **Per-object config** — TOML file passed via `--config`
4. **Sampling profile** — merged into per-object config layer
5. **CLI flags** — `--seed`, `--width`, `--height`, `--depth`, `--param`

Higher-priority values override lower-priority values. Nested sections (e.g.,
`[container]`) are deep-merged: only explicitly set keys override defaults.

## Project Config (mathviz.toml)

Place a `mathviz.toml` file in the working directory to set project-wide
defaults. MathViz auto-discovers this file on startup.

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

Pass a TOML config file via `--config` for per-object overrides:

```bash
mathviz generate lorenz --config block_config.toml --output lorenz.ply
```

The per-object config has the same format as the project config and overrides
project-level values.

## Sampling Profiles

Named sampling profiles live in the `profiles/` directory within the package.
Use them via `--profile`:

```bash
mathviz generate gyroid --profile production --output gyroid.ply
```

### Built-in profiles

#### preview

Fast iteration with low point budget.

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

Template for user-defined settings.

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
| `shell_thickness` | float | null | Shell thickness for surface_shell |
| `volume_density` | float | null | Density for volume_fill |
| `slice_count` | int | null | Number of slices for slice_stack |
| `slice_axis` | string | `"z"` | Axis for slicing: x, y, z |
| `wireframe_thickness` | float | null | Line thickness for wireframe |
| `surface_density` | float | null | Density for sparse_shell |
| `density_weight_function` | string | null | Weight function expression for weighted_cloud |

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

## JSON Schema Generation

Generate JSON Schema files for all config models using:

```bash
mathviz schema schemas/
```

This writes schema files for Container, PlacementPolicy, SamplerConfig,
RepresentationConfig, EngravingProfile, and per-generator parameter schemas.
Use these for editor autocompletion and validation of TOML config files.
