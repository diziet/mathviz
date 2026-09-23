# Pipeline

The MathViz pipeline is a linear sequence of stages, and each stage can be
called on its own. Each stage operates on a `MathObject` dataclass and calls
`validate_or_raise()` at every boundary, which raises `ValueError` for an
invalid `MathObject`.

## Stages

```
Generate → Represent → Transform → Sample → Validate → Export
```

### 1. Generate

The generator produces raw geometry in abstract coordinate space. Each
generator is deterministic for a given seed. It draws random numbers from
`numpy.random.default_rng(seed)`.

- **Input**: generator name, parameters, seed
- **Output**: `MathObject` with mesh and/or point cloud in `ABSTRACT` coordinate space
- **Key rule**: generators never assume a specific container size or aspect ratio

### 2. Represent

The Represent stage applies a fabrication policy, which controls how the raw
geometry appears when engraved in glass. It keeps what the math produces
separate from how it looks engraved.

- **Input**: `MathObject` from Generate, `RepresentationConfig`
- **Output**: `MathObject` with geometry modified according to the chosen strategy
- **Key rule**: without a `RepresentationConfig`, the stage uses the generator's default strategy

See [representation.md](representation.md) for the nine available strategies.

### 3. Transform

The Transform stage fits geometry from abstract coordinate space into a
physical glass block container. It scales the geometry, centers it, and applies
the placement policy (anchor, viewing axis, depth bias, rotation).

- **Input**: `MathObject` in `ABSTRACT` space, `Container`, `PlacementPolicy`
- **Output**: `MathObject` in `PHYSICAL` space, fitted within the container margins
- **Key rule**: aspect ratio is preserved by default. Depth bias changes the z scaling.

### 4. Sample (optional)

Converts mesh geometry into a point cloud for laser engraving. The stage runs
only when a `SamplerConfig` is provided.

- **Input**: `MathObject` with mesh, `SamplerConfig`
- **Output**: `MathObject` with point cloud added
- **Methods**: `uniform_surface` (even distribution), `random_surface` (fast, stochastic), `volume_fill` (fills interior)

### 5. Validate

Runs mesh and engraving checks on the final geometry: bounding box
containment, mesh integrity, point spacing, and the point budget.

- **Input**: `MathObject`, `Container`, optional `EngravingProfile`
- **Output**: `ValidationResult` with a list of pass/fail checks
- **Key rule**: a failed check produces a warning and does not stop the pipeline

### 6. Export (optional)

Writes the final geometry to disk, as a mesh (STL, OBJ, GLB) or a point cloud
(PLY, XYZ, PCD).

- **Input**: `MathObject`, `ExportConfig` (path, format, export type)
- **Output**: file written to disk

## Running the Pipeline

### Via CLI

The `generate` command runs the full pipeline:

```bash
mathviz generate lorenz --output lorenz.ply --seed 42
```

The `validate` command runs Generate through Validate without exporting:

```bash
mathviz validate lorenz --seed 42
```

`--dry-run` shows what the command would do without running it:

```bash
mathviz generate lorenz --output lorenz.ply --dry-run
```

### Via Python

```python
from mathviz.core.container import Container, PlacementPolicy
from mathviz.pipeline.runner import ExportConfig, run

result = run(
    generator="lorenz",
    params={"sigma": 12.0},
    seed=42,
    container=Container(),
    placement=PlacementPolicy(),
    export_config=ExportConfig(path="lorenz.ply"),
)

print(result.validation.passed)
print(result.timings)
```

## Stage Timing

The pipeline records the run time of each stage in `PipelineResult.timings`:

```python
result = run(generator="lorenz", container=Container(), placement=PlacementPolicy())
for stage, seconds in result.timings.items():
    print(f"{stage}: {seconds:.3f}s")
```

## Coordinate Spaces

MathViz enforces two coordinate spaces:

- **ABSTRACT**: raw generator output, arbitrary units
- **PHYSICAL**: fitted to the glass block container, in millimeters

The Transform stage converts from ABSTRACT to PHYSICAL. Generators always
produce ABSTRACT coordinates. Exported files contain PHYSICAL coordinates.
