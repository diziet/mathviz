# Python API

MathViz can be used as a Python library for programmatic access to generators,
the pipeline, and individual processing stages.

## Running the Full Pipeline

```python
from mathviz.core.container import Container, PlacementPolicy
from mathviz.pipeline.runner import ExportConfig, run

result = run(
    generator="lorenz",
    params={"sigma": 12.0, "rho": 28.0},
    seed=42,
    container=Container(width_mm=100, height_mm=100, depth_mm=100),
    placement=PlacementPolicy(),
    export_config=ExportConfig(path="lorenz.ply"),
)

# Check validation
print(f"Passed: {result.validation.passed}")

# Access timings
for stage, seconds in result.timings.items():
    print(f"  {stage}: {seconds:.3f}s")

# Access the final geometry
obj = result.math_object
print(f"Has mesh: {obj.mesh is not None}")
print(f"Has point cloud: {obj.point_cloud is not None}")
```

## Listing Generators

```python
from mathviz.core.generator import list_generators, get_generator_meta

# List all generators
for meta in list_generators():
    print(f"{meta.name} ({meta.category}): {meta.description}")

# Get info about a specific generator
meta = get_generator_meta("lorenz")
print(f"Aliases: {meta.aliases}")
print(f"Resolution params: {meta.resolution_params}")
```

## Running a Generator Directly

```python
from mathviz.core.generator import get_generator

# Get the generator class
gen_class = get_generator("lorenz")

# Create an instance
gen = gen_class.create(resolved_name="lorenz")

# View default parameters
defaults = gen.get_default_params()
print(defaults)  # {'sigma': 10.0, 'rho': 28.0, 'beta': 2.667, ...}

# View parameter schema
schema = gen.get_param_schema()

# Generate raw geometry
obj = gen.generate(params={"sigma": 12.0}, seed=42)
print(f"Coord space: {obj.coord_space}")  # ABSTRACT
```

## Individual Pipeline Stages

### Representation

```python
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.pipeline import representation_strategy

# Get default representation for a generator
config = representation_strategy.get_default("lorenz")

# Or create a custom one
config = RepresentationConfig(
    type=RepresentationType.TUBE,
    tube_radius=0.15,
    tube_sides=24,
)

# Apply representation
obj = representation_strategy.apply(obj, config)
```

### Transform

```python
from mathviz.core.container import Container, PlacementPolicy
from mathviz.pipeline.transformer import fit

container = Container(width_mm=100, height_mm=100, depth_mm=100)
policy = PlacementPolicy(anchor="center", depth_bias=1.2)

obj = fit(obj, container, policy)
# obj is now in PHYSICAL coordinate space
```

### Sample

```python
from mathviz.pipeline.sampler import SamplerConfig, SamplingMethod, sample

config = SamplerConfig(
    method=SamplingMethod.UNIFORM_SURFACE,
    num_points=50000,
    seed=42,
)

obj = sample(obj, config)
print(f"Points: {len(obj.point_cloud.points)}")
```

### Validate

```python
from mathviz.core.container import Container
from mathviz.core.engraving import EngravingProfile
from mathviz.core.validator import validate_mesh, validate_engraving

# Mesh validation
if obj.mesh is not None:
    result = validate_mesh(obj.mesh, container=Container())
    for check in result.checks:
        print(f"{check.name}: {'PASS' if check.passed else 'FAIL'}")

# Engraving validation
if obj.point_cloud is not None:
    profile = EngravingProfile(point_budget=2_000_000)
    result = validate_engraving(obj.point_cloud, profile, container=Container())
    print(f"Overall: {'PASS' if result.passed else 'FAIL'}")
```

### Export

```python
from mathviz.pipeline.mesh_exporter import export_mesh
from mathviz.pipeline.point_cloud_exporter import export_point_cloud

# Export mesh to STL
if obj.mesh is not None:
    export_mesh(obj, "output.stl", fmt="stl")

# Export point cloud to PLY
if obj.point_cloud is not None:
    export_point_cloud(obj, "output.ply", fmt="ply")
```

## Configuration

### Loading Config Files

```python
from mathviz.core.config import (
    load_project_config,
    load_object_config,
    load_sampling_profile,
    resolve_config,
)

# Load project config (auto-discovers mathviz.toml)
project_cfg = load_project_config()

# Load per-object config
object_cfg = load_object_config("block_config.toml")

# Load sampling profile
profile = load_sampling_profile("production")

# Resolve all layers into a typed config
resolved = resolve_config(
    project=project_cfg,
    object_config=object_cfg,
    cli_overrides={"seed": 7},
)

print(resolved.container)   # Container(width_mm=100, ...)
print(resolved.placement)   # PlacementPolicy(anchor='center', ...)
print(resolved.seed)        # 7
```

### Config Models

All configuration models are Pydantic BaseModel subclasses (except
`ResolvedConfig` which is a dataclass). They support JSON serialization,
validation, and schema generation:

```python
from mathviz.core.container import Container

# Create with defaults
container = Container()

# Create with overrides
container = Container(width_mm=120, height_mm=120, depth_mm=60)

# Export to dict
print(container.model_dump())

# Generate JSON schema
print(container.model_json_schema())
```

## Grid Manifest

```python
from mathviz.core.grid import GridManifest, BlockStatus

# Create a new grid
manifest = GridManifest.create(rows=10, cols=10, path="grid.toml")
manifest.save()

# Load existing grid
manifest = GridManifest.load("grid.toml")

# Assign presets
manifest.assign(0, 0, "lorenz")
manifest.assign(0, 1, "gyroid", config_path="configs/gyroid.toml")

# Check status
block = manifest.get_block(0, 0)
print(f"Preset: {block.preset}, Status: {block.status}")

# Update status
manifest.set_status(0, 0, BlockStatus.EXPORTED)

# Get neighbors
neighbors = manifest.neighbors(5, 5)

# Summary
counts = manifest.summary()
print(counts)  # {'empty': 98, 'assigned': 1, 'exported': 1}

manifest.save()
```
