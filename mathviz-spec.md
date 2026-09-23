# MathViz — System Specification

## Mathematical Visualization & Laser Engraving Pipeline

**Version:** 0.4
**Purpose:** Programmatic generation of 3D mathematical forms for subsurface laser engraving in crystal glass blocks, with interactive preview and multi-format export.

---

## 1. Project Context

The project is a wall-mounted installation of 100–256 crystal glass blocks. Each block contains a different three-dimensional mathematical form, made by subsurface laser engraving. The blocks are arranged in a grid, nominally 10×10, and adjacent blocks show related forms. For example, knot complexity increases across one region, fractal zoom depth varies in another, and prime spirals change density elsewhere.

Each block is a separate sub-project. This system generates one object at a time, with configurable parameters. The human operator maps objects to grid positions outside the system.

Subsurface laser engraving focuses a laser at discrete (x, y, z) coordinates inside the glass, which creates micro-fractures. Under backlighting, the result is a monochrome point cloud of glowing fracture points. The system must produce both surface meshes (STL) and point clouds. Engraver software may accept either format, but the physical output is always a point cloud.

---

## 2. Pipeline Architecture

```
Generator ──▶ MathObject(raw) ──▶ RepresentationStrategy ──▶ Transformer ──▶ Sampler ──▶ EngravingOptimizer ──▶ Export
                                                                                │                                  │
                                                                                ▼                                  ▼
                                                                            Validate                           Preview
                                                                        (mesh + engraving)
```

### Pipeline Stages

1. **Generate** — A `Generator` evaluates a mathematical definition with the given parameters and a deterministic seed. It produces a `MathObject` that contains raw geometry in abstract coordinate space.

2. **Representation** — A `RepresentationStrategy` decides which physical form the raw geometry takes when engraved. This is a fabrication policy, and generators do not make this decision. Examples: keep a Lorenz trajectory as a raw point cloud; thicken a torus knot curve into a tube mesh; convert a gyroid implicit field into a surface shell clipped to a slab; render a Mandelbulb as a sparse shell rather than a solid fill. The strategy produces one or more candidate representations as new MathObjects. The operator selects which one to use, or the default is applied automatically.

3. **Transform** — The `Transformer` scales and centers the geometry and fits it into a physical container, defined in millimeters, according to a `PlacementPolicy`. It preserves aspect ratio by default. Depth bias, anchor mode, and perceptual corrections for the glass block are optional.

4. **Sample** — The `Sampler` converts surface meshes or volumes into point clouds at a specified density. The stage is skipped when the object is already a point cloud and no resampling is requested.

5. **Engraving Optimization** — Adjustments after sampling that are specific to the engraving medium: volumetric occlusion thinning, so that a dense object does not engrave as an opaque white volume; depth-dependent density compensation, because deeper points are less visible; and point budget enforcement.

6. **Validate** — Check mesh properties (watertight, manifold, no degenerate faces, bounding box within container) and engraving properties (point count within budget, density distribution, no degenerate clusters, visibility estimate).

7. **Export** — Write to disk in the requested format or formats. Export requires the requested representation to exist on the MathObject and never converts silently. Pass `--auto-sample` to permit implicit conversion.

8. **Preview** — Launch an interactive 3D viewer in the browser. It manages level of detail so that interaction stays responsive.

Each stage can be called on its own. The pipeline is composable, not monolithic.

---

## 3. Core Abstractions

### Validation Strategy: Pydantic for Config, Dataclasses for Geometry

The system uses two data-modeling approaches, chosen by what each model contains:

**Pydantic `BaseModel`** for configuration, metadata, and API boundaries: Container, PlacementPolicy, EngravingProfile, RepresentationConfig, Preset, all CLI/API request/response models. For these models, Pydantic's automatic validation, JSON serialization, and schema generation are useful, because they contain scalar fields (floats, ints, strings, enums) that Pydantic can fully validate.

**Plain `@dataclass`** for geometry containers: Mesh, PointCloud, Curve, MathObject. These carry `np.ndarray` fields. Pydantic's `arbitrary_types_allowed` would disable validation for those fields, which matter most. Instead, these dataclasses have explicit `validate()` methods that check array shape, dtype, NaN presence, and dimensional consistency. Code calls the validation explicitly at construction boundaries (generator output, pipeline stage transitions), not implicitly through Pydantic.

This avoids paying Pydantic's model construction overhead on million-point arrays without getting any validation from it.

### 3.1 Geometry Containers

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np

class CoordSpace(str, Enum):
    ABSTRACT = "abstract"   # unitless, as generated
    PHYSICAL = "physical"   # millimeters, fitted to container

@dataclass
class Mesh:
    vertices: np.ndarray    # (N, 3) float64
    faces: np.ndarray       # (M, 3) int
    normals: Optional[np.ndarray] = None  # (M, 3) or (N, 3)

    def validate(self) -> list[str]:
        """Return list of validation errors. Empty list = valid."""
        errors = []
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            errors.append(f"vertices shape {self.vertices.shape}, expected (N, 3)")
        if self.vertices.dtype != np.float64:
            errors.append(f"vertices dtype {self.vertices.dtype}, expected float64")
        if np.any(np.isnan(self.vertices)):
            errors.append("vertices contain NaN")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            errors.append(f"faces shape {self.faces.shape}, expected (M, 3)")
        if self.faces.dtype.kind not in ('i', 'u'):
            errors.append(f"faces dtype {self.faces.dtype}, expected integer")
        if self.faces.max() >= len(self.vertices):
            errors.append("face index out of bounds")
        if self.normals is not None:
            if self.normals.ndim != 2 or self.normals.shape[1] != 3:
                errors.append(f"normals shape {self.normals.shape}, expected (K, 3)")
        return errors

@dataclass
class PointCloud:
    points: np.ndarray      # (N, 3) float64
    normals: Optional[np.ndarray] = None
    intensities: Optional[np.ndarray] = None  # per-point scalar

    def validate(self) -> list[str]:
        errors = []
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            errors.append(f"points shape {self.points.shape}, expected (N, 3)")
        if self.points.dtype != np.float64:
            errors.append(f"points dtype {self.points.dtype}, expected float64")
        if np.any(np.isnan(self.points)):
            errors.append("points contain NaN")
        if self.intensities is not None and len(self.intensities) != len(self.points):
            errors.append(f"intensities length {len(self.intensities)} != points {len(self.points)}")
        return errors

@dataclass
class Curve:
    points: np.ndarray      # (K, 3) float64
    closed: bool = False

    def validate(self) -> list[str]:
        errors = []
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            errors.append(f"points shape {self.points.shape}, expected (K, 3)")
        if self.points.dtype != np.float64:
            errors.append(f"points dtype {self.points.dtype}, expected float64")
        return errors
```

### 3.2 MathObject

```python
@dataclass
class BoundingBox:
    min_corner: tuple[float, float, float]
    max_corner: tuple[float, float, float]

@dataclass
class MathObject:
    """Universal geometry container.

    At least one of mesh, point_cloud, or curves must be populated.
    coord_space tracks whether geometry is in abstract or physical units.
    """
    # Geometry — at least one must be populated
    mesh: Optional[Mesh] = None
    point_cloud: Optional[PointCloud] = None
    curves: Optional[list[Curve]] = None

    # Metadata
    generator_name: str = ""                # e.g. "lorenz_attractor"
    category: str = ""                      # e.g. "attractors"
    parameters: dict = field(default_factory=dict)
    seed: int = 42                          # RNG seed used
    coord_space: CoordSpace = CoordSpace.ABSTRACT
    bounding_box: BoundingBox = field(default_factory=lambda: BoundingBox((0,0,0),(0,0,0)))
    representation: Optional[str] = None    # e.g. "surface_shell", "raw_point_cloud"

    # Optional
    scalar_field: Optional[np.ndarray] = None
    description: str = ""

    # Performance
    generation_time_seconds: Optional[float] = None

    def validate(self) -> list[str]:
        """Validate all geometry components. Returns list of errors."""
        errors = []
        has_geometry = False
        if self.mesh is not None:
            has_geometry = True
            errors.extend(f"mesh: {e}" for e in self.mesh.validate())
        if self.point_cloud is not None:
            has_geometry = True
            errors.extend(f"point_cloud: {e}" for e in self.point_cloud.validate())
        if self.curves is not None:
            has_geometry = True
            for i, c in enumerate(self.curves):
                errors.extend(f"curve[{i}]: {e}" for e in c.validate())
        if not has_geometry:
            errors.append("MathObject has no geometry (mesh, point_cloud, and curves are all None)")
        return errors

    def validate_or_raise(self) -> None:
        """Validate and raise ValueError if invalid."""
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid MathObject: {'; '.join(errors)}")
```

**Coordinate space enforcement:**
- Generators return MathObjects with `coord_space=CoordSpace.ABSTRACT`
- The Transformer accepts ABSTRACT and returns PHYSICAL
- Exporters check `coord_space == CoordSpace.PHYSICAL` and raise if not

This is a runtime check on an enum field, not a generic type parameter. The generic approach (`MathObject[CoordSpace.ABSTRACT]`) added complexity without benefit: mypy cannot enforce it on dataclasses in a useful way, and the runtime check is simple and reliable. Agents get this error message: "Cannot export: MathObject is in abstract coordinate space, run the transformer first."

**Validation call sites:** `validate_or_raise()` is called at every pipeline stage boundary:
- After `Generator.generate()` returns
- After `RepresentationStrategy.apply()` returns
- After `Transformer.fit()` returns
- Before any exporter writes to disk

Malformed geometry fails at the stage that produced it, instead of in a later stage.

### 3.3 Container and PlacementPolicy

```python
from pydantic import BaseModel
from typing import Literal, Optional

class Container(BaseModel):
    """Physical glass block dimensions. Margins are always per-axis."""
    width_mm: float = 100.0      # x-axis
    height_mm: float = 100.0     # y-axis
    depth_mm: float = 100.0      # z-axis
    margin_x_mm: float = 5.0
    margin_y_mm: float = 5.0
    margin_z_mm: float = 5.0

    @property
    def usable_volume(self) -> tuple[float, float, float]:
        return (
            self.width_mm - 2 * self.margin_x_mm,
            self.height_mm - 2 * self.margin_y_mm,
            self.depth_mm - 2 * self.margin_z_mm,
        )

    @classmethod
    def with_uniform_margin(cls, w: float = 100, h: float = 100, d: float = 100, margin: float = 5) -> "Container":
        """Convenience constructor for uniform margins."""
        return cls(width_mm=w, height_mm=h, depth_mm=d,
                   margin_x_mm=margin, margin_y_mm=margin, margin_z_mm=margin)

class PlacementPolicy(BaseModel):
    """Controls how geometry is positioned and scaled within the container.

    The default glass block is 100×100×100mm. When using shallow blocks, many forms
    that are mathematically balanced in xyz will read badly when compressed. This
    policy lets the operator tune placement for the medium.
    """
    anchor: Literal["center", "front", "back", "top", "bottom", "left", "right"] = "center"
    viewing_axis: Literal["+z", "-z", "+x", "-x", "+y", "-y"] = "+z"
    preserve_aspect_ratio: bool = True
    depth_bias: float = 1.0
        # 1.0 = no distortion. <1.0 = compress depth further. >1.0 = exaggerate depth.
    offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale_override: Optional[float] = None
    rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0)
```

Margins are always per-axis. The convenience constructor `Container.with_uniform_margin()` covers the common case of equal margins. No code path checks for None to choose between a uniform margin and per-axis margins.

Default: 100×100×100mm block with 5mm margins → 90×90×90mm usable volume.

### 3.3 RepresentationStrategy

This is the architectural layer between generation and transformation. It separates the mathematical definition (what the object *is*) from the fabrication decision (how it should *look* when engraved in glass).

```python
class RepresentationType(str, Enum):
    SURFACE_SHELL = "surface_shell"         # surface mesh, no interior
    TUBE = "tube"                           # curve thickened into tube mesh
    RAW_POINT_CLOUD = "raw_point_cloud"     # native points (e.g., attractor trajectory)
    VOLUME_FILL = "volume_fill"             # interior filled with points
    SPARSE_SHELL = "sparse_shell"           # surface sampled sparsely
    SLICE_STACK = "slice_stack"             # parallel slices through volume
    WIREFRAME = "wireframe"                 # edges only, thickened into thin tubes
    WEIGHTED_CLOUD = "weighted_cloud"       # point cloud with importance-weighted density
    HEIGHTMAP_RELIEF = "heightmap_relief"   # 2D data extruded as relief surface

class RepresentationConfig(BaseModel):
    type: RepresentationType
    # Type-specific parameters:
    tube_radius: Optional[float] = None
    tube_sides: Optional[int] = 16
    shell_thickness: Optional[float] = None
    volume_density: Optional[float] = None
    slice_count: Optional[int] = None
    slice_axis: Literal["x", "y", "z"] = "z"
    wireframe_thickness: Optional[float] = None
    density_weight_function: Optional[str] = None  # e.g. "inverse_distance_from_center"

class RepresentationStrategy:
    """Decides how raw geometry should be realized for engraving."""

    @staticmethod
    def apply(obj: MathObject, config: RepresentationConfig) -> MathObject:
        """Transform raw geometry into the requested representation."""
        ...

    @staticmethod
    def get_default(generator_name: str) -> RepresentationConfig:
        """Return the default representation for a given generator.

        Examples:
            lorenz_attractor  -> RAW_POINT_CLOUD (ghostly trail, no tube)
            torus_knot        -> TUBE (tube_radius=0.3, tube_sides=16)
            gyroid             -> SURFACE_SHELL (clipped to container)
            mandelbulb        -> SPARSE_SHELL (surface only, no interior fill)
            ulam_spiral       -> WEIGHTED_CLOUD (raised particle distribution)
            voronoi           -> WIREFRAME (cell edges as thin tubes)
        """
        ...
```

**Why this matters:** Without this layer, each generator would collect its own fabrication-specific workarounds. The Lorenz generator should not know about tube thickening, and the gyroid generator should not know about slab clipping. The generator produces the mathematical geometry. The representation strategy turns it into physical geometry for engraving.

The operator can override the default representation per object with the CLI flag `--representation surface_shell` or in the config file. `--representation candidates` outputs several representations to compare visually in the preview.

### 3.4 EngravingOptimizer

Adjustments after sampling that are specific to the laser engraving medium.

```python
class EngravingProfile(BaseModel):
    """Fabrication constraints and optical corrections."""
    point_budget: int = 2_000_000           # max points for this block
    min_point_spacing_mm: float = 0.05      # minimum distance between points
    max_point_spacing_mm: float = 2.0       # maximum gap (for uniform coverage)

    # Volumetric occlusion: a dense point cloud of a solid object becomes
    # an opaque white brick. These controls thin outer layers so the viewer
    # can see internal structure.
    occlusion_mode: Literal["none", "shell_fade", "radial_gradient", "custom"] = "none"
    occlusion_shell_layers: int = 3         # for shell_fade: number of outer layers to thin
    occlusion_density_falloff: float = 0.5  # 0=fully thin outer, 1=uniform

    # Depth compensation: points deeper in the glass are less visible
    # due to light attenuation. Increase density for deeper points.
    depth_compensation: bool = False
    depth_compensation_factor: float = 1.5  # multiply density at max depth by this factor

class EngravingOptimizer:
    @staticmethod
    def optimize(cloud: PointCloud, profile: EngravingProfile, container: Container) -> PointCloud:
        """Apply engraving-specific adjustments to a point cloud."""
        ...
```

### 3.5 Generator Base Class

```python
from abc import ABC, abstractmethod
from numpy.random import Generator as RNG, default_rng

class GeneratorBase(ABC):
    """Base class for all mathematical object generators."""

    name: str               # unique identifier, e.g. "lorenz_attractor"
    category: str           # canonical category (see §4)
    aliases: list[str] = [] # alternative names for registry lookup
    description: str        # human-readable description

    # Each generator declares which resolution type(s) it uses.
    # This replaces the overloaded single `resolution` parameter.
    resolution_params: dict[str, str]  # e.g. {"grid_resolution": "UV grid density (N×N)"}

    @abstractmethod
    def get_default_params(self) -> dict:
        """Return default parameter dict with descriptions and valid ranges."""
        ...

    @abstractmethod
    def generate(self, params: dict | None = None, seed: int = 42,
                 **resolution_kwargs) -> MathObject:
        """Generate geometry.

        Args:
            params: Mathematical parameters (sigma, rho, etc.)
            seed: Deterministic RNG seed. All randomness must use
                  numpy.random.default_rng(seed). Never np.random.seed().
            **resolution_kwargs: Resolution parameters specific to this
                generator type. See resolution_params for valid keys.

        Returns:
            MathObject in abstract coordinate space.
        """
        ...

    def get_param_schema(self) -> dict:
        """Return JSON-schema-like description of all parameters for CLI/UI."""
        ...

    def get_default_representation(self) -> RepresentationConfig:
        """Return the recommended representation for engraving."""
        ...
```

**Resolution types** (generators declare which they use):

| Resolution Type | Meaning | Scaling | Used By |
|----------------|---------|---------|---------|
| `grid_resolution` | N×N UV grid for parametric surfaces | O(N²) | Parametric surfaces |
| `voxel_resolution` | N×N×N voxel grid for marching cubes | O(N³) | Implicit surfaces, Mandelbulb, Julia 3D |
| `integration_steps` | Number of ODE integration steps | O(N) | Strange attractors, double pendulum, N-body |
| `pixel_resolution` | N×N 2D grid for heightmaps | O(N²) | Mandelbrot heightmap, fractal slices |
| `curve_points` | Number of points along a curve | O(N) | Knots, Lissajous curves, spirals |
| `iteration_depth` | Max iterations for escape-time fractals | O(1) per point | Mandelbulb, Julia, Mandelbrot |

A generator may use several resolution types. For example, Mandelbulb uses both `voxel_resolution` and `iteration_depth`. Each type has its own default, and the CLI exposes each as a separate flag.

**Deterministic seeding:** Every generator that uses randomness (Voronoi seed points, noise fields, reaction-diffusion initial conditions, n-body with chaotic sensitivity) must accept a `seed` parameter and use only `numpy.random.default_rng(seed)`. It must never call the global `np.random.seed()`. The seed is recorded in `MathObject.seed` and in the sidecar metadata, so the object can be reproduced exactly.

Deterministic generators (parametric surfaces, knots, pure-math fractals) accept the seed parameter but do not use it. The seed still appears in their metadata, for consistency across the pipeline.

### 3.6 Exporters

Exporters are split into separate classes and never convert silently. Export fails when the requested representation does not exist on the MathObject, unless `--auto-sample` is passed.

```python
class MeshExporter:
    """Exports mesh geometry. Fails if MathObject has no mesh."""

    @staticmethod
    def to_stl(obj: MathObject, path: str) -> None:
        """Binary STL. Always binary — no ASCII option."""
        ...

    @staticmethod
    def to_obj(obj: MathObject, path: str) -> None: ...

    @staticmethod
    def to_ply_mesh(obj: MathObject, path: str, binary: bool = True) -> None: ...

class PointCloudExporter:
    """Exports point cloud geometry. Fails if MathObject has no point_cloud."""

    @staticmethod
    def to_ply(obj: MathObject, path: str, binary: bool = True) -> None: ...

    @staticmethod
    def to_xyz(obj: MathObject, path: str) -> None: ...

    @staticmethod
    def to_pcd(obj: MathObject, path: str) -> None: ...

class MetadataExporter:
    """Exports MathObject metadata for reproducibility."""

    @staticmethod
    def to_json(obj: MathObject, path: str) -> None:
        """Full parameter dict, generator name, seed, container spec, timestamp,
        generation time, pipeline stages applied, and version info."""
        ...
```

STL is always binary, so `to_stl` has no `binary` parameter. Every geometry export also writes a sidecar `.meta.json`.

---

## 4. Generator Taxonomy

### Canonical Homes and Aliases

Each generator has exactly one canonical location in the directory tree. A generator that could belong to several categories lives in the category that best matches its primary mathematical identity. The registry supports aliases, so a generator can be called by another name.

**Alias resolution rule:** The registry maps every name, canonical or alias, to the same generator class. `mathviz generate torus_knot` and `mathviz generate trefoil` resolve to the same generator with different default parameters.

| Generator | Canonical Category | Aliases |
|-----------|-------------------|---------|
| `torus_knot` | `knots/` | `trefoil` (p=2,q=3), `cinquefoil` (p=2,q=5) |
| `lissajous_knot` | `knots/` | — |
| `lissajous_curve` | `curves/` | — |
| `lissajous_surface` | `parametric/` | — |
| `double_pendulum` | `attractors/` | — |

The object's mathematical identity decides its canonical home. A torus knot is a knot that is defined parametrically, so it lives in `knots/`. A Lissajous surface is a surface that shares a name with Lissajous curves, so it lives in `parametric/`.

### 4.1 Parametric Surfaces

**Definition:** A function `f: (u, v) → (x, y, z)` evaluated over a parameter domain.

**Resolution type:** `grid_resolution` (default: 128). O(N²) scaling.

**Shared implementation pattern:**
1. Define `f(u, v, **params) -> (x, y, z)`
2. Evaluate on a regular `(u, v)` grid of size `grid_resolution × grid_resolution`
3. Build triangle mesh from grid (two triangles per grid cell)
4. Optionally close the mesh at boundaries (for periodic surfaces)

**Default representation:** `SURFACE_SHELL`

| Name | Key Parameters | Notes |
|------|---------------|-------|
| `klein_bottle` | immersion type (figure-8, Lawson) | Not embeddable in 3D; immersion has self-intersection |
| `mobius_strip` | half-twists, width | Boundary is a single closed curve |
| `torus` | major radius R, minor radius r | R/r ratio controls appearance |
| `superellipsoid` | e1 (squareness lat), e2 (squareness lon) | Interpolates sphere ↔ cube ↔ octahedron |
| `spherical_harmonics` | l, m (degree, order) or coefficient vector | Deformation of a sphere |
| `lissajous_surface` | a, b, c (frequencies), δ₁, δ₂ (phases) | Extension of Lissajous curves to surfaces |
| `boy_surface` | — | Immersion of real projective plane |
| `enneper_surface` | order | Minimal surface; complexity increases with order |

### 4.2 Implicit Surfaces

**Definition:** The zero-level set of a function `f: (x, y, z) → ℝ`, extracted by marching cubes.

**Resolution type:** `voxel_resolution` (default: 128). O(N³) scaling. Agents must account for the cubic cost.

**Shared implementation pattern:**
1. Define `f(x, y, z, **params) -> float`
2. Evaluate on a regular 3D grid of size `voxel_resolution³`
3. Run marching cubes at isolevel=0 (via `skimage.measure.marching_cubes`)
4. Optionally smooth (Laplacian) and decimate

**Default representation:** `SURFACE_SHELL`

| Name | Key Parameters | Notes |
|------|---------------|-------|
| `gyroid` | cell size, number of periods | TPMS; `sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) = 0` |
| `schwarz_p` | cell size, periods | TPMS; `cos(x) + cos(y) + cos(z) = 0` |
| `schwarz_d` | cell size, periods | Diamond TPMS |
| `costa_surface` | genus | Weierstrass representation; may need parametric approach |
| `genus2_surface` | — | Various constructions |

**TPMS note:** These surfaces tile space without end. The `periods` parameter sets how many unit cells are included. Marching cubes clips the surface to the evaluation box.

### 4.3 Strange Attractors / Dynamical Systems

**Definition:** Trajectories of ODEs or iterated maps, producing curves in 3D space.

**Resolution type:** `integration_steps` (default: 100,000). O(N) scaling.

**Shared implementation pattern:**
1. Define the system of ODEs: `dx/dt = f(x, y, z, **params)`
2. Integrate using `scipy.integrate.solve_ivp` (RK45 or DOP853)
3. Output is a `Curve` (polyline of N points)
4. Representation strategy decides: raw point cloud, tube, or weighted cloud

**Default representation:** `RAW_POINT_CLOUD`. Raw trajectory points give a faint, translucent look, which works best in glass. `--representation tube` thickens the trajectory into a tube for objects that should look solid.

| Name | Equations | Key Parameters |
|------|-----------|---------------|
| `lorenz_attractor` | dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy-βz | σ=10, ρ=28, β=8/3 |
| `rossler_attractor` | dx/dt = -y-z, dy/dt = x+ay, dz/dt = b+z(x-c) | a=0.2, b=0.2, c=5.7 |
| `chen_attractor` | dx/dt = a(y-x), dy/dt = (c-a)x-xz+cy, dz/dt = xy-bz | a=35, b=3, c=28 |
| `aizawa_attractor` | 6-parameter system | — |
| `thomas_attractor` | dx/dt = sin(y)-bx, etc. (cyclic) | b=0.208186 |
| `halvorsen_attractor` | dx/dt = -ax-4y-4z-y², etc. | a=1.89 |
| `double_pendulum` | 4D phase space, project to 3D | m1, m2, l1, l2, initial angles |

**Attractor-specific considerations:**
- **Transient removal:** Discard the first N steps (configurable, default ~1000) before recording.
- **Multiple trajectories:** Optionally integrate from multiple initial conditions (controlled by seed) for denser coverage.
- **Both representations available:** Every attractor can be exported as a raw point cloud or a tube mesh. The CLI flag `--representation tube --tube-radius 0.3` overrides the default.

### 4.4 Fractals

**Resolution types:** `voxel_resolution` (for 3D extraction) + `iteration_depth` (escape-time cutoff). `pixel_resolution` for 2D heightmaps. For agents: `voxel_resolution=256` means 256³ = 16M evaluations, so use `numba` for the inner loop (see §10.2).

#### 4.4.1 Mandelbulb

**Default representation:** `SPARSE_SHELL` (surface only, no interior fill). A solid Mandelbulb engraves as an opaque white volume. The sparse shell keeps the fractal surface detail.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `power` | 8 | The "n" in the Mandelbulb formula |
| `max_iterations` | 20 | Escape-time cutoff |
| `bailout` | 2.0 | Escape radius |
| `center` | (0,0,0) | Center of evaluation volume |
| `zoom` | 1.0 | Inverse of evaluation volume radius |

#### 4.4.2 Mandelbrot Cross-Sections

**Resolution type:** `pixel_resolution` (default: 512). O(N²) scaling.

**Default representation:** `HEIGHTMAP_RELIEF`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `center_real` | -0.5 | Real-axis center |
| `center_imag` | 0.0 | Imaginary-axis center |
| `zoom` | 1.0 | Zoom level (higher = deeper) |
| `max_iterations` | 256 | Escape-time cutoff |
| `height_scale` | 1.0 | Vertical exaggeration |
| `smoothing` | True | Use smooth iteration count |

#### 4.4.3 3D Julia Sets

Same approach as Mandelbulb, but with a fixed c parameter. Default representation: `SPARSE_SHELL`.

#### 4.4.4 Fractal Cross-Sections

Slice a 3D fractal with a plane at various angles and offsets. This reuses the 3D evaluator and outputs the 2D slice as a heightmap or contour. Default representation: `HEIGHTMAP_RELIEF`.

### 4.5 Knot Theory

**Resolution type:** `curve_points` (default: 1024). O(N) scaling.

**Default representation:** `TUBE` (tube_radius=0.3, tube_sides=16)

| Name | Parametric Form | Key Parameters |
|------|----------------|---------------|
| `torus_knot` | Standard torus knot formula | p, q, R, r. Aliases: `trefoil` (p=2,q=3), `cinquefoil` (p=2,q=5) |
| `figure_eight_knot` | Explicit parametric formula | — |
| `lissajous_knot` | `(cos(nₓt+φₓ), cos(nᵧt+φᵧ), cos(n_zt+φ_z))` | frequencies, phases |
| `seven_crossing_knots` | Various; may need explicit coordinate data | knot_index |

**Knot-specific considerations:**
- Detect self-intersection after tube thickening. Warn, but do not fail.
- Progressions: `torus_knot` with a list of (p,q) pairs generates a sequence of knots for adjacent grid positions.

### 4.6 Number Theory / Constants

**Resolution type:** varies per generator, and each generator documents its own.

**Default representation:** varies, typically `WEIGHTED_CLOUD` or `HEIGHTMAP_RELIEF`.

| Name | Approach | Default Representation |
|------|----------|----------------------|
| `ulam_spiral` | Integers on spiral; primes elevated | `WEIGHTED_CLOUD` |
| `sacks_spiral` | Archimedean spiral, primes marked | `WEIGHTED_CLOUD` |
| `prime_gaps` | Consecutive gaps mapped to 3D ribbon | `TUBE` |
| `digit_encoding` | Digits of π/e/constants → point positions/heights | `WEIGHTED_CLOUD` |

### 4.7 Curves and Dynamics

**Resolution type:** `curve_points` (default: 1024) or `integration_steps`.

**Default representation:** `TUBE` (thin tube, radius varies per curve)

| Name | Approach | Key Parameters |
|------|----------|---------------|
| `lissajous_curve` | Parametric 3D curve | frequencies a, b, c; phase δ |
| `parabolic_envelope` | Family of lines, envelope as surface | number of lines, curvature |
| `cardioid` | Parametric curve | — |
| `logarithmic_spiral` | r = ae^(bθ) | a, b, turns |
| `fibonacci_spiral` | Discrete spiral of arcs | number of arcs |

### 4.8 Geometry and Topology

| Name | Approach | Default Representation |
|------|----------|----------------------|
| `voronoi_3d` | 3D Voronoi, extract cell boundaries | `WIREFRAME` |
| `generic_parametric` | User-supplied `f(u,v) → (x,y,z)` | `SURFACE_SHELL` |

`voronoi_3d` requires a seed for point placement, which is deterministic through `default_rng(seed)`.

### 4.9 Physics and Astronomy

**Resolution type:** `integration_steps` for N-body, `curve_points` for Keplerian.

| Name | Approach | Default Representation |
|------|----------|----------------------|
| `kepler_orbit` | Elliptical orbits as 3D curves | `TUBE` |
| `nbody` | Numerical integration of N-body problem | `RAW_POINT_CLOUD` |
| `planetary_positions` | Solar system at epoch, spheres + trails | `TUBE` + point markers |

### 4.10 Data-Driven Forms

| Name | Input | Default Representation |
|------|-------|----------------------|
| `heightmap` | GeoTIFF or image | `HEIGHTMAP_RELIEF` |
| `building_extrude` | GeoJSON | `SURFACE_SHELL` |
| `soundwave` | WAV/MP3 | `TUBE` or `SURFACE_SHELL` |

### 4.11 Procedural and Generative

All require `seed` for reproducibility.

| Name | Approach | Default Representation |
|------|----------|----------------------|
| `noise_surface` | Perlin/Simplex noise as implicit surface or heightmap | `SURFACE_SHELL` |
| `terrain` | Noise-based terrain generation | `HEIGHTMAP_RELIEF` |
| `reaction_diffusion` | Gray-Scott on 2D surface, mapped to 3D | `HEIGHTMAP_RELIEF` |

---

## 5. Shared Components

### 5.1 Tube Thickening

Used by: `RepresentationStrategy` when `type=TUBE`.

**Input:** Curve (polyline, N×3 array)
**Output:** Mesh (tube surface)
**Algorithm:** Parallel transport frame extrusion. Not Frenet-Serret: Frenet frames are undefined at inflection points and produce twisting artifacts.
**Parameters:** `radius`, `sides` (default 16), `closed` (default True for closed curves)

### 5.2 Marching Cubes

Used by: all implicit surface generators, Mandelbulb, 3D Julia sets.

**Wraps:** `skimage.measure.marching_cubes`
**Input:** 3D scalar field (voxel_resolution³ array) + isolevel
**Output:** Mesh
**Post-processing:** vertex normal computation, optional Laplacian smoothing, optional mesh decimation

### 5.3 Point Cloud Sampler

**Algorithms:**

1. **Uniform surface sampling** — Poisson disk sampling on mesh surface. Parameter: `density` (points/mm²) or `num_points`.
2. **Random surface sampling** — Random barycentric sampling, face-area-weighted. Faster, less uniform.
3. **Volume fill** — Fill interior of watertight mesh with jittered grid. Parameter: `density` (points/mm³).

**Default:** Uniform surface sampling at 10 points/mm².

**Library choice:** `trimesh.sample.sample_surface` for surface sampling, which adds no heavy dependency. `open3d` is an **optional** dependency, installed only for PCD export and advanced point cloud processing. The core pipeline does not require it.

### 5.4 Validation

Validation has two tiers: mesh validation and engraving validation.

#### Mesh Validation

- Watertight (all edges shared by exactly two faces)
- Manifold (no edge shared by >2 faces, no isolated vertices)
- No degenerate faces (zero-area or near-zero)
- Consistent normals
- Bounding box within container

Repair options (best-effort): fill small holes, remove degenerate faces, fix normals. Uses `trimesh.repair`.

#### Engraving Validation

- Point count within budget (default 2M, configurable)
- No point outside container volume
- Minimum point spacing (no clusters denser than min_point_spacing_mm)
- Maximum gap check (no region of usable volume >max_point_spacing_mm from nearest point, for objects that should have continuous coverage)
- Estimated visual density: flag if >70% of voxels in any axis-aligned projection are occupied (opacity warning: the object may look like a solid white block)
- Depth distribution: report what fraction of points are in each depth quartile

Returns a `ValidationResult` with pass/fail per check, severity (error/warning/info), and human-readable messages.

### 5.5 Transformer (Bounding Box Fitting)

**Input:** `MathObject[CoordSpace.ABSTRACT]` + `Container` + `PlacementPolicy`
**Output:** `MathObject[CoordSpace.PHYSICAL]`

**Algorithm:**
1. Optionally apply rotation (from PlacementPolicy)
2. Compute axis-aligned bounding box of rotated geometry
3. Compute usable volume of the container (after margins)
4. Apply depth_bias: multiply the z-axis usable extent by `depth_bias` before computing scale
5. Compute uniform scale factor: `scale = min(usable[i] / bbox_size[i] for i in xyz)` (or per-axis if `preserve_aspect_ratio=False`)
6. Center at container center (or anchor point per policy)
7. Apply offset

---

## 6. Export Formats

| Format | Extension | Exporter | Content | Use Case |
|--------|-----------|----------|---------|----------|
| STL | `.stl` | `MeshExporter` | Triangle mesh (binary) | Engraver input (mesh mode) |
| OBJ | `.obj` | `MeshExporter` | Triangle mesh | 3D software interop |
| PLY (mesh) | `.ply` | `MeshExporter` | Triangle mesh | Archival |
| PLY (cloud) | `.ply` | `PointCloudExporter` | Point cloud | Engraver input (point cloud mode) |
| XYZ | `.xyz` | `PointCloudExporter` | Point cloud (ASCII) | Simple exchange |
| PCD | `.pcd` | `PointCloudExporter` | Point cloud (PCL) | PCL interop (requires open3d) |
| JSON | `.meta.json` | `MetadataExporter` | Parameters + pipeline metadata | Reproducibility |

**Export rules:**
- `MeshExporter.to_stl(obj)` fails with a clear error if `obj.mesh` is None.
- `PointCloudExporter.to_ply(obj)` fails if `obj.point_cloud` is None.
- Pass `--auto-sample` on the CLI to permit implicit mesh→cloud conversion at export time.
- Every geometry export writes a sidecar `.meta.json` automatically.
- The `.meta.json` includes: generator name, full parameters dict, seed, container spec, placement policy, representation config, pipeline stages applied, generation time, export timestamp, MathViz version.

---

## 7. Studio — Interactive Exploration Workbench

This project produces an artwork. Before engraving, the operator needs to see, rotate, adjust, compare, and select from 100+ mathematical forms. The **Studio** is a browser-based GUI application for exploring the full generator library, tuning parameters interactively, saving configurations, and queueing production exports. It is a primary interface, equal to the CLI, not an add-on to it.

The CLI remains the stable interface for agents and batch automation. The Studio is the interface for the human artist.

The Studio has two tiers. **Tier 1** is a minimal, working viewer built alongside the core pipeline. Agents can build it in one phase, and it provides about 80% of the value for artistic exploration. **Tier 2** is the full React application with presets, gallery, comparison, and export queue. It is built later, likely with more hands-on work, after the operator has generated 20–30 blocks with the CLI and Tier 1 and has concrete UX requirements.

### 7.1 Tier 1: Preview Viewer (build with pipeline)

A single HTML page with a Three.js viewport and basic controls, served by a small FastAPI server. It is the minimum tool that supports artistic exploration.

**Architecture:**

```
Single HTML page (vanilla JS + Three.js)  ◄── HTTP ──►  Python Backend (FastAPI)
       │                                                        │
       │  • Three.js viewport                                  │  • Generator registry
       │  • View mode toggles                                   │  • Pipeline runner
       │  • Container wireframe                                 │  • LOD decimation
       │  • URL query params for generator + params             │  • GLB/binary PLY serving
       │  • Screenshot button                                   │
       └────────────────────────────────────────────────────────┘
```

**Technology:** FastAPI backend. Tier 2 needs it anyway, and it uses Pydantic natively for the config models. The frontend is a single `index.html` with inline JS: no build step, no npm, no React. Three.js loads from a CDN. This choice is critical for agents: an agent can produce and debug a single HTML file with vanilla JS much more easily than a React + r3f + Zustand application.

**Features:**

- Three.js OrbitControls for rotate / pan / zoom
- Toggle between: shaded mesh, wireframe, point cloud
- Container bounding box wireframe (glass block outline with margins)
- Background toggle: dark (simulates backlit glass) / light
- Point size slider (for point cloud view)
- Screenshot button (PNG at viewport resolution)
- Info display: generator name, face count, point count, generation time
- Axis indicator

**Geometry loading:**
- `mathviz preview <file>` — loads a specific STL/PLY file into the viewer
- `mathviz preview <generator_name>` — generates at preview resolution, serves result
- URL query params: `?generator=lorenz_attractor&sigma=12&rho=28`, so a configuration can be bookmarked
- Hot-reload: the viewer watches the file and reloads it when it changes

**LOD strategy:**
- Server-side decimation: meshes ≤100K faces, point clouds ≤200K points
- "High-res" button loads full geometry with loading spinner
- Transfer format: GLB for meshes (Three.js native), binary PLY for point clouds

**What Tier 1 does NOT include:** No parameter sliders, no preset management, no gallery, no comparison view, no WebSocket, no SQLite. To change a parameter, rerun the CLI and refresh the browser, or edit the URL query params. Presets are TOML files that the operator manages by hand. Tier 1 is deliberately minimal: it works, it is fast, and it needs no React build toolchain.

CLI commands:
```
mathviz preview <file_or_generator> [--port N] [--no-open]
    # Launches Tier 1 viewer with the specified file or generator
```

### 7.2 Tier 2: Full Studio (build later, possibly different tooling)

The full Studio application with parameter controls, presets, gallery, comparison, and export queue. It is built after the operator has used Tier 1 and the CLI for 20–30 blocks and has concrete UX requirements from that work.

**Architecture:**

```
Browser (React + Three.js)  ◄──── WebSocket ────►  Python Backend (FastAPI)
       │                                                    │
       │  • 3D viewer (Three.js)                           │  • Generator registry
       │  • Parameter controls                              │  • Pipeline runner
       │  • Preset manager                                  │  • Session/preset storage
       │  • Gallery / comparison                            │  • LOD decimation
       │  • Export queue                                     │  • Export execution
       │                                                    │
       └────────────────────────────────────────────────────┘
```

**Technology:**
- **Backend:** FastAPI (same server as Tier 1, extended with WebSocket + preset endpoints)
- **Frontend:** React (Vite) + Three.js (via `@react-three/fiber` and `@react-three/drei`) + Zustand for state
- **Communication:** WebSocket for live parameter updates and generation progress. REST for CRUD.
- **Storage:** SQLite (stdlib) for presets, generation history, gallery metadata

**Why this requires different tooling:** For an agent, a React + Three.js + FastAPI + WebSocket + SQLite application with debounced regeneration and LOD switching is much harder to build than the Python pipeline. r3f state management with live WebSocket geometry updates needs fast visual iteration, which is where agents are weakest. The spec below is detailed enough for a strong agent to produce the skeleton, but the UX polish will almost certainly need hands-on work.

### 7.3 Tier 2 Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  MathViz Studio                                          [settings] │
├──────────────┬───────────────────────────────────┬───────────────────┤
│              │                                   │                   │
│  Generator   │        3D Viewport                │   Parameters      │
│  Browser     │                                   │                   │
│              │   ┌───────────────────────────┐   │   [Generator]     │
│  ▸ Parametric│   │                           │   │   [Math params]   │
│  ▸ Implicit  │   │                           │   │   [Resolution]    │
│  ▸ Attractors│   │      Three.js scene       │   │   [Representation]│
│  ▸ Fractals  │   │                           │   │   [Container]     │
│  ▸ Knots     │   │                           │   │   [Placement]     │
│  ▸ Num Theory│   │                           │   │   [Engraving]     │
│  ▸ Curves    │   │                           │   │   [Seed]          │
│  ▸ Physics   │   └───────────────────────────┘   │                   │
│  ▸ Data      │   [mesh] [wireframe] [cloud]      │   ── Presets ──   │
│  ▸ Procedural│   [container] [dark/light]         │   [Save as...]   │
│              │   [screenshot] [high-res]          │   [Load preset]  │
│              │                                   │   [Compare]       │
│  ── Recent ──│   ── Info bar ──                  │                   │
│  lorenz_v3   │   faces: 52K | pts: 0 | 0.4s     │   ── Actions ──   │
│  gyroid_v1   │   validated: ✓ | warnings: 0      │   [Generate]      │
│  trefoil_v2  │                                   │   [Export STL]    │
│              │                                   │   [Export Cloud]  │
├──────────────┴───────────────────────────────────┴───────────────────┤
│  Gallery / Comparison View (expandable)                              │
└──────────────────────────────────────────────────────────────────────┘
```

### 7.4 Tier 2 Features

#### Generator Browser (left panel)

- Tree view of all generators organized by category
- Search/filter by name
- Each generator shows: name, one-line description, small thumbnail (cached)
- Click to select → loads defaults, generates preview, displays in viewport
- Recently used generators pinned at bottom

#### 3D Viewport (center)

Same rendering capabilities as Tier 1 (OrbitControls, view modes, container wireframe, backgrounds, screenshot), plus:
- Parameter changes regenerate at preview resolution with debounced updates
- Transform-only changes (placement, rotation, offset) update instantly without regeneration

#### Parameter Panel (right side)

Auto-generated from the generator's Pydantic parameter schema. Organized into collapsible sections:

- **Generator Parameters** — mathematical parameters (e.g., σ, ρ, β for Lorenz). Rendered as: float → slider + numeric input, int → stepper, bool → toggle, Literal → dropdown, tuple → linked inputs.
- **Resolution** — resolution parameters this generator uses, with logarithmic sliders and cost estimates
- **Representation** — dropdown + type-specific sub-parameters
- **Container** — dimensions and per-axis margins, with live bounding box update
- **Placement** — anchor, depth bias, rotation, offset — all instant (transform-only)
- **Engraving** — point budget, occlusion, depth compensation
- **Seed** — numeric input + "randomize" button + "increment" button

**Update behavior:**
- Transform/placement changes: instant re-render (no regeneration)
- Math/resolution changes: debounced (~300ms) regeneration at preview resolution
- "Regenerate" button: explicit full pipeline run

#### Preset System

A preset is a complete, named, saveable pipeline configuration.

```python
class Preset(BaseModel):
    """A saved configuration for a generator + pipeline. (Pydantic — this is config, not geometry.)"""
    id: str                             # UUID
    name: str                           # e.g. "lorenz_butterfly_v3"
    description: str = ""
    generator_name: str
    params: dict
    seed: int
    resolution_kwargs: dict
    representation: RepresentationConfig
    container: Container
    placement: PlacementPolicy
    engraving: EngravingProfile
    sampling_profile: str               # "preview" | "production" | "custom"
    tags: list[str] = []
    created_at: datetime
    updated_at: datetime
    thumbnail_path: Optional[str] = None
    notes: str = ""

    # Camera state (return to exactly the same view)
    camera_position: Optional[tuple[float, float, float]] = None
    camera_target: Optional[tuple[float, float, float]] = None
    camera_zoom: Optional[float] = None

    # Grid assignment (see §9.4)
    grid_position: Optional[tuple[int, int]] = None  # (row, col) in the installation grid
```

**Preset operations:** Save / Save as variant / Load / Quick-save (Cmd+S) / Export as TOML / Delete / Rename / Tag.

**Storage:** SQLite in the project directory (`mathviz_studio.db`). Presets can also be exported as TOML for CLI use and version control.

#### Gallery View

Expandable bottom panel or full-page view showing saved presets as a thumbnail grid. Filter by category, tags, date. Sort by name, date, generator. Click → load into main view. Multi-select for comparison. Drag-and-drop reordering for layout planning.

#### Comparison Mode

Select 2–4 presets to view side by side in split viewports. Each viewport has its own orbit controls, and an optional toggle links the cameras. Each viewport shows the preset name and highlights the key parameters that differ.

#### Export Queue

- "Export STL" / "Export Point Cloud" buttons → export dialog (format, sampling profile, output path)
- Background thread execution with progress bar
- Several exports can be queued
- Batch export from gallery multi-select

### 7.5 API Endpoints (shared by Tier 1 and Tier 2)

The FastAPI backend exposes a REST API. Tier 1 uses a subset; Tier 2 uses all of it.

#### Tier 1 endpoints (minimal)

```
GET    /api/generators                     # list all generators
GET    /api/generators/{name}              # generator info + param schema
POST   /api/generate                       # run pipeline, return geometry URLs
GET    /api/geometry/{id}/mesh?lod=preview  # decimated mesh (GLB)
GET    /api/geometry/{id}/cloud?lod=preview # subsampled point cloud (binary PLY)
GET    /api/geometry/{id}/mesh?lod=full     # full mesh
GET    /api/geometry/{id}/cloud?lod=full    # full point cloud
```

#### Tier 2 additional endpoints

```
POST   /api/generate/dry-run               # estimate without running
GET    /api/presets                         # list all presets
POST   /api/presets                         # create preset
GET    /api/presets/{id}                    # get preset
PUT    /api/presets/{id}                    # update preset
DELETE /api/presets/{id}                    # delete preset
POST   /api/presets/{id}/export-toml       # export as CLI-compatible TOML
POST   /api/presets/{id}/export-geometry    # queue production export
GET    /api/gallery                         # gallery with thumbnails
POST   /api/gallery/batch-export            # batch export
GET    /api/history                         # recent generations with timing
GET    /api/grid                            # grid manifest
PUT    /api/grid                            # update grid manifest
```

#### WebSocket (Tier 2 only)

```
ws://localhost:{port}/ws

Client → Server:
  { "type": "generate", "generator": "lorenz_attractor", "params": {...}, "seed": 42 }
  { "type": "transform", "placement": {...}, "container": {...} }
  { "type": "update_param", "key": "sigma", "value": 12.0 }

Server → Client:
  { "type": "progress", "stage": "generate", "percent": 45 }
  { "type": "geometry_ready", "mesh_url": "/api/geometry/abc/mesh?lod=preview", "stats": {...} }
  { "type": "validation", "result": {...} }
  { "type": "error", "message": "..." }
```

### 7.6 Geometry Transfer Format

**For meshes:** GLB (binary glTF). Three.js loads it natively, and trimesh exports it. It is compact and fast.

**For point clouds:** A custom binary format: a header (uint32 point count) followed by packed float32 XYZ triplets. For this use it is simpler and faster than PLY. 200K points take ~2.4 MB.

Both load in <1 second on localhost.

### 7.7 Session and State Management

**Session state** (in-memory, lost on server restart): current selection, current geometry cache.

**Persistent state** (SQLite, Tier 2 only): presets, generation history, gallery, grid manifest, user preferences.

**Geometry cache:** LRU in temp directory, keyed by hash(generator + params + seed + resolution). Default 1GB limit.

### 7.8 Static Image Rendering

PyVista (VTK) for high-resolution offline renders. CLI: `mathviz render <file> --width 4096 --height 4096 --output render.png`. Also available in the Tier 2 Studio.

### 7.9 2D Rendering

Projection or native 2D evaluation. CLI: `mathviz render-2d <file_or_generator> --projection top --output flat.png`. Tier 2 Studio: "2D projection" toggle in viewport.

---

## 8. CLI Interface

Built with **Typer**. Typer validates parameters from type hints, and agents find it easier to use than Click.

```
mathviz generate <generator_name>
    [--param key=value ...]
    [--seed N]
    [--grid-resolution N] [--voxel-resolution N] [--integration-steps N]
    [--pixel-resolution N] [--curve-points N] [--iteration-depth N]
    [--container WxHxD] [--margin-x M] [--margin-y M] [--margin-z M]
    [--representation TYPE] [--tube-radius R] [--tube-sides N]
    [--anchor center|front|back|...] [--rotation X,Y,Z] [--depth-bias F]
    [--offset X,Y,Z]
    [--point-budget N]
    [--sampling-profile preview|production|custom]
    [--output path] [--format stl|ply|xyz|pcd|obj]
    [--auto-sample]
    [--preview]
    [--no-validate]
    [--dry-run]
    [--report path.json]
    [--json]
    [--verbose] [--quiet]

mathviz studio [--port N] [--no-open]
mathviz list [--category CATEGORY] [--json]
mathviz info <generator_name> [--json]
mathviz preview <file_or_generator> [--port N]
mathviz render <file> [--width W] [--height H] [--output path.png]
mathviz render-2d <file> [--projection top|front|side|angle] [--output path.png]
mathviz validate <file> [--engraving] [--point-budget N] [--json]
mathviz convert <input> <output> [--auto-sample] [--density D]
mathviz sample <input> <output> [--density D] [--method uniform|random|volume] [--point-budget N]
mathviz transform <input> <output> [--container WxHxD] [--anchor A] [--rotation X,Y,Z] [--depth-bias F]
mathviz grid show [--json]
mathviz grid assign <row> <col> <preset_or_config>
mathviz grid status <row> <col> <status>
mathviz grid neighbors <row> <col> [--json]
mathviz grid summary [--json]
mathviz grid export-all [--sampling-profile production] [--output-dir exports/]
```

### Key CLI flags

| Flag | Purpose |
|------|---------|
| `mathviz studio` | Launch the Tier 2 Studio (§7.2–7.4): the full exploration GUI with parameter controls, presets, gallery, comparison, and export queue. Available only after Tier 2 is built. Default port 8457. |
| `mathviz preview` | Launch the Tier 1 preview viewer (§7.1): a minimal Three.js viewport with orbit controls, view mode toggles, and the container wireframe. Available from Phase 4. |
| `--dry-run` | Run the full pipeline without writing files. Print what would be generated: generator, params, resolution, estimated point/face count, estimated generation time. Agents rely on it when iterating. |
| `--report path.json` | Write a structured JSON report of the full pipeline run: timing per stage, validation results, output file paths, parameters used. |
| `--json` | Every command writes structured, machine-readable JSON to stdout. Agents are part of the development loop, so this is required from the start. |
| `--representation TYPE` | Override the default representation strategy for this generator. |
| `--sampling-profile` | `preview` (fast, ~100K points), `production` (full density, ~2M points), or `custom` (use explicit density/budget flags). |
| `--point-budget N` | Hard cap on output point count. |
| `--auto-sample` | Allow implicit mesh→cloud conversion at export. Without this, exporting a mesh-only object to XYZ format fails with an explicit error. |
| `--seed N` | RNG seed for reproducibility. Default: 42. |

### CLI conventions
- All flags use `--long-name` (no single-letter shortcuts) for agent readability
- Exit codes: 0 = success, 1 = validation warning (output still produced), 2 = error
- `--json` output always includes a `timing` object with per-stage durations

---

## 9. Configuration

### 9.1 Project Config File

An optional `mathviz.toml` in the working directory:

```toml
[container]
width_mm = 100
height_mm = 100
depth_mm = 100
margin_x_mm = 5
margin_y_mm = 5
margin_z_mm = 5

[placement]
anchor = "center"
viewing_axis = "+z"
preserve_aspect_ratio = true
depth_bias = 1.0

[sampling]
default_profile = "preview"   # "preview" | "production" | "custom"
preview_point_budget = 200_000
production_point_budget = 2_000_000
default_method = "uniform"

[engraving]
point_budget = 2_000_000
min_point_spacing_mm = 0.05
occlusion_mode = "none"
depth_compensation = false

[export]
default_format = "stl"
write_sidecar_meta = true

[preview]
# Tier 1 viewer settings
port = 8457
auto_open_browser = true
background = "dark"
lod_mesh_faces = 100_000
lod_cloud_points = 200_000

[studio]
# Tier 2 settings (only relevant when Tier 2 Studio is built)
port = 8457
auto_open_browser = true
background = "dark"
lod_mesh_faces = 100_000
lod_cloud_points = 200_000
geometry_cache_size_mb = 1024
db_path = "mathviz_studio.db"
thumbnail_size = [256, 256]
debounce_ms = 300

[grid]
rows = 10
cols = 10
manifest_path = "grid.toml"

[performance]
log_timing = true
```

### 9.2 Per-Object Config

```toml
generator = "lorenz_attractor"
seed = 42

[params]
sigma = 10.0
rho = 28.0
beta = 2.6667

[resolution]
integration_steps = 100000

[representation]
type = "raw_point_cloud"

[placement]
anchor = "center"
depth_bias = 1.2
rotation_degrees = [0, 0, 0]

[container]
width_mm = 100
height_mm = 100
depth_mm = 100
margin_x_mm = 5
margin_y_mm = 5
margin_z_mm = 5
```

CLI: `mathviz generate --config lorenz.toml`

### 9.3 Sampling Profiles

Predefined profiles in `profiles/`:

```toml
# profiles/preview.toml
name = "preview"
point_budget = 200_000
density_surface = 5.0       # points/mm²
density_volume = 2.0        # points/mm³
method = "random"           # faster for preview

# profiles/production.toml
name = "production"
point_budget = 2_000_000
density_surface = 20.0
density_volume = 10.0
method = "uniform"          # highest quality
```

### 9.4 Grid Manifest

The installation is 100+ blocks in a 10×10 or larger grid. Each block is managed as a preset and a per-object TOML config, but one file must map grid positions to blocks. Without it, the operator would track 100 blocks across weeks of iteration in a separate spreadsheet.

The grid manifest is a TOML file (`grid.toml`) in the project root:

```toml
[grid]
rows = 10
cols = 10
title = "Mathematical Atlas"
description = "100 crystal glass blocks, 10x10 grid"

# Each entry maps (row, col) to a preset or config file.
# row 0 = top, col 0 = left.
# Positions can be empty (block not yet assigned).

[[blocks]]
row = 0
col = 0
preset = "gyroid_v2"                    # preset name (from Studio DB or presets/ dir)
config = "blocks/block_0_0.toml"        # per-object config file
export_path = "exports/block_0_0.stl"   # most recent production export
status = "exported"                     # "draft" | "preview" | "exported" | "sent_to_engraver"
notes = "First block in TPMS region"
exported_at = 2025-07-15T14:30:00Z

[[blocks]]
row = 0
col = 1
preset = "schwarz_p_v1"
config = "blocks/block_0_1.toml"
export_path = "exports/block_0_1.stl"
status = "preview"
notes = "Adjacent to gyroid — same period count, different surface type"

[[blocks]]
row = 3
col = 5
preset = "lorenz_butterfly_v3"
config = "blocks/block_3_5.toml"
status = "draft"
notes = "Attractor cluster center"

# ... up to 100 entries
```

**Data model:**

```python
class GridBlock(BaseModel):
    """A single block assignment in the installation grid. (Pydantic — this is config.)"""
    row: int
    col: int
    preset: Optional[str] = None        # preset name
    config: Optional[str] = None        # path to per-object TOML
    export_path: Optional[str] = None   # path to most recent production export
    status: Literal["empty", "draft", "preview", "exported", "sent_to_engraver"] = "empty"
    notes: str = ""
    exported_at: Optional[datetime] = None

class GridManifest(BaseModel):
    """The full installation grid."""
    rows: int = 10
    cols: int = 10
    title: str = ""
    description: str = ""
    blocks: list[GridBlock] = []

    def get_block(self, row: int, col: int) -> Optional[GridBlock]:
        """Get the block at (row, col), or None if unassigned."""
        for b in self.blocks:
            if b.row == row and b.col == col:
                return b
        return None

    def summary(self) -> dict:
        """Return counts by status."""
        from collections import Counter
        statuses = Counter(b.status for b in self.blocks)
        statuses["empty"] = (self.rows * self.cols) - len(self.blocks) + statuses.get("empty", 0)
        return dict(statuses)
```

**CLI commands for grid management:**

```
mathviz grid show                       # print grid as ASCII table with status colors
mathviz grid show --json                # machine-readable grid state
mathviz grid assign <row> <col> <preset_or_config>  # assign a block
mathviz grid status <row> <col> <status>             # update status
mathviz grid export-all [--sampling-profile production] [--output-dir exports/]
                                        # batch export all assigned blocks
mathviz grid summary                    # counts by status
```

**The grid manifest is a data model from Phase 1.** The data model and CLI commands exist from early on, because they need only a TOML file and a Pydantic model. The Tier 2 Studio's grid layout view (drag-and-drop preset assignment, adjacency preview, installation mockup) is built on this data model later. Without the Studio, the manifest is usable through the CLI and a text editor.

**Adjacency tracking:** The manifest stores positions but does not enforce adjacency constraints. The operator keeps adjacent blocks visually coherent by hand, or with CLI queries such as `mathviz grid neighbors 3 5`, which shows the 8 surrounding blocks. Adjacency is an artistic judgment, not a constraint in code.

---

## 10. Technology Stack

### 10.1 Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **pydantic** | ≥2.0 | Data models, runtime validation, schema generation |
| **numpy** | ≥1.24 | Array operations, linear algebra |
| **scipy** | ≥1.10 | ODE integration, spatial algorithms |
| **trimesh** | ≥4.0 | Mesh I/O, manipulation, repair, surface sampling |
| **scikit-image** | ≥0.21 | Marching cubes |
| **typer** | ≥0.9 | CLI framework (auto-validation from type hints) |
| **rich** | ≥13.0 | Terminal formatting, progress bars, timing display |
| **fastapi** | ≥0.110 | Tier 1 preview server + Tier 2 Studio backend |
| **uvicorn** | ≥0.27 | ASGI server |

### 10.2 Performance Dependencies

| Package | Version | Purpose | Scope |
|---------|---------|---------|-------|
| **numba** | ≥0.58 | JIT compilation | **Fractal inner loops only**: Mandelbulb, Julia 3D, and Mandelbrot iteration kernels. Not a general optimization strategy. Agents should not use numba unless the inner loop is a tight numerical kernel with millions of iterations. The first call has ~2-5 seconds of cold-start latency. |

### 10.3 Optional Dependencies

| Package | Version | Purpose | When needed |
|---------|---------|---------|-------------|
| **open3d** | ≥0.17 | PCD export, advanced point cloud ops | Only when PCD output is needed. Large install. |
| **pyvista** | ≥0.42 | High-res static rendering | Only for `mathviz render` command |
| **meshio** | — | Additional mesh format I/O | If export format breadth becomes insufficient |
| **pygalmesh** / **gmsh** | — | Higher quality mesh generation | If marching cubes quality is insufficient |

Install groups: `pip install mathviz` (core + CLI + Tier 1 preview), `pip install mathviz[studio]` (adds Tier 2: React frontend build), `pip install mathviz[render]` (adds pyvista), `pip install mathviz[open3d]` (adds open3d), `pip install mathviz[all]`. The Tier 2 Studio frontend is a separate npm project in `studio-frontend/`. It is built to static files that the Python backend serves. The Tier 1 preview has no npm dependency: it is a single HTML file that loads Three.js from a CDN.

### 10.4 Tier 2 Studio Dependencies

These are needed only to build the full Tier 2 Studio (Phase 7). The Tier 1 preview needs no dependencies beyond core.

**Backend (already in core):**
FastAPI and uvicorn are core dependencies, because Tier 1 also uses them. Tier 2 adds:

| Package | Version | Purpose |
|---------|---------|---------|
| **websockets** | ≥12.0 | WebSocket protocol support (Tier 2 live updates) |

**Frontend (separate `studio-frontend/` directory, built with Vite):**

| Package | Purpose |
|---------|---------|
| **react** + **react-dom** | UI component framework |
| **@react-three/fiber** | Declarative Three.js for React |
| **@react-three/drei** | OrbitControls, helpers, loaders |
| **three** | 3D rendering engine |
| **vite** | Build tool / dev server |
| **zustand** | Lightweight state management |

The frontend builds to static files (`studio/static/`) served by FastAPI. During development, the Vite dev server proxies API calls to the backend.

**Storage:**
- **sqlite3** (stdlib) — preset database, generation history, gallery metadata, grid manifest. No external DB dependency.

### 10.5 Development Dependencies

| Package | Purpose |
|---------|---------|
| **pytest** ≥7.0 | Testing |
| **pytest-benchmark** | Performance regression tests |
| **mypy** | Type checking |
| **ruff** | Linting and formatting |

### 10.6 Platform

**Python 3.11+** (tomllib in stdlib, performance improvements).

Primary platform: macOS on Apple Silicon. The code is pure Python plus compiled extensions that all have ARM64 wheels. There is no platform-specific code.

---

## 11. Performance

### 11.1 Timing Instrumentation

Every pipeline stage logs its wall-clock duration. This is built into the core pipeline runner and is not optional.

```python
class PipelineTimer:
    """Context manager that records timing for each pipeline stage."""
    stages: dict[str, float]  # stage_name -> seconds

    def stage(self, name: str) -> ContextManager: ...

# Usage in pipeline:
with timer.stage("generate"):
    obj = generator.generate(params, seed=seed, **resolution_kwargs)
with timer.stage("represent"):
    obj = strategy.apply(obj, rep_config)
with timer.stage("transform"):
    obj = transformer.fit(obj, container, placement)
with timer.stage("sample"):
    obj = sampler.sample(obj, profile)
with timer.stage("validate"):
    result = validator.validate(obj, container, engraving_profile)
with timer.stage("export"):
    exporter.export(obj, path)
```

Timing appears in:
- Terminal output (via `rich` table) unless `--quiet`
- The `--report` JSON file
- The sidecar `.meta.json`
- The `--json` output
- The `MathObject.generation_time_seconds` field

### 11.2 Performance Expectations

| Operation | Typical Time (M-series Mac) | Notes |
|-----------|---------------------------|-------|
| Parametric surface, grid_resolution=128 | <0.5s | NumPy vectorized |
| Implicit surface, voxel_resolution=128 | 1-3s | Marching cubes dominates |
| Implicit surface, voxel_resolution=256 | 5-15s | 8× more voxels |
| Lorenz attractor, 100K steps | <0.5s | scipy ODE |
| Mandelbulb, voxel_resolution=128 | 2-5s | With numba JIT (after warmup) |
| Mandelbulb, voxel_resolution=256 | 10-30s | With numba JIT |
| Mandelbulb, voxel_resolution=256, no numba | 2-10min | Pure NumPy — avoid |
| Point cloud sampling, 1M points | 1-3s | trimesh |
| Tube thickening, 100K curve points | <1s | |
| STL export, 500K faces | <0.5s | Binary |

### 11.3 Preview / Studio Responsiveness

Both Tier 1 and Tier 2 viewers must keep ≥30fps during orbit/pan/zoom. This is the primary UX constraint: a slow viewport makes artistic exploration hard.

- **LOD defaults:** 100K faces for meshes, 200K points for clouds (configurable in mathviz.toml)
- **Decimation is server-side:** FastAPI sends pre-decimated GLB/binary geometry. The browser never processes full-resolution data during interaction.
- **High-res mode:** An explicit user action (a button click) loads the full geometry, with a loading spinner. The frame rate may drop, which is acceptable because the user asked for it.
- **Point size:** Three.js `Points` material with `sizeAttenuation: true` and configurable base size.
- **Parameter debounce (Tier 2 only):** Slider changes are debounced at 300ms, and regeneration runs at preview resolution. A "Regenerate full-res" button runs a high-quality generation on request.
- **Transform-only updates (Tier 2 only):** Placement changes re-transform the existing geometry without regeneration, so they appear instant.
- **Geometry cache:** Recent generations are cached in memory (LRU, 1GB default), so returning to a recent configuration is instant.

---

## 12. Project Structure

```
mathviz/
├── pyproject.toml
├── mathviz.toml                    # Default project config
├── grid.toml                       # Grid manifest (§9.4)
├── README.md
├── src/
│   └── mathviz/
│       ├── __init__.py
│       ├── cli.py                  # Typer-based CLI entry point
│       ├── pipeline.py             # Pipeline runner with timing + validation at boundaries
│       ├── core/
│       │   ├── __init__.py
│       │   ├── math_object.py      # dataclasses: MathObject, Mesh, PointCloud, Curve + validate()
│       │   ├── container.py        # Pydantic: Container, PlacementPolicy
│       │   ├── generator.py        # GeneratorBase ABC + registry + alias resolution
│       │   ├── representation.py   # RepresentationStrategy, RepresentationConfig (Pydantic)
│       │   ├── transformer.py      # Bounding box fitting with PlacementPolicy
│       │   ├── sampler.py          # Mesh → point cloud sampling
│       │   ├── engraving.py        # EngravingOptimizer, EngravingProfile (Pydantic)
│       │   ├── validator.py        # Mesh validation + engraving validation
│       │   ├── mesh_exporter.py    # STL, OBJ, PLY (mesh)
│       │   ├── cloud_exporter.py   # PLY (cloud), XYZ, PCD
│       │   ├── meta_exporter.py    # JSON metadata
│       │   └── grid.py             # GridManifest, GridBlock (Pydantic) + grid CLI commands
│       ├── generators/
│       │   ├── __init__.py         # Generator registry (auto-discovers)
│       │   ├── parametric/
│       │   │   ├── __init__.py
│       │   │   ├── klein_bottle.py
│       │   │   ├── mobius_strip.py
│       │   │   ├── torus.py
│       │   │   ├── superellipsoid.py
│       │   │   ├── spherical_harmonics.py
│       │   │   ├── lissajous_surface.py
│       │   │   ├── boy_surface.py
│       │   │   └── enneper_surface.py
│       │   ├── implicit/
│       │   │   ├── __init__.py
│       │   │   ├── gyroid.py
│       │   │   ├── schwarz_p.py
│       │   │   ├── schwarz_d.py
│       │   │   └── costa_surface.py
│       │   ├── attractors/
│       │   │   ├── __init__.py
│       │   │   ├── lorenz.py
│       │   │   ├── rossler.py
│       │   │   ├── chen.py
│       │   │   ├── thomas.py
│       │   │   ├── halvorsen.py
│       │   │   └── double_pendulum.py
│       │   ├── fractals/
│       │   │   ├── __init__.py
│       │   │   ├── mandelbulb.py
│       │   │   ├── mandelbrot_heightmap.py
│       │   │   ├── julia_3d.py
│       │   │   └── fractal_slice.py
│       │   ├── knots/
│       │   │   ├── __init__.py
│       │   │   ├── torus_knot.py       # canonical home; aliases: trefoil, cinquefoil
│       │   │   ├── figure_eight.py
│       │   │   └── lissajous_knot.py
│       │   ├── number_theory/
│       │   │   ├── __init__.py
│       │   │   ├── ulam_spiral.py
│       │   │   ├── sacks_spiral.py
│       │   │   ├── prime_gaps.py
│       │   │   └── digit_encoding.py
│       │   ├── curves/
│       │   │   ├── __init__.py
│       │   │   ├── lissajous_curve.py
│       │   │   ├── logarithmic_spiral.py
│       │   │   ├── cardioid.py
│       │   │   └── fibonacci_spiral.py
│       │   ├── physics/
│       │   │   ├── __init__.py
│       │   │   ├── kepler_orbit.py
│       │   │   ├── nbody.py
│       │   │   └── planetary_positions.py
│       │   ├── data_driven/
│       │   │   ├── __init__.py
│       │   │   ├── heightmap.py
│       │   │   ├── building_extrude.py
│       │   │   └── soundwave.py
│       │   └── procedural/
│       │       ├── __init__.py
│       │       ├── noise_surface.py
│       │       ├── terrain.py
│       │       └── reaction_diffusion.py
│       ├── shared/
│       │   ├── __init__.py
│       │   ├── tube_thickening.py
│       │   ├── marching_cubes.py
│       │   ├── mesh_utils.py
│       │   └── math_utils.py
│       └── preview/
│           ├── __init__.py
│           ├── server.py               # FastAPI: Tier 1 endpoints + geometry serving
│           ├── lod.py                  # LOD decimation for browser transfer
│           ├── geometry_cache.py       # LRU geometry cache
│           ├── renderer.py             # PyVista high-res (optional dep)
│           ├── renderer_2d.py          # matplotlib 2D projections
│           └── static/
│               └── index.html          # Tier 1: single-file vanilla JS + Three.js viewer
│       └── studio/                     # Tier 2 (built later)
│           ├── __init__.py
│           ├── app.py                  # FastAPI extensions: WebSocket, preset CRUD, grid API
│           ├── models.py               # Pydantic models for Tier 2 API request/response
│           ├── database.py             # SQLite preset/history/gallery storage
│           └── static/                 # Built React frontend (generated by Vite build)
├── studio-frontend/                    # Tier 2 React app (built later)
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── store.ts                    # Zustand state management
│       ├── api.ts                      # REST + WebSocket client
│       ├── components/
│       │   ├── Layout.tsx              # Main layout shell
│       │   ├── GeneratorBrowser.tsx    # Left panel: generator tree + search
│       │   ├── Viewport.tsx            # Center: Three.js scene
│       │   ├── ViewportControls.tsx    # View mode toggles, screenshot, etc.
│       │   ├── ParameterPanel.tsx      # Right panel: auto-generated controls
│       │   ├── ParamSlider.tsx         # Individual parameter controls
│       │   ├── ParamInput.tsx
│       │   ├── PresetManager.tsx       # Save/load/export presets
│       │   ├── Gallery.tsx             # Thumbnail grid of saved presets
│       │   ├── ComparisonView.tsx      # Side-by-side viewports
│       │   ├── ExportDialog.tsx        # Export format/profile selection
│       │   ├── ExportQueue.tsx         # Background export progress
│       │   ├── InfoBar.tsx             # Stats, validation, timing
│       │   └── ContainerWireframe.tsx  # Glass block outline in viewport
│       ├── hooks/
│       │   ├── useGenerator.ts         # Generator selection + param state
│       │   ├── useGeometry.ts          # Geometry loading + LOD
│       │   ├── useWebSocket.ts         # WS connection management
│       │   └── usePresets.ts           # Preset CRUD
│       └── types/
│           └── index.ts                # TypeScript types matching Pydantic models
├── profiles/
│   ├── preview.toml
│   ├── production.toml
│   └── high_density.toml
├── schemas/
│   ├── math_object.schema.json         # auto-generated from Pydantic models
│   ├── container.schema.json
│   └── generator_params/
│       ├── lorenz_attractor.schema.json
│       └── ...                         # auto-generated per generator
├── fixtures/
│   ├── reference_stls/
│   │   ├── torus_default.stl
│   │   ├── lorenz_default.stl
│   │   └── gyroid_default.stl
│   └── reference_meta/
│       ├── torus_default.meta.json
│       └── ...
├── docs/
│   └── architecture/
│       ├── adr-001-pydantic-over-dataclass.md
│       ├── adr-002-representation-strategy.md
│       ├── adr-003-deterministic-seeding.md
│       ├── adr-004-split-resolution-types.md
│       └── adr-005-explicit-export.md
├── tests/
│   ├── conftest.py
│   ├── test_core/
│   │   ├── test_math_object.py
│   │   ├── test_container.py
│   │   ├── test_transformer.py
│   │   ├── test_sampler.py
│   │   ├── test_representation.py
│   │   ├── test_engraving.py
│   │   ├── test_validator.py
│   │   ├── test_mesh_exporter.py
│   │   ├── test_cloud_exporter.py
│   │   └── test_pipeline.py
│   ├── test_generators/
│   │   ├── test_parametric.py
│   │   ├── test_implicit.py
│   │   ├── test_attractors.py
│   │   ├── test_fractals.py
│   │   ├── test_knots.py
│   │   └── ...
│   └── test_shared/
│       ├── test_tube_thickening.py
│       └── test_marching_cubes.py
└── examples/
    ├── lorenz.toml
    ├── gyroid.toml
    ├── trefoil.toml
    └── mandelbulb.toml
blocks/                                 # Per-block config files (one per grid position)
    ├── block_0_0.toml
    └── ...
exports/                                # Production export output directory
    ├── block_0_0.stl
    ├── block_0_0.meta.json
    └── ...
```

### Generator Registration

Generators register themselves with a decorator, which also accepts aliases:

```python
@register(aliases=["trefoil", "cinquefoil"])
class TorusKnot(GeneratorBase):
    name = "torus_knot"
    category = "knots"
    aliases = ["trefoil", "cinquefoil"]
    resolution_params = {"curve_points": "Number of points along the knot curve"}
    ...
```

Aliases resolve to the same class with different default parameters. `mathviz generate trefoil` calls `TorusKnot.generate(params={"p": 2, "q": 3})`.

---

## 13. Testing Strategy

### Unit Tests (per generator)

Every generator test:
1. Generate with default parameters at low resolution
2. Assert MathObject has valid geometry (non-empty)
3. Assert bounding box is finite, non-degenerate
4. Assert mesh validity if mesh output
5. Full pipeline: generate → represent → transform → validate → export → reload → compare
6. **Determinism test:** generate twice with same seed, assert identical output

### Integration Tests

1. Full CLI pipeline from `mathviz generate` to exported file
2. File round-trip: export → reimport → compare
3. `--dry-run` produces expected output without writing files
4. `--json` output is valid JSON and contains expected fields
5. Grid manifest: create → assign blocks → export-all → verify exports match presets
6. Tier 1 preview: server starts, `/api/generators` returns valid JSON, `/api/generate` returns geometry, `index.html` loads without errors
7. MathObject.validate_or_raise() catches malformed geometry at stage boundaries

### Performance Tests (on-demand, not every commit)

- Mandelbulb at voxel_resolution 128, 256, 512
- Marching cubes at various grid sizes
- Point cloud sampling at various densities
- Tube thickening with various point counts
- Track regressions. Alert if >2× slower than baseline.

### Fixture Tests

Reference STL and meta files live in `fixtures/`. Tests compare generated output against the reference: vertex count within tolerance, bounding box match, face count within tolerance.

---

## 14. Agent Development Guidelines

### Module Independence

Each generator is a self-contained file that imports only from `core/`, `shared/`, and external libraries, never from another generator. An agent implements a new generator from one example and the GeneratorBase ABC.

### Progressive Implementation

**Phase 1: Core infrastructure**
1. `core/math_object.py` — dataclasses (Mesh, PointCloud, Curve, MathObject) with validate()
2. `core/container.py` — Pydantic: Container, PlacementPolicy
3. `core/generator.py` — GeneratorBase ABC + registry + aliases
4. `core/representation.py` — RepresentationStrategy + RepresentationConfig (Pydantic)
5. `core/transformer.py` — bounding box fitting
6. `core/mesh_exporter.py` — binary STL export
7. `core/cloud_exporter.py` — PLY, XYZ export
8. `core/meta_exporter.py` — JSON metadata
9. `core/validator.py` — mesh + engraving validation
10. `pipeline.py` — pipeline runner with timing + validate_or_raise at stage boundaries
11. `cli.py` — minimal CLI (`generate`, `list`, `info`, `validate`, `--dry-run`, `--json`)
12. Tests for all of the above

**Phase 2: Shared components**
1. `shared/tube_thickening.py`
2. `shared/marching_cubes.py`
3. `shared/mesh_utils.py`
4. Tests

**Phase 3: First generators (one per category)**
1. `generators/parametric/torus.py`
2. `generators/implicit/gyroid.py`
3. `generators/attractors/lorenz.py`
4. `generators/knots/torus_knot.py`
5. `generators/fractals/mandelbrot_heightmap.py`
6. Tests, end-to-end pipeline validation, fixture generation

**Phase 4: Tier 1 Preview Viewer**
1. FastAPI backend: `/api/generators`, `/api/generate`, `/api/geometry/{id}/mesh|cloud`
2. LOD decimation, GLB export, geometry cache (LRU)
3. Single `index.html` with vanilla JS + Three.js (CDN): viewport, OrbitControls, view toggles, container wireframe, dark background, screenshot, info display
4. CLI `mathviz preview` command
5. Tests: server smoke test, geometry endpoint returns valid GLB
— **Phase 4 is the checkpoint for artistic exploration.** After Phase 4 you can generate objects with the CLI, preview them in the browser, iterate on parameters, and export production files. Phases 5 and later run in parallel with block production.

**Phase 5: Remaining generators** — in any order; each is independent. The grid manifest (§9.4) is maintained as TOML during production.

**Phase 6: Polish**
1. EngravingOptimizer (occlusion, depth compensation)
2. Sampling profiles
3. High-res renderer (PyVista, optional dep)
4. 2D rendering
5. Config file support
6. `convert`, `sample`, `transform` CLI commands
7. Schema auto-generation

**Phase 7: Tier 2 Studio (when needed, likely hands-on)**
1. FastAPI extensions: WebSocket endpoint, SQLite database, preset CRUD, grid manifest API
2. React + Vite + TypeScript scaffold
3. Three-panel layout, r3f viewport, generator browser
4. Parameter panel (auto-generated from JSON Schema), debounced regeneration
5. Preset save/load/export, gallery, comparison mode
6. Export dialog and background export queue
7. UX polish: keyboard shortcuts, dark mode, responsive layout

### Code Style

- Type hints everywhere (enforced by mypy)
- Pydantic models for config/metadata structures (Container, PlacementPolicy, Preset, EngravingProfile, RepresentationConfig, GridManifest)
- Plain dataclasses with explicit `validate()` methods for geometry containers (Mesh, PointCloud, Curve, MathObject)
- `validate_or_raise()` called at every pipeline stage boundary
- Docstrings on all public methods (NumPy style)
- No global mutable state
- All randomness via `numpy.random.default_rng(seed)` — never `np.random.seed()`
- All numerical code uses float64
- Imports are explicit (no `from x import *`)
- numba only in fractal inner loops, nowhere else

### Test-Driven for Agents

Each implementation PR includes:
1. The implementation file
2. Tests: default params + at least one non-default combination + determinism test
3. An example TOML config

Verify: `pytest tests/test_generators/test_<category>.py -v`

---

## 15. Open Questions and Future Considerations

### Engraver Format

Confirm the format, resolution, and point density with the vendor. The production sampling profile depends on these values.

### Mesh Boolean Operations

Add when needed (trimesh CSG, libigl, pymeshlab). The architecture allows it, as a shared component or a representation strategy variant.

### Parameter Sweeps

Future: a `mathviz sweep` command that generates N objects with interpolated parameters. The architecture supports it through config files.

### GPU Acceleration

If fractal generation at voxel_resolution=512+ becomes a bottleneck, use CUDA or Metal compute shaders. The architecture allows it: the numba inner loops are the place to swap in GPU code.

### Studio Evolution (Tier 2 and beyond)

The Tier 2 Studio (§7.2–7.4) covers parameter controls, presets, gallery, and comparison. Further additions:
- **Grid layout view** — visual 10×10 grid backed by the grid manifest (§9.4), with drag-and-drop preset assignment
- **Adjacency preview** — select a block in the grid and see its neighbors rendered side-by-side
- **Batch parameter sweep** — generate N presets with interpolated parameters in one action
- **Installation mockup** — a 3D render of the full grid with all blocks, simulating the wall-mounted appearance with backlighting

### Additional Libraries (adopt when needed)

- **networkx** — if topology analysis is needed (e.g., knot invariant computation)
- **meshio** — if export format breadth becomes insufficient
- **attrs** — not needed given Pydantic adoption

---

## Appendix A: Mathematical Reference

### Lorenz System
```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz

Default: σ = 10, ρ = 28, β = 8/3
```

### Torus Knot (p, q)
```
x(t) = (R + r·cos(q·t)) · cos(p·t)
y(t) = (R + r·cos(q·t)) · sin(p·t)
z(t) = r · sin(q·t)

t ∈ [0, 2π], R > r > 0
Trefoil: p=2, q=3
Cinquefoil: p=2, q=5
```

### Gyroid
```
f(x, y, z) = sin(x)·cos(y) + sin(y)·cos(z) + sin(z)·cos(x)
Surface: f = 0
```

### Mandelbulb
```
For each point c in ℝ³:
  z = c
  For i = 1 to max_iter:
    r = |z|
    If r > bailout: escape at iteration i
    θ = arccos(z_z / r)
    φ = arctan2(z_y, z_x)
    z = r^n · (sin(nθ)cos(nφ), sin(nθ)sin(nφ), cos(nθ)) + c

Default: n = 8, max_iter = 20, bailout = 2.0
```

### Superellipsoid
```
x(η, ω) = a₁ · C(η, 2/e₁) · C(ω, 2/e₂)
y(η, ω) = a₂ · C(η, 2/e₁) · S(ω, 2/e₂)
z(η, ω) = a₃ · S(η, 2/e₁)

Where C(θ, n) = sign(cos θ) · |cos θ|^n
      S(θ, n) = sign(sin θ) · |sin θ|^n
η ∈ [-π/2, π/2], ω ∈ [-π, π]
```

### Spherical Harmonics (real)
```
r(θ, φ) = Σ (a_lm · Y_lm(θ, φ))

Where Y_lm are the real spherical harmonics.
Surface point: r(θ,φ) · (sin θ cos φ, sin θ sin φ, cos θ)
```

---

## Appendix B: Engraving-Specific Considerations

### Point Density

Subsurface laser engraving typically operates at 500–1000 DPI equivalent in 3D (~20–40 points per mm per axis). For a 90×90×90mm usable volume:
- At 20 pts/mm: 1800 × 1800 × 1800 = ~5.8 billion potential points (maximum theoretical)
- Actual objects use far fewer (surface-only, sparse structures)
- Typical: 100K–5M points

The `point_budget` parameter caps output. Default: 2M for production.

### Engraving Depth and Aspect Ratio

The default container is a 100×100×100mm cube. In a non-cubic container, such as a shallow block, mathematically balanced objects may look compressed. The `PlacementPolicy.depth_bias` parameter lets the operator exaggerate depth or compress it further, so the form stays readable in the glass. The default of 1.0 applies no correction. Values of 1.2–1.5 may help objects with important z-axis structure in shallow containers.

### Fracture Point Visibility

Points deeper in the glass are less visible, because the glass attenuates light. The `EngravingOptimizer.depth_compensation` flag increases point density for deeper points. `depth_compensation_factor=1.5` means the deepest layer gets 1.5× the density of the front layer, with linear interpolation between.

### Volumetric Occlusion

A dense point cloud of a solid object engraves as an opaque white volume that hides the internal structure. The `EngravingOptimizer` provides three strategies:

- **shell_fade:** Keep surface points at full density and thin inner layers progressively. Suits objects with important surface detail (Mandelbulb, minimal surfaces).
- **radial_gradient:** Density decreases from the center outward, so the core is visible through a sparse outer shell. Suits objects with important internal structure.
- **none:** No thinning. For objects that are already sparse (attractors, wireframes, thin surfaces).

---

## Appendix C: Resolution Quick Reference

For agents implementing generators, the cost of each resolution parameter:

| Resolution Type | N=64 | N=128 | N=256 | N=512 |
|----------------|------|-------|-------|-------|
| `grid_resolution` (N²) | 4K pts | 16K pts | 65K pts | 262K pts |
| `voxel_resolution` (N³) | 262K vox | 2M vox | 16M vox | 134M vox |
| `integration_steps` (N) | 64 steps | 128 steps | 256 steps | 512 steps |
| `curve_points` (N) | 64 pts | 128 pts | 256 pts | 512 pts |
| `pixel_resolution` (N²) | 4K px | 16K px | 65K px | 262K px |

Rule of thumb: `voxel_resolution` above 256 requires numba for fractal kernels. Above 512, expect minutes even with numba.
