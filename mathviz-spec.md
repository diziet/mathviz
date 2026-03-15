# MathViz — System Specification

## Mathematical Visualization & Laser Engraving Pipeline

**Version:** 0.4
**Purpose:** Programmatic generation of 3D mathematical forms for subsurface laser engraving in crystal glass blocks, with interactive preview and multi-format export.

---

## 1. Project Context

A wall-mounted installation of 100–256 crystal glass blocks, each containing a unique three-dimensional mathematical form rendered through subsurface laser engraving. The blocks are arranged in a grid (nominally 10×10), where adjacent blocks show related forms — knot complexity increasing across one region, fractal zoom depth varying in another, prime spirals changing density elsewhere.

Each block is a separate sub-project. This system generates one object at a time with configurable parameters. The operator (human) manages the mapping of objects to grid positions externally.

Subsurface laser engraving works by focusing a laser at discrete (x, y, z) coordinates inside glass, creating micro-fractures. The visual result is a monochrome point cloud of glowing fracture points illuminated by backlighting. The system must produce both surface meshes (STL) and point clouds, since engraver software may accept either format but the physical output is always a point cloud.

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

1. **Generate** — A `Generator` evaluates a mathematical definition with given parameters and a deterministic seed, producing a `MathObject` containing raw geometry in abstract coordinate space.

2. **Representation** — A `RepresentationStrategy` decides how the raw geometry should be physically realized for engraving. This is a fabrication policy, not a generator concern. Examples: keep a Lorenz trajectory as a raw point cloud; thicken a torus knot curve into a tube mesh; convert a gyroid implicit field into a surface shell clipped to a slab; render a Mandelbulb as a sparse shell rather than a solid fill. The strategy produces one or more candidate representations as new MathObjects. The operator selects which to use (or the default is applied automatically).

3. **Transform** — The `Transformer` scales, centers, and fits the geometry into a physical container (defined in millimeters) according to a `PlacementPolicy`. Preserves aspect ratio by default, with optional depth bias, anchor mode, and perceptual corrections for the glass block.

4. **Sample** — The `Sampler` converts surface meshes or volumes into point clouds at a specified density. This stage is skipped if the object is already a point cloud and no resampling is requested.

5. **Engraving Optimization** — Post-sampling adjustments specific to the engraving medium: volumetric occlusion thinning (so dense objects don't become opaque white bricks), depth-dependent density compensation (deeper points are less visible), and point budget enforcement.

6. **Validate** — Check mesh properties (watertight, manifold, no degenerate faces, bounding box within container) AND engraving properties (point count within budget, density distribution, no degenerate clusters, visibility estimate).

7. **Export** — Write to disk in requested format(s). Export requires the requested representation to exist on the MathObject; no silent conversion. Pass `--auto-sample` to permit implicit conversion.

8. **Preview** — Launch an interactive browser-based 3D viewer with level-of-detail management for responsive interaction.

Each stage is independently callable. The pipeline is composable, not monolithic.

---

## 3. Core Abstractions

### Validation Strategy: Pydantic for Config, Dataclasses for Geometry

The system uses two different data modeling approaches, chosen to match what each model actually contains:

**Pydantic `BaseModel`** for configuration, metadata, and API boundaries: Container, PlacementPolicy, EngravingProfile, RepresentationConfig, Preset, all CLI/API request/response models. These are the models where Pydantic's automatic validation, JSON serialization, and schema generation provide real value — they contain scalar fields (floats, ints, strings, enums) that Pydantic can fully validate.

**Plain `@dataclass`** for geometry containers: Mesh, PointCloud, Curve, MathObject. These carry `np.ndarray` fields where Pydantic's `arbitrary_types_allowed` would disable validation for the fields that matter most. Instead, these dataclasses have explicit `validate()` methods that check the things we actually care about: array shape, dtype, NaN presence, and dimensional consistency. The validation is called explicitly at construction boundaries (generator output, pipeline stage transitions) rather than implicitly via Pydantic.

This avoids the worst-of-both-worlds trap: Pydantic model construction overhead on million-point arrays with zero validation benefit.

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

This is a runtime check on a simple enum field rather than a generic type parameter. The generic approach (`MathObject[CoordSpace.ABSTRACT]`) added complexity without real benefit — mypy can't enforce it on dataclasses meaningfully, and the runtime check is trivially reliable. Agents get a clear error message: "Cannot export: MathObject is in abstract coordinate space, run the transformer first."

**Validation call sites:** `validate_or_raise()` is called at every pipeline stage boundary:
- After `Generator.generate()` returns
- After `RepresentationStrategy.apply()` returns
- After `Transformer.fit()` returns
- Before any exporter writes to disk

This catches malformed geometry immediately at the source rather than letting it propagate.

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

Margins are always per-axis. The convenience constructor `Container.with_uniform_margin()` covers the common case. No dual-path None-check logic.

Default: 100×100×100mm block with 5mm margins → 90×90×90mm usable volume.

### 3.3 RepresentationStrategy

This is the key architectural insert between generation and transformation. It separates the mathematical definition (what the object *is*) from the fabrication decision (how it should *look* when engraved in glass).

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

**Why this matters:** Without this layer, each generator accumulates fabrication-specific hacks. The Lorenz generator shouldn't know about tube thickening; the gyroid generator shouldn't know about slab clipping. The generator produces pure math; the representation strategy makes it physical.

The operator can override the default representation per object via CLI `--representation surface_shell` or in the config file. The system also supports `--representation candidates` which outputs multiple representations for visual comparison in the preview.

### 3.4 EngravingOptimizer

Post-sampling adjustments specific to the laser engraving medium.

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

A generator might use multiple resolution types (e.g., Mandelbulb uses both `voxel_resolution` and `iteration_depth`). Each has independent defaults and the CLI exposes them as separate flags.

**Deterministic seeding:** Every generator that uses randomness (Voronoi seed points, noise fields, reaction-diffusion initial conditions, n-body with chaotic sensitivity) must accept a `seed` parameter and use `numpy.random.default_rng(seed)` exclusively. No global `np.random.seed()`. The seed is recorded in `MathObject.seed` and in the sidecar metadata, guaranteeing exact reproducibility.

For deterministic generators (parametric surfaces, knots, pure-math fractals), the seed parameter is accepted but unused. It still appears in the metadata for pipeline consistency.

### 3.6 Exporters

Split into separate classes. No silent conversion. Export fails unless the requested representation exists on the MathObject, unless `--auto-sample` is passed.

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

STL is always binary — there is no `binary` parameter. The sidecar `.meta.json` is written automatically alongside every geometry export.

---

## 4. Generator Taxonomy

### Canonical Homes and Aliases

Each generator has exactly one canonical location in the directory tree. Generators that could logically belong to multiple categories live in the category that best matches their primary mathematical identity. The registry supports aliases so they can be invoked by alternate names.

**Alias resolution rule:** The registry maps all names (canonical + aliases) to the same generator class. `mathviz generate torus_knot` and `mathviz generate trefoil` resolve to the same generator with different default parameters.

| Generator | Canonical Category | Aliases |
|-----------|-------------------|---------|
| `torus_knot` | `knots/` | `trefoil` (p=2,q=3), `cinquefoil` (p=2,q=5) |
| `lissajous_knot` | `knots/` | — |
| `lissajous_curve` | `curves/` | — |
| `lissajous_surface` | `parametric/` | — |
| `double_pendulum` | `attractors/` | — |

The canonical home is determined by: what is the object's mathematical identity? A torus knot is a knot that happens to be defined parametrically. A Lissajous surface is a surface that happens to share a naming convention with Lissajous curves.

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
| `enneper_surface` | order | Minimal surface with increasing complexity |

### 4.2 Implicit Surfaces

**Definition:** The zero-level set of a function `f: (x, y, z) → ℝ`, extracted by marching cubes.

**Resolution type:** `voxel_resolution` (default: 128). O(N³) scaling — agents must be aware this is cubic.

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

**TPMS note:** These tile space infinitely. The `periods` parameter controls how many unit cells are included. Marching cubes naturally clips to the evaluation box.

### 4.3 Strange Attractors / Dynamical Systems

**Definition:** Trajectories of ODEs or iterated maps, producing curves in 3D space.

**Resolution type:** `integration_steps` (default: 100,000). O(N) scaling.

**Shared implementation pattern:**
1. Define the system of ODEs: `dx/dt = f(x, y, z, **params)`
2. Integrate using `scipy.integrate.solve_ivp` (RK45 or DOP853)
3. Output is a `Curve` (polyline of N points)
4. Representation strategy decides: raw point cloud, tube, or weighted cloud

**Default representation:** `RAW_POINT_CLOUD` — raw trajectory points produce the ghostly, ethereal look that works best in glass. Tube thickening is available via `--representation tube` for objects where a solid form is preferred.

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
- **Both representations available:** Every attractor can be exported as raw point cloud OR tube mesh. The CLI flag `--representation tube --tube-radius 0.3` overrides the default.

### 4.4 Fractals

**Resolution types:** `voxel_resolution` (for 3D extraction) + `iteration_depth` (escape-time cutoff). `pixel_resolution` for 2D heightmaps. Agents: `voxel_resolution=256` means 256³ = 16M evaluations — use `numba` for the inner loop (see §10.2).

#### 4.4.1 Mandelbulb

**Default representation:** `SPARSE_SHELL` — surface only, no interior fill. A solid Mandelbulb becomes an opaque white blob in glass. The sparse shell preserves the fractal surface detail.

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

Same approach as Mandelbulb but with fixed c parameter. Default representation: `SPARSE_SHELL`.

#### 4.4.4 Fractal Cross-Sections

Slice a 3D fractal with a plane at various angles/offsets. Reuses the 3D evaluator, outputs a 2D slice as heightmap or contour. Default representation: `HEIGHTMAP_RELIEF`.

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
- Self-intersection detection after tube thickening — warn but don't fail.
- Progression convenience: `torus_knot` with a list of (p,q) pairs generates a sequence for grid adjacency.

### 4.6 Number Theory / Constants

**Resolution type:** varies per generator. Each documents its own.

**Default representation:** varies — typically `WEIGHTED_CLOUD` or `HEIGHTMAP_RELIEF`.

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

`voronoi_3d` requires a seed for point placement — deterministic via `default_rng(seed)`.

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
**Algorithm:** Parallel transport frame extrusion (not Frenet-Serret — Frenet frames are undefined at inflection points and produce twisting artifacts)
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

**Library choice:** `trimesh.sample.sample_surface` for surface sampling (no heavy dependency). `open3d` is an **optional** dependency, pulled in only for PCD export and advanced point cloud processing. The core pipeline does not require it.

### 5.4 Validation

Two-tier validation: mesh validation + engraving validation.

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
- Estimated visual density: flag if >70% of voxels in any axis-aligned projection are occupied (opacity warning — the object may read as a white brick)
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

This project is fundamentally artistic. The operator needs to see, rotate, tweak, compare, and curate 100+ mathematical forms before committing them to glass. The **Studio** is a first-class browser-based GUI application for exploring the full generator library, tuning parameters interactively, saving configurations, and queueing production exports. It is not an afterthought bolted onto the CLI — it is a primary interface alongside the CLI.

The CLI remains the contract for agents and batch automation. The Studio is the contract for the human artist.

The Studio is designed in two tiers. **Tier 1** is a minimal but functional viewer built alongside the core pipeline — achievable by agents in one phase, providing ~80% of the artistic exploration value. **Tier 2** is the full React application with presets, gallery, comparison, and export queue — built later, likely with more hands-on intervention, after the operator has generated 20–30 blocks via CLI + Tier 1 and has concrete UX requirements.

### 7.1 Tier 1: Preview Viewer (build with pipeline)

A single HTML page served by a lightweight FastAPI server with a Three.js viewport and basic controls. This is the minimum viable artistic exploration tool.

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

**Technology:** FastAPI backend (already needed for Tier 2 anyway, and it's Pydantic-native for the config models). Frontend is a single `index.html` with inline JS — no build step, no npm, no React. Three.js loaded from CDN. This is critical for agent buildability: a single HTML file with vanilla JS is dramatically easier for an agent to produce and debug than a React + r3f + Zustand application.

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
- URL query params: `?generator=lorenz_attractor&sigma=12&rho=28` — allows bookmarking specific configurations
- Hot-reload: watches for file changes, auto-reloads

**LOD strategy:**
- Server-side decimation: meshes ≤100K faces, point clouds ≤200K points
- "High-res" button loads full geometry with loading spinner
- Transfer format: GLB for meshes (Three.js native), binary PLY for point clouds

**What Tier 1 does NOT include:** No parameter sliders, no preset management, no gallery, no comparison view, no WebSocket, no SQLite. Parameter changes happen via CLI + browser refresh (or URL query params). Presets are TOML files managed manually. This is deliberately spartan — it works, it's fast, and it doesn't require a React build toolchain.

CLI commands:
```
mathviz preview <file_or_generator> [--port N] [--no-open]
    # Launches Tier 1 viewer with the specified file or generator
```

### 7.2 Tier 2: Full Studio (build later, possibly different tooling)

The full Studio application with parameter controls, presets, gallery, comparison, and export queue. Built after the operator has used Tier 1 + CLI for 20–30 blocks and has concrete UX requirements from that experience.

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

**Why this requires different tooling:** Agent-driven development of a React + Three.js + FastAPI + WebSocket + SQLite application with debounced regeneration and LOD switching is significantly harder than the Python pipeline. r3f state management with live WebSocket geometry updates is the kind of thing that requires tight visual iteration — exactly where agents are weakest. The spec below is detailed enough that a strong agent could produce the skeleton, but the UX polish will almost certainly require hands-on intervention.

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

**Storage:** SQLite in the project directory (`mathviz_studio.db`). Presets also exportable as TOML for CLI use and version control.

#### Gallery View

Expandable bottom panel or full-page view showing saved presets as a thumbnail grid. Filter by category, tags, date. Sort by name, date, generator. Click → load into main view. Multi-select for comparison. Drag-and-drop reordering for layout planning.

#### Comparison Mode

Select 2–4 presets, view side-by-side in split viewports. Independent orbit controls per viewport, with optional linked camera toggle. Each viewport shows preset name and key differing parameters highlighted.

#### Export Queue

- "Export STL" / "Export Point Cloud" buttons → export dialog (format, sampling profile, output path)
- Background thread execution with progress bar
- Multiple exports queueable
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

**For meshes:** GLB (binary glTF). Three.js native loading, trimesh exports it. Compact, fast.

**For point clouds:** Custom binary format: header (uint32 point count) + packed float32 XYZ triplets. Simpler and faster than PLY for this specific use case. ~2.4 MB for 200K points.

Both load in <1 second on localhost.

### 7.7 Session and State Management

**Session state** (in-memory, lost on server restart): current selection, current geometry cache.

**Persistent state** (SQLite, Tier 2 only): presets, generation history, gallery, grid manifest, user preferences.

**Geometry cache:** LRU in temp directory, keyed by hash(generator + params + seed + resolution). Default 1GB limit.

### 7.8 Static Image Rendering

PyVista (VTK) for high-resolution offline renders. CLI: `mathviz render <file> --width 4096 --height 4096 --output render.png`. Also accessible from Tier 2 Studio.

### 7.9 2D Rendering

Projection or native 2D evaluation. CLI: `mathviz render-2d <file_or_generator> --projection top --output flat.png`. Tier 2 Studio: "2D projection" toggle in viewport.

---

## 8. CLI Interface

Built with **Typer** (automatic parameter validation from type hints, better agent ergonomics than Click).

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
| `mathviz studio` | Launch the Tier 2 Studio workbench (§7.2–7.4). Full exploration GUI with parameter controls, presets, gallery, comparison, and export queue. Only available after Tier 2 is built. Default port 8457. |
| `mathviz preview` | Launch the Tier 1 preview viewer (§7.1). Minimal Three.js viewport with orbit controls, view mode toggles, container wireframe. Available from Phase 4. |
| `--dry-run` | Run the full pipeline without writing files. Print what would be generated: generator, params, resolution, estimated point/face count, estimated generation time. Essential for agent iteration. |
| `--report path.json` | Write a structured JSON report of the full pipeline run: timing per stage, validation results, output file paths, parameters used. |
| `--json` | All commands produce structured machine-readable JSON to stdout. Since agents are part of the development loop, this is a day-one requirement, not a nice-to-have. |
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

Optional `mathviz.toml` in working directory:

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

The installation is 100+ blocks in a 10×10 (or larger) grid. Individual blocks are managed as presets and per-object TOML configs, but there needs to be a single artifact that maps grid positions to blocks. Without this, managing 100 blocks across weeks of iteration becomes a spreadsheet problem.

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

**The grid manifest is a data model from Phase 1.** The data model and CLI commands exist from early on (it's just a TOML file and a Pydantic model). The Tier 2 Studio's grid layout view (drag-and-drop preset assignment, adjacency preview, installation mockup) is built on top of this data model later. But even without the Studio, the manifest is usable via CLI and a text editor.

**Adjacency tracking:** The manifest stores positions but doesn't enforce adjacency constraints. The operator maintains visual coherence manually (or with CLI queries like `mathviz grid neighbors 3 5` which shows the 8 surrounding blocks). Adjacency is an artistic judgment, not a programmatic constraint.

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
| **numba** | ≥0.58 | JIT compilation | **Fractal inner loops only.** Mandelbulb, Julia 3D, Mandelbrot iteration kernels. NOT a general optimization strategy. Agents should not reach for numba unless the inner loop is a tight numerical kernel with millions of iterations. Cold-start latency is ~2-5 seconds on first call. |

### 10.3 Optional Dependencies

| Package | Version | Purpose | When needed |
|---------|---------|---------|-------------|
| **open3d** | ≥0.17 | PCD export, advanced point cloud ops | Only if PCD format needed. Heavy install. |
| **pyvista** | ≥0.42 | High-res static rendering | Only for `mathviz render` command |
| **meshio** | — | Additional mesh format I/O | If export format breadth becomes insufficient |
| **pygalmesh** / **gmsh** | — | Higher quality mesh generation | If marching cubes quality is insufficient |

Install groups: `pip install mathviz` (core + CLI + Tier 1 preview), `pip install mathviz[studio]` (adds Tier 2: React frontend build), `pip install mathviz[render]` (adds pyvista), `pip install mathviz[open3d]` (adds open3d), `pip install mathviz[all]`. The Tier 2 Studio frontend is a separate npm project in `studio-frontend/` — built to static files and served by the Python backend. Tier 1 preview has no npm dependency (single HTML file with CDN-loaded Three.js).

### 10.4 Tier 2 Studio Dependencies

These are only needed if building the full Tier 2 Studio (Phase 7). Tier 1 preview has zero additional dependencies beyond core.

**Backend (already in core):**
FastAPI and uvicorn are in core dependencies since Tier 1 also uses them. Tier 2 adds:

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

The frontend builds to static files (`studio/static/`) served by FastAPI. During development, Vite dev server proxies API calls to the backend.

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

Primary: macOS (Apple Silicon). Pure Python + compiled extensions that all have ARM64 wheels. No platform-specific code.

---

## 11. Performance

### 11.1 Timing Instrumentation

Every pipeline stage logs its wall-clock duration. This is not optional — it's built into the core pipeline runner.

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

Both Tier 1 and Tier 2 viewers must maintain ≥30fps during orbit/pan/zoom. This is the primary UX constraint — a laggy viewport makes artistic exploration painful.

- **LOD defaults:** 100K faces for meshes, 200K points for clouds (configurable in mathviz.toml)
- **Decimation is server-side:** FastAPI sends pre-decimated GLB/binary geometry. The browser never processes full-resolution data during interaction.
- **High-res mode:** Explicit user action (button click) loads full geometry. Loading spinner shown. Framerate may drop — acceptable since user explicitly requested it.
- **Point size:** Three.js `Points` material with `sizeAttenuation: true` and configurable base size.
- **Parameter debounce (Tier 2 only):** Slider changes debounced at 300ms. Regeneration at preview resolution. "Regenerate full-res" button for deliberate high-quality generation.
- **Transform-only updates (Tier 2 only):** Placement changes re-transform existing geometry without regeneration — effectively instant.
- **Geometry cache:** Recent generations cached in memory (LRU, 1GB default). Revisiting a recent configuration is instant.

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

Self-registering via decorator, with alias support:

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

Reference STL/meta files in `fixtures/`. Tests compare generated output against reference: vertex count within tolerance, bounding box match, face count within tolerance.

---

## 14. Agent Development Guidelines

### Module Independence

Each generator is a self-contained file importing only from `core/`, `shared/`, and external libraries. Never from other generators. An agent implements a new generator by looking at one example + the GeneratorBase ABC.

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
— **This is the artistic exploration checkpoint.** After Phase 4 you can generate objects via CLI, preview them in the browser, iterate on parameters, and export production files. Phases 5+ happen in parallel with block production.

**Phase 5: Remaining generators** — any order, each independent. Grid manifest (§9.4) maintained as TOML alongside production.

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
- numba used ONLY in fractal inner loops — nowhere else

### Test-Driven for Agents

Each implementation PR includes:
1. The implementation file
2. Tests: default params + at least one non-default combination + determinism test
3. An example TOML config

Verify: `pytest tests/test_generators/test_<category>.py -v`

---

## 15. Open Questions and Future Considerations

### Engraver Format

Confirm with vendor: what format, what resolution, what point density. This anchors the production sampling profile.

### Mesh Boolean Operations

Add as needed (trimesh CSG, libigl, pymeshlab). Architecture supports it — it would be a shared component or a representation strategy variant.

### Parameter Sweeps

Future: `mathviz sweep` command generating N objects with interpolated parameters. Architecture supports it via config files.

### GPU Acceleration

If fractal generation at voxel_resolution=512+ becomes a bottleneck: CUDA or Metal compute shaders. Architecture doesn't preclude it — the numba inner loops are the natural swap point.

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

The default container is a 100×100×100mm cube. When using non-cubic containers (e.g. shallow blocks), mathematically balanced objects may appear compressed. The `PlacementPolicy.depth_bias` parameter lets the operator exaggerate or further compress depth to optimize visual readability in the medium. The default of 1.0 applies no correction; values of 1.2–1.5 may help for objects with important z-axis structure in shallow containers.

### Fracture Point Visibility

Points deeper in the glass are less visible (light attenuation through glass). The `EngravingOptimizer.depth_compensation` flag increases point density for deeper points. `depth_compensation_factor=1.5` means the deepest layer gets 1.5× the density of the front layer, with linear interpolation between.

### Volumetric Occlusion

A dense point cloud of a solid object becomes an opaque white brick, obscuring internal structure. The `EngravingOptimizer` provides three strategies:

- **shell_fade:** Keep surface points at full density, thin inner layers progressively. Good for objects with interesting surface detail (Mandelbulb, minimal surfaces).
- **radial_gradient:** Density decreases from center outward, so the core is visible through a sparse outer shell. Good for objects with interesting internal structure.
- **none:** No thinning. Appropriate for inherently sparse objects (attractors, wireframes, thin surfaces).

---

## Appendix C: Resolution Quick Reference

For agents implementing generators — know what your resolution parameters cost:

| Resolution Type | N=64 | N=128 | N=256 | N=512 |
|----------------|------|-------|-------|-------|
| `grid_resolution` (N²) | 4K pts | 16K pts | 65K pts | 262K pts |
| `voxel_resolution` (N³) | 262K vox | 2M vox | 16M vox | 134M vox |
| `integration_steps` (N) | 64 steps | 128 steps | 256 steps | 512 steps |
| `curve_points` (N) | 64 pts | 128 pts | 256 pts | 512 pts |
| `pixel_resolution` (N²) | 4K px | 16K px | 65K px | 262K px |

Rule of thumb: `voxel_resolution` above 256 requires numba for fractal kernels. Above 512, expect minutes even with numba.
