# MathViz — Tasks

---

## Task 1: Project scaffold and geometry dataclasses

**Objective:**

Establish the package structure and the core geometry containers that every
other module depends on. After this task, a developer can import `MathObject`,
populate it with a mesh or point cloud, and call `validate_or_raise()` to get
a clear error if anything is malformed. The `CoordSpace` enum must exist from
the start since the transformer and exporters rely on it for runtime
coordinate-space enforcement.

**Suggested path:**

Set up `pyproject.toml` with the package metadata and core dependency list
(numpy, scipy, trimesh, scikit-image, pydantic, typer, rich, fastapi, uvicorn).
Create the `src/mathviz/` package tree — the directory skeleton should include
empty `__init__.py` files for all planned subpackages so later tasks don't need
to create package structure. The geometry dataclasses (Mesh, PointCloud, Curve,
BoundingBox, MathObject) are plain `@dataclass` with explicit `validate()`
methods — not Pydantic, since the interesting fields are `np.ndarray`. Each
`validate()` returns a list of error strings; `validate_or_raise()` raises
`ValueError` if that list is non-empty.

**Tests:** `tests/test_core/test_math_object.py`

- MathObject with no geometry (all None) fails validation
- Mesh with out-of-bounds face indices fails validation
- PointCloud with mismatched intensities length fails validation
- NaN in any coordinate array fails validation
- `validate_or_raise()` raises ValueError with all errors joined

---

## Task 2: Configuration models

**Objective:**

Provide the Pydantic models that describe the physical glass block, placement
policy, engraving constraints, and representation options. These are used by
nearly every pipeline stage. The Container must be fully flexible on
dimensions — the operator may use cubes, tall rectangles, or shallow blocks,
so nothing in the codebase should assume a specific aspect ratio or that one
axis is shorter than another.

**Suggested path:**

All models in this task are Pydantic `BaseModel` — they hold scalar config,
not arrays. Container gets a `usable_volume` property and a
`with_uniform_margin()` classmethod. RepresentationConfig holds the
RepresentationType enum plus optional type-specific parameters (tube_radius,
shell_thickness, etc.). EngravingProfile holds point budget, spacing, occlusion
mode, and depth compensation settings. Keep these in separate files under
`core/` since they'll be imported independently by different stages.

**Tests:** `tests/test_core/test_container.py`, `tests/test_core/test_config_models.py`

- Container usable_volume with non-uniform margins computes correctly
- Container with equal width/height/depth (cube) computes correct usable volume
- Container with zero margin on one axis works (margin_z_mm = 0)
- PlacementPolicy defaults are center anchor, +z viewing axis, aspect ratio preserved
- EngravingProfile point_budget must be positive

---

## Task 3: Generator base class and registry

**Objective:**

Create the abstract base class that all generators implement and the registry
that maps names (and aliases) to generator classes. The registry must support
alias resolution — e.g., "trefoil" and "torus_knot" resolve to the same
generator class but with different default parameters. A `@register` decorator
handles self-registration at import time. The registry also needs a discovery
mechanism so that importing the `generators` package auto-registers everything
in its subpackages.

**Suggested path:**

GeneratorBase is an ABC with `generate()`, `get_default_params()`, and
`get_param_schema()` as abstract methods. The `@register` decorator adds the
class to a module-level dict keyed by `name` and all `aliases`. Auto-discovery
can walk the `generators/` subpackages on first access (importlib-based). For
now, implement a tiny placeholder generator inside the test suite to validate
the registry mechanics — real generators come in later tasks.

**Tests:** `tests/test_core/test_generator.py`

- Registering a generator makes it discoverable by canonical name
- Registering with aliases makes it discoverable by each alias
- Alias and canonical name resolve to the same class
- Duplicate name registration raises an error
- Registry listing returns all registered generators with metadata

---

## Task 4: Transformer

**Objective:**

Implement the stage that takes a MathObject in abstract coordinate space and
fits it into a physical container, producing a MathObject in physical
coordinate space (millimeters). The transformer must handle all geometry types
on the MathObject (mesh, point_cloud, curves — whichever are populated) and
must respect the PlacementPolicy: anchor point, aspect ratio preservation,
depth bias, rotation, and offset. Output MathObject must have
`coord_space=PHYSICAL` and an updated bounding box.

**Suggested path:**

The algorithm: optionally rotate, compute AABB, compute usable volume, compute
uniform scale factor (min of per-axis ratios when preserving aspect ratio, or
per-axis when not), translate to anchor point within container, apply offset.
**Note on depth_bias:** the spec's §5.5 describes multiplying usable z by
depth_bias before computing scale, but this doesn't achieve the stated
semantics — it either grows the object uniformly (when z is the bottleneck) or
does nothing. The correct behavior is: apply depth_bias as a z-axis-only scale
factor *after* uniform fitting. `depth_bias=1.5` stretches z to 1.5× what
uniform scaling gives; `depth_bias=0.7` compresses it to 0.7×. This may push
z coordinates past the container margin — that's the intentional trade-off. The
transformer should refuse (raise) if the input is already in PHYSICAL space.

**Tests:** `tests/test_core/test_transformer.py`

- Abstract→Physical sets coord_space to PHYSICAL
- Already-PHYSICAL input raises an error
- Output bounding box fits within container usable volume (with depth_bias=1.0)
- Aspect ratio preserved: a 2:1:1 object in a cube container stays 2:1:1
- depth_bias > 1.0 stretches z relative to x/y (z extent grows, x/y unchanged)
- depth_bias < 1.0 compresses z relative to x/y
- Anchor "bottom" places geometry against the bottom face of the container
- Rotation applies before scaling (a rotated object still fits the container)
- All three geometry types (mesh, cloud, curves) are transformed when present

---

## Task 5: Exporters

**Objective:**

Implement MeshExporter, PointCloudExporter, and MetadataExporter. Each exporter
must fail with a clear error if the required geometry is absent on the
MathObject — no silent conversion between representations. The metadata
exporter writes a sidecar `.meta.json` capturing full reproducibility info.
Every geometry export must automatically write the sidecar alongside it. PCD
export (requires open3d) should be stubbed with a clear error message pointing
to the optional install group.

**Suggested path:**

MeshExporter handles STL (binary only — no ASCII option), OBJ, and PLY (mesh
mode). PointCloudExporter handles PLY (cloud mode) and XYZ. PCD raises an
import error directing the user to `pip install mathviz[open3d]`. Use trimesh
for mesh I/O. MetadataExporter writes JSON containing generator name, full
params dict, seed, container spec, placement policy, representation config,
pipeline stages applied, generation time, export timestamp, and version string.
The sidecar auto-write logic belongs in each geometry exporter method.

**Tests:** `tests/test_core/test_exporters.py`

- STL export of a MathObject with no mesh raises a clear error
- PLY cloud export of a MathObject with no point_cloud raises a clear error
- STL round-trip: export then reimport via trimesh, vertex count matches
- Sidecar `.meta.json` is auto-written alongside every geometry export
- Sidecar contains generator_name, params, seed, and timestamp
- PCD export without open3d installed raises ImportError with install instructions

---

## Task 6: Validator

**Objective:**

Implement two-tier validation: mesh checks (watertight, manifold, no
degenerate faces, normals, bounding box within container) and engraving checks
(point count within budget, spacing, opacity warning, depth distribution).
Return a structured `ValidationResult` with pass/fail per check, severity
levels (error / warning / info), and human-readable messages. Mesh repair is
best-effort and opt-in, not automatic.

**Suggested path:**

ValidationResult can be a dataclass with a list of check results, each having
a name, passed bool, severity, and message. Use trimesh for mesh checks. The
engraving checks operate on PointCloud: count vs budget, min spacing via
KDTree nearest neighbor, max gap check, and axis-projection density for the
opacity warning (>70% of voxels occupied in any projection). Container
bounding-box check needs the container passed in.

**Tests:** `tests/test_core/test_validator.py`

- Mesh with boundary edges (open surface like Möbius strip) triggers a warning, not an error
- Point cloud exceeding point budget fails engraving validation
- Point cloud with points outside container fails
- Opacity warning fires when >70% of voxels in a projection are occupied
- ValidationResult.passed is True only when there are zero errors (warnings allowed)

---

## Task 7: Sampler

**Objective:**

Implement mesh-to-point-cloud conversion with three algorithms: uniform
surface sampling (Poisson disk), random surface sampling (barycentric,
face-area-weighted), and volume fill (jittered grid inside watertight mesh).
The sampler takes a MathObject with a mesh and populates its point_cloud field.
It should be skippable when the MathObject already has a point cloud and no
resampling is requested.

**Suggested path:**

Use trimesh for surface sampling. Volume fill: generate a jittered 3D grid
within the mesh's bounding box, then use trimesh's `contains` to keep only
interior points. Parameters: `density` (points/mm² for surface, points/mm³
for volume) or `num_points` as an explicit count. The sampler should accept
either, with density as the default interface.

**Tests:** `tests/test_core/test_sampler.py`

- Volume fill of a cube produces points only inside the cube
- Sampler skips when point_cloud already exists and no resampling flag is set
- Sampler with density parameter scales point count with surface area
- Requesting N points produces approximately N points (within 10%)

---

## Task 8: Tube thickening

**Objective:**

Implement the shared tube-thickening component that converts a curve (polyline)
into a triangle mesh by extruding a circular cross-section along the path. This
is used by the representation strategy for knots, attractors with tube mode,
curves, and any generator that outputs a Curve. The algorithm must handle closed
curves without seam artifacts and avoid the twisting problems of Frenet-Serret
frames.

**Suggested path:**

Use the parallel transport frame (Bishop frame) method — at each point,
compute a normal and binormal that rotate minimally from one point to the next.
Place a ring of `sides` vertices at each point, connect adjacent rings with
triangles. For closed curves, blend the last frame back into the first to
avoid a seam. Parameters: radius, sides (default 16), closed (auto-detected
from the Curve's `closed` flag). Self-intersection from large radii should
produce a warning but not fail.

**Tests:** `tests/test_shared/test_tube_thickening.py`

- Closed circular curve produces a torus-like mesh with no seam gap
- Output mesh is watertight for closed curves
- Vertex count equals curve_points × sides (plus caps for open curves)
- A helix (high curvature, no inflection) produces no twisting artifacts
- Self-intersection warning fires for large radius relative to curvature

---

## Task 9: Marching cubes wrapper

**Objective:**

Wrap `skimage.measure.marching_cubes` with a clean interface and
post-processing steps (vertex normal computation, optional Laplacian smoothing,
optional mesh decimation). This is used by all implicit surface generators and
3D fractals. The wrapper should accept a 3D scalar field and evaluation bounds
and return a Mesh dataclass with proper coordinates (not voxel indices).

**Suggested path:**

Input: a 3D numpy array (the scalar field), the spatial bounds (min/max
corner in abstract coordinates), an isolevel (default 0), and optional
smoothing / decimation parameters. The wrapper calls marching cubes, then
rescales vertices from voxel space back to the original coordinate space.
Normals: compute from the mesh after extraction. Smoothing: Laplacian via
trimesh if requested. Decimation: `trimesh.simplify` if target face count
is provided.

**Tests:** `tests/test_shared/test_marching_cubes.py`

- Sphere implicit field `x²+y²+z²-1` produces a roughly spherical mesh
- Output vertices are in the coordinate space of the provided bounds, not voxel indices
- Mesh is watertight for a closed implicit surface
- Decimation reduces face count to approximately the target
- Higher voxel_resolution produces more faces (sanity check on scaling)

---

## Task 10: Representation strategy

**Objective:**

Implement `RepresentationStrategy.apply()` — the fabrication layer that
transforms raw generator output into the physical form suitable for engraving.
This is the key architectural seam between mathematical definition and
fabrication. Implement these representation types: SURFACE_SHELL (pass-through
for meshes), RAW_POINT_CLOUD (extract points from curves into a PointCloud),
TUBE (apply tube thickening to curves), and HEIGHTMAP_RELIEF (extrude a 2D
scalar field into a surface mesh). Also implement `get_default()` which
returns the recommended representation for each generator name. Remaining types
(SPARSE_SHELL, VOLUME_FILL, WIREFRAME, WEIGHTED_CLOUD, SLICE_STACK) can be
stubs that raise NotImplementedError — they're filled in by later tasks.

**Suggested path:**

`apply()` dispatches on `RepresentationConfig.type`. SURFACE_SHELL is a no-op
for mesh inputs; RAW_POINT_CLOUD extracts curve points into a PointCloud;
TUBE calls tube thickening from Task 8; HEIGHTMAP_RELIEF builds a grid mesh
from `MathObject.scalar_field`. For the HEIGHTMAP_RELIEF test, synthesize a
simple 2D gradient array as the scalar_field — no real generator needed yet.
`get_default()` maps generator names to default configs. The interface should
also accept a `candidates` mode that returns multiple representations for
comparison — this can be a stub initially but the flag must exist for the CLI.

**Tests:** `tests/test_core/test_representation.py`

- SURFACE_SHELL on a mesh input returns a MathObject with mesh intact
- RAW_POINT_CLOUD on a curve input produces a PointCloud from curve points
- TUBE on a curve input produces a watertight mesh via tube thickening
- TUBE on a mesh-only input (no curves) raises a clear error
- HEIGHTMAP_RELIEF on a scalar field produces a mesh with z-range matching the field
- Applying a representation sets the `representation` field on MathObject
- NotImplementedError for unimplemented types is clear and names the type

---

## Task 11: Pipeline runner and timing

**Objective:**

Implement the pipeline runner that chains stages together with timing
instrumentation and `validate_or_raise()` at every stage boundary. The runner
is the single entry point for executing the full or partial pipeline. Each
stage is independently skippable (representation, sampling, engraving
optimization), and the runner must handle these optional stages cleanly.
PipelineTimer records wall-clock time per stage and makes timing available
in the result.

**Suggested path:**

The runner takes a generator name (or instance), params, seed, resolution
kwargs, container, placement, representation config (optional), sampling config
(optional), engraving profile (optional), and export config. It resolves the
generator from the registry, calls generate, applies representation (using
`get_default()` if no explicit config), transforms, optionally samples,
optionally optimizes, validates, and exports. PipelineTimer is a
context-manager utility. Use a placeholder generator in tests — real
generators arrive in the next task.

**Tests:** `tests/test_core/test_pipeline.py`

- Full pipeline with all stages produces valid output and timing for each stage
- Pipeline with no sampling config skips sampling
- validate_or_raise fires between stages (inject a generator that produces invalid geometry, confirm it fails before reaching the transformer)
- Timing dict has entries for every stage that ran, none for skipped stages
- Pipeline result includes the final MathObject and ValidationResult

---

## Task 12: First generator — torus

**Objective:**

Implement the torus as the first real parametric surface generator, establishing
the pattern all other parametric generators will follow. The torus is chosen
because it's mathematically simple, always valid, and exercises the full
pipeline (mesh output → representation → transform → export). After this task,
`mathviz generate torus` via the pipeline runner should produce a valid STL
file end-to-end. This generator also serves as the reference implementation
that agents will study when building subsequent generators.

**Suggested path:**

The shared parametric pattern: define `f(u, v, **params) → (x, y, z)`, evaluate
on a regular grid of `grid_resolution × grid_resolution`, build a triangle
mesh (two triangles per grid cell), close at boundaries for periodic surfaces.
The torus is periodic in both u and v so the grid wraps. Key parameters: major
radius R (default 1.0), minor radius r (default 0.4). Resolution parameter:
`grid_resolution` (default 128). Default representation: SURFACE_SHELL.
Register with no aliases.

**Tests:** `tests/test_generators/test_parametric.py`

- Default torus produces a watertight mesh
- Mesh face count is consistent with grid_resolution (2 × N² faces for N×N grid)
- Bounding box is roughly ±(R+r) in x/y and ±r in z
- Determinism: same seed (even though torus is deterministic, seed is accepted and recorded)
- Full pipeline: generate → transform → export STL → reimport → valid mesh
- Low resolution (grid_resolution=8) still produces valid geometry

---

## Task 13: CLI foundation

**Objective:**

Implement the Typer-based CLI with the core commands: `generate`, `list`,
`info`, and `validate`. The CLI is a first-class interface for agents — every
command must support `--json` for structured output. `--dry-run` on generate
must report what would happen without writing files. `--report path.json`
writes a structured pipeline report. The CLI should use long flag names only
(no single-letter shortcuts) for agent readability. Exit codes: 0 = success,
1 = validation warning (output still produced), 2 = error.

**Suggested path:**

`cli.py` as the Typer app entry point. The `generate` command takes a
generator name as a positional argument, plus the full flag set from §8. Not
all flags need to be fully wired yet — the ones that matter now are: `--param`,
`--seed`, `--output`, `--format`, `--dry-run`, `--report`, `--json`,
`--verbose`/`--quiet`. The rest can accept values and pass them through to the
pipeline runner. Use `rich` for terminal output in non-json mode.

**Tests:** `tests/test_cli.py`

- `mathviz generate torus --json` outputs valid JSON with timing and metadata
- `mathviz generate torus --dry-run` writes no files but prints expected output info
- `mathviz generate torus --report /tmp/report.json` writes valid JSON report
- `mathviz list --json` produces valid JSON array of generator info
- `mathviz info torus --json` produces valid JSON param schema
- Unknown generator name exits with code 2 and a clear error
- `--param R=2.0 --param r=0.5` parses multiple key=value params correctly

---

## Task 14: Lorenz attractor

**Objective:**

Implement the Lorenz attractor as the first dynamical system generator,
establishing the pattern for all attractor/ODE generators. The Lorenz system
outputs a Curve (trajectory). The representation strategy converts it — the
default RAW_POINT_CLOUD representation extracts the curve points into a
PointCloud, producing the ghostly trail aesthetic. The generator must NOT
pre-populate point_cloud itself; that's the representation layer's job.

**Suggested path:**

Use `scipy.integrate.solve_ivp` with RK45 or DOP853. Parameters: σ (default
10), ρ (default 28), β (default 8/3), plus transient_steps (default 1000) to
discard. Resolution parameter: `integration_steps` (default 100,000). Initial
condition: `(1, 1, 1)` perturbed by `default_rng(seed)` for variation. Output
MathObject has `curves` populated only; the representation strategy (already
in Task 10) handles RAW_POINT_CLOUD conversion.

**Tests:** `tests/test_generators/test_attractors.py`

- Default Lorenz produces a curve with `integration_steps - transient_steps` points
- Bounding box is finite and non-degenerate (no NaN, no collapse to a line)
- Different seeds produce different trajectories (initial condition perturbation)
- Same seed produces identical output (determinism)
- Output has curves populated but NOT point_cloud (representation handles that)
- Full pipeline with RAW_POINT_CLOUD representation produces a valid PointCloud

---

## Task 15: Torus knot with aliases

**Objective:**

Implement the torus knot generator with alias support, establishing the
pattern for knot-family generators. "trefoil" and "cinquefoil" must resolve to
this generator with different default (p, q) values. The torus knot outputs a
Curve; its default representation is TUBE. After this task, the full pipeline
for `mathviz generate trefoil --format stl` should produce a tube mesh STL.

**Suggested path:**

Standard torus knot parametric formula. Parameters: p (default 2), q (default
3), R (major radius, default 1.0), r (minor radius, default 0.4). Resolution
parameter: `curve_points` (default 1024). Register with aliases: "trefoil"
maps to `{p:2, q:3}`, "cinquefoil" to `{p:2, q:5}`. The representation
strategy's TUBE mode (Task 10) handles tube thickening via the pipeline.

**Tests:** `tests/test_generators/test_knots.py`

- "trefoil" alias resolves to torus_knot and produces p=2, q=3 defaults
- "cinquefoil" alias produces p=2, q=5 defaults
- Output curve is closed (first and last points are coincident or close)
- Full pipeline with TUBE representation produces a watertight mesh
- Explicit `--param p=3 --param q=5` via alias still works (param override)

---

## Task 16: Gyroid

**Objective:**

Implement the gyroid as the first implicit surface generator, exercising the
marching cubes wrapper from Task 9. The gyroid tiles space infinitely —
the `periods` parameter controls how many unit cells are included. This
generator demonstrates the implicit surface pattern: define scalar field,
evaluate on voxel grid, run marching cubes, return mesh.

**Suggested path:**

Scalar field: `sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x)`. Parameters:
cell_size (default 1.0), periods (default 2). Resolution parameter:
`voxel_resolution` (default 128). Evaluate the field on an N³ grid, pass to
the marching cubes wrapper. Default representation: SURFACE_SHELL. Note the
O(N³) cost in the generator's docstring and param schema.

**Tests:** `tests/test_generators/test_implicit.py`

- Default gyroid produces a non-empty manifold mesh
- Increasing periods increases face count (more cells = more geometry)
- Bounding box extent is proportional to periods × cell_size
- voxel_resolution=32 still produces valid geometry (low-res sanity check)
- Full pipeline: generate → transform to container → export STL → valid file

---

## Task 17: Mandelbrot heightmap

**Objective:**

Implement the Mandelbrot set as a heightmap generator, where escape-time
iteration count becomes the z-height of a relief surface. This is the first
fractal generator and the first user of the HEIGHTMAP_RELIEF representation.
After this task, `mathviz generate mandelbrot_heightmap` should produce a 3D
relief surface viewable as STL.

**Suggested path:**

Standard escape-time algorithm on a 2D grid of `pixel_resolution²` points.
Parameters: center_real (-0.5), center_imag (0.0), zoom (1.0),
max_iterations (256), height_scale (1.0), smoothing (True). Store iteration
counts as a 2D array in `MathObject.scalar_field`. The HEIGHTMAP_RELIEF
representation (Task 10) extrudes this into a mesh. No numba needed here —
pixel_resolution=512 is only 262K evaluations, fast in vectorized NumPy.

**Tests:** `tests/test_generators/test_fractals.py`

- Default parameters produce a non-empty scalar field of shape (pixel_resolution, pixel_resolution)
- Smooth iteration count produces non-integer values (when smoothing=True)
- Full pipeline with HEIGHTMAP_RELIEF representation produces a valid mesh
- height_scale multiplier scales the z-range of the resulting mesh proportionally
- Zooming in changes the detail pattern (different scalar field values)

---

## Task 18: Tier 1 preview — FastAPI backend

**Objective:**

Implement the FastAPI server that powers the Tier 1 preview viewer. The server
exposes REST endpoints for listing generators, running the pipeline, and
serving decimated geometry in browser-friendly formats. LOD decimation is
server-side — the browser never processes full-resolution geometry during
interactive viewing. A geometry cache (LRU) avoids re-generating identical
configurations.

**Suggested path:**

Endpoints: `GET /api/generators`, `GET /api/generators/{name}`,
`POST /api/generate` (accepts generator name + params + seed + resolution,
runs pipeline, returns geometry ID and URLs), `GET /api/geometry/{id}/mesh`
and `/cloud` with `?lod=preview|full`. Mesh LOD: decimate to ≤100K faces,
serve as GLB. Cloud LOD: subsample to ≤200K points, serve as binary PLY.
Geometry cache: key by hash of (generator + params + seed + resolution),
LRU with configurable size limit.

**Tests:** `tests/test_preview/test_server.py`

- `GET /api/generators` returns JSON list including the torus
- `POST /api/generate` with torus returns geometry URLs
- `GET /api/geometry/{id}/mesh?lod=preview` returns valid binary data
- LOD preview mesh has ≤100K faces
- Same generation request hits cache on second call (faster response)
- Unknown generator returns 404 with helpful error

---

## Task 19: Tier 1 preview — Three.js viewer and preview CLI command

**Objective:**

Build the single-file HTML/JS viewer (`static/index.html`) and wire it up to
the `mathviz preview` CLI command. The viewer must maintain ≥30fps during
orbit/pan/zoom — this is the primary UX constraint. It provides orbit controls,
view mode toggles (shaded mesh / wireframe / point cloud), container bounding
box wireframe, background toggle (dark / light), point size slider, screenshot
button, and info display. No npm, no build step — vanilla JS with Three.js
from CDN.

**Suggested path:**

The page reads URL query params (`?generator=lorenz_attractor&sigma=12`) to
determine what to generate. GLB for meshes (GLTFLoader), binary PLY for point
clouds. The `mathviz preview` CLI command starts the FastAPI server, optionally
opens the browser, and accepts either a file path or a generator name.

**Tests:** `tests/test_preview/test_viewer.py`

- `mathviz preview torus --no-open` starts server, `/` serves HTML with Three.js
- HTML contains OrbitControls setup and all view mode toggles
- URL query params are parsed and forwarded to generate endpoint
- `mathviz preview <stl_file> --no-open` serves the file in the viewer
- Server shuts down cleanly on interrupt

---

*After Task 19, the artistic exploration checkpoint is reached. The operator
can generate objects via CLI, preview them in the browser, iterate on
parameters, and export production files. All remaining tasks happen in parallel
with block production.*

---

## Task 20: Remaining parametric surface generators

**Objective:**

Implement the remaining parametric surface generators: klein_bottle,
mobius_strip, superellipsoid, spherical_harmonics, lissajous_surface,
boy_surface, and enneper_surface. Each follows the pattern established by the
torus in Task 12. Each generator is a self-contained file importing only from
`core/` and external libraries — never from other generators.

**Suggested path:**

All use `grid_resolution` as their resolution parameter. Watch for: Klein
bottle immersion has self-intersection (valid, not an error); Möbius strip has
a boundary (not watertight); superellipsoid uses signed-power trig functions;
spherical harmonics can accept either (l, m) or a coefficient vector. Default
representation for all: SURFACE_SHELL.

**Tests:** `tests/test_generators/test_parametric.py` (extend existing)

- Each generator produces a non-empty mesh at default params
- Klein bottle has self-intersection (not watertight) — no error
- Möbius strip is not watertight (open boundary) — no error
- Superellipsoid with e1=e2=1 approximates a sphere
- Spherical harmonics with all-zero coefficients except Y₀₀ approximates a sphere
- All generators accept and record seed, even if they don't use randomness

---

## Task 21: Remaining implicit surface generators

**Objective:**

Implement schwarz_p, schwarz_d, costa_surface, and genus2_surface. Schwarz P
and D are TPMS like the gyroid and share the same structural parameters
(cell_size, periods). Costa surface may need a Weierstrass representation
approach rather than a simple implicit form — a parametric fallback is
acceptable. genus2_surface is a genus-2 implicit surface.

**Suggested path:**

Schwarz P: `cos(x) + cos(y) + cos(z) = 0`. Schwarz D: the standard diamond
TPMS form. The TPMS generators can share a helper for grid evaluation +
marching cubes if natural, but code duplication is fine. Costa surface and
genus2 are more specialized — implement whatever approach produces valid
geometry and document the mathematical construction in the docstring.

**Tests:** `tests/test_generators/test_implicit.py` (extend existing)

- Schwarz P and Schwarz D produce manifold meshes
- TPMS generators with periods=1 vs periods=3 differ in face count proportionally
- Costa surface produces geometry (basic non-empty check — topology is complex)
- genus2_surface produces a mesh with genus > 1 (higher Euler characteristic check via trimesh)

---

## Task 22: Remaining attractors and dynamical systems

**Objective:**

Implement the remaining attractor generators: rossler, chen, thomas, halvorsen,
aizawa, and double_pendulum. Each follows the ODE integration pattern from the
Lorenz task. All default to RAW_POINT_CLOUD representation. The double
pendulum is special — it's a 4D phase space system projected to 3D; the
projection choice must be documented.

**Suggested path:**

Each attractor is a separate file under `generators/attractors/`. All share
the pattern: define derivatives, call `solve_ivp`, discard transient, output
Curve. The double pendulum requires choosing a 3D projection of the 4D phase
space — document the choice. Each generator should include its equations in
the docstring.

**Tests:** `tests/test_generators/test_attractors.py` (extend existing)

- Each attractor produces finite, non-degenerate geometry
- Rössler bounding box is wider than tall (characteristic folded-band shape)
- Thomas attractor with default b≈0.208186 produces a bounded attractor (not divergent)
- Double pendulum projection produces 3D points (not collapsed to a plane)
- All are deterministic given seed

---

## Task 23: Mandelbulb, Julia 3D, and fractal cross-sections

**Objective:**

Implement the Mandelbulb, 3D Julia set, and fractal cross-section (fractal
slice) generators. These are the most computationally expensive generators
and the primary use case for numba JIT compilation. The inner escape-time
loop must be JIT-compiled for acceptable performance at voxel_resolution ≥128.
Default representation for Mandelbulb and Julia 3D: SPARSE_SHELL. This task
also adds the SPARSE_SHELL representation type to `RepresentationStrategy`.
Fractal slice reuses the 3D evaluator, outputs a 2D cut as heightmap.

**Suggested path:**

The Mandelbulb formula is in the spec's Appendix A. For Julia 3D, same
iteration but with a fixed c parameter. Both evaluate escape-time on a voxel
grid, then marching cubes extracts the boundary. Use `@numba.njit` on the
inner iteration kernel — add numba to the dependency list (performance group).
SPARSE_SHELL: sample the mesh surface at reduced density to preserve fractal
detail without opacity. Fractal slice: evaluate the 3D field on a 2D plane,
output as scalar_field for HEIGHTMAP_RELIEF.

**Tests:** `tests/test_generators/test_fractals.py` (extend existing)

- Mandelbulb at voxel_resolution=32 produces a non-empty mesh (low-res smoke test)
- Julia 3D with a known c value produces geometry
- Same seed + params produces identical output (determinism despite numba)
- SPARSE_SHELL representation produces a point cloud with fewer points than full surface sampling
- Fractal slice at a known plane produces a non-trivial 2D heightmap
- numba-compiled kernel is used (check via timing or a flag)

---

## Task 24: Remaining knots and curves

**Objective:**

Implement the remaining knot generators (figure_eight_knot, lissajous_knot,
seven_crossing_knots) and the curve generators (lissajous_curve,
logarithmic_spiral, cardioid, fibonacci_spiral, parabolic_envelope). Knots
default to TUBE representation; curves default to TUBE with a thinner radius.
All use `curve_points` as their resolution parameter.

**Suggested path:**

Each is a parametric curve evaluated at `curve_points` evenly spaced parameter
values. Lissajous knot uses three frequency/phase pairs. Figure-eight knot
has a standard parametric formula. seven_crossing_knots may need explicit
coordinate data for specific knot types — a knot_index parameter selects which.
Parabolic envelope is a family of lines whose envelope forms a surface. Default
tube radius should vary per generator — knots want thicker tubes than spirals.

**Tests:** `tests/test_generators/test_knots.py`, `tests/test_generators/test_curves.py`

- Figure-eight knot produces a closed curve
- Lissajous knot with coprime frequencies produces a closed curve
- seven_crossing_knots with different knot_index values produce distinct curves
- Logarithmic spiral extent scales with the number of turns
- TUBE representation on closed curves produces watertight meshes
- Parabolic envelope produces a surface mesh (not just a curve)

---

## Task 25: Number theory generators

**Objective:**

Implement ulam_spiral, sacks_spiral, prime_gaps, and digit_encoding. These
generators map mathematical sequences to 3D geometry. They introduce the
WEIGHTED_CLOUD representation type — point clouds where each point has an
intensity weight. This task adds WEIGHTED_CLOUD support to
`RepresentationStrategy.apply()`.

**Suggested path:**

Ulam spiral: integers on a rectangular spiral grid, primes elevated in z.
Sacks spiral: Archimedean spiral where primes are marked. Prime gaps:
consecutive gaps mapped to a 3D ribbon. Digit encoding: digits of π, e, or
other constants mapped to point positions/heights. WEIGHTED_CLOUD: the
`intensities` field on PointCloud carries per-point weights. The
representation strategy can modulate density based on weights or pass them
through to the engraving optimizer.

**Tests:** `tests/test_generators/test_number_theory.py`

- Ulam spiral: primes have higher z-values than composites
- Prime gaps: gap sizes are mathematically correct for the first N primes
- Digit encoding with π produces correct digit sequence (spot-check first 10)
- WEIGHTED_CLOUD representation preserves intensities on the PointCloud

---

## Task 26: Physics generators

**Objective:**

Implement kepler_orbit, nbody, and planetary_positions. Kepler orbit is a
straightforward parametric curve. N-body is a gravitational integration
requiring seed for reproducible initial conditions. Planetary positions
renders the solar system at a given epoch with spheres and orbital trails.

**Suggested path:**

Kepler orbit: elliptical orbit from orbital elements, output as Curve.
N-body: integrate gravitational system via scipy, output trajectories as
curves or merged point cloud. Initial conditions from seed. Planetary
positions: hardcode or compute planet positions at a given Julian date, render
each orbit as a Curve and each body as a small sphere (point or mesh marker).
Default representations: TUBE for orbits, RAW_POINT_CLOUD for n-body.

**Tests:** `tests/test_generators/test_physics.py`

- Kepler orbit with eccentricity=0 produces a circular curve
- N-body with 2 bodies produces recognizable orbital curves
- N-body with same seed is deterministic
- Planetary positions produces multiple curves (one per planet)

---

## Task 27: Procedural generators

**Objective:**

Implement noise_surface, terrain, and reaction_diffusion. All require seed for
reproducibility. Noise generators use seed-controlled Perlin/simplex noise.
Reaction-diffusion runs Gray-Scott on a 2D grid and maps the concentration
field to 3D.

**Suggested path:**

Noise surface: evaluate noise as an implicit surface or heightmap. Terrain:
multi-octave noise heightmap. Reaction-diffusion: Gray-Scott model, output
the concentration field as scalar_field for HEIGHTMAP_RELIEF. For noise,
use a lightweight library or implement simplex noise directly if dependency
is a concern.

**Tests:** `tests/test_generators/test_procedural.py`

- Noise surface with same seed produces identical mesh
- Terrain heightmap has plausible elevation range (not flat, not degenerate)
- Reaction-diffusion produces non-trivial pattern (not uniform field)
- Different seeds produce different outputs for all three generators

---

## Task 28: Voronoi 3D and generic parametric

**Objective:**

Implement voronoi_3d (3D Voronoi cell boundaries as wireframe geometry) and
generic_parametric (user-supplied `f(u,v) → (x,y,z)` as a string expression).
Voronoi requires seed for reproducible point placement. Generic parametric
is the escape hatch for forms not covered by dedicated generators — it must
evaluate user expressions safely with no code injection risk.

**Suggested path:**

Voronoi: generate N random points via `default_rng(seed)`, compute 3D Voronoi
(scipy.spatial.Voronoi), extract finite cell ridges as edges, output as Curves.
Default representation: WIREFRAME. Generic parametric: accept three string
expressions for x(u,v), y(u,v), z(u,v), evaluate safely using a restricted
namespace (numpy math functions only — no exec, no imports, no file I/O).

**Tests:** `tests/test_generators/test_geometry.py`

- Voronoi with seed=42 produces consistent geometry across runs
- Voronoi cell count scales with the number of seed points
- Generic parametric with torus formula matches the dedicated torus generator (within tolerance)
- Malicious expressions (import, exec, file I/O) are rejected

---

## Task 29: Data-driven generators

**Objective:**

Implement heightmap (from image or GeoTIFF), building_extrude (from GeoJSON),
and soundwave (from WAV/MP3). These consume external data files and convert
them to 3D geometry. Input validation is critical — the data file must exist,
be in the expected format, and have reasonable dimensions. File paths are
passed via `--param input_file=path`.

**Suggested path:**

Heightmap: read an image (PIL/Pillow) or GeoTIFF (rasterio, optional dep) as
a 2D array, use as scalar_field for HEIGHTMAP_RELIEF. Building extrude: parse
GeoJSON polygons, extrude to height, output as mesh. Soundwave: extract
amplitude envelope or spectrogram, map to 3D. These generators should check
file existence early with clear errors.

**Tests:** `tests/test_generators/test_data_driven.py`

- Heightmap from a test PNG produces a mesh with z-range proportional to pixel values
- Building extrude from a simple GeoJSON rectangle produces a box mesh
- Missing input file raises a clear error before any computation
- Unsupported file format raises a descriptive error

---

## Task 30: Engraving optimizer

**Objective:**

Implement the post-sampling adjustments specific to the laser engraving medium:
volumetric occlusion thinning, depth-dependent density compensation, and point
budget enforcement. The optimizer operates on a PointCloud and returns a
modified PointCloud. It must be idempotent — running it twice should not
degrade the result further.

**Suggested path:**

Three occlusion modes: `shell_fade` (thin outer layers progressively),
`radial_gradient` (density decreases from center outward), and `none`. Depth
compensation: linearly interpolate density multiplier from 1.0 at front to
`depth_compensation_factor` at back. Point budget enforcement: downsample
uniformly if count exceeds budget after other adjustments. The Container is
needed to know depth axis orientation.

**Tests:** `tests/test_core/test_engraving.py`

- `occlusion_mode="none"` returns point cloud unchanged (except budget trim)
- `shell_fade` reduces point count in outer layers vs inner
- Depth compensation increases point density at max depth
- Point budget enforcement never exceeds the budget
- Idempotent: optimize(optimize(cloud)) ≈ optimize(cloud) in point count

---

## Task 31: Config file support and sampling profiles

**Objective:**

Add support for the `mathviz.toml` project config file and per-object TOML
configs, plus the predefined sampling profiles (preview / production / custom).
Config files provide defaults; CLI flags always override. The `--config` flag
on `generate` loads a per-object TOML. Sampling profiles bundle point budget,
density, and method into named presets.

**Suggested path:**

Use Python 3.11's `tomllib`. The project config (`mathviz.toml`) is
auto-discovered in the working directory. Per-object configs specify generator,
seed, params, resolution, representation, placement, and container. Merge
order: built-in defaults → project config → per-object config → CLI flags.
Sampling profiles live in `profiles/` as TOML files.

**Tests:** `tests/test_core/test_config.py`

- Project config sets container dimensions, CLI flag overrides them
- Per-object config specifies generator and params, pipeline uses them
- Missing config file produces a clear error with the path it tried
- Sampling profile "preview" loads correct point budget and method
- Merge order: CLI > per-object > project > built-in defaults

---

## Task 32: Grid manifest and grid CLI

**Objective:**

Implement the GridManifest and GridBlock data models and the `mathviz grid`
CLI commands. The grid manifest tracks which preset/config is assigned to each
position in the installation grid, along with export status. Grid dimensions
are configurable — not hardcoded to any specific size.

**Suggested path:**

GridManifest and GridBlock are Pydantic models. The manifest lives in
`grid.toml` (path configurable). CLI commands: `grid show` (ASCII table),
`grid show --json`, `grid assign`, `grid status`, `grid neighbors` (8
surrounding blocks), `grid summary` (counts by status), and `grid export-all`
(batch export all assigned blocks sequentially).

**Tests:** `tests/test_core/test_grid.py`

- Assigning a block and retrieving it returns correct preset/config
- Grid summary counts statuses correctly, including empties
- Neighbors at corner position returns only 3 neighbors (not 8)
- Grid with non-square dimensions (e.g., 8×12) works correctly
- Assigning to an out-of-bounds position raises an error
- Status transitions update the manifest file on disk

---

## Task 33: Remaining representation strategies

**Objective:**

Implement the representation types that were stubbed in Task 10: VOLUME_FILL,
SLICE_STACK, and WIREFRAME. VOLUME_FILL fills the interior of a watertight
mesh with points. SLICE_STACK cuts parallel slices through a volume.
WIREFRAME extracts mesh edges and thickens them into thin tubes. These
complete the full set of fabrication strategies from the spec.

**Suggested path:**

VOLUME_FILL: use the sampler's volume fill (Task 7). Requires watertight mesh.
SLICE_STACK: slice the mesh with parallel planes along the chosen axis, extract
contour curves, output as Curves or PointCloud. WIREFRAME: extract unique edges,
treat each as a short Curve, apply tube thickening at wireframe_thickness,
merge into a single mesh.

**Tests:** `tests/test_core/test_representation.py` (extend existing)

- VOLUME_FILL on a watertight cube fills the interior (points inside, none outside)
- VOLUME_FILL on a non-watertight mesh raises an error
- SLICE_STACK with slice_count=5 produces roughly 5 discrete layers
- WIREFRAME produces thin tubes along mesh edges
- WIREFRAME vertex count scales with edge count × tube_sides

---

## Task 34: High-resolution renderer and 2D rendering

**Objective:**

Implement the `mathviz render` command (PyVista/VTK high-resolution offline
rendering) and `mathviz render-2d` command (projection or native 2D
evaluation). These require optional dependencies — the system must fail
gracefully with a clear install instruction if pyvista is not available.

**Suggested path:**

`mathviz render` loads geometry, sets up a PyVista scene with lighting
appropriate for simulating backlit glass, and renders to PNG. `mathviz
render-2d` supports projections (top/front/side/angle). For the optional
dependency pattern, catch ImportError with a message pointing to
`pip install mathviz[render]`.

**Tests:** `tests/test_preview/test_renderer.py`

- Render command produces a PNG file at the specified dimensions
- Missing pyvista import produces a clear error message with install instructions
- 2D top projection of a sphere produces a circular outline
- Render output is non-trivial (not all-black or all-white)

---

## Task 35: Utility CLI commands and schema generation

**Objective:**

Implement the remaining CLI commands: `convert` (format conversion), `sample`
(standalone mesh→cloud sampling), and `transform` (standalone bounding-box
fitting). Also add automatic JSON Schema generation from the Pydantic models
and generator parameter schemas. These commands make individual pipeline stages
accessible without running the full pipeline.

**Suggested path:**

`mathviz convert <input> <output>` reads geometry, optionally samples (if
`--auto-sample`), writes to target format. `mathviz sample` runs only the
sampler. `mathviz transform` runs only the transformer. For schema generation,
use Pydantic's `.model_json_schema()` on all config models and generators.

**Tests:** `tests/test_cli.py` (extend existing)

- `mathviz convert` STL to OBJ produces a valid OBJ file
- `mathviz convert` mesh STL to PLY point cloud with `--auto-sample` produces a valid cloud PLY
- `mathviz convert` PLY(cloud) to STL without `--auto-sample` fails with clear error
- `mathviz sample` on an STL produces a PLY point cloud
- `mathviz transform` fits geometry within specified container dimensions
- Schema generation produces valid JSON Schema files

---

## Task 36: End-to-end fixture tests and reference outputs

**Objective:**

Generate reference output files (STL, PLY, meta.json) for at least one
generator per category using default parameters. Store them in `fixtures/`.
Write tests that regenerate and compare against the reference: vertex/point
counts within tolerance, bounding boxes match, metadata fields are consistent.
This catches regressions across the entire pipeline and validates that
deterministic seeding produces reproducible output across runs.

**Suggested path:**

Pick one generator per category: torus (parametric), gyroid (implicit), lorenz
(attractor), mandelbulb (fractal), torus_knot (knot), ulam_spiral (number
theory), lissajous_curve (curve). Generate each at default params + seed=42
at low resolution. The comparison tests regenerate and compare: exact match on
metadata, near-match on vertex count (±1%), bounding box within ε.

**Tests:** `tests/test_fixtures.py`

- Each fixture regeneration matches reference vertex/point count within tolerance
- Each fixture regeneration matches reference bounding box within ε
- Metadata generator_name and seed match exactly
- A deliberately wrong seed produces a mismatch (sanity check on the test itself)

---

## Task 37: Add thorough documentation in README and /docs files

**Objective:**

Produce comprehensive user-facing documentation covering every feature in the
project. The top-level README should be expanded into a complete project
overview with install instructions, quickstart examples, and a feature summary
table. A `docs/` directory should contain detailed per-topic guides covering
generators, the pipeline, CLI commands, configuration, representation
strategies, rendering, grid layouts, and the API. Every CLI command, every
generator category, every config option, and every representation strategy
must be documented with usage examples. A developer or user reading only the
docs should be able to use every feature without reading source code.

**Suggested path:**

Start by auditing every CLI command (`mathviz generate`, `mathviz preview`,
`mathviz render`, `mathviz render-2d`, `mathviz grid`, `mathviz convert`,
`mathviz sample`, `mathviz transform`, `mathviz schema`) and every generator
category (parametric, implicit, attractor, fractal, knot, number theory,
curve). For each, document purpose, parameters, defaults, and at least one
usage example. Structure `docs/` as:

- `docs/generators.md` — all generator categories with parameter tables and examples
- `docs/pipeline.md` — the generate→sample→transform→export pipeline, how stages connect
- `docs/cli.md` — every CLI command with flags, options, and example invocations
- `docs/configuration.md` — config file format, precedence rules, sampling profiles, all config models
- `docs/representation.md` — representation strategies (surface, cloud, volume fill, slice stack, wireframe) with visual descriptions
- `docs/rendering.md` — `render` and `render-2d` commands, optional dependencies, output formats
- `docs/grid.md` — grid manifest format, grid CLI, layout options
- `docs/api.md` — using mathviz as a Python library (importing generators, running pipeline stages programmatically)

Update the top-level README to include: project description, feature list,
install instructions (including optional extras like `[render]`), quickstart
showing a simple generate command, a table of all generators, and links to
each doc page. Remove or replace any placeholder content.

**Tests:** `tests/test_docs.py`

- README.md exists and contains sections: install, quickstart, generators, CLI
- Every file in docs/ is valid markdown (no broken headers, no empty files)
- Every CLI command mentioned in the codebase appears in docs/cli.md
- Every generator registered in the registry appears in docs/generators.md
- Every representation strategy enum value appears in docs/representation.md
- docs/ index or README links to all doc files that exist in the directory

---

## Task 38: Fix export routing to auto-detect geometry type

**Objective:**

Fix the bug where `mathviz generate <name> --output <file>` always fails with
`PointCloudExportError` regardless of the actual geometry produced. The root
cause is that `ExportConfig.export_type` defaults to `"point_cloud"` and is
never set based on the actual geometry present on the `MathObject`. This
affects every generator — even `torus --output torus.ply` fails because the
pipeline produces a mesh (via `SURFACE_SHELL`) but export always routes to
`export_point_cloud`. The preview server (`mathviz preview`) has a related
issue: it returns HTTP 500 for generators that produce mesh-only geometry
(e.g., lorenz with tube thickening) because the client-side viewer doesn't
gracefully handle a null `cloud_url`.

**Suggested path:**

The fix has two parts:

1. **Export auto-detection in `_run_export`** (`runner.py:155-159`): Instead of
   relying solely on `config.export_type`, inspect the `MathObject` to decide
   which exporter to call. If `export_type` is explicitly set by the caller,
   honour it. Otherwise, auto-detect: if `obj.mesh is not None`, export as
   mesh; if `obj.point_cloud is not None`, export as point cloud; if both
   exist, prefer whichever matches the file extension (`.stl`/`.obj` → mesh,
   `.xyz`/`.pcd` → cloud, `.ply` → whichever is present). The
   `ExportConfig.export_type` default should change from `"point_cloud"` to
   `"auto"` (add `"auto"` to `_VALID_EXPORT_TYPES`). The CLI in `cli.py:218`
   already passes no explicit `export_type`, so it will pick up `"auto"`.

2. **Preview server fix** (`server.py`): Ensure the `/api/generate` response
   and the JavaScript viewer handle mesh-only and cloud-only geometry
   correctly. The viewer must not attempt to fetch a geometry URL that is null.
   If only `mesh_url` is present, render the mesh. If only `cloud_url` is
   present, render the cloud.

**Tests:** `tests/test_pipeline/test_export_routing.py`

- `mathviz generate torus --output torus.stl` succeeds (mesh via SURFACE_SHELL)
- `mathviz generate torus --output torus.ply` succeeds (mesh, PLY format)
- `mathviz generate lorenz --output lorenz.ply` succeeds (mesh via tube thickening)
- `ExportConfig(export_type="auto")` with mesh-only MathObject routes to mesh exporter
- `ExportConfig(export_type="auto")` with cloud-only MathObject routes to cloud exporter
- `ExportConfig(export_type="mesh")` still forces mesh export (explicit override)
- Preview server returns 200 for lorenz (mesh-only geometry after tube)
- Preview server returns 200 for generators producing point-cloud-only geometry

---

## Task 39: Fix representation fallback for curve-producing generators

**Objective:**

Fix the bug where `mathviz render kepler_orbit` (and any other generator not
listed in `_GENERATOR_DEFAULTS`) crashes with `ValueError: SURFACE_SHELL
requires a mesh input, but MathObject has no mesh`. The root cause is that
`_FALLBACK_DEFAULT` in `representation_strategy.py` is hardcoded to
`SURFACE_SHELL`, which requires a mesh, but many generators produce only
curves (kepler_orbit, cardioid, fibonacci_spiral, logarithmic_spiral,
parabolic_envelope, etc.) or only point clouds (sacks_spiral, prime_gaps,
ulam_spiral, etc.). The fallback must inspect the actual geometry and choose a
compatible representation.

**Suggested path:**

Replace the single `_FALLBACK_DEFAULT` with a `get_fallback` function that
inspects the `MathObject` and returns an appropriate default:

- If `obj.mesh is not None` → `RepresentationType.SURFACE_SHELL`
- If `obj.curves` is non-empty → `RepresentationType.TUBE` (with a sensible
  default `tube_radius` based on bounding box, e.g. 1% of the diagonal)
- If `obj.point_cloud is not None` → `RepresentationType.SPARSE_SHELL`
- If none of the above → raise a clear error

Update `get_default()` to call this fallback when the generator has no entry
in `_GENERATOR_DEFAULTS`. Add entries to `_GENERATOR_DEFAULTS` for all
currently-registered generators that are missing (kepler_orbit, cardioid,
fibonacci_spiral, logarithmic_spiral, parabolic_envelope, sacks_spiral,
prime_gaps, digit_encoding, nbody, planetary_positions, etc.) so they get
appropriate representation types.

**Tests:** `tests/test_pipeline/test_representation_fallback.py`

- kepler_orbit (curve) gets TUBE representation by default, not SURFACE_SHELL
- cardioid (curve) gets TUBE representation by default
- sacks_spiral (point cloud) gets SPARSE_SHELL representation by default
- torus (mesh) still gets SURFACE_SHELL representation
- A MathObject with only curves and no `_GENERATOR_DEFAULTS` entry gets TUBE fallback
- A MathObject with only point_cloud and no entry gets SPARSE_SHELL fallback
- A MathObject with no geometry raises a clear error from the fallback
- All generators registered in the registry can run through the representation stage without error

---

## Task 40: Fix planetary_positions generator producing single-point curves

**Objective:**

Fix the bug where `mathviz generate planetary_positions` and
`mathviz render planetary_positions` crash with `ValueError: Curve needs >= 2
points, got 1`. The planetary_positions generator produces 8 orbital curves
(one per planet), but at least one orbit produces a `Curve` with only 1 point.
The tube thickening representation requires at least 2 points per curve. This
affects `generate`, `render`, and `preview` commands.

**Suggested path:**

Investigate the `planetary_positions` generator
(`src/mathviz/generators/physics/planetary_positions.py`) to determine which
orbit(s) produce single-point curves and why. Likely causes:

1. A planet's orbital calculation degenerates (e.g., near-circular orbit with
   insufficient angular sampling, or a parameter that produces a single
   position instead of a full orbit).
2. The generator might be producing planet _positions_ (single points at an
   epoch) alongside full orbits, and those single-point "orbits" are returned
   as Curve objects.

The fix should ensure every Curve in the output has at least 2 points. Options:

- If some entries are positions (not orbits), output them as a PointCloud
  instead of as single-point Curves.
- If an orbit calculation produces too few points, increase sampling or filter
  out degenerate curves with a warning.
- Add a minimum-points guard in the generator's `generate()` method so the
  error is caught early with a clear message rather than deep in tube
  thickening.

Additionally, consider adding a defensive check in `_apply_tube` or
`_thicken_all_curves` to skip curves with fewer than 2 points (with a
warning) rather than crashing, since this could affect any generator that
produces short curves.

**Tests:** `tests/test_generators/test_planetary_positions.py`

- `planetary_positions` with default params produces all curves with >= 2 points
- `planetary_positions` generate + tube representation succeeds without error
- `planetary_positions` render produces a PNG without error
- No curve in the output has fewer than 2 points (parameterized across planet count)
- If the fix includes a defensive skip in tube thickening: a single-point curve is skipped with a warning, not a crash

---

## Task 41: Fix preview server static files not included in package distribution

**Objective:**

Fix the bug where `mathviz preview` returns HTTP 500 with
`"Viewer HTML not found."` when mathviz is installed as a non-editable package
(e.g., `pip install mathviz`). The root cause is that `pyproject.toml` only
declares `profiles/*.toml` as package data but omits the `static/` directory
containing `index.html`. When installed non-editable, the `static/` directory
is not copied into `site-packages/mathviz/`, so
`Path(__file__).resolve().parent.parent / "static"` resolves to a path that
doesn't exist.

**Suggested path:**

1. **Add static files to package data** in `pyproject.toml`:

   ```toml
   [tool.setuptools.package-data]
   mathviz = ["profiles/*.toml", "static/*.html", "static/**/*"]
   ```

   This ensures `index.html` (and any future static assets like CSS/JS) are
   included when the package is built and installed.

2. **Use `importlib.resources` for robust path resolution** (optional but
   recommended): Instead of `Path(__file__).resolve().parent.parent / "static"`,
   use `importlib.resources.files("mathviz") / "static"` which correctly
   resolves package data in all install scenarios (editable, non-editable,
   zipped eggs, etc.).

3. **Verify the fix** by doing a non-editable install into a venv and
   confirming `mathviz preview <generator>` serves the viewer HTML at `GET /`.

**Tests:** `tests/test_preview/test_static_files.py`

- `index.html` is accessible via `importlib.resources` from the installed package
- Preview server `GET /` returns 200 with HTML content
- The HTML content contains expected markers (e.g., Three.js script tag or app container div)
- Package data glob in pyproject.toml includes `static/*.html`

---

## Task 42: Add generator switcher with type-ahead search to preview UI

**Objective:**

Add a generator selection UI to the preview viewer so users can browse and
switch between all available generators without restarting the server or
editing the URL. The UI should include a searchable dropdown (type-ahead /
autocomplete) in the top controls panel that lists all registered generators,
grouped or sortable by category. Selecting a generator loads it immediately
in the viewer. This replaces the current workflow of manually editing the
`?generator=` query parameter in the URL.

**Suggested path:**

1. **Generator selector in the controls panel**: Add a combobox/dropdown at
   the top of the existing `#controls` div in `index.html`. It should:
   - Fetch the full generator list from `GET /api/generators` on page load
   - Display generators in a searchable dropdown with type-ahead filtering
     (user types "lor" and sees "lorenz", "lorenz_attractor" etc.)
   - Show the generator category as secondary text or group headers
     (e.g., "attractors", "parametric", "fractals")
   - Pre-select the current generator from the URL query param if present
   - On selection, call `POST /api/generate` with the new generator name
     and reload the 3D scene without a full page refresh

2. **Implementation approach**: Since the viewer is a single HTML file using
   vanilla JS + Three.js from CDN, keep the combobox implementation
   lightweight — no framework dependencies. Options:
   - A custom `<input>` + filtered `<div>` dropdown (straightforward, no deps)
   - An HTML `<datalist>` element for native browser autocomplete (simplest
     but limited styling)
   - The custom approach is preferred for consistent styling with the existing
     dark/light theme controls

3. **URL sync**: When a generator is selected, update the browser URL
   (`history.replaceState`) with the new `?generator=` param so the URL
   remains shareable and bookmarkable.

4. **Seed control**: Add a small seed input field next to the generator
   dropdown so users can regenerate with a different seed without URL editing.
   Include a "randomize" button (dice icon or "Random" text) that picks a
   random seed and regenerates.

5. **Styling**: Match the existing controls panel style — dark translucent
   background, light text, compact layout. The dropdown should overlay the
   3D canvas when open and not push other controls around.

**Tests:** `tests/test_preview/test_generator_switcher.py`

- `GET /api/generators` returns a non-empty list with name and category fields
- Preview `GET /` HTML contains the generator selector input element
- Preview `GET /` HTML contains JavaScript that fetches `/api/generators`
- Selecting a generator triggers a `POST /api/generate` with the correct name
- URL is updated with the selected generator name after switching
- Seed input field is present and changing it triggers regeneration
- Generator list includes all registered generators (compare against registry)
- Type-ahead filtering narrows the visible list correctly

---

## Task 43: Add container/bounding box editor panel to preview UI

**Objective:**

Add a second controls panel to the preview viewer that lets users edit the
container (glass block) dimensions and margins, then re-render the geometry
to fit the new bounding box. Currently the preview server hardcodes
`Container.with_uniform_margin()` (100x100x100mm, 5mm margins). This panel
gives users interactive control over the physical dimensions their geometry
will be fitted into, so they can see how the object scales and fits before
exporting.

**Suggested path:**

1. **New panel in `index.html`**: Add a collapsible panel on the left side
   (opposite the existing controls on the right) or below the existing
   controls. The panel should contain:
   - **Dimensions section**: Three numeric inputs for Width (X), Height (Y),
     and Depth (Z) in mm, defaulting to 100, 100, 100
   - **Margins section**: Three numeric inputs for X, Y, Z margins in mm,
     defaulting to 5, 5, 5. Optionally a "uniform margin" checkbox that
     locks all three to the same value (common case)
   - **Usable volume display**: Read-only text showing the computed usable
     volume after margins (e.g., "Usable: 90 x 90 x 90 mm"), updated live
     as the user types
   - **Apply button**: Triggers re-generation with the new container
     dimensions. Does NOT auto-regenerate on every keystroke (generation can
     be slow for complex generators like mandelbulb)
   - **Reset button**: Restores defaults (100x100x100, 5mm margins)

2. **Server changes** (`server.py`): Extend the `GenerateRequest` model to
   accept optional container parameters:
   ```python
   container: dict[str, float] | None = None  # width_mm, height_mm, depth_mm, margin_x/y/z_mm
   ```
   When present, construct a `Container` from these values instead of using
   the default. Pass validation errors (e.g., margin exceeds half dimension)
   back as 422 with a clear message.

3. **Visual feedback**: When the container dimensions change, update the
   bounding box wireframe in the 3D scene to reflect the new container size.
   The existing `#show-bbox` checkbox already controls a bounding box helper —
   ensure it reflects the container dimensions, not just the geometry bounds.

4. **Styling**: Match the existing panel style (dark translucent, blur
   backdrop). Input fields should be compact — use `<input type="number">`
   with `step`, `min` constraints. Label each field clearly (e.g., "W", "H",
   "D" or "Width (mm)").

**Tests:** `tests/test_preview/test_container_editor.py`

- `POST /api/generate` with custom container dimensions returns 200
- Custom container dimensions change the transformer scale in the result
- Invalid margins (margin >= half dimension) return 422 with error message
- Default container values (100x100x100, margins 5) match current behavior
- Usable volume calculation is correct (dimension - 2*margin per axis)
- Preview HTML contains the container editor panel with dimension inputs
- Apply button triggers a new `POST /api/generate` with container params
- Reset button restores default values in the input fields

---

## Task 44: Add generator parameter editor panel to preview UI

**Objective:**

Add a panel to the preview viewer that shows editable parameter fields for
the currently selected generator. Each generator exposes different parameters
(e.g., Lorenz has sigma, rho, beta, transient_steps; Mandelbulb has power,
max_iterations, extent). The panel should dynamically populate with the
correct fields whenever the generator changes, pre-filled with default values,
and allow the user to edit them and re-render.

**Suggested path:**

1. **New API endpoint** `GET /api/generators/{name}/params`: Return the
   generator's default parameters and their types. Use the existing
   `get_default_params()` method on each generator class, which returns a dict
   like `{"sigma": 10.0, "rho": 28.0, "beta": 2.667, "transient_steps": 1000}`.
   Infer types from the default values (float → number input with decimal
   step, int → number input with integer step, bool → checkbox, str → text).
   Also include `resolution_params` from `GeneratorMeta` (e.g.,
   `{"voxel_resolution": "Voxels per axis (N³ cost)"}`) so the UI can show
   resolution controls too. Response format:
   ```json
   {
     "params": {"sigma": 10.0, "rho": 28.0, "beta": 2.667},
     "resolution": {"integration_steps": 100000},
     "descriptions": {"integration_steps": "Total number of integration time steps"}
   }
   ```

2. **Parameter panel in `index.html`**: Add a collapsible "Parameters" panel
   (similar to the existing controls panel styling). When a generator is
   loaded or switched (via Task 42's selector):
   - Fetch `GET /api/generators/{name}/params`
   - Clear the panel and dynamically create one labeled input per parameter
   - Float params: `<input type="number" step="0.1">` with the default value
   - Int params: `<input type="number" step="1">` with the default value
   - Bool params: `<input type="checkbox">`
   - Show parameter descriptions as tooltips or small helper text
   - Separate resolution params into their own sub-section with a label
     like "Resolution" so the user knows these affect quality/speed

3. **Apply button**: A single "Apply" button at the bottom of the panel that
   collects all current parameter values and calls `POST /api/generate` with
   them in the `params` and `resolution` fields. Do not auto-regenerate on
   every keystroke — some generators take seconds to run. Show a loading
   indicator while generating.

4. **Reset to defaults button**: Restores all fields to the generator's
   default values (re-fetches from the API or uses cached defaults).

5. **Integration with generator switcher (Task 42)**: When the user switches
   generators via the dropdown, the parameter panel should clear and
   repopulate with the new generator's params. The previous generator's
   edited values are discarded.

6. **Validation**: If the server returns a validation error (e.g., "sigma
   must be positive"), display it near the Apply button or inline next to
   the offending field. The existing `POST /api/generate` already propagates
   generator `ValueError` exceptions.

**Tests:** `tests/test_preview/test_param_editor.py`

- `GET /api/generators/lorenz/params` returns sigma, rho, beta, transient_steps with correct defaults
- `GET /api/generators/mandelbulb/params` returns power, max_iterations, extent with correct defaults
- `GET /api/generators/unknown/params` returns 404
- Resolution params are included in the response (e.g., integration_steps for lorenz)
- `POST /api/generate` with custom params produces different geometry than defaults
- `POST /api/generate` with invalid params returns error with descriptive message
- Preview HTML contains the parameter editor panel
- Parameter panel updates when generator selection changes
- Apply button sends params in the POST body
- Reset button restores default values

---

## Task 45: Add resolution controls to preview UI

**Objective:**

Add a "Resolution" section to the preview UI that displays the current
generator's resolution parameters (e.g., `integration_steps`,
`voxel_resolution`, `grid_resolution`, `curve_points`, `pixel_resolution`)
with their default values, and lets the user increase or decrease them to
get higher or lower fidelity renders. The output stats (vertex count, face
count, point count) should also be displayed after each render so the user
can see the effect of resolution changes.

Currently, resolution parameters are hardcoded as module-level constants in
each generator (e.g., `_DEFAULT_INTEGRATION_STEPS = 100_000`,
`_DEFAULT_VOXEL_RESOLUTION = 128`) and passed via `resolution_kwargs` to
`generate()`. The `POST /api/generate` request body already has a `resolution`
field, but the default values are not programmatically exposed.

**Suggested path:**

1. **Add `get_default_resolution()` to the Generator base class**: A new
   method returning a dict of resolution param names to their default values,
   e.g., `{"integration_steps": 100000}` for lorenz or
   `{"voxel_resolution": 128}` for mandelbulb. Each generator subclass
   overrides this. The values should match the existing module-level
   `_DEFAULT_*` constants. This complements the existing `resolution_params`
   dict on `GeneratorMeta` which has descriptions but not default values.

2. **Expose via API**: The `GET /api/generators/{name}/params` endpoint
   (from Task 44) should include resolution defaults in its response:
   ```json
   {
     "params": {"sigma": 10.0, ...},
     "resolution": {"integration_steps": {"default": 100000, "description": "Total number of integration time steps"}},
   }
   ```
   If Task 44 isn't implemented yet, add a standalone
   `GET /api/generators/{name}/resolution` endpoint instead.

3. **Resolution section in the UI**: Add a distinct "Resolution" section in
   the parameter/controls area (visually separated from mathematical params).
   For each resolution parameter:
   - Show the parameter name and description
   - A numeric input pre-filled with the default value
   - Show a cost warning where applicable (descriptions already include
     hints like "N² cost" or "N³ cost")
   - Reasonable input constraints: `min="1"`, no upper limit but warn the
     user visually if they pick a very high value (e.g., voxel_resolution
     > 256 or integration_steps > 500000)

4. **Output stats display**: After each render completes, update the info
   panel to show the resulting vertex count, face count, and point count.
   The existing `#info-panel` already has `#info-vertices`, `#info-faces`,
   and `#info-points` elements — ensure these are populated from the
   `MathObject` stats after generation. Add a "Triangles" or "Faces" count
   if not already shown.

5. **Interaction**: The resolution fields are sent in the `resolution` field
   of `POST /api/generate` (this field already exists in `GenerateRequest`).
   Changing resolution and clicking Apply (shared with Task 44's Apply
   button, or a separate "Re-render" button) triggers regeneration. Defaults
   stay unchanged — the inputs just let the user override them.

**Tests:** `tests/test_preview/test_resolution_controls.py`

- `get_default_resolution()` returns correct defaults for lorenz (integration_steps=100000)
- `get_default_resolution()` returns correct defaults for mandelbulb (voxel_resolution=128)
- `get_default_resolution()` returns correct defaults for torus (grid_resolution=128)
- Every generator with `resolution_params` also returns values from `get_default_resolution()`
- API endpoint returns resolution defaults with descriptions
- `POST /api/generate` with higher resolution produces more vertices/points than default
- `POST /api/generate` with lower resolution produces fewer vertices/points
- `POST /api/generate` with no resolution field uses defaults (unchanged behavior)
- Preview HTML contains resolution input fields
- Info panel displays vertex/face/point counts after generation

---

## Task 46: Default preview UI view mode to Point Cloud

**Objective:**

Change the default view mode in the preview viewer from "Shaded Mesh" to
"Point Cloud". Currently the `#view-mode` select element in `index.html`
defaults to `"shaded"` (the first `<option>`). The default should be
`"points"` instead, so the viewer starts in point cloud mode. The user can
still switch to "Shaded Mesh" or "Wireframe" via the dropdown.

**Suggested path:**

In `src/mathviz/static/index.html`, move the `selected` attribute (or
reorder the options) so `"points"` is the default:

```html
<select id="view-mode">
  <option value="shaded">Shaded Mesh</option>
  <option value="wireframe">Wireframe</option>
  <option value="points" selected>Point Cloud</option>
</select>
```

Also update the JavaScript `state` object to match:

```js
const state = {
  ...
  viewMode: 'points',
  ...
};
```

Ensure the initial render applies point cloud mode (not just setting the
dropdown value — the scene objects' visibility must also default correctly).

**Tests:** `tests/test_preview/test_view_mode_default.py`

- Preview HTML has `"points"` as the default selected view mode option
- JavaScript state initializes viewMode to `"points"`
- Initial scene render shows point cloud, not shaded mesh

---

## Task 47: Add render style option to `mathviz render` CLI

**Objective:**

Add a `--style` flag to `mathviz render` that controls the rendering style:
`shaded` (current default behavior), `wireframe`, or `points` (point cloud).
The default should be `points` to match the preview UI default (Task 46).
This lets users generate point cloud PNGs from the command line, matching
what they see in the web preview.

Currently `renderer.py` always renders with `plotter.add_mesh()` and
`smooth_shading=True`. There is no way to render as a point cloud or
wireframe from the CLI.

**Suggested path:**

1. **Add `style` to `RenderConfig`**: Add a `style` field with type
   `Literal["shaded", "wireframe", "points"]` defaulting to `"points"`.

2. **Update `_setup_backlit_scene`** in `renderer.py` to use the style:
   - `"shaded"`: current behavior — `plotter.add_mesh(pv_mesh, smooth_shading=True, ...)`
   - `"wireframe"`: `plotter.add_mesh(pv_mesh, style='wireframe', ...)`
   - `"points"`: `plotter.add_mesh(pv_mesh, style='points', point_size=..., render_points_as_spheres=True, ...)`

3. **Add `--style` flag to CLI** in `cli_render.py`: Pass it through to
   `RenderConfig`. The flag should accept `shaded`, `wireframe`, or `points`.

4. **Point size**: For point cloud style, add a `--point-size` flag
   (default 2.0 or 3.0) that controls `point_size` in PyVista. Larger
   values make individual points more visible at high resolution.

**Tests:** `tests/test_preview/test_renderer.py` (extend existing)

- `mathviz render lorenz -o out.png --style points` produces a non-trivial PNG
- `mathviz render torus -o out.png --style wireframe` produces a non-trivial PNG
- `mathviz render torus -o out.png --style shaded` produces a non-trivial PNG (current behavior)
- Default style (no flag) renders as points
- `--point-size 5` with `--style points` produces visually different output than default
- Invalid style value is rejected with a clear error

---

## Task 48: Fix bounding box and geometry in different coordinate spaces in preview

**Objective:**

Fix the bug where the bounding box wireframe and the 3D geometry are rendered
in completely different positions in the preview viewer. The bounding box
appears at the origin while the geometry is far away, making the bounding box
useless for visualizing fit.

**Root cause:** The `addBoundingBox()` function in `index.html` draws the
container box centered at the origin (±half-dimensions scaled by 0.01). But
the geometry from the pipeline is in mm space — the transformer places
vertices at coordinates like (5, 5, 7) to (95, 95, 95) for a 100x100x100mm
container. The mesh is centered around (50, 50, 20), not the origin. The
bounding box and geometry are in completely different coordinate spaces with
different origins and different scales.

**Suggested path:**

There are two clean approaches — pick one:

**Option A — Compute bounding box from actual geometry (simplest):**
Replace the container-based bounding box with one computed from the actual
mesh geometry. Change `addBoundingBox(object3d)` to use
`new THREE.Box3().setFromObject(object3d)` instead of hardcoded container
dimensions. This always matches the geometry regardless of coordinate space.
The container dimensions would only be shown as text info, not as a 3D box.

**Option B — Align coordinate spaces (more correct):**
Ensure both the bounding box and geometry share the same coordinate space.
Either:
- Re-center the geometry to the origin in the GLB serialization step
  (`mesh_to_glb` in `lod.py`), translating vertices so the mesh center is
  at (0,0,0). Then the container box at ±half-dimensions will align.
- Or move the bounding box to match the geometry's actual position: compute
  the mesh center and offset the box to match.

Option B is preferred because it preserves the container visualization, which
becomes important for Task 43 (container editor). The re-centering should
happen in the serialization layer (`mesh_to_glb`) so the Three.js scene
always receives geometry centered at the origin. The bounding box then
correctly represents the container the geometry fits within.

If re-centering in `mesh_to_glb`, also re-center in `cloud_to_binary_ply`
so point cloud data is consistent. Store the translation offset in the API
response if needed for round-tripping back to absolute coordinates.

Additionally, **normalize coordinates to unit scale** during serialization.
Currently the geometry is in raw mm (vertices at 5-95 for a 100mm container).
The Three.js camera is set up with `near=0.01, far=100` (line 199 of
`index.html`). When the container is large (e.g., 100x100x100mm), `fitCamera`
places the camera ~162 units from the center — well beyond the far clipping
plane of 100. **This causes the entire scene to be invisible** (clipped).

The fix should either:
- **Scale geometry to unit space** in the serialization layer (divide by
  container max dimension so the scene fits in roughly ±1 units), OR
- **Update `fitCamera` to adjust the camera's near/far planes** dynamically
  based on the geometry size: `camera.near = dist * 0.01;
  camera.far = dist * 10; camera.updateProjectionMatrix();`

The dynamic clipping plane approach is simpler and handles any geometry size.
The normalization approach is cleaner but requires the bounding box to also
be scaled to match.

**Tests:** `tests/test_preview/test_bounding_box.py`

- Bounding box and geometry overlap visually (box contains the geometry)
- Bounding box center matches geometry center (within tolerance)
- Geometry is centered at approximately the origin after serialization
- Point cloud data is also centered at the origin
- Bounding box dimensions match the container dimensions
- Toggling the bounding box checkbox shows/hides the box (existing behavior preserved)
- `fitCamera` centers the view on the geometry (not on the origin if different)
- Large container (100x100x100mm) renders correctly without clipping
- Small container (10x10x10mm) renders correctly without near-plane clipping
- Camera far plane is sufficient to contain the geometry at any container size

---

## Task 49: Add "Reset View" button to preview UI

**Objective:**

Add a button to the preview controls panel that resets the camera to the
default view — centered on the geometry, zoomed to fit the entire object in
frame, with the default viewing angle. After orbiting, zooming, or panning,
the user should be able to click this button to snap back to the initial
overview position.

**Suggested path:**

1. **Add a "Reset View" button** in the `#controls` div in `index.html`,
   next to the existing "Screenshot" button. Label it "Reset View" or use a
   home/reset icon.

2. **On click, call `fitCamera`** on the current geometry (whichever of
   `state.meshGroup` or `state.cloudPoints` is active). `fitCamera` already
   computes the correct camera position from the geometry's bounding box and
   sets the orbit controls target. Also reset the OrbitControls:
   ```js
   controls.reset();  // resets zoom/pan to initial state
   fitCamera(activeObject);  // reposition based on geometry bounds
   ```

3. **Also reset the orbit controls' internal state**: `OrbitControls.reset()`
   resets to the controls' saved state, but since `fitCamera` modifies
   `controls.target` and `camera.position` after construction, the saved
   state may be stale. Either call `controls.saveState()` after each
   `fitCamera` so `reset()` returns to the right place, or just call
   `fitCamera` directly without `controls.reset()`.

4. **Keyboard shortcut**: Optionally bind the `Home` key or `0` key to
   trigger the same reset for quick access.

**Tests:** `tests/test_preview/test_reset_view.py`

- Preview HTML contains a "Reset View" button
- Clicking the button calls fitCamera on the active geometry
- After orbiting/zooming, reset returns camera to the default overview position
- Button works when viewing mesh geometry
- Button works when viewing point cloud geometry
- Button is disabled or hidden when no geometry is loaded

---

## Task 50: Add `--view` flag to `mathviz render` with comprehensive camera angles

**Objective:**

Add a `--view` flag to the `mathviz render` command (3D perspective renderer)
that lets the user choose a camera angle. Currently `render` has a single
hardcoded camera position. Also expand the set of named views for both
`render` and `render-2d` to cover all face-on, edge-on, and vertex views
of the bounding box.

Currently available views (only in `render-2d`):
- `top`, `front`, `side`, `angle`

Missing views:
- The opposing face-on views: `bottom`, `back`, `right`
- Edge-on views: looking at where two faces meet (e.g., front-right edge)
- Vertex/corner views: looking at where three faces meet from all 8 corners

**Suggested path:**

1. **Expand the named views** in `renderer.py`. Add camera positions for a
   complete set:

   **6 face-on views** (looking straight at one face):
   - `front` — (0, -1, 0), `back` — (0, 1, 0)
   - `left` — (-1, 0, 0), `right`/`side` — (1, 0, 0)
   - `top` — (0, 0, 1), `bottom` — (0, 0, -1)

   **12 edge-on views** (looking at an edge where two faces meet):
   - `front-right` — (1, -1, 0), `front-left` — (-1, -1, 0)
   - `back-right` — (1, 1, 0), `back-left` — (-1, 1, 0)
   - `front-top` — (0, -1, 1), `front-bottom` — (0, -1, -1)
   - `back-top` — (0, 1, 1), `back-bottom` — (0, 1, -1)
   - `top-right` — (1, 0, 1), `top-left` — (-1, 0, 1)
   - `bottom-right` — (1, 0, -1), `bottom-left` — (-1, 0, -1)

   **8 vertex/corner views** (looking at a corner where three faces meet):
   - `front-right-top` — (1, -1, 1) (this is the current `angle`)
   - `front-left-top` — (-1, -1, 1)
   - `back-right-top` — (1, 1, 1)
   - `back-left-top` — (-1, 1, 1)
   - `front-right-bottom` — (1, -1, -1)
   - `front-left-bottom` — (-1, -1, -1)
   - `back-right-bottom` — (1, 1, -1)
   - `back-left-bottom` — (-1, 1, -1)

   Keep `angle` as an alias for `front-right-top` for backwards compatibility.

2. **Add `--view` flag to `mathviz render`** in `cli_render.py`. Default to
   `front-right-top` (current approximate camera position). Unlike
   `render-2d`, this uses perspective projection — the camera is positioned
   on a sphere around the geometry center at the angle specified by the view
   name, at a distance computed from the bounding box size.

3. **Add `--view all` option** that renders all 26 views (or a curated
   subset like the 6 faces + 8 vertices = 14 views) and saves them as
   separate files with the view name appended (e.g.,
   `torus_front.png`, `torus_top.png`, `torus_front-right-top.png`).

4. **Update `render-2d`** to accept the same expanded view names so both
   commands share the same vocabulary.

**Tests:** `tests/test_preview/test_render_views.py`

- `mathviz render torus -o out.png --view front` produces a non-trivial PNG
- `mathviz render torus -o out.png --view top` produces a different image than `--view front`
- `mathviz render torus -o out.png --view front-right-top` matches `--view angle`
- `mathviz render-2d torus -o out.png --view back` produces a non-trivial PNG
- `mathviz render-2d torus -o out.png --view right` works (new view name)
- All 6 face-on views produce distinct images
- Edge-on view `front-right` produces an image distinct from `front` and `right`
- `--view all` produces multiple files with correct naming
- Default view (no flag) renders from `front-right-top` perspective
- Invalid view name is rejected with a clear error listing available views

---

## Task 51: Add resolution and parameter editor UI elements to preview HTML

**Objective:**

Task 45 added the backend for resolution controls (`get_default_resolution()`,
API endpoint, `POST /api/generate` resolution field) but did not add the
frontend UI elements to `index.html`. Task 44 specified a parameter editor
panel but it also has not been implemented in the HTML. Add the missing HTML,
CSS, and JavaScript to make both the parameter editor and resolution controls
visible and functional in the preview viewer.

**Suggested path:**

1. **Parameter editor panel**: Add a collapsible "Parameters" section to
   `index.html` that:
   - On page load (or generator switch), fetches
     `GET /api/generators/{name}/params` to get parameter defaults and types
   - Dynamically creates one labeled `<input>` per parameter:
     float → `<input type="number" step="0.1">`,
     int → `<input type="number" step="1">`,
     bool → `<input type="checkbox">`
   - Shows parameter descriptions as tooltips or helper text
   - Pre-fills inputs with default values

2. **Resolution controls section**: Add a visually separated "Resolution"
   sub-section within or below the parameter panel:
   - Fetches resolution defaults from the same
     `GET /api/generators/{name}/params` endpoint (which returns both
     `params` and `resolution` fields per Task 45)
   - Creates numeric inputs for each resolution param (e.g.,
     `integration_steps`, `voxel_resolution`, `grid_resolution`)
   - Shows the description text (e.g., "Voxels per axis (N³ cost)")
   - Warns visually for high values (e.g., voxel_resolution > 256)

3. **Apply / Reset buttons**: A single "Apply" button at the bottom that
   collects all parameter and resolution values and calls
   `POST /api/generate` with both `params` and `resolution` fields. A
   "Reset" button restores all fields to defaults. Show a loading indicator
   during generation.

4. **Generator switch integration**: When the generator changes (Task 42's
   switcher), clear and repopulate both panels with the new generator's
   params and resolution defaults.

5. **Geometry info display**: After each generation completes, update the
   info panel (`#info-vertices`, `#info-faces`, `#info-points`) with the
   actual counts from the generated geometry so users can see the effect
   of resolution changes.

6. **Styling**: Match existing dark/light theme. Panels should be compact,
   scrollable if too many parameters, and not obscure the 3D canvas.

**Tests:** `tests/test_preview/test_param_ui.py`

- Preview HTML contains a parameter editor section with input elements
- Preview HTML contains a resolution editor section
- Fetching `/api/generators/lorenz/params` returns params and resolution fields
- Parameter inputs are dynamically created based on generator defaults
- Apply button sends both params and resolution in POST body
- Reset button restores default values
- Switching generators clears and repopulates the parameter panel
- Info panel updates with vertex/face/point counts after generation

---

## Task 52: Add demo/placeholder mode for data-driven generators

**Objective:**

The three data-driven generators (`building_extrude`, `heightmap`,
`soundwave`) crash with `ValueError: input_file parameter is required` when
run without an `input_file` parameter. This makes them unusable via
`mathviz render`, `mathviz render-2d`, `mathviz preview`, or any other
pipeline entry point unless the user supplies an external file.

Each generator should synthesize a built-in demo when `input_file` is not
provided, so they work out of the box like every other generator.

**Suggested path:**

1. **`soundwave`**: When `input_file` is omitted, synthesize a short WAV-like
   signal procedurally (e.g., a 2-second sine wave at 440 Hz with amplitude
   envelope). Use `numpy` to generate the samples directly — no file I/O
   needed. Log `INFO: No input_file provided, using built-in demo waveform`.

2. **`heightmap`**: When `input_file` is omitted, generate a procedural
   grayscale field (e.g., a radial gradient, concentric rings, or a simple
   Perlin-like pattern using numpy). Produce a 2D float array as if it were
   loaded from an image. Log `INFO: No input_file provided, using built-in
   demo heightmap`.

3. **`building_extrude`**: When `input_file` is omitted, construct a small
   in-memory GeoJSON FeatureCollection with a handful of rectangular
   polygons at varying heights (simulating a city block). Pass the dict
   directly to `_extract_polygons` instead of loading from disk. Log
   `INFO: No input_file provided, using built-in demo buildings`.

4. Each generator's `generate()` method should check `input_file` early and
   branch to the demo path before calling `validate_input_file`. The demo
   data should be deterministic (seeded by the `seed` parameter) so renders
   are reproducible.

5. Do not add new dependencies — use only `numpy` and existing stdlib.

**Tests:** `tests/test_generators/test_data_driven_demo.py`

- `soundwave` generates successfully with no `input_file` parameter
- `heightmap` generates successfully with no `input_file` parameter
- `building_extrude` generates successfully with no `input_file` parameter
- Each demo generator produces a valid MathObject with non-empty geometry
- Demo output is deterministic: same seed produces identical geometry
- Providing an `input_file` still works as before (no regression)
- `mathviz render-2d <generator> -o out.png` succeeds for all three

---

## Task 53: Fix view mode override in `displayGenerateResult` (Task 46 regression)

**Objective:**

Task 46 (PR #46) set the default `state.viewMode` to `'points'` and added
`selected` to the Point Cloud dropdown option. However, `displayGenerateResult()`
unconditionally overrides the view mode after every generation. For mesh-only
generators (e.g., `schwarz_d`), the logic at line ~459:

```js
if (!hasMesh || hasCloud) {
  state.viewMode = 'points';        // only when no mesh, or cloud present
} else {
  state.viewMode = 'shaded';        // ← forces shaded for mesh-only generators
}
```

This means the default of `'points'` is immediately overridden to `'shaded'`
for any generator that only produces a mesh (no separate point cloud). The
user sees "Shaded Mesh" as the active view despite the dropdown defaulting to
Point Cloud.

**Suggested path:**

1. In `displayGenerateResult()`, only force a view mode switch when the
   current `state.viewMode` is incompatible with the available data — e.g.,
   if `viewMode` is `'shaded'` or `'wireframe'` but there's no mesh, switch
   to `'points'`. If the current mode can be rendered (mesh-based views work
   when `hasMesh` is true; `'points'` always works since mesh vertices can
   be rendered as points), leave `state.viewMode` unchanged.

2. Remove the unconditional `state.viewMode = 'shaded'` fallback in the
   `else` branch — the default from state initialization should persist.

3. Same fix in `loadFromFile()` — only override when incompatible, not
   unconditionally.

4. Verify the dropdown `<select>` value stays in sync with `state.viewMode`
   after generation completes.

**Tests:** `tests/test_preview/test_view_mode_default.py` (extend existing)

- After generating a mesh-only object (e.g., torus), view mode remains `'points'`
- After generating a cloud-only object, view mode is `'points'`
- After generating an object with both mesh and cloud, view mode is `'points'`
- Dropdown `<select>` value matches `state.viewMode` after generation
- User-selected view mode (e.g., switching to wireframe) is preserved across regeneration if compatible
- Switching generators (e.g., from schwarz_d to lorenz) preserves the current view mode if compatible

---

## Task 54: Enter key triggers regeneration in preview UI input fields

**Objective:**

When editing numeric input fields in the preview UI (container dimensions,
seed, point size, generator parameters, resolution settings), pressing
Enter should trigger the same action as clicking the Apply/Generate button.
Currently the user must click the button after typing values, which is
cumbersome.

**Suggested path:**

1. Add a `keydown` event listener to all numeric `<input>` fields in the
   controls panel (seed, container dimensions, point size, and any future
   parameter/resolution inputs from Tasks 43/51). On `Enter` keypress,
   call the same generation function that the Apply/Generate button uses.

2. Use event delegation on the controls container rather than attaching
   individual listeners to each input — this automatically covers
   dynamically-added inputs from Tasks 43, 44, and 51.

3. For the seed input specifically, Enter should regenerate with the new
   seed value (same as clicking Generate).

4. Do not trigger on other keys — only Enter (keyCode 13 / key === 'Enter').

5. Blur the input after triggering so the user sees the result immediately
   without the cursor staying in the field.

**Tests:** `tests/test_preview/test_enter_key.py`

- Preview HTML includes keydown event handling on input fields
- Pressing Enter in seed input triggers generation (same as clicking Generate)
- Pressing Enter in a container dimension input triggers regeneration
- Event delegation covers dynamically-added parameter inputs
- Non-Enter keys do not trigger regeneration

---

## Task 55: Lock Camera toggle in preview UI

**Objective:**

Add a "Lock Camera" checkbox to the preview controls panel. When enabled,
regenerating geometry (changing parameters, seed, generator, etc.) does not
call `fitCamera()` or reset the OrbitControls. The camera stays at the
user's current position, rotation, and zoom so they can compare the effect
of different settings from the exact same viewpoint.

**Suggested path:**

1. Add a `<label><input type="checkbox" id="lock-camera"> Lock Camera</label>`
   to the Options section of the controls panel, near the existing
   bounding box and background toggles.

2. Add `cameraLocked: false` to the `state` object.

3. Wire the checkbox to toggle `state.cameraLocked`.

4. In `displayMesh()` and `displayCloud()`, gate the `fitCamera()` and
   `addBoundingBox()` calls behind `!state.cameraLocked`. When locked,
   skip camera repositioning — just replace the geometry in place.

5. When locked, also preserve the OrbitControls target (the look-at point).
   Store `controls.target.clone()` before clearing the scene and restore
   it after adding new geometry.

6. The Reset View button (Task 49) should still work even when camera is
   locked — it explicitly overrides the lock to reframe.

**Tests:** `tests/test_preview/test_lock_camera.py`

- Preview HTML contains a "Lock Camera" checkbox with id `lock-camera`
- JS state object includes `cameraLocked` initialized to `false`
- `fitCamera` is not called during regeneration when `cameraLocked` is true
- Reset View button works regardless of lock state
- Toggling the checkbox updates `state.cameraLocked`

---

## Task 56: Auto-apply toggle and controls panel layout reorganization

**Objective:**

Two changes to the preview UI controls:

1. **Auto-Apply toggle**: Add an "Auto-Apply" checkbox next to parameter
   inputs. When enabled, changing any parameter value via the up/down
   spinner buttons (or arrow keys) immediately triggers regeneration
   without needing to click Apply. This is useful for fast generators
   where the user wants to interactively explore parameter space by
   stepping values up and down.

2. **Panel layout**: Move the parameter/resolution editor panel (from
   Tasks 44/51) to sit below the container dimensions and margins section
   (from Task 43), instead of to the right of it. The current right-side
   placement overlaps with the 3D visualization. The controls panel should
   flow vertically: generator selector → container dimensions/margins →
   parameters/resolution → view mode/options.

**Suggested path:**

1. Add `<label><input type="checkbox" id="auto-apply"> Auto-Apply</label>`
   near the parameter editor section heading.

2. Add `autoApply: false` to the `state` object.

3. Listen for `input` events (not just `change`) on numeric parameter
   inputs. The `input` event fires on every spinner click and arrow key
   press. When `state.autoApply` is true, debounce and trigger
   regeneration (e.g., 300ms debounce to avoid flooding during rapid
   clicks).

4. The debounce should reset on each new input event so only the final
   value triggers a render.

5. For the layout change: restructure the HTML so the parameter/resolution
   panel is a vertical section within the existing right-side controls
   column, placed after the container editor section. Remove any
   side-by-side or separate column positioning.

6. Ensure the controls panel is scrollable if the combined sections exceed
   the viewport height.

**Tests:** `tests/test_preview/test_auto_apply.py`

- Preview HTML contains an "Auto-Apply" checkbox with id `auto-apply`
- JS state object includes `autoApply` initialized to `false`
- When auto-apply is enabled, changing an input value triggers regeneration
- Regeneration is debounced (not fired on every keystroke immediately)
- When auto-apply is disabled, changing inputs does not trigger regeneration
- Parameter section appears below container dimensions in DOM order
- Controls panel is scrollable when content overflows

---

## Task 57: Save configuration and geometry snapshot from preview UI

**Objective:**

Add a "Save" button in the bottom-right of the preview UI that persists
the current session state — generator name, all parameters, container
dimensions, seed, and the generated geometry file — to a local directory.
This lets users bookmark interesting configurations for later comparison
or re-use.

**Suggested path:**

1. Add a `POST /api/snapshots` endpoint to the preview server. The request
   body includes: `generator`, `params`, `seed`, `container`, and the
   current `geometry_id`. The server saves:
   - A JSON metadata file: `{generator, params, seed, container, created_at, geometry_id}`
   - A copy of the generated geometry (mesh and/or point cloud files)

2. Snapshots are stored under `~/.mathviz/snapshots/<snapshot_id>/` where
   `snapshot_id` is a short timestamp-based ID (e.g., `20260315-143022`).
   Each directory contains `metadata.json` plus the geometry files
   (`mesh.glb`, `cloud.ply`, or both).

3. Also render a thumbnail PNG (small resolution, e.g., 256x256) using the
   existing PyVista renderer and save it as `thumbnail.png` in the snapshot
   directory. If PyVista is not available, skip the thumbnail gracefully.

4. Add a floating "Save" button in the bottom-right corner of the preview
   UI. On click, POST to `/api/snapshots` with the current state. Show a
   brief confirmation toast (e.g., "Saved!" for 2 seconds). Disable the
   button during save to prevent double-clicks.

5. The snapshot directory path should be configurable via
   `MATHVIZ_SNAPSHOTS_DIR` env var, defaulting to `~/.mathviz/snapshots`.

**Tests:** `tests/test_preview/test_snapshots_save.py`

- `POST /api/snapshots` creates a snapshot directory with metadata.json
- metadata.json contains generator, params, seed, container, created_at
- Geometry files (mesh.glb and/or cloud.ply) are copied to snapshot dir
- Snapshot ID is timestamp-based and unique
- Preview HTML contains a Save button
- Saving without a generated geometry returns 400
- Snapshot directory is configurable via env var

---

## Task 58: Load and browse saved snapshots in preview UI

**Objective:**

Add a "Load" panel to the preview UI that displays previously saved
snapshots (from Task 57) as a browsable gallery. Users can see what they
explored before and restore any snapshot to the current session.

**Suggested path:**

1. Add `GET /api/snapshots` endpoint that returns a list of all saved
   snapshots, sorted by date (newest first). Each entry includes:
   `snapshot_id`, `generator`, `params`, `seed`, `container`, `created_at`,
   `has_thumbnail`, and `thumbnail_url` (if available).

2. Add `GET /api/snapshots/<snapshot_id>/thumbnail` to serve the thumbnail
   PNG. Return 404 if no thumbnail exists.

3. Add `GET /api/snapshots/<snapshot_id>/geometry/<filename>` to serve
   saved geometry files for restoring.

4. Add `DELETE /api/snapshots/<snapshot_id>` to delete a snapshot.

5. In the preview UI, add a "Load" button (next to Save) that opens a
   modal/slide-out panel showing the snapshot gallery. Each card shows:
   - Thumbnail image (or a placeholder icon if none)
   - Generator name
   - Date/time saved
   - Key parameters (compact summary, e.g., "sigma=10, rho=28")
   - "Load" button to restore the snapshot
   - "Delete" button (with confirmation) to remove it

6. Loading a snapshot: set the generator selector, populate parameter
   fields, set seed/container values, then load the saved geometry files
   directly (no regeneration needed — serve from the snapshot directory).

7. The gallery should be paginated or scrollable if there are many
   snapshots.

**Tests:** `tests/test_preview/test_snapshots_browse.py`

- `GET /api/snapshots` returns a list of saved snapshots sorted by date
- Each snapshot entry contains generator, params, created_at, has_thumbnail
- `GET /api/snapshots/<id>/thumbnail` serves the thumbnail PNG
- `GET /api/snapshots/<id>/thumbnail` returns 404 when no thumbnail exists
- `DELETE /api/snapshots/<id>` removes the snapshot directory
- Loading a snapshot restores generator, params, seed, and container values
- Loading a snapshot displays the saved geometry without regeneration
- Preview HTML contains a Load button that opens the snapshot gallery

---

## Task 59: Change default container to 100×100×100 cube

**Objective:**

Change the default container dimensions from 100×100×40 mm to
100×100×100 mm (a cube). Update the `Container` dataclass default for
`depth_mm` from `40.0` to `100.0`, and update all documentation,
docstrings, tests, comments, CLI help text, and config examples that
reference the old default dimensions.

**Suggested path:**

1. In `src/mathviz/core/container.py`, change `depth_mm` default from
   `40.0` to `100.0`.

2. In the preview UI (`index.html`), update the default value of the
   depth input field from `40` to `100`.

3. Search the entire codebase for references to the old defaults:
   - `40.0` or `40` in context of depth/container dimensions
   - `100x100x40`, `100×100×40`, or similar dimension strings
   - Any docstrings, comments, or help text mentioning "40mm depth"

4. Update all documentation files (`docs/`, `README.md`) that mention
   container dimensions to reflect 100×100×100.

5. Update any test fixtures or assertions that hardcode the old 40mm depth.

6. Update any sampling profile TOML files or config examples that
   reference the old dimensions.

**Tests:** `tests/test_core/test_container.py` (extend existing)

- `Container()` with no arguments produces 100×100×100 dimensions
- `Container().usable_volume()` returns (90, 90, 90) with default 5mm margins
- No references to the old 40mm depth default remain in docs or comments
- CLI `mathviz info` or help text shows 100×100×100 as default

---

## Task 60: Pipeline benchmark suite with per-stage timing

**Objective:**

Create a `mathviz benchmark` CLI command that runs every generator through
the full pipeline, measures wall-clock time for each stage (generate,
represent, transform, validate, serialize), and outputs results as an HTML
report with sortable tables. The benchmark runs generators in parallel for
speed and produces a clear picture of where time is spent.

**Suggested path:**

1. Add a `mathviz benchmark` CLI command in a new `cli_benchmark.py` module.
   Options:
   - `--generators` (optional): comma-separated list, defaults to all
     non-data-driven generators
   - `--workers N`: number of parallel workers (default: `os.cpu_count()`)
   - `--output PATH`: output HTML file path (default: `benchmark_report.html`)
   - `--runs N`: number of runs per generator for averaging (default: 3)

2. The pipeline runner (`runner.py`) already has a `timer` context manager
   with `timer.stage()` calls. Expose the per-stage durations from the
   `PipelineResult` (or `PipelineTimer`). If not already returned, add
   a `stage_timings: dict[str, float]` field to the result.

3. Use `concurrent.futures.ProcessPoolExecutor` to run generators in
   parallel. Each worker runs a single generator through the full pipeline
   N times and returns the timing data. Collect results from all workers.

4. Generate an HTML report containing:
   - **Summary table**: one row per generator, columns for each pipeline
     stage (generate, represent, transform, validate, serialize, total).
     Show mean time across runs. Sortable by clicking column headers
     (use inline JS).
   - **Color coding**: cells green (<100ms), yellow (100ms–1s), red (>1s)
   - **Bar chart**: horizontal bars showing relative time per stage for
     each generator (pure CSS, no external dependencies)
   - **System info header**: CPU, Python version, date, number of workers
   - **Fastest/slowest generators** highlighted

5. Also print a compact text summary table to stdout (using `rich` or
   plain formatting) so results are visible without opening the HTML file.

6. Skip data-driven generators that require `input_file` (unless they
   have demo mode from Task 52). Handle generator errors gracefully —
   record the error in the report rather than crashing the whole suite.

**Tests:** `tests/test_cli/test_benchmark.py`

- `mathviz benchmark` runs without error on at least 3 generators
- Output HTML file is created and contains a `<table>` element
- Each generator row has timing columns for all pipeline stages
- `--generators lorenz,torus` limits the benchmark to specified generators
- `--runs 1` produces results with a single run per generator
- Failed generators appear in the report with error messages, not crashes
- Per-stage timings sum approximately to total time (within 10% tolerance)
- Text summary is printed to stdout

---

## Task 61: Batch render CLI command for all generators

**Objective:**

Add a `mathviz render-all` CLI command that renders every generator across
multiple views in parallel and organizes outputs into a structured directory.
This replaces the ad-hoc shell one-liner with a proper, repeatable command.

**Suggested path:**

1. Add a `mathviz render-all` command in `cli_render.py` (or a new
   `cli_render_batch.py` if the file would exceed 300 lines). Options:
   - `--output-dir PATH`: base output directory (default: `renders/`)
   - `--views`: comma-separated views to render (default: `top,front,side,angle`)
   - `--workers N`: parallel workers (default: `os.cpu_count()`)
   - `--generators`: comma-separated list to limit scope (default: all
     non-data-driven generators, plus data-driven if demo mode is available)
   - `--style`: render style — `shaded`, `wireframe`, `points`
     (default: `points`)
   - `--width` / `--height`: image dimensions (default: 1920×1080)

2. Output directory structure:
   ```
   renders/
     lorenz/
       lorenz_top.png
       lorenz_front.png
       lorenz_side.png
       lorenz_angle.png
     torus/
       torus_top.png
       ...
   ```
   Each generator gets its own subdirectory so renders from different
   runs or styles don't collide.

3. Use `concurrent.futures.ProcessPoolExecutor` to parallelize across
   generator+view combinations. Show a progress indicator (e.g., rich
   progress bar or simple counter: `[42/180] Rendering lorenz angle...`).

4. At the end, print a summary: total renders, successes, failures, total
   time. List any failed generator+view combinations with their error
   messages.

5. Skip generators that error out and continue with the rest. Do not abort
   the entire batch on a single failure.

6. If `--output-dir` already exists, render into it without clearing
   existing files (overwrite only matching filenames).

**Tests:** `tests/test_cli/test_render_all.py`

- `mathviz render-all --generators lorenz,torus --views top` creates
  `renders/lorenz/lorenz_top.png` and `renders/torus/torus_top.png`
- Output directory structure has one subdirectory per generator
- `--workers 1` runs sequentially without error
- Failed generators are reported in the summary, not raised as exceptions
- `--output-dir custom_dir/` creates renders in the specified directory
- Default views produce 4 images per generator

---

## Task 62: Multi-panel comparison view in preview UI

**Objective:**

Add a comparison mode to the preview UI that renders 2×2 or 3×3 viewports
of the same generator with different parameters/seeds side by side. All
viewports share a single camera — rotating one rotates all. Each viewport
has a compact, collapsible parameter overlay so the user can tweak
individual panels without taking up much screen space.

**Suggested path:**

1. **Layout toggle**: Add a "Compare" button or dropdown in the controls
   panel that switches between single view (current) and grid mode
   (2×2 or 3×3). Store the layout in `state.compareMode` (`null`, `2x2`,
   `3x3`).

2. **Viewport splitting**: Use a single `WebGLRenderer` with
   `renderer.setViewport()` and `renderer.setScissor()` to divide the
   canvas into equal regions. In the render loop, iterate over each
   panel and render its scene from the shared camera. One `OrbitControls`
   instance controls the camera for all panels.

3. **Per-panel scenes**: Each panel gets its own `THREE.Scene` containing
   its geometry. All panels share the same `THREE.PerspectiveCamera` and
   lighting setup. When the user rotates/zooms, all panels update
   identically.

4. **Per-panel state**: Each panel stores: `{generator, params, seed,
   scene, meshGroup, cloudPoints}`. On first entering compare mode,
   populate all panels with the current generator and params. The user
   can then modify individual panels.

5. **Compact parameter overlay**: Each viewport has a small overlay in
   its bottom-left corner showing the seed and any parameters that differ
   from panel 1 (the "base"). Clicking the overlay expands it into a
   minimal inline editor (small inputs, no labels — just
   `param_name: value` rows). Clicking outside or pressing Escape
   collapses it. The overlay should be semi-transparent so it doesn't
   fully obscure the geometry.

6. **Generation**: Each panel generates independently. When a panel's
   params change (and Auto-Apply is on, or the user hits Enter/Apply),
   only that panel regenerates. Show a small per-panel loading spinner.

7. **Shared controls**: View mode (points/wireframe/shaded), point size,
   background color, bounding box toggle, and Lock Camera all apply
   globally to every panel.

8. **Switching back**: Exiting compare mode returns to single-view,
   keeping panel 1's geometry as the active scene.

9. **Panel labels**: Each panel shows a small label in the top-left
   (e.g., "A", "B", "C", "D" or "1", "2", "3", "4") to help the user
   reference them.

**Tests:** `tests/test_preview/test_compare_view.py`

- Preview HTML contains a compare mode toggle (button or dropdown)
- Selecting 2×2 mode creates 4 viewport regions in the canvas
- Selecting 3×3 mode creates 9 viewport regions
- All viewports share the same camera object
- Each viewport has its own scene with independent geometry
- Per-panel parameter overlay is present in the DOM for each viewport
- Exiting compare mode returns to single-view with panel 1's geometry
- Shared controls (view mode, point size, background) apply to all panels
- Changing params in one panel does not affect other panels

---

## Task 63: Generation timeout and cancel mechanism in preview server

**Objective:**

Add a 5-minute timeout to the pipeline generation in the preview server.
If generation exceeds 5 minutes, kill it and return an error to the UI.
Also add a cancel button so users can abort long-running generations
without waiting for the timeout.

**Suggested path:**

1. **Server-side timeout**: Run the pipeline in a separate process (using
   `concurrent.futures.ProcessPoolExecutor` or `multiprocessing.Process`)
   so it can be killed cleanly. Set a 300-second (5-minute) timeout. If
   the process exceeds the timeout, terminate it and return HTTP 504 with
   a JSON body: `{"error": "Generation timed out after 5 minutes"}`.

2. **Cancel endpoint**: Add `POST /api/generate/cancel` that terminates
   the running generation process. Return 200 if a generation was
   cancelled, 404 if nothing is running. Store the running process/future
   in server state so it can be referenced.

3. **UI cancel button**: While generation is in progress (loading spinner
   is visible), show a "Cancel" button next to the spinner. On click,
   POST to `/api/generate/cancel`. On success, hide the spinner and show
   "Generation cancelled".

4. **Timeout configuration**: The timeout should be configurable via
   `MATHVIZ_GENERATION_TIMEOUT` env var (in seconds), defaulting to 300.

5. **Cleanup**: When a generation is timed out or cancelled, ensure any
   temporary files or partial results are cleaned up. The process should
   be terminated, not just abandoned.

6. **UI feedback**: Show elapsed time next to the loading spinner
   (e.g., "Generating... 12s") so users know how long they've been
   waiting and can decide whether to cancel.

**Tests:** `tests/test_preview/test_generation_timeout.py`

- Generation that exceeds timeout returns HTTP 504
- Response body contains a meaningful error message
- `POST /api/generate/cancel` returns 200 when generation is running
- `POST /api/generate/cancel` returns 404 when nothing is running
- Cancelled generation does not leave orphan processes
- Preview HTML shows a cancel button during generation
- Timeout is configurable via environment variable
- Normal (fast) generation still works correctly with timeout in place

---

## Task 64: Improve shaded mesh lighting and material quality in preview UI

**Objective:**

The shaded mesh view looks flat and lifeless — complex surfaces like
schwarz_d appear as featureless blobs. The root causes are: ambient light
too strong (washes out directional contrast), no shadow mapping, only two
directional lights, and no environment-based lighting. Fix the lighting,
enable shadows, and improve materials so shaded mesh view reveals surface
detail and depth.

**Suggested path:**

1. **Reduce ambient light**: Lower `AmbientLight` intensity from 0.5 to
   ~0.15. Replace with a `HemisphereLight` (sky color: soft blue, ground
   color: warm dark) at intensity ~0.3 to give subtle top-bottom gradient
   that adds depth without washing out shadows.

2. **Improve directional lights**: Use a three-point lighting setup:
   - Key light: intensity ~1.2, positioned high and to one side
   - Fill light: intensity ~0.4, opposite side, softer
   - Rim/back light: intensity ~0.3, behind and above, to define edges

3. **Enable shadow mapping**: Set `renderer.shadowMap.enabled = true` and
   `renderer.shadowMap.type = THREE.PCFSoftShadowMap`. Configure the key
   light to cast shadows (`castShadow = true`) with appropriate shadow
   map size (1024 or 2048). Set meshes to both `castShadow` and
   `receiveShadow`.

4. **Improve material**: Consider using `MeshPhysicalMaterial` instead of
   `MeshStandardMaterial` for richer surface response. Add subtle
   `envMapIntensity` with a simple procedural environment (e.g.,
   `PMREMGenerator` with a basic scene). Current metalness (0.1) and
   roughness (0.6) are reasonable but may need tuning.

5. **Add tone mapping**: Set `renderer.toneMapping = THREE.ACESFilmicToneMapping`
   and `renderer.toneMappingExposure = 1.0` for more cinematic/realistic
   output.

6. **Normal computation**: Ensure loaded meshes have computed vertex
   normals (`geometry.computeVertexNormals()`) so shading interpolates
   smoothly across faces. Without normals, `MeshStandardMaterial` renders
   flat-shaded.

7. **Light background mode**: When toggling to light background, adjust
   light intensities so the mesh is still clearly visible (dark
   backgrounds need brighter lights, light backgrounds need less).

**Tests:** `tests/test_preview/test_lighting.py`

- Preview HTML configures at least 3 light sources
- Shadow mapping is enabled on the renderer
- At least one light has `castShadow = true`
- Mesh materials are created with `castShadow` and `receiveShadow`
- `computeVertexNormals()` is called on loaded geometries
- Tone mapping is configured on the renderer
- HemisphereLight is present in the scene

---

## Task 65: Randomize parameters button in preview UI

**Objective:**

Add a "Randomize" button next to the parameter editor that shuffles all
generator parameters to random values within reasonable ranges. This lets
users quickly explore the parameter space and discover interesting
configurations, especially for the ~76% of generators whose output is
purely determined by parameters (not seed).

**Suggested path:**

1. Add a "Randomize" (or dice icon) button in the parameter editor
   section, next to Apply/Reset.

2. Add `GET /api/generators/<name>/param-ranges` endpoint (or extend the
   existing params endpoint) that returns min/max/step for each parameter.
   Each generator should define reasonable exploration ranges — not just
   validation bounds. For example, Lorenz sigma might validate at `> 0`
   but the exploration range should be `[5, 20]` to stay in the chaotic
   regime.

3. If a generator does not define explicit exploration ranges, derive them
   from the defaults: `[default * 0.25, default * 2.0]` for positive
   floats, `[0, default * 2]` for non-negative ints. For parameters with
   known physical meaning (angles, frequencies), clamp to meaningful
   ranges.

4. On click, for each parameter: pick a random value uniformly within
   its exploration range, respecting the step size. Populate the input
   fields with the new values.

5. If Auto-Apply (Task 56) is enabled, trigger regeneration immediately
   after randomizing. Otherwise, just populate the fields and wait for
   the user to click Apply.

6. Also randomize the seed (random integer 0–999999) since for
   seed-dependent generators this adds further variation.

7. Add a keyboard shortcut (e.g., `R` key when no input is focused) to
   trigger randomize for rapid exploration.

**Tests:** `tests/test_preview/test_randomize_params.py`

- Preview HTML contains a Randomize button
- Clicking Randomize changes parameter input values
- Randomized values fall within the generator's exploration ranges
- Seed is also randomized
- Default exploration ranges are derived from defaults when not specified
- Generators can define explicit exploration ranges
- Keyboard shortcut triggers randomization when no input is focused

---

## Task 66: Fix Lock Camera — disable interaction and fix regeneration race

**Objective:**

The Lock Camera checkbox has two bugs:

1. **Does not prevent camera movement**: Checking "Lock Camera" sets
   `state.cameraLocked = true` but never disables `OrbitControls`. The
   user can still rotate, pan, and zoom freely.

2. **Intermittent failure with Auto-Apply**: Camera lock sometimes fails
   during rapid Auto-Apply regeneration. Root causes:
   - `addBoundingBox()` is called unconditionally in `displayMesh()` and
     may affect the scene even when locked
   - No guard against concurrent `displayGenerateResult()` calls — rapid
     auto-apply can fire overlapping async operations where the second
     call's `saveCameraIfLocked()` captures a partially-modified state
     from the first call

**Suggested path:**

1. In the lock-camera change event listener, set
   `controls.enabled = !e.target.checked`. This disables all
   OrbitControls interaction (rotate, pan, zoom) when locked.

2. Gate `addBoundingBox()` calls in `displayMesh()` and `displayCloud()`
   behind `!state.cameraLocked` — when locked, keep the existing
   bounding box or hide it.

3. Add a generation guard to `displayGenerateResult()`: set a flag like
   `state.isGenerating = true` at the start. If another call arrives
   while generating, cancel/ignore the previous one. This prevents
   overlapping async operations from corrupting camera state.

4. When camera is locked and geometry regenerates, update only
   `camera.near` and `camera.far` (for clipping) but do NOT change
   position, target, or projection.

5. The Reset View button should still work when locked — temporarily
   re-enable controls, reframe, then re-disable.

6. Visual feedback: change cursor to `not-allowed` over the canvas when
   locked.

**Tests:** `tests/test_preview/test_lock_camera.py` (extend existing)

- Toggling Lock Camera on sets `controls.enabled` to false
- Toggling Lock Camera off sets `controls.enabled` to true
- `addBoundingBox` is skipped during regeneration when locked
- Camera position is identical before and after 5 rapid auto-apply cycles
- Concurrent `displayGenerateResult` calls are serialized or debounced
- Reset View still works when camera is locked
- Near/far clipping planes update even when locked

---

## Task 67: Fix snapshot save crash — VTK NSWindow threading error

**Objective:**

The `POST /api/snapshots` endpoint crashes the entire preview server on
macOS. The save endpoint calls PyVista to render a thumbnail, but VTK's
Cocoa renderer requires NSWindow on the main thread. FastAPI handles
requests on worker threads, so VTK throws
`NSInternalInconsistencyException: NSWindow should only be instantiated
on the main thread` and kills the process. All subsequent requests fail
with `NetworkError`.

**Suggested path:**

1. Remove the server-side PyVista thumbnail rendering from the save
   endpoint entirely.

2. Instead, capture the thumbnail client-side from the Three.js canvas
   using `canvas.toDataURL('image/png')` (the renderer already has
   `preserveDrawingBuffer: true`). Downscale to ~256×256 using an
   offscreen canvas. Send the base64-encoded PNG in the POST body as
   a `thumbnail` field.

3. On the server, decode the base64 PNG and save it as `thumbnail.png`
   in the snapshot directory.

4. If no thumbnail is provided in the request (e.g., from a CLI caller),
   simply skip thumbnail generation — do not attempt PyVista rendering.

5. Ensure the save endpoint cannot crash the server under any
   circumstances — wrap in appropriate error handling and return a proper
   HTTP error response instead of crashing.

**Tests:** `tests/test_preview/test_snapshots_save.py` (extend existing)

- `POST /api/snapshots` does not import or call PyVista
- Saving with a base64 thumbnail field creates thumbnail.png on disk
- Saving without a thumbnail field succeeds (no thumbnail.png created)
- Server remains responsive after a save operation
- Invalid base64 thumbnail returns 400, does not crash the server

---

## Task 68: Parallel generation for multi-panel comparison view

**Objective:**

When the preview UI is in comparison mode (2×2 or 3×3 grid from Task 62),
each viewport's geometry should be generated in parallel using multiple
CPU cores — one process per viewport. Currently the preview server
generates geometry sequentially in a single process, which means a 2×2
grid takes 4× as long and a 3×3 grid takes 9× as long as a single render.

**Suggested path:**

1. Add a `POST /api/generate-batch` endpoint to the preview server. The
   request body is a list of generation configs:
   ```json
   {
     "panels": [
       {"generator": "lorenz", "params": {"sigma": 10}, "seed": 42, "container": {...}},
       {"generator": "lorenz", "params": {"sigma": 15}, "seed": 42, "container": {...}},
       {"generator": "lorenz", "params": {"sigma": 20}, "seed": 42, "container": {...}},
       {"generator": "lorenz", "params": {"sigma": 25}, "seed": 42, "container": {...}}
     ]
   }
   ```

2. On the server, use `concurrent.futures.ProcessPoolExecutor` to run
   each panel's pipeline in a separate process. Limit the pool size to
   `min(len(panels), os.cpu_count())`. Each worker runs the full
   pipeline (generate → represent → transform → validate) and returns
   the serialized geometry (mesh GLB and/or point cloud PLY bytes).

3. The response returns a list of geometry results in the same order as
   the input panels, each with `geometry_id`, `mesh_url`, `cloud_url`
   (or `error` if that panel failed).

4. Important: do NOT use PyVista/VTK in the worker processes — only run
   the pipeline and serialization. Rendering is done client-side by
   Three.js. This avoids the macOS NSWindow threading crash (Task 67).

5. In the preview UI, when in comparison mode, call
   `POST /api/generate-batch` instead of making N separate
   `POST /api/generate` calls. Load all returned geometries into their
   respective viewport scenes.

6. Apply the same 5-minute timeout (Task 63) to the entire batch, not
   per-panel. If the batch times out, return partial results for any
   panels that completed.

7. Show a single loading indicator for the batch with progress
   (e.g., "Generating 2/4...") as each panel completes.

**Tests:** `tests/test_preview/test_batch_generate.py`

- `POST /api/generate-batch` with 4 panels returns 4 geometry results
- Panels are generated in parallel (total time < 2× single panel time)
- Failed panels return error without crashing the batch
- Response order matches request order
- Batch respects the generation timeout
- Empty panels list returns 400
- Single-panel batch works the same as regular generate

---

## Task 69: Fix stale "Usable: 90 x 90 x 30 mm" on page load

**Objective:**

The preview HTML has `Usable: 90 x 90 x 30 mm` hardcoded at line 282,
reflecting the old 100×100×40 container defaults. The JS function
`updateUsableVolume()` recalculates it dynamically but is not called on
initial page load, so users see the stale value until they interact with
a dimension or margin input.

**Suggested path:**

1. Call `updateUsableVolume()` on page load (in the initialization
   section, after DOM is ready) so the displayed value always matches
   the actual input field values.

2. Change the hardcoded initial text from `Usable: 90 x 90 x 30 mm` to
   `Usable: 90 x 90 x 90 mm` to match the new 100×100×100 defaults
   (with 5mm margins). This way even before JS runs, the HTML is correct.

3. Audit for any other hardcoded values in the HTML that reference the
   old 40mm depth default.

**Tests:** `tests/test_preview/test_usable_volume.py`

- Initial HTML usable volume text matches current container defaults
- `updateUsableVolume()` is called during page initialization
- Changing depth input updates the usable volume display
- Usable volume calculation is correct: dimension - 2 * margin per axis

---

## Task 70: Strange attractors gallery — Clifford, Dequan Li, Sprott

**Objective:**

Add three new chaotic attractor generators that produce visually striking
point clouds and tube meshes: Clifford attractor, Dequan Li attractor,
and a selection of Sprott systems. These complement the existing attractor
family (Lorenz, Rossler, Thomas, etc.) with more exotic geometries.

**Suggested path:**

1. **Clifford attractor** (`generators/attractors/clifford.py`):
   A 2D iterated map extended to 3D. Equations:
   ```
   x_{n+1} = sin(a * y_n) + c * cos(a * x_n)
   y_{n+1} = sin(b * x_n) + d * cos(b * y_n)
   ```
   Extend to 3D by using iteration count as z-coordinate (scaled) or by
   stacking multiple attractors with varying parameters. Default params:
   `a=-1.4, b=1.6, c=1.0, d=0.7`. This is an iterated map, not an ODE —
   do not use `solve_ivp`. Iterate directly for N points (default 500,000).
   Use SPARSE_SHELL representation since this naturally produces a point
   cloud, not a curve.

2. **Dequan Li attractor** (`generators/attractors/dequan_li.py`):
   A 3D continuous chaotic system. Equations:
   ```
   dx/dt = a*(y - x) + d*x*z
   dy/dt = k*x + f*y - x*z
   dz/dt = c*z + x*y - e*x^2
   ```
   Default params: `a=40, c=1.833, d=0.16, e=0.65, f=20, k=55`.
   Integrates via the existing attractor base class with `solve_ivp`.
   Use TUBE representation.

3. **Sprott systems** (`generators/attractors/sprott.py`):
   Implement 3–5 of the most visually interesting Sprott minimal chaotic
   systems (from Sprott's catalog of simple chaotic flows). Each is a 3D
   ODE with minimal terms. Expose a `system` parameter to select which
   Sprott variant (e.g., `sprott_a`, `sprott_b`, `sprott_g`, `sprott_n`,
   `sprott_s`). Use the attractor base class. Use TUBE representation.

4. All three generators should follow the existing attractor patterns:
   seed-dependent via initial condition perturbation, resolution controlled
   by `integration_steps` (for ODE types) or `num_points` (for Clifford).

**Tests:** `tests/test_generators/test_strange_attractors.py`

- Clifford attractor produces a point cloud with expected number of points
- Clifford output varies with seed
- Dequan Li attractor produces a curve with > 1000 points
- Dequan Li default params produce a bounded (non-diverging) trajectory
- Sprott generator accepts a `system` parameter to select variant
- Each Sprott variant produces a distinct trajectory
- All three generators register correctly and appear in `mathviz list`
- `mathviz render-2d <name> -o test.png` succeeds for each

---

## Task 71: Reaction-diffusion patterns on surfaces

**Objective:**

Add a generator that runs Gray-Scott reaction-diffusion on a curved
surface (torus, sphere, or arbitrary mesh) and produces a 3D mesh with
Turing-pattern geometry. Unlike the existing `reaction_diffusion`
generator (which runs on a flat 2D grid), this simulates directly on a
surface, displacing vertices along normals to create organic bumps, spots,
and stripes.

**Suggested path:**

1. Create `generators/procedural/rd_surface.py` with a
   `ReactionDiffusionSurface` generator.

2. Parameters:
   - `base_surface`: `sphere`, `torus`, `klein_bottle` (default: `torus`)
   - `feed_rate` (f): controls pattern type (default: 0.055)
   - `kill_rate` (k): controls pattern type (default: 0.062)
   - `diffusion_u`, `diffusion_v`: diffusion rates (defaults: 0.16, 0.08)
   - `iterations`: simulation steps (default: 5000)
   - `displacement_scale`: vertex displacement along normals (default: 0.1)
   - `grid_resolution`: base surface mesh resolution (default: 128)

3. Algorithm:
   - Generate the base surface mesh (reuse existing parametric generators)
   - Build a vertex adjacency graph (Laplacian) from mesh topology
   - Initialize U=1, V=0 everywhere, seed V with small random patches
   - Run Gray-Scott iteration using the mesh Laplacian for diffusion
   - Displace each vertex along its normal by `V * displacement_scale`
   - Recompute normals on the displaced mesh

4. Different `(f, k)` values produce different patterns: spots (0.035,
   0.065), stripes (0.055, 0.062), maze (0.029, 0.057). Document these
   presets as parameter suggestions.

5. Seed controls the initial V perturbation placement.

6. Use SURFACE_SHELL representation (output is already a mesh).

**Tests:** `tests/test_generators/test_rd_surface.py`

- Generator produces a valid mesh with expected vertex count
- Different feed/kill rates produce different vertex displacements
- Output is seed-dependent
- Base surface parameter switches between torus, sphere, klein_bottle
- Displacement scale of 0 produces the unmodified base surface
- Generator registers and appears in `mathviz list`

---

## Task 72: L-system / fractal tree generator

**Objective:**

Add a generator that produces 3D branching structures from L-system
(Lindenmayer system) grammars. L-systems are a classic way to generate
fractal trees, bushes, ferns, and other organic branching patterns. These
look striking as both tube meshes and point clouds when laser-engraved.

**Suggested path:**

1. Create `generators/procedural/lsystem.py` with an `LSystemGenerator`.

2. Parameters:
   - `preset`: named preset — `tree`, `bush`, `fern`, `hilbert3d`,
     `sierpinski` (default: `tree`)
   - `iterations`: number of L-system rewriting steps (default: 5)
   - `angle`: branching angle in degrees (default: 25.0)
   - `length_scale`: initial segment length (default: 1.0)
   - `length_decay`: segment length multiplier per generation
     (default: 0.7)
   - `thickness_decay`: branch thickness decay per generation
     (default: 0.6)

3. Each preset defines an axiom, production rules, and a turtle
   interpretation. For example, `tree`:
   - Axiom: `F`
   - Rules: `F → F[+F]F[-F]F`
   - Turtle: `F` = move forward, `+` = turn right by angle,
     `-` = turn left, `[` = push state, `]` = pop state

4. Extend the classic 2D turtle to 3D: add pitch (`^`, `&`), roll
   (`/`, `\\`), and random rotation for organic variation (seed-dependent).

5. The turtle produces a set of line segments (start, end, thickness).
   Convert these to a `Curve` per branch (or a single connected curve
   tree). Use TUBE representation for mesh output.

6. For presets like `hilbert3d` and `sierpinski`, the L-system produces
   space-filling curves and fractal geometry respectively.

7. Seed controls random angle jitter for organic-looking trees.

**Tests:** `tests/test_generators/test_lsystem.py`

- Tree preset produces a branching curve structure
- Increasing iterations increases the number of segments
- Different presets produce distinct geometries
- Angle parameter affects branching spread
- Output is seed-dependent (random jitter varies)
- Hilbert 3D preset produces a space-filling curve
- Generator registers and appears in `mathviz list`
- `mathviz render-2d lsystem -o test.png` succeeds

---

## Task 73: Spherical Voronoi generator

**Objective:**

Add a generator that creates Voronoi tessellation on a sphere surface,
producing geodesic cell structures that look organic and architectural.
The existing `voronoi_3d` generator fills a 3D volume; this one
tessellates the surface of a sphere into polygonal cells with raised
edges, producing a structure that resembles a soccer ball, biological
cells, or a geodesic dome.

**Suggested path:**

1. Create `generators/geometry/voronoi_sphere.py` with a
   `VoronoiSphereGenerator`.

2. Parameters:
   - `num_cells`: number of Voronoi seed points on the sphere
     (default: 64)
   - `radius`: sphere radius (default: 1.0)
   - `edge_width`: width of the ridge along cell boundaries
     (default: 0.05)
   - `edge_height`: how far ridges protrude above the sphere surface
     (default: 0.1)
   - `cell_style`: `ridges_only`, `cells_only`, `both` (default:
     `ridges_only`) — ridges show just the edges, cells show filled
     polygonal faces, both shows the full structure

3. Algorithm:
   - Distribute `num_cells` points on the unit sphere (use Fibonacci
     spiral for even spacing, perturbed by seed for variation)
   - Compute the spherical Voronoi diagram using
     `scipy.spatial.SphericalVoronoi`
   - For `ridges_only`: extract cell boundary arcs as curves on the
     sphere, thicken them into tube meshes along the surface
   - For `cells_only`: triangulate each Voronoi cell face
   - For `both`: combine ridges and cell faces

4. Seed controls the perturbation of the initial Fibonacci spiral points.

5. Use SURFACE_SHELL representation for cell faces, TUBE for ridges.

**Tests:** `tests/test_generators/test_voronoi_sphere.py`

- Generator produces a mesh with the expected cell structure
- `num_cells=6` produces roughly an icosahedron-like structure
- `num_cells=64` produces a denser tessellation
- Output is seed-dependent
- `cell_style` parameter changes the output geometry
- `edge_height=0` produces a flat sphere with visible cell boundaries
- Generator registers and appears in `mathviz list`
- `mathviz render-2d voronoi_sphere -o test.png` succeeds

---

## Task 74: Extended mathematical knot gallery

**Objective:**

Expand the knot generators beyond the current set (figure-eight, torus
knot, lissajous knot, seven-crossing knots) with more exotic knots and
linked structures. These multi-component objects look particularly
striking in glass when rendered as tubes.

**Suggested path:**

1. Create `generators/knots/exotic_knots.py` with the following knots,
   each as a separate registered generator:

2. **Pretzel knot** (`pretzel_knot`): A (p, q) pretzel knot
   parameterization. Parameters: `p` (number of left-hand twists,
   default: 2), `q` (right-hand twists, default: 3), `curve_points`
   (default: 1024). Produces a single closed curve.

3. **Borromean rings** (`borromean_rings`): Three mutually linked rings
   where no two are directly linked — removing any one frees the other
   two. Parameterize as three elliptical curves in orthogonal planes
   with slight deformation to create the linking. Parameters:
   `ring_radius` (default: 1.0), `ring_thickness` for tube
   (default: 0.08), `curve_points` (default: 512). Produces three
   closed curves.

4. **Chain links** (`chain_links`): A chain of N interlocking torus
   links. Parameters: `num_links` (default: 5), `link_radius`
   (default: 0.5), `link_thickness` (default: 0.1), `curve_points`
   (default: 256). Each link is a torus curve oriented alternately.
   Produces N closed curves.

5. **Trefoil on torus** (`trefoil_on_torus`): A (2,3) torus knot
   rendered alongside a transparent/wireframe torus surface showing
   how the knot sits on the torus. Parameters: `torus_R` (default: 1.0),
   `torus_r` (default: 0.4), `curve_points` (default: 1024). Produces
   curves (the knot) plus a mesh (the torus surface). The knot uses TUBE
   representation; the torus surface uses WIREFRAME.

6. **Cinquefoil knot** (`cinquefoil_knot`): The (2,5) torus knot — a
   five-lobed knot that looks like a star. Simple parametric form.
   Parameters: `curve_points` (default: 1024).

7. All knot generators produce closed curves and use TUBE representation.
   Multi-component knots (Borromean rings, chain links) return multiple
   curves in the MathObject.

**Tests:** `tests/test_generators/test_exotic_knots.py`

- Pretzel knot produces a closed curve
- Borromean rings produces exactly 3 closed curves
- Chain links with `num_links=5` produces 5 closed curves
- Chain links with `num_links=1` produces a single ring
- Trefoil on torus produces both curves and a mesh
- Cinquefoil knot produces a closed curve distinct from trefoil
- All generators register and appear in `mathviz list`
- `mathviz render-2d <name> -o test.png` succeeds for each

---

## Task 75: Update generator documentation for all new generators

**Objective:**

Update all documentation (README, docs/, CLI help) to cover every
generator currently in the codebase, including all generators added in
Tasks 70–74 (strange attractors, reaction-diffusion surfaces, L-systems,
spherical Voronoi, exotic knots) and any other generators added since the
last documentation pass (Task 37).

**Suggested path:**

1. **`docs/generators.md`**: Add an entry for every registered generator.
   Each entry should include:
   - Name and aliases
   - Category
   - One-line description
   - Parameters table (name, type, default, description)
   - Resolution parameters (if any)
   - Whether output varies with seed
   - Recommended representation strategy
   - A sample render command

2. **Organize by category**: Group generators under headings — Attractors,
   Parametric Surfaces, Implicit Surfaces, Curves, Knots, Fractals,
   Number Theory, Physics, Procedural, Geometry, Data-Driven.

3. **`README.md`**: Update the generator count and list. Add a summary
   table of all categories with generator counts.

4. **Parameter presets**: For generators with interesting parameter
   combinations (e.g., reaction-diffusion spots vs stripes vs maze,
   Sprott system variants, L-system presets), document recommended
   parameter sets with example commands.

5. **Audit for completeness**: Run `mathviz list` and verify every
   generator in the registry has a corresponding entry in
   `docs/generators.md`. Flag any missing ones.

6. **Sample renders**: Reference the `mathviz render-all` command
   (Task 61) for generating a visual catalog. If a `renders/` directory
   with sample images exists, link to them from the docs.

**Tests:** `tests/test_docs/test_generator_docs.py`

- Every generator in the registry has an entry in `docs/generators.md`
- Every entry includes name, category, parameters, and description
- No generator is listed in docs that doesn't exist in the registry
- README generator count matches the actual registry count
- `docs/generators.md` has a heading for each category

---

## Task 76: Comprehensive documentation update for all features

**Objective:**

Perform a full documentation audit and update covering all implemented
features, CLI commands, preview UI capabilities, pipeline concepts, and
configuration options. This goes beyond generator docs (Task 75) to cover
the entire project.

**Suggested path:**

1. **`README.md`**: Rewrite to cover:
   - Project overview and purpose (glass engraving pipeline)
   - Installation (with `[render]` extras)
   - Quick start: generate, render, preview
   - Full CLI command reference summary
   - Link to detailed docs

2. **`docs/cli.md`**: Document every CLI command with usage, options,
   and examples:
   - `mathviz generate` — full pipeline with export
   - `mathviz render` / `render-2d` — offline PNG rendering
   - `mathviz render-all` — batch parallel rendering (Task 61)
   - `mathviz preview` — interactive browser preview
   - `mathviz benchmark` — performance benchmarking (Task 60)
   - `mathviz list` / `info` / `validate` / `convert` / `sample`
   - `mathviz transform` / `schema` / `grid`

3. **`docs/preview.md`**: Document the preview UI features:
   - Generator switcher (Task 42)
   - Container/dimensions editor (Task 43)
   - Parameter editor (Tasks 44/51)
   - View modes (shaded, wireframe, point cloud)
   - Lock Camera, Auto-Apply, Reset View
   - Save/Load snapshots (Tasks 57/58)
   - Comparison mode 2×2/3×3 (Task 62)
   - Keyboard shortcuts (Enter to regenerate, R to randomize)

4. **`docs/pipeline.md`**: Document the pipeline stages:
   - Generate → Represent → Transform → Validate → Export
   - Representation strategies (SURFACE_SHELL, TUBE, SPARSE_SHELL, etc.)
   - Container model and placement policy
   - Export formats (STL, OBJ, PLY, GLB)

5. **`docs/configuration.md`**: Document configuration options:
   - Config file support and sampling profiles
   - Environment variables (MATHVIZ_SNAPSHOTS_DIR,
     MATHVIZ_GENERATION_TIMEOUT, etc.)
   - CLI flag precedence

6. **Audit**: Cross-reference every public feature against docs. Every
   CLI flag, every UI control, every config option should be documented.

**Tests:** `tests/test_docs/test_docs_completeness.py`

- README exists and contains install, run, and test sections
- Every CLI command has an entry in `docs/cli.md`
- `docs/preview.md` exists and covers all UI features
- `docs/pipeline.md` exists and covers all pipeline stages
- Every doc file is valid markdown with no broken headers
- No dead links between doc files
