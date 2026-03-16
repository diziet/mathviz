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

---

## Task 77: Calabi-Yau manifold generator

**Objective:**

Add a generator for Calabi-Yau manifold cross-sections — the iconic
string theory shape. Uses the standard parameterization projecting a
complex algebraic surface to 3D, producing a crystalline flower-like form.

**Suggested path:**

1. Create `generators/parametric/calabi_yau.py`. Use the standard
   parameterization:
   ```
   x = Re(e^{2πi*k/n} * (x1 + i*x2))
   y = Re(e^{2πi*k/n} * (y1 + i*y2))
   z = cos(α) * Re(z_val) + sin(α) * Im(z_val)
   ```
   where the surface satisfies `z1^n + z2^n = 1` in C².

2. Parameters: `n` (exponent, default: 5), `alpha` (projection angle,
   default: π/4), `grid_resolution` (default: 128).

3. Generate multiple patches (one per value of k in 0..n-1) and combine.
   Use SURFACE_SHELL representation.

**Tests:** `tests/test_generators/test_calabi_yau.py`

- Produces a valid mesh
- Different `n` values produce distinct geometries
- Registers and renders successfully

---

## Task 78: Roman / Steiner surface generator

**Objective:**

Add a generator for the Roman surface (Steiner surface) — a
self-intersecting non-orientable surface with tetrahedral symmetry.

**Suggested path:**

1. Create `generators/parametric/roman_surface.py`. Parameterization:
   ```
   x = a² * sin(2u) * cos(v)² / 2
   y = a² * sin(u) * sin(2v) / 2
   z = a² * cos(u) * sin(2v) / 2
   ```
   with u ∈ [0, π], v ∈ [0, π].

2. Parameters: `scale` (default: 1.0), `grid_resolution` (default: 128).
   Use SURFACE_SHELL representation.

**Tests:** `tests/test_generators/test_roman_surface.py`

- Produces a valid mesh with self-intersections
- Registers and renders successfully

---

## Task 79: Seifert surface generator

**Objective:**

Add a generator for Seifert surfaces — the orientable surface bounded by
a knot. Given a knot (trefoil, figure-eight, etc.), generate the minimal
surface that fills its interior.

**Suggested path:**

1. Create `generators/parametric/seifert_surface.py`. For the trefoil
   Seifert surface, use the Milnor fiber parameterization:
   the surface satisfying `f(z1, z2) / |f(z1, z2)| = e^{iθ}` where
   `f(z1, z2) = z1^p + z2^q` intersected with S³.

2. Parameters: `knot_type` (`trefoil`, `figure_eight`, default: `trefoil`),
   `theta` (Milnor fiber angle, default: 0), `grid_resolution`
   (default: 128).

3. Use SURFACE_SHELL representation. The boundary of the surface should
   match the knot curve.

**Tests:** `tests/test_generators/test_seifert_surface.py`

- Produces a valid mesh
- Boundary curve approximates the expected knot
- Different knot_type values produce distinct surfaces
- Registers and renders successfully

---

## Task 80: Dini's surface generator

**Objective:**

Add a generator for Dini's surface — a twisted pseudospherical surface
that looks like a seashell or spiral horn.

**Suggested path:**

1. Create `generators/parametric/dini_surface.py`. Parameterization:
   ```
   x = a * cos(u) * sin(v)
   y = a * sin(u) * sin(v)
   z = a * (cos(v) + log(tan(v/2))) + b * u
   ```
   with u ∈ [0, 4π], v ∈ (0.01, π-0.01).

2. Parameters: `a` (default: 1.0), `b` (twist rate, default: 0.2),
   `turns` (number of u-wraps, default: 2), `grid_resolution`
   (default: 128). Use SURFACE_SHELL representation.

**Tests:** `tests/test_generators/test_dini_surface.py`

- Produces a valid mesh
- More turns produce a longer spiral
- Registers and renders successfully

---

## Task 81: Dupin cyclide generator

**Objective:**

Add a generator for the Dupin cyclide — a smooth inversive geometry shape
that generalizes a torus, cylinder, and cone.

**Suggested path:**

1. Create `generators/parametric/dupin_cyclide.py`. Use the standard
   parameterization with parameters `a`, `b`, `c`, `d` controlling the
   shape family (torus-like to horn-like). Use the inversion-of-torus
   approach: parameterize a torus and apply a Möbius transformation.

2. Parameters: `a` (default: 1.0), `b` (default: 0.8), `c` (default: 0.5),
   `d` (offset, default: 0.6), `grid_resolution` (default: 128).
   Use SURFACE_SHELL representation.

**Tests:** `tests/test_generators/test_dupin_cyclide.py`

- Produces a valid mesh
- Different parameter ratios produce torus-like vs horn-like shapes
- Registers and renders successfully

---

## Task 82: Cross-cap surface generator

**Objective:**

Add a generator for the cross-cap — a non-orientable surface and
a model of the real projective plane immersed in 3D.

**Suggested path:**

1. Create `generators/parametric/cross_cap.py`. Parameterization:
   ```
   x = sin(u) * sin(2v) / 2
   y = sin(2u) * cos(v)²
   z = cos(2u) * cos(v)²
   ```
   with u ∈ [0, π], v ∈ [0, π].

2. Parameters: `scale` (default: 1.0), `grid_resolution` (default: 128).
   Use SURFACE_SHELL representation.

**Tests:** `tests/test_generators/test_cross_cap.py`

- Produces a valid mesh
- Registers and renders successfully

---

## Task 83: Enable parallel test execution with pytest-xdist

**Objective:**

Tests currently run sequentially even though `pytest-xdist` is installed.
Enable parallel test execution by default so the full test suite runs
faster, especially on multi-core machines.

**Suggested path:**

1. Add `addopts = "-n auto"` under `[tool.pytest.ini_options]` in
   `pyproject.toml`. This uses all available CPU cores.

2. Ensure no tests share mutable global state, temp directories, or
   fixed port numbers that would cause conflicts under parallel execution.
   Use `tmp_path` fixtures instead of hardcoded paths.

3. If any tests require serial execution (e.g. they bind to a fixed port
   for the preview server), mark them with `@pytest.mark.serial` or
   isolate them in a separate test group.

4. Verify the full suite passes with `-n auto`. Fix any flaky or
   order-dependent tests that surface.

**Files:**

- `pyproject.toml`
- Any test files that need isolation fixes

**Tests:**

- `pytest -n auto` runs the full suite without failures
- Test suite wall-clock time is measurably faster than sequential
- No tests fail intermittently due to resource conflicts

## Task 84: Bour's minimal surface generator

**Objective:**

Add a generator for Bour's minimal surface — a shape that interpolates
between a helicoid and a catenoid.

**Suggested path:**

1. Create `generators/parametric/bour_surface.py`. Parameterization:
   ```
   x = r * cos(θ) - r^n * cos(nθ) / (2n)
   y = -r * sin(θ) - r^n * sin(nθ) / (2n)
   z = 2 * r^(n/2) * cos(nθ/2) / n
   ```
   with r ∈ [0, 1], θ ∈ [0, 2π].

2. Parameters: `n` (order, default: 2 for helicoid-catenoid),
   `r_max` (default: 1.0), `grid_resolution` (default: 128).
   Use SURFACE_SHELL representation.

**Tests:** `tests/test_generators/test_bour_surface.py`

- Produces a valid mesh
- Different `n` values produce distinct surfaces
- Registers and renders successfully

---

## Task 85: Menger sponge generator

**Objective:**

Add a generator for the Menger sponge — the iconic recursive cube
fractal created by repeatedly removing sub-cubes.

**Suggested path:**

1. Create `generators/fractals/menger_sponge.py`. Recursively subdivide
   a cube into 27 sub-cubes (3×3×3 grid) and remove the center of each
   face and the center cube (7 removals per level), keeping 20 sub-cubes.
   Recurse on each remaining sub-cube.

2. Parameters: `level` (recursion depth, default: 3, max: 4),
   `size` (default: 1.0).

3. At level 3: 20³ = 8000 cubes. Generate a mesh by creating 6 quad
   faces per visible cube face (skip internal faces between adjacent
   cubes for efficiency). Use SURFACE_SHELL representation.

**Tests:** `tests/test_generators/test_menger_sponge.py`

- Level 0 produces a single cube (8 vertices, 12 triangles)
- Level 1 produces 20 sub-cubes
- Level 3 produces expected vertex count
- Registers and renders successfully

---

## Task 86: Sierpinski tetrahedron generator

**Objective:**

Add a generator for the Sierpinski tetrahedron (tetrix) — the 3D
analogue of the Sierpinski triangle, created by recursively removing
octahedra from a tetrahedron.

**Suggested path:**

1. Create `generators/fractals/sierpinski_tetrahedron.py`. Start with
   a regular tetrahedron. At each level, replace it with 4 half-scale
   tetrahedra at the corners.

2. Parameters: `level` (recursion depth, default: 5, max: 8),
   `size` (default: 1.0).

3. At level N: 4^N tetrahedra. Generate mesh faces for each. Use
   SURFACE_SHELL representation.

**Tests:** `tests/test_generators/test_sierpinski_tetrahedron.py`

- Level 0 produces a single tetrahedron (4 faces)
- Level 1 produces 4 tetrahedra
- Output vertex count scales as expected with level
- Registers and renders successfully

---

## Task 87: Speed up slow tests — reduce integration steps and share benchmark runs

**Objective:**

Several test files use unnecessarily high iteration counts, causing
individual tests to take 1–4 seconds. The benchmark tests are the worst
offenders — each test re-runs the entire benchmark CLI independently.
Reduce step counts and share expensive fixtures to cut test suite time.

**Changes needed:**

1. **Reduce `_TEST_STEPS` in attractor/dynamics tests**:
   - `tests/test_generators/test_attractors.py`: 5000 → 500
   - `tests/test_generators/test_attractors_extended.py`: 5000 → 500,
     and the 20,000-step test on line 73 → 2000
   - `tests/test_generators/test_double_pendulum.py`: 5000 → 500
   - `tests/test_generators/test_strange_attractors.py`: 5000 → 500
   - 500 steps is sufficient to verify trajectory properties (finite,
     non-NaN, non-degenerate, deterministic, seed-varying). These are
     unit tests, not integration tests.

2. **Share benchmark fixture**: The 6 tests in `TestBenchmarkCommand`
   each call `_run_benchmark()` independently (~4s each). Refactor to
   use a class-scoped or session-scoped fixture that runs the benchmark
   once and shares the `(result, output_path, html_content)` across all
   tests that inspect the output.

3. **Verify all tests still pass** after the reductions. If any test
   relies on high step counts for numerical stability (e.g. checking
   attractor convergence), increase that specific test's count with a
   comment explaining why.

4. **Audit for other slow tests**: Check if any other test files have
   unnecessarily high iteration counts or redundant generation calls.

**Files:**

- `tests/test_generators/test_attractors.py`
- `tests/test_generators/test_attractors_extended.py`
- `tests/test_generators/test_double_pendulum.py`
- `tests/test_generators/test_strange_attractors.py`
- `tests/test_cli/test_benchmark.py`

**Tests:**

- Full test suite passes with reduced step counts
- No individual test takes more than 500ms (except benchmark which should
  be under 2s total for the shared fixture)
- Total test suite wall-clock time is measurably faster


---

## Task 88: Fix voronoi_sphere representation default — SURFACE_SHELL → TUBE

**Objective:**

`voronoi_sphere` generates curves (ridge edges), not a mesh, but
`representation_defaults.py` maps it to `SURFACE_SHELL` which requires
a mesh. This causes a `ValueError: SURFACE_SHELL requires a mesh input`
when rendering voronoi_sphere through the pipeline.

**Suggested path:**

1. In `src/mathviz/pipeline/representation_defaults.py`, change
   `"voronoi_sphere": _SURFACE_CONFIG` to `"voronoi_sphere": _TUBE_CONFIG`.

2. Verify `mathviz render-2d voronoi_sphere -o test.png --view angle`
   completes without error.

**Files:**

- `src/mathviz/pipeline/representation_defaults.py`

**Tests:** `tests/test_pipeline/test_representation_fallback.py`

- voronoi_sphere renders through the full pipeline without error
- voronoi_sphere default representation is TUBE, not SURFACE_SHELL

## Task 89: Apollonian gasket 3D generator

**Objective:**

Add a generator for the 3D Apollonian gasket — recursive sphere packing
where each gap between tangent spheres is filled with the largest fitting
sphere.

**Suggested path:**

1. Create `generators/fractals/apollonian_3d.py`. Start with 4 mutually
   tangent spheres (Soddy configuration). Recursively fill each gap with
   the unique sphere tangent to the three surrounding spheres, using
   Descartes' circle theorem extended to 3D.

2. Parameters: `max_depth` (recursion, default: 5), `min_radius`
   (stop when sphere radius < this, default: 0.01).

3. Render each sphere as a low-poly icosphere mesh. Combine all sphere
   meshes into one MathObject. Use SURFACE_SHELL representation.

4. Seed controls initial configuration perturbation.

**Tests:** `tests/test_generators/test_apollonian_3d.py`

- Produces a mesh containing multiple spheres
- Deeper recursion produces more spheres
- All spheres are non-overlapping (centers separated by sum of radii)
- Registers and renders successfully

---

## Task 90: Epsilon separation for self-intersecting parametric surfaces

**Objective:**

Add a tiny epsilon offset at self-intersection regions of non-orientable
and self-intersecting parametric surfaces to eliminate coincident vertices,
z-fighting, degenerate normals, and visual pinching in the renderer.

**Background:**

Four parametric surfaces produce coincident vertex pairs where the surface
passes through itself in 3D:

| Surface | Coincident pairs | Cause |
|---|---|---|
| `klein_bottle` | 128 | v=0 and v=π map to same circle |
| `cross_cap` | 510 | Self-intersection along a line segment |
| `roman_surface` | 258 | Self-intersection curves + pinch points |
| `boy_surface` | 258 | Triple point and self-intersection curves |

When vertices from different parameter-space regions land at identical 3D
positions, the mesh has degenerate triangles, undefined normals, and
z-fighting artifacts. An epsilon separation (displacing one sheet slightly
along the local surface normal at the intersection) fixes the visual
without meaningfully affecting the mathematical shape.

**Suggested path:**

1. **Shared helper in `_mesh_utils.py`**: Add a function
   `separate_coincident_vertices(vertices, faces, epsilon=1e-3)` that:
   - Builds a KDTree of all vertices
   - Finds pairs closer than `epsilon * 0.1`
   - For each pair, computes approximate face normals from adjacent faces
   - Offsets one vertex in each pair by `epsilon` along its face normal
   - Returns the modified vertex array

2. **Per-surface integration**: Call the helper after mesh construction in
   each of the four generators:
   - `klein_bottle.py` — after `_generate_klein_mesh`
   - `cross_cap.py` — after `_generate_cross_cap_mesh`
   - `roman_surface.py` — after `_generate_roman_mesh`
   - `boy_surface.py` — after `_generate_boy_mesh`

3. **Alternative per-surface approach**: If a shared helper is awkward
   (different surfaces have different self-intersection geometry), add
   surface-specific epsilon logic. For example, the Klein bottle could
   offset vertices near v=0 vs v=π by ±epsilon along z.

4. **Epsilon as parameter**: Expose `separation_epsilon` as an optional
   parameter (default ~0.005) so users can tune or disable it (set to 0).

**Files:**

- `src/mathviz/generators/parametric/_mesh_utils.py`
- `src/mathviz/generators/parametric/klein_bottle.py`
- `src/mathviz/generators/parametric/cross_cap.py`
- `src/mathviz/generators/parametric/roman_surface.py`
- `src/mathviz/generators/parametric/boy_surface.py`
- `tests/test_generators/test_epsilon_separation.py`

**Tests:**

- For each of the four surfaces, generate mesh and confirm zero coincident
  vertex pairs (no pair closer than `epsilon * 0.5`)
- Default epsilon produces visually distinct sheets (min NN distance > 0)
- Setting `separation_epsilon=0` disables separation and coincident pairs
  return
- Mesh vertex count and face count are unchanged (no vertices added/removed)
- Bounding box is not significantly changed (within 1% of original)
- All four surfaces still pass their existing test suites


---

## Task 91: Consolidate benchmark tests to eliminate redundant CLI runs

**Objective:**

The benchmark test file runs the full benchmark CLI 3 separate times
(~11s total), making it the single slowest test file. Consolidate to
a single shared run where possible.

**Background:**

Current benchmark test structure:
- `shared_benchmark` fixture (class-scoped): runs lorenz+torus+mobius_strip — **~6s**
- `test_generators_flag_limits_selection`: runs its own lorenz+torus — **~4s**
- `test_failed_generator_in_report_not_crash`: runs torus+nonexistent — **~1s**
- `torus_benchmark` fixture: runs torus only (shared, fast)
- `test_stage_timings_sum_to_total`: calls `_run_single_generator` directly (fast)

The first test that touches `shared_benchmark` pays a 6s setup cost.
Then `test_generators_flag_limits_selection` runs a second, nearly
identical benchmark. These two can be merged.

**Suggested path:**

1. **Merge `test_generators_flag_limits_selection`** into the shared
   fixture tests. The shared benchmark already runs 3 generators — test
   that filtering works by verifying the HTML contains exactly those 3,
   not others.

2. **Make `test_failed_generator_in_report_not_crash` use a fixture too**,
   or keep it standalone but with a single fast generator
   (`generators="torus,nonexistent_xyz"` — torus is the fastest).

3. **Consider reducing `FAST_GENERATORS`** from 3 to 2 (drop mobius_strip
   or lorenz) since the tests don't need 3 generators to verify behavior.

4. **Target**: benchmark tests complete in < 4s total (down from ~11s).

**Files:**

- `tests/test_cli/test_benchmark.py`

**Tests:**

- All existing benchmark test assertions still pass
- Total benchmark test file runtime < 4 seconds
- No redundant benchmark CLI invocations (at most 2 total runs)


---

## Task 92: Fix flaky `test_parallel_faster_than_sequential` test

**Objective:**

The test `test_parallel_faster_than_sequential` in
`tests/test_preview/test_batch_generate.py` is a timing-based assertion
that intermittently fails depending on system load. Make it reliable.

**Background:**

The test asserts that parallel batch generation is faster than sequential.
This is inherently flaky because:
- On low-core machines or under load, parallelism overhead can negate gains
- Timing-based assertions are non-deterministic
- The test fails sporadically in CI and local runs, then passes on retry

**Suggested path:**

1. **Remove the timing assertion entirely.** Instead, verify that the
   parallel endpoint *works correctly* (returns correct results for all
   panels) without asserting it's faster. The parallelism is an
   implementation detail, not a correctness property.

2. **Alternative**: If a speed assertion is desired, use a generous margin
   (e.g., parallel must complete in < 2× sequential time) to avoid flakes,
   or skip the test with `@pytest.mark.slow` and only run it in dedicated
   perf test suites.

3. **Alternative**: Replace the wall-clock comparison with a structural
   check — verify that the parallel endpoint uses `ProcessPoolExecutor`
   or `concurrent.futures` (mock-based), confirming parallelism is wired
   up without depending on timing.

**Files:**

- `tests/test_preview/test_batch_generate.py`

**Tests:**

- The replacement test passes reliably on 10 consecutive runs
- No timing-dependent assertions remain
- Parallel batch generation correctness is still verified


---

## Task 93: Quaternion Julia set generator

**Objective:**

Add a generator for quaternion Julia sets — 4D fractals sliced to 3D,
producing smoother, more organic shapes than the Mandelbulb.

**Suggested path:**

1. Create `generators/fractals/quaternion_julia.py`. Iterate
   `q → q² + c` in quaternion space (4D), where c is a quaternion
   constant. Extract a 3D isosurface by fixing one quaternion component.

2. Parameters: `c_real` (default: -0.2), `c_i` (default: 0.8),
   `c_j` (default: 0.0), `c_k` (default: 0.0), `max_iter`
   (default: 10), `escape_radius` (default: 2.0), `voxel_resolution`
   (default: 128), `slice_w` (4th dimension slice, default: 0.0).

3. Use marching cubes on a 3D grid to extract the isosurface. Use
   SURFACE_SHELL representation.

**Tests:** `tests/test_generators/test_quaternion_julia.py`

- Produces a valid mesh
- Different `c` values produce distinct shapes
- Higher `voxel_resolution` produces more detailed mesh
- Registers and renders successfully

---

## Task 94: Burning ship fractal heightmap generator

**Objective:**

Add a generator for the Burning Ship fractal — the asymmetric,
aggressive-looking cousin of the Mandelbrot set, rendered as a 3D
heightmap.

**Suggested path:**

1. Create `generators/fractals/burning_ship.py`. Iteration:
   `z → (|Re(z)| + i|Im(z)|)² + c`. Count escape iterations on a 2D
   grid and use as a heightmap.

2. Parameters: `center_x` (default: -0.4), `center_y` (default: -0.6),
   `zoom` (default: 3.0), `max_iter` (default: 256),
   `pixel_resolution` (default: 512), `height_scale` (default: 0.3).

3. Use HEIGHTMAP_RELIEF representation, same as `mandelbrot_heightmap`.

**Tests:** `tests/test_generators/test_burning_ship.py`

- Produces a valid scalar field
- Output is distinct from Mandelbrot (asymmetric)
- Higher pixel_resolution produces a finer grid
- Registers and renders successfully

---

## Task 95: IFS fractal generator

**Objective:**

Add a generator for Iterated Function System (IFS) fractals — including
Barnsley fern in 3D, Sierpinski variants, and custom affine transforms.

**Suggested path:**

1. Create `generators/fractals/ifs_fractal.py`. An IFS is defined by a
   set of affine transformations with associated probabilities. Iterate
   by randomly choosing a transform and applying it to the current point.

2. Parameters: `preset` (`barnsley_fern`, `maple_leaf`, `spiral`,
   `custom`, default: `barnsley_fern`), `num_points` (default: 500,000),
   `dimensions` (`2d_extruded`, `3d`, default: `3d`).

3. For `barnsley_fern`: use the classic 4-transform IFS extended to 3D
   by adding z-axis rotation or thickness. For `3d` mode, use 3D affine
   matrices.

4. Output as a point cloud. Use SPARSE_SHELL representation. Seed
   controls the random iteration sequence.

**Tests:** `tests/test_generators/test_ifs_fractal.py`

- Barnsley fern preset produces a point cloud with expected shape bounds
- Different presets produce distinct point distributions
- Output is seed-dependent
- Registers and renders successfully

---

## Task 96: Fix Klein bottle u-seam — vertex duplication for smooth normals

**Objective:**

Fix the remaining Klein bottle shaded mesh artifact. Task 96's earlier
fix (`build_klein_wrapped_faces` with v-reflection) eliminated the
stretched seam triangles, but 128 flipped normal pairs remain at u=0
where seam faces meet interior faces with opposite winding. This causes
a dark seam line in shaded view because `computeVertexNormals()` averages
opposing normals to near-zero at those vertices.

**Root cause:**

The Klein bottle is non-orientable — face winding must flip somewhere.
The flip is at u=0, where seam faces (coming from u=127 with reversed
winding) share vertices with interior faces (u=0 to u=1 with normal
winding). When Three.js averages face normals per vertex, the opposing
contributions cancel, producing near-zero normals along the entire u=0
ring.

Measured: u=0 vertex normal magnitudes are ~0.001 vs ~0.06 for interior
vertices (60× weaker).

**Suggested path:**

1. **Duplicate u=0 row vertices**: Add 128 extra vertices (copies of
   row 0). Interior faces reference the original u=0 vertices; seam
   faces reference the duplicates. Each copy gets normals only from
   its own side, eliminating the cancellation.

2. **Update `build_klein_wrapped_faces`**: Seam faces should reference
   vertex indices `n_u * n_v + v` (the duplicate row) instead of `v`
   for their row-0 connections.

3. **Update `klein_bottle.py`**: Append the duplicate row to the vertex
   array after mesh construction.

**Files:**

- `src/mathviz/generators/parametric/_mesh_utils.py`
- `src/mathviz/generators/parametric/klein_bottle.py`
- `tests/test_generators/test_klein_bottle.py`

**Tests:**

- No face edge > 2× average edge length
- Zero coincident vertex pairs (epsilon separation still works)
- Vertex normals at u=0 have magnitude > 50% of interior mean
- No adjacent face pair has dot product < -0.5
- Existing Klein bottle tests still pass


---

## Task 97: Koch snowflake 3D generator

**Objective:**

Add a generator for the 3D Koch snowflake — the classic fractal curve
extruded or revolved into a 3D solid.

**Suggested path:**

1. Create `generators/fractals/koch_3d.py`. Generate the 2D Koch
   snowflake curve at a given recursion level, then produce 3D geometry
   by either extrusion (along z-axis) or revolution (around y-axis).

2. Parameters: `level` (recursion, default: 4, max: 6), `mode`
   (`extrude`, `revolve`, default: `extrude`), `height` (extrusion
   height, default: 0.3). Use SURFACE_SHELL representation.

**Tests:** `tests/test_generators/test_koch_3d.py`

- Level 0 produces an equilateral triangle
- Higher levels produce more vertices (4^n scaling)
- Extrude and revolve modes produce distinct geometries
- Registers and renders successfully

---

## Task 98: Electron orbital generator

**Objective:**

Add a generator for hydrogen atom electron orbitals — probability density
isosurfaces for s, p, d, and f orbitals.

**Suggested path:**

1. Create `generators/physics/electron_orbital.py`. Compute the hydrogen
   wavefunction `|ψ(r,θ,φ)|²` using the radial function `R_nl(r)` and
   spherical harmonics `Y_lm(θ,φ)`.

2. Parameters: `n` (principal quantum number, default: 3), `l` (angular,
   default: 2), `m` (magnetic, default: 0), `voxel_resolution`
   (default: 128), `iso_level` (probability cutoff, default: 0.01).

3. Evaluate on a 3D grid and extract isosurface with marching cubes.
   Use SURFACE_SHELL representation.

4. Different (n,l,m) produce the iconic orbital shapes: (1,0,0)=sphere,
   (2,1,0)=dumbbell, (3,2,0)=cloverleaf, etc.

**Tests:** `tests/test_generators/test_electron_orbital.py`

- (1,0,0) produces a roughly spherical isosurface
- (2,1,0) produces a dumbbell shape (two lobes)
- Invalid quantum numbers (l >= n) raise ValueError
- Registers and renders successfully

---

## Task 99: Magnetic field lines generator

**Objective:**

Add a generator for 3D magnetic field line visualizations — dipole and
quadrupole configurations rendered as tube curves.

**Suggested path:**

1. Create `generators/physics/magnetic_field.py`. Compute field lines by
   numerically integrating the magnetic field vector from seed points.

2. Parameters: `field_type` (`dipole`, `quadrupole`, default: `dipole`),
   `num_lines` (default: 24), `line_points` (integration steps per line,
   default: 500), `spread` (seed point distribution radius, default: 0.3).

3. Seed points are distributed on a ring around the source. Integrate
   using RK4 in both directions along the field. Each field line becomes
   a Curve. Use TUBE representation.

4. Seed controls the initial ring angle offset.

**Tests:** `tests/test_generators/test_magnetic_field.py`

- Dipole produces field lines that loop from one pole to the other
- Quadrupole produces a more complex line pattern
- `num_lines` controls the number of curves output
- Registers and renders successfully

---

## Task 100: DNA double helix generator

**Objective:**

Add a generator for a DNA double helix — twin parametric helices with
base pair rungs connecting them.

**Suggested path:**

1. Create `generators/parametric/dna_helix.py`. Two helices offset by
   180° (major and minor groove), connected by rungs at regular intervals.

2. Parameters: `turns` (number of helix turns, default: 3),
   `radius` (helix radius, default: 1.0), `rise_per_turn` (default: 3.4,
   matching real DNA ~34Å per turn), `base_pairs_per_turn` (default: 10),
   `curve_points` (default: 512).

3. Output: two helix curves + rung curves connecting them. Use TUBE
   representation with different tube_radius for backbone vs rungs.

**Tests:** `tests/test_generators/test_dna_helix.py`

- Produces two helix curves plus rung curves
- Number of rungs = turns × base_pairs_per_turn
- More turns produce a longer structure
- Registers and renders successfully

---

## Task 101: Hopf fibration generator

**Objective:**

Add a generator for the Hopf fibration — circles in S³ projected to R³,
forming nested tori of linked rings. This is one of the most visually
stunning mathematical objects.

**Suggested path:**

1. Create `generators/parametric/hopf_fibration.py`. The Hopf map sends
   points on S² to great circles on S³. Project S³ → R³ via
   stereographic projection. For each point on a chosen set of circles
   on S², compute the corresponding fiber (a circle in S³) and project.

2. Parameters: `num_fibers` (default: 32), `num_circles` (number of
   base circles on S², default: 5), `fiber_points` (points per fiber
   curve, default: 256), `projection_point` (stereographic projection
   offset, default: (0,0,0,2)).

3. Output: many closed curves (one per fiber). Use TUBE representation
   with thin tube_radius. The fibers naturally form tori that nest
   inside each other.

4. Different base circle configurations produce different visual patterns.

**Tests:** `tests/test_generators/test_hopf_fibration.py`

- Produces `num_fibers × num_circles` closed curves
- Each fiber is a closed loop
- Different num_circles values produce different torus configurations
- Registers and renders successfully

---

## Task 102: Gravitational lensing grid generator

**Objective:**

Add a generator for a gravitational lensing visualization — a warped
coordinate grid showing spacetime curvature around a point mass.

**Suggested path:**

1. Create `generators/physics/gravitational_lensing.py`. Start with a
   flat 2D grid of lines. Apply the Schwarzschild deflection formula to
   bend each grid line around a central mass, then extrude to 3D using
   the deflection magnitude as z-displacement.

2. Parameters: `mass` (controls curvature strength, default: 1.0),
   `grid_lines` (per axis, default: 20), `grid_extent` (default: 5.0),
   `grid_points` (points per line, default: 200).

3. Output as curves (grid lines). Use TUBE representation with thin
   tubes or SPARSE_SHELL as point cloud.

**Tests:** `tests/test_generators/test_gravitational_lensing.py`

- Grid lines near center are more deflected than far ones
- `mass=0` produces a flat grid
- Registers and renders successfully

---

## Task 103: Wave interference pattern generator

**Objective:**

Add a generator for 3D standing wave interference patterns from multiple
point sources.

**Suggested path:**

1. Create `generators/physics/wave_interference.py`. Place N point
   sources on a plane. Compute the superposition of their spherical waves
   `A * sin(k*r - ωt + φ) / r` at each point in a 3D grid. Extract an
   isosurface at a threshold amplitude.

2. Parameters: `num_sources` (default: 3), `wavelength` (default: 0.5),
   `source_spacing` (default: 1.0), `voxel_resolution` (default: 128),
   `iso_level` (default: 0.5), `time` (phase, default: 0).

3. Use marching cubes for isosurface. Use SURFACE_SHELL representation.
   Seed controls source position jitter.

**Tests:** `tests/test_generators/test_wave_interference.py`

- Produces a valid mesh with wave-like structure
- More sources produce more complex patterns
- Registers and renders successfully

---

## Task 104: Hilbert curve 3D generator

**Objective:**

Add a generator for the 3D Hilbert curve — a space-filling curve that
visits every cell in a cubic grid exactly once.

**Suggested path:**

1. Create `generators/curves/hilbert_3d.py`. Implement the 3D Hilbert
   curve using recursive coordinate transformation (Gray code mapping
   or Butz algorithm).

2. Parameters: `order` (recursion level, default: 4, max: 6),
   `size` (default: 1.0).

3. At order N: 8^N points connected in sequence. Output as a single
   Curve. Use TUBE representation for mesh or SPARSE_SHELL for points.

**Tests:** `tests/test_generators/test_hilbert_3d.py`

- Order 1 produces 8 points
- Order N produces 8^N points
- Curve visits each grid cell exactly once
- Registers and renders successfully

---

## Task 105: Penrose tiling 3D generator

**Objective:**

Add a generator for 3D Penrose tiling — aperiodic tilings extruded as
a relief surface.

**Suggested path:**

1. Create `generators/procedural/penrose_3d.py`. Generate a 2D Penrose
   tiling (P3 rhombus tiling via de Bruijn's method or Robinson triangle
   subdivision). Extrude each tile to a height based on its type (thick
   vs thin rhombus).

2. Parameters: `generations` (subdivision depth, default: 5),
   `tile_height_ratio` (height difference between tile types,
   default: 0.3), `extent` (default: 5.0).

3. Use SURFACE_SHELL representation. Output is a mesh of extruded tiles.

**Tests:** `tests/test_generators/test_penrose_3d.py`

- Produces a valid mesh with non-periodic structure
- Higher generations produce more tiles
- Registers and renders successfully

---

## Task 106: Weaire-Phelan foam structure generator

**Objective:**

Add a generator for the Weaire-Phelan structure — the most efficient
known foam partition of space into equal-volume cells.

**Suggested path:**

1. Create `generators/geometry/weaire_phelan.py`. The structure has two
   cell types: irregular dodecahedra and tetrakaidecahedra, arranged in
   a repeating unit cell. Construct the cell geometry from known vertex
   coordinates and tile the unit cell.

2. Parameters: `cells_per_axis` (repeats of unit cell, default: 2),
   `edge_only` (boolean — show only cell edges as wireframe, default: true).

3. If `edge_only`: output cell edges as curves with TUBE representation.
   Otherwise: output cell faces as mesh with SURFACE_SHELL.

**Tests:** `tests/test_generators/test_weaire_phelan.py`

- Produces geometry with the expected cell count
- `cells_per_axis=1` produces 8 cells (2 dodecahedra + 6 tetrakaidecahedra)
- Edge mode produces curves, face mode produces mesh
- Registers and renders successfully

---

## Task 107: Geodesic sphere generator

**Objective:**

Add a generator for geodesic spheres — triangulated spheres at various
frequencies, like Buckminster Fuller domes.

**Suggested path:**

1. Create `generators/geometry/geodesic_sphere.py`. Start with an
   icosahedron. Subdivide each face into a frequency-N triangular grid,
   project vertices onto the unit sphere.

2. Parameters: `frequency` (subdivision level, default: 4, max: 32),
   `radius` (default: 1.0), `dual` (boolean — if true, produce the
   dual polyhedron with pentagonal/hexagonal faces, default: false).

3. Use SURFACE_SHELL representation. The dual mode produces the
   classic soccer-ball / Goldberg polyhedron pattern.

**Tests:** `tests/test_generators/test_geodesic_sphere.py`

- Frequency 1 produces an icosahedron (12 vertices, 20 faces)
- Higher frequency produces more faces (20 * N² faces)
- Dual mode produces pentagonal and hexagonal faces
- All vertices are equidistant from center (on sphere)
- Registers and renders successfully

---

## Task 108: Möbius trefoil generator

**Objective:**

Add a generator for a Möbius trefoil — a Möbius strip twisted into a
trefoil knot shape, combining non-orientability with knot topology.

**Suggested path:**

1. Create `generators/parametric/mobius_trefoil.py`. Parameterize a
   Möbius-like strip whose centerline follows a trefoil knot path.
   The strip cross-section makes a half-twist as it traverses the knot.

2. Parameters: `width` (strip width, default: 0.3), `curve_points`
   (default: 1024), `grid_resolution` (cross-section resolution,
   default: 32). Use SURFACE_SHELL representation.

**Tests:** `tests/test_generators/test_mobius_trefoil.py`

- Produces a valid mesh
- Surface is non-orientable (single-sided)
- Registers and renders successfully

---

## Task 109: Linked tori generator

**Objective:**

Add a generator for linked tori — two or more interlocking torus shapes,
like links in a chain.

**Suggested path:**

1. Create `generators/parametric/linked_tori.py`. Generate N tori with
   centers and orientations arranged so each passes through its neighbor.

2. Parameters: `num_tori` (default: 2), `major_radius` (default: 1.0),
   `minor_radius` (default: 0.3), `link_spacing` (default: 1.5),
   `grid_resolution` (default: 64). Use SURFACE_SHELL representation.

**Tests:** `tests/test_generators/test_linked_tori.py`

- 2 linked tori produce two distinct mesh components
- Tori geometrically interlock (bounding boxes overlap)
- Registers and renders successfully

---

## Task 110: Twisted torus generator

**Objective:**

Add a generator for a twisted torus — a torus where the circular
cross-section rotates N times as it goes around the loop.

**Suggested path:**

1. Create `generators/parametric/twisted_torus.py`. Standard torus
   parameterization but the cross-section angle `v` gets an additional
   `twist * u / (2π)` term.

2. Parameters: `twist` (number of half-twists, default: 3),
   `major_radius` (default: 1.0), `minor_radius` (default: 0.3),
   `grid_resolution` (default: 128). Use SURFACE_SHELL representation.

3. `twist=1` produces a Möbius-torus. Even twists produce orientable
   surfaces.

**Tests:** `tests/test_generators/test_twisted_torus.py`

- twist=0 produces a standard torus
- twist=1 produces a non-orientable surface
- Different twist values produce distinct geometries
- Registers and renders successfully

---

## Task 111: Rose surface generator

**Objective:**

Add a generator for the rose surface — a rhodonea (rose) curve revolved
into 3D, producing flower-like petals.

**Suggested path:**

1. Create `generators/parametric/rose_surface.py`. Start with the 2D
   rose curve `r = cos(kθ)` and revolve it around the z-axis, or use
   a 3D extension: `r = cos(k₁θ) * cos(k₂φ)` on a sphere.

2. Parameters: `k1` (petal count parameter, default: 3), `k2`
   (secondary frequency, default: 2), `grid_resolution` (default: 128).
   Use SURFACE_SHELL representation.

**Tests:** `tests/test_generators/test_rose_surface.py`

- Integer k1 values produce symmetric petal patterns
- Different k1/k2 combinations produce distinct geometries
- Registers and renders successfully

---

## Task 112: Shell spiral generator

**Objective:**

Add a generator for a realistic seashell spiral — a logarithmic spiral
with expanding cross-section, producing a nautilus-like form.

**Suggested path:**

1. Create `generators/parametric/shell_spiral.py`. Parameterize a tube
   whose centerline is a logarithmic spiral and whose cross-section
   radius grows exponentially. The cross-section can be circular or
   elliptical with a lip.

2. Parameters: `growth_rate` (spiral expansion, default: 0.1),
   `turns` (default: 3), `opening_rate` (cross-section growth,
   default: 0.08), `ellipticity` (cross-section shape, default: 1.0),
   `curve_points` (default: 1024), `radial_segments` (default: 32).
   Use SURFACE_SHELL representation.

**Tests:** `tests/test_generators/test_shell_spiral.py`

- Produces a valid mesh with spiral structure
- More turns produce a longer shell
- Growth rate affects how quickly the spiral expands
- Registers and renders successfully

---

## Task 113: Gear / involute curve generator

**Objective:**

Add a generator for mechanical gear tooth profiles — involute gear
geometry extruded into a 3D solid.

**Suggested path:**

1. Create `generators/geometry/gear.py`. Generate a 2D involute gear
   tooth profile (standard gear geometry: base circle, involute curve,
   tip circle, root circle, fillet). Extrude along z-axis for a spur
   gear, or along a helix for a helical gear.

2. Parameters: `num_teeth` (default: 20), `module` (tooth size,
   default: 1.0), `pressure_angle` (default: 20°), `face_width`
   (extrusion height, default: 0.5), `helix_angle` (0 for spur gear,
   default: 0), `curve_points` (per tooth, default: 32).
   Use SURFACE_SHELL representation.

**Tests:** `tests/test_generators/test_gear.py`

- Produces a valid mesh with the expected number of teeth
- helix_angle=0 produces a straight spur gear
- helix_angle>0 produces twisted teeth
- Registers and renders successfully

---

## Task 114: Update documentation for Tasks 77–109 generators

**Objective:**

Update `docs/generators.md` and `README.md` to cover all 30 new
generators added in Tasks 77–106 (Calabi-Yau, Roman surface, Seifert,
Dini, Dupin cyclide, cross-cap, Bour, Menger sponge, Sierpinski
tetrahedron, Apollonian 3D, quaternion Julia, burning ship, IFS, Koch 3D,
electron orbitals, magnetic field, DNA helix, Hopf fibration,
gravitational lensing, wave interference, Hilbert 3D, Penrose 3D,
Weaire-Phelan, geodesic sphere, Möbius trefoil, linked tori, twisted
torus, rose surface, shell spiral, gear).

This is a follow-up to Task 75 (which covers Tasks 70–74 generators).

**Suggested path:**

1. Add an entry in `docs/generators.md` for each of the 30 generators,
   following the same format as Task 75: name, category, description,
   parameters table, sample command.

2. Group the new generators under their categories:
   - Parametric Surfaces: Calabi-Yau, Roman, Seifert, Dini, Dupin,
     cross-cap, Bour, DNA helix, Hopf fibration, Möbius trefoil,
     linked tori, twisted torus, rose surface, shell spiral
   - Fractals: Menger sponge, Sierpinski tetrahedron, Apollonian 3D,
     quaternion Julia, burning ship, IFS, Koch 3D
   - Physics: electron orbitals, magnetic field, gravitational lensing,
     wave interference
   - Curves: Hilbert 3D
   - Procedural: Penrose 3D
   - Geometry: Weaire-Phelan, geodesic sphere, gear

3. Update the README generator count to reflect the full catalog.

4. For visually striking generators (Hopf fibration, electron orbitals,
   Calabi-Yau, Menger sponge), include recommended parameter presets
   with example commands.

**Tests:** `tests/test_docs/test_generator_docs.py` (extend existing)

- Every generator from Tasks 77–106 has an entry in `docs/generators.md`
- README generator count matches the registry
- Each entry includes name, category, parameters, and description

---

## Task 115: Realistic K9 glass crystal preview mode

**Objective:**

Add a "Crystal Preview" view mode to the preview UI that simulates how
the geometry would actually look as a subsurface laser engraving inside
a K9 glass block. Real SSLE creates thousands of tiny white micro-fracture
dots (~0.1mm) suspended inside a clear crystal block. On a dark background
the dots appear as bright white points floating in space; on an LED base
they glow and scatter light.

**Suggested path:**

1. **New view mode**: Add `crystal` to the view mode dropdown alongside
   shaded/wireframe/points. When selected, render the scene in crystal
   simulation mode.

2. **Glass block mesh**: Create a `MeshPhysicalMaterial` box matching
   the container dimensions with:
   - `transmission: 0.95` (nearly fully transparent)
   - `ior: 1.5` (K9 glass refractive index)
   - `thickness: 50` (simulates solid block refraction)
   - `roughness: 0.02` (polished surface)
   - `clearcoat: 1.0`, `clearcoatRoughness: 0.05`
   - `color: 0xffffff`, `opacity: 1`
   - `envMapIntensity` for subtle reflections
   Render the box around the point cloud geometry. Requires an
   environment map — use `PMREMGenerator` with a simple scene or load
   a small HDRI.

3. **Point cloud rendering**: Render the geometry as white points using
   a custom `ShaderMaterial` or `PointsMaterial` with a soft radial
   gradient sprite texture (gaussian falloff, not hard circles). Point
   color: white with slight blue tint (`0xe8f0ff`). Points should appear
   to be physically inside the glass block.

4. **Bloom post-processing**: Add `UnrealBloomPass` from Three.js
   post-processing (`EffectComposer` + `RenderPass` + `UnrealBloomPass`).
   Settings: `threshold: 0.6`, `strength: 0.4`, `radius: 0.3`. This
   creates the subtle glow that simulates light scattering from
   micro-fractures. Only enable bloom in crystal mode to avoid
   performance cost in other modes.

5. **LED base simulation**: Add an optional colored point light or area
   light below the glass block to simulate an LED display base. Toggle
   with a "LED Base" checkbox. Color picker for base color (default:
   warm white). The light should refract through the glass and
   illuminate the points from below.

6. **Dark background**: Crystal mode should force a very dark or black
   background regardless of the light/dark toggle setting, since the
   glass engraving effect only looks right on dark backgrounds.

7. **Performance**: The `MeshPhysicalMaterial` transmission requires an
   extra render pass per transmissive object. This is acceptable for a
   single glass block but document the performance cost. The bloom pass
   adds another full-screen pass.

8. **Controls**: When in crystal mode, expose:
   - Glass tint color (default: clear/white)
   - Bloom intensity slider (0–1)
   - LED base on/off + color
   - Point brightness slider

**Tests:** `tests/test_preview/test_crystal_mode.py`

- Preview HTML contains a "Crystal" option in the view mode dropdown
- Crystal mode creates a MeshPhysicalMaterial box with transmission > 0
- Crystal mode uses PointsMaterial or ShaderMaterial for inner points
- EffectComposer with UnrealBloomPass is set up in crystal mode
- Switching away from crystal mode removes the glass block and bloom
- LED base toggle adds/removes a light below the scene
- Crystal mode forces dark background

---

## Task 116: Disk-based generation cache with UI indicator and invalidation

**Objective:**

Add a disk cache for generated geometry so identical requests return
instantly without re-running the pipeline. Show cache status in the UI
and provide a button to force regeneration (bypass cache).

**Suggested path:**

1. **Cache key**: SHA256 hash of deterministic JSON string of
   `(generator_name, sorted params, seed, container)`.

2. **Disk cache**: Save serialized geometry files (GLB, PLY) to
   `~/.mathviz/cache/<hash>/` along with a `metadata.json` containing
   the request params and timestamp. On a generate request, check disk
   first — if hit, serve the cached files directly.

3. **Cache directory**: Configurable via `MATHVIZ_CACHE_DIR` env var,
   defaulting to `~/.mathviz/cache`. Create on first use.

4. **Cache headers**: Return `X-Cache: HIT` or `X-Cache: MISS` in the
   `/api/generate` response so the frontend knows whether the result
   was cached.

5. **UI indicator**: When the response has `X-Cache: HIT`, show a small
   "Cached" badge near the loading time / info panel. When `MISS`, show
   nothing (or "Generated" briefly).

6. **Force regenerate button**: Add a "Regenerate" button (or a small
   refresh icon next to Generate/Apply) that sends the request with a
   `force: true` flag. The server skips the cache lookup, runs the
   pipeline fresh, and overwrites the cache entry with the new result.
   This lets users force a clean generation when they suspect stale
   data or want to verify reproducibility.

7. **Cache cleanup**: Add `POST /api/cache/clear` endpoint that deletes
   all cached entries. Also add `mathviz cache clear` CLI command.
   No automatic pruning — cache persists until manually cleared or
   size limit is hit.

8. **Cache size limit**: Cap total disk usage (default 5GB, configurable
   via `MATHVIZ_CACHE_MAX_SIZE`). When exceeded, evict oldest entries
   first.

**Tests:** `tests/test_preview/test_cache.py`

- Same request twice: second call serves from cache (faster, X-Cache: HIT)
- Different params produce a cache miss
- `force: true` bypasses cache and overwrites the entry
- `POST /api/cache/clear` removes all cached files
- `X-Cache` header is present in all generate responses
- Cache directory is created automatically on first use
- Cache respects 5GB max size limit and evicts oldest entries when full
- Preview HTML contains a force-regenerate button
- UI shows "Cached" badge when result is from cache

---

## Task 117: Remove redundant `cell_size` parameter from TPMS generators

**Objective:**

The TPMS generators (gyroid, schwarz_p, schwarz_d) have two parameters —
`cell_size` and `periods` — that are mathematically redundant. The spatial
extent is computed as `cell_size * periods * 2π`, so `(cell_size=1, periods=2)`
produces identical output to `(cell_size=2, periods=1)`. Remove `cell_size`
and keep only `periods` to eliminate user confusion.

**Suggested path:**

1. **`_tpms_base.py`**: Remove `cell_size` from `get_default_params()` and
   from `_compute_bounds()`. Hard-code `cell_size = 1.0` internally (or
   inline the constant). Update `_validate_params()` to drop the cell_size
   check.

2. **`gyroid.py`**: Remove `cell_size` from `get_default_params()` and
   `generate()`. Update `_compute_bounds()` call to use a fixed cell size
   of 1.0. Update `_validate_params()`.

3. **`schwarz_p.py`** and **`schwarz_d.py`**: Same changes — remove
   `cell_size` from params, use fixed 1.0 internally.

4. **Tests**: Update any tests that pass `cell_size` as a parameter.
   Add a test confirming that `periods=2` with no `cell_size` produces
   the same output as the old `(cell_size=1, periods=2)`.

5. **Schema/docs**: If `cell_size` appears in generated JSON schemas or
   CLI help text, verify it's removed after the change.

**Files:**

- `src/mathviz/generators/implicit/_tpms_base.py`
- `src/mathviz/generators/implicit/gyroid.py`
- `src/mathviz/generators/implicit/schwarz_p.py`
- `src/mathviz/generators/implicit/schwarz_d.py`
- `tests/test_generators/test_implicit/`

**Tests:** `tests/test_generators/test_implicit/`

- Gyroid generates successfully with only `periods` param
- Schwarz P generates successfully with only `periods` param
- Schwarz D generates successfully with only `periods` param
- Passing unknown `cell_size` param doesn't break generation (ignored gracefully)
- `periods=3` produces larger extent than `periods=1`
- Default params dict does not contain `cell_size`

---

## Task 118: Split Lock Camera into two modes — render lock and full lock

**Objective:**

Replace the single Lock Camera checkbox with a three-state toggle cycling
through: **Off → Render Lock → Full Lock**.

- **Render Lock** (default): Camera position is preserved across
  regenerations (no `fitCamera`, no target reset), but the user can still
  freely orbit, pan, and zoom with the mouse. This is the most common use
  case — compare different parameters from the same viewpoint while still
  being able to adjust the angle.
- **Full Lock** (first click from default): Same as Render Lock, plus mouse
  interaction is disabled (`controls.enabled = false`, cursor `not-allowed`).
  Camera is completely frozen. Useful for pixel-exact comparisons.
- **Off** (second click): Camera reframes on regeneration; mouse
  orbit/pan/zoom works normally.

Clicking the toggle cycles: **Render Lock → Full Lock → Off → Render Lock**.

**Suggested path:**

1. Replace the `<input type="checkbox" id="lock-camera">` with a
   `<button id="lock-camera">` that cycles through three visual states.
   Display the current mode as text or icon on the button:
   - Off: "Camera: Free"
   - Render Lock: "Camera: Locked (movable)" or a lock icon with a move
     indicator
   - Full Lock: "Camera: Frozen" or a solid lock icon

2. Change `state.cameraLocked` from a boolean to a string enum:
   `"off" | "render" | "full"`, initialized to `"render"`. Update all
   existing code that checks `state.cameraLocked` (currently used as a
   boolean) to check for the appropriate mode:
   - `saveCameraIfLocked()` / `restoreCameraIfSaved()`: activate when
     mode is `"render"` or `"full"`
   - `setupCameraForObject()`: skip `fitCamera()` when mode is `"render"`
     or `"full"`
   - `controls.enabled`: set to `false` only when mode is `"full"`
   - Cursor `not-allowed`: only when mode is `"full"`

3. On the toggle button click, cycle: `"render"` → `"full"` → `"off"` →
   `"render"`. Update `controls.enabled` and cursor accordingly on each
   transition.

4. Reset View button should work in all modes — temporarily override the
   lock, reframe, then restore the lock state.

5. Supersedes the controls-disabling parts of Task 66. The race condition
   fix (generation guard / debounce) from Task 66 is still needed
   independently.

**Files:**

- `src/mathviz/static/index.html`

**Tests:** `tests/test_preview/test_lock_camera.py`

- Lock Camera button exists and defaults to "render" state
- Clicking cycles through off → render → full → off
- In "off" mode: `fitCamera` is called on regeneration, controls enabled
- In "render" mode: `fitCamera` is NOT called on regeneration, controls
  remain enabled, camera position/target preserved across regenerations
- In "full" mode: `fitCamera` is NOT called, controls disabled, cursor
  is `not-allowed`
- Reset View works in all three modes
- `saveCameraIfLocked` / `restoreCameraIfSaved` activate in both render
  and full modes
- Transitioning from full → off re-enables controls and resets cursor

---

## Task 119: Fix randomize ranges and add editable min/max to parameter UI

**Objective:**

Two related issues with the dice (randomize) button:

1. **Bug**: `_derive_param_range` in `server.py` returns `min: 0` for
   positive integer defaults (e.g. torus knot `p=2` gets range `[0, 4]`).
   Many generators require `p >= 1`, so randomizing to 0 causes a
   validation error like "p must be >= 1, got 0". The derived min for
   positive integers should be `1`, not `0`.

2. **Bug**: `randomizeParams()` only calls `applyParams()` when
   `state.autoApply` is true. Randomize should always trigger apply —
   the whole point is to see a new random shape immediately.

3. **Bug**: Randomizer doesn't respect inter-parameter constraints. For
   example, trefoil-on-torus requires `torus_r < torus_R`, but independent
   randomization can produce `torus_R=0.65, torus_r=0.78`. After
   randomizing, if the generator rejects the params, either: (a) retry
   with re-rolled values (up to a few attempts), or (b) clamp dependent
   params to satisfy constraints (e.g. ensure `torus_r < torus_R`). A
   simple retry loop (randomize → try generate → if validation error,
   re-roll, up to 5 attempts) is the most general fix since it doesn't
   require encoding every generator's constraints in the UI.

3. **Feature**: Show editable min and max fields next to each numeric
   parameter in the UI. Currently the randomization range is invisible
   and not user-controllable. Interesting shapes often live outside the
   default range (e.g. pretzel knot at `p=11, q=1` but default max is 4).
   Users should be able to widen or narrow the range before hitting
   randomize.

**Suggested path:**

1. **Fix `_derive_param_range`** in `server.py`:
   - For positive integers: change `min: 0` to `min: 1`
   - Review negative integer and float cases for similar off-by-one issues
   - Generators with explicit `get_param_ranges()` are unaffected

2. **Add min/max inputs to parameter editor**: For each numeric parameter
   row in the UI, add small editable `min` and `max` fields (e.g. two
   narrow number inputs to the right of the value input). Pre-populate
   them from the `/api/generators/{name}/param-ranges` response.

3. **Wire randomize to use UI min/max**: When the dice button is clicked,
   read the min/max values from the UI inputs (not the cached server
   response) so user edits take effect immediately. The `randomizeInRange`
   function already accepts min/max/step — just feed it from the UI fields
   instead of the cached `ranges` object.

4. **Visual design**: Keep the min/max fields compact — they should not
   dominate the parameter row. Consider placing them as small inputs
   flanking the value, or as a collapsible "Range" row below each param.
   Show them by default so users know they exist.

5. **Persist per-session**: Store edited min/max values in `state` so they
   survive parameter refreshes within the same session. Reset them when
   the generator changes (new defaults from the server).

**Files:**

- `src/mathviz/preview/server.py` (fix `_derive_param_range`)
- `src/mathviz/static/index.html` (UI for editable min/max)

**Tests:** `tests/test_preview/test_param_ranges.py`

- `_derive_param_range(2)` returns `min: 1` (not 0) for positive integers
- `_derive_param_range(0)` returns `min: 0` (zero is a valid min for zero default)
- `_derive_param_range(-3)` returns a range that includes the default
- Randomize with server-derived ranges never produces values below generator minimums
- Preview HTML contains min/max input fields for numeric parameters
- Editing min/max in the UI affects subsequent randomize results
- Min/max fields are pre-populated from the param-ranges endpoint

---

## Task 120: Collapsible Dimensions/Margins panel, collapsed by default

**Objective:**

The Dimensions (mm) and Margins (mm) sections in the container panel take
up vertical space that is rarely needed. Make the entire container panel
collapsible with a click-to-toggle header, and collapse it by default so
it stays out of the way until the user needs it.

**Suggested path:**

1. Wrap the Dimensions and Margins content inside a collapsible container.
   Add a clickable header row with the section title ("Dimensions / Margins")
   and a chevron indicator (▸ when collapsed, ▾ when expanded).

2. On click, toggle a CSS class (e.g. `.collapsed`) that hides the content
   with `display: none` or `max-height: 0` with a smooth transition.

3. Default state: collapsed. The header row is always visible so users know
   the panel exists and can expand it.

4. When collapsed, only the header row with the chevron is visible — the
   W/H/D inputs, margin inputs, and uniform checkbox are all hidden.

5. Store the collapsed/expanded state in `state` so it persists within the
   session. Optionally save to `localStorage` so it persists across reloads.

**Files:**

- `src/mathviz/static/index.html`

**Tests:** `tests/test_preview/test_collapsible_panel.py`

- Container panel has a clickable header with collapse toggle
- Panel is collapsed by default (dimension inputs not visible)
- Clicking the header expands the panel (inputs become visible)
- Clicking again collapses it
- Dimension/margin values are preserved when collapsing and expanding

---

## Task 121: Fix save after load, show params in gallery, save camera state

**Objective:**

Three issues with the save/load snapshot system:

1. **Bug — cannot save after load + tweak**: After loading a snapshot and
   tweaking parameters, the Save button stays disabled or fails. Loading
   a snapshot sets `state.geometryId = null` (line 1179), and save requires
   a non-null `geometryId` (line 983). If the user tweaks params and hits
   Apply (which calls the generate endpoint and sets a new `geometryId`),
   save should work. Verify this flow works end-to-end. If the user tweaks
   params but does NOT re-generate (just edits input fields), save should
   still work — either by re-enabling save when params change, or by
   auto-generating on param change when auto-apply is on.

2. **Feature — show parameters in load gallery**: The snapshot cards in the
   gallery already show a small `snap-params` line (line 1095-1097), but
   it's a dense single-line summary. Make the parameters more prominent
   and readable in the gallery card:
   - Show each parameter on its own line or in a key=value grid
   - Include seed, container dimensions, and view mode
   - Make it easy to scan and compare snapshots at a glance

3. **Feature — save and restore full UI state**: The save payload should
   include ALL display/control state so loading a snapshot restores the
   exact experience. This includes:
   - Camera position, target (look-at point), and zoom level
   - View mode (points, shaded, wireframe)
   - Per-axis stretch values (X, Y, Z scale factors)
   - Lock camera mode (off / render / full)
   - Show bounding box toggle
   - Show axes toggle
   - Light/dark background toggle
   - Compare mode and layout (if applicable)
   When loading a snapshot, restore all of these to match the saved state.

**Suggested path:**

1. **Fix save flow**: After `loadSnapshot`, if the user triggers a
   generate (via Apply, auto-apply, or randomize), `state.geometryId`
   gets set and save re-enables. Ensure this path works. Additionally,
   consider enabling save immediately after load by storing the snapshot's
   existing geometry files as the "current" state — the geometry is already
   on the server, so re-saving with tweaked display params should be
   possible.

2. **Gallery params display**: In `buildSnapshotCard`, replace the single
   `snap-params` line with a formatted block showing:
   - Parameters as a two-column grid (`param: value`)
   - Seed value
   - Container dimensions (e.g. "100×100×100 mm, margin 5mm")
   - Camera view info if saved

3. **Full UI state in save payload**: When saving, capture and include:
   ```js
   ui_state: {
     camera: {position: {x,y,z}, target: {x,y,z}, zoom},
     view_mode: "points" | "shaded" | "wireframe",
     stretch: {x: 1.0, y: 1.0, z: 1.0},
     camera_lock: "off" | "render" | "full",
     show_bbox: true,
     show_axes: false,
     light_bg: false,
   }
   ```
   On load, restore all of these after geometry is displayed.

4. **Server changes**: The snapshot storage (server-side) needs to accept
   and return the `ui_state` field. Add it to the snapshot metadata JSON
   schema. Unknown fields should be ignored for forward compatibility.

**Files:**

- `src/mathviz/static/index.html`
- `src/mathviz/preview/server.py` (snapshot endpoint — accept camera field)

**Tests:** `tests/test_preview/test_snapshots.py`

- Save button is enabled after loading a snapshot and re-generating
- Snapshot save payload includes full `ui_state` (camera, view mode, stretch, toggles)
- Snapshot load restores camera, view mode, stretch, and all toggles
- Gallery card displays parameters in readable format (not single line)
- Gallery card shows seed and container dimensions
- Round-trip: save → load → all UI state matches original (camera, view mode,
  stretch, toggles)
- Loading a snapshot saved before `ui_state` existed still works (graceful
  fallback to defaults)

---

## Task 122: Colored axis labels on bounding box and per-axis stretch controls

**Objective:**

Two related features for the preview UI:

1. **Axis labels with colors**: Draw labeled X, Y, Z axes on the bounding
   box, each in a distinct color (e.g. X=red, Y=green, Z=blue — the
   standard RGB/XYZ convention). This should be a toggle in the Options
   panel so the user can show/hide them. Labels should be visible text
   ("X", "Y", "Z") at the ends of the axes.

2. **Per-axis stretch (scale) controls**: Add three scale factor inputs
   (X, Y, Z) that non-uniformly stretch the geometry along each axis.
   Default is 1.0 for all three. For example, Dini's surface fills one
   axis but uses only 5–10% of another — the user should be able to
   stretch it to fill the volume more evenly. This is a visual/display
   transform only; it does not change the underlying geometry data, just
   the Three.js scene scale.

**Suggested path:**

1. **Axis labels**: Use Three.js `AxesHelper` for the colored axis lines,
   and `CSS2DRenderer` or `Sprite` with `CanvasTexture` for the "X", "Y",
   "Z" text labels at the axis endpoints. Color convention: X=red (#ff4444),
   Y=green (#44ff44), Z=blue (#4488ff). Add a checkbox toggle "Show Axes"
   in the Options panel, off by default.

2. **Axis lines on bounding box**: Position the axes at the bounding box
   origin or center, scaled to match the bounding box extents. Update
   the axes when geometry changes (new bounding box).

3. **Stretch controls**: Add three number inputs (X, Y, Z scale) in the
   Options or a new "Transform" section. Default 1.0 each, step 0.1,
   min 0.1. When changed, apply `meshGroup.scale.set(sx, sy, sz)` (or
   the equivalent for point clouds). This scales the Three.js object
   in-place without modifying geometry data.

4. **Auto-apply stretch**: If auto-apply is on, changing a stretch value
   should immediately update the scene. Otherwise, apply on next
   generate/apply.

5. **Stretch persists across regeneration**: The stretch values should
   remain set when regenerating with new params. Reset them only when
   the user manually sets them back to 1.0 or clicks a "Reset Scale"
   button.

6. **Interaction with bounding box**: The bounding box wireframe should
   NOT be stretched — it represents the true container dimensions. Only
   the geometry inside is stretched. This makes it visually clear how the
   stretched shape relates to the actual container.

**Files:**

- `src/mathviz/static/index.html`

**Tests:** `tests/test_preview/test_axes_and_stretch.py`

- Preview HTML contains a "Show Axes" toggle checkbox
- Axes toggle defaults to off
- Enabling axes adds AxesHelper or equivalent to the scene
- Axis labels use correct colors (X=red, Y=green, Z=blue)
- Preview HTML contains three scale factor inputs (X, Y, Z)
- Scale inputs default to 1.0
- Setting scale X=2.0 doubles the geometry width without changing the
  bounding box
- Scale values persist across regeneration
- Bounding box is not affected by stretch values

---

## Task 123: Generator thumbnail endpoint with persistent disk cache

**Objective:**

Add a server endpoint that generates and caches preview thumbnails for
each generator. This is the backend prerequisite for the visual generator
browser (Tasks 118–119).

**Suggested path:**

1. **Endpoint**: `GET /api/generators/{name}/thumbnail?view_mode=points`
   returns a 256×256 PNG thumbnail of the generator rendered with default
   parameters. The `view_mode` query param selects the rendering style
   (points, shaded, wireframe). Defaults to points if omitted.

2. **Generation flow**: On a cache miss, the endpoint runs the generator
   with default params at a low resolution (e.g. half the default
   `voxel_resolution` or `curve_points`), converts to geometry, renders
   a small preview image, and returns it as `image/png`.

3. **Rendering**: Use Three.js on the client side or PyVista server-side.
   Client-side is preferred for consistency with the main view — the
   endpoint can return geometry and let the client render thumbnails via
   an offscreen canvas. Alternatively, server-side rendering via PyVista
   produces a PNG directly. Choose whichever is simpler; the key
   requirement is that thumbnails use the same lighting/materials as the
   main view and look polished.

4. **Persistent disk cache**: Store rendered thumbnails at
   `~/.mathviz/thumbnails/<view_mode>/<generator_name>.png`. Keyed by
   generator name + view mode. Cache survives server restarts.

5. **Cache invalidation**: Changing view mode requests a different cache
   key, so each mode has its own set of thumbnails. Add a
   `DELETE /api/thumbnails` endpoint to clear all cached thumbnails.

6. **Batch endpoint**: `GET /api/generators/thumbnails?view_mode=points`
   returns a JSON map of `{generator_name: thumbnail_url}` for all
   generators, triggering background generation for any missing
   thumbnails. This lets the browser modal pre-fetch all thumbnails
   efficiently.

**Files:**

- `src/mathviz/preview/server.py` (thumbnail endpoint)
- `src/mathviz/preview/thumbnails.py` (thumbnail generation + caching logic)

**Tests:** `tests/test_preview/test_thumbnails.py`

- Thumbnail endpoint returns a valid PNG for known generators
- Thumbnail endpoint returns 404 for unknown generators
- Same request twice serves from disk cache (second call is fast)
- Different `view_mode` values produce different cached files
- `DELETE /api/thumbnails` clears all cached thumbnails
- Batch endpoint returns URLs for all registered generators
- Thumbnails are stored at the expected disk path

---

## Task 124: Visual generator browser modal with category grid

**Objective:**

Add a near-fullscreen visual browser modal that replaces the type-ahead
generator dropdown. Users can visually browse generators organized by
category, each with a thumbnail preview. Depends on Task 123 (thumbnail
endpoint).

**Layout:**

- **Opening**: Clicking the generator selector input or pressing `Cmd+K`
  opens the browser modal, covering ~95% of the viewport. Dark backdrop
  with blur, large rounded panel.

- **Top bar**: Search input (auto-focused on open) + close button (X).
  The search input replaces the old type-ahead — typing filters results
  in real-time.

- **Category grid** (default view): A responsive grid (roughly 4×3 for
  12 categories) showing all categories. Each category card shows:
  - Category name and generator count (e.g. "Parametric (15)")
  - Thumbnails of up to 3 representative generators as a preview strip
  - Shortcut number (1–12) displayed in the corner

- **Category detail view**: Clicking a category (or pressing its shortcut
  number) transitions to a grid of all generators in that category (e.g.
  3×5 for parametric's 15 items). Each generator card shows:
  - Generator name
  - Single thumbnail rendered with current view mode
  - Shortcut number within the category

- **Selecting a generator**: Clicking a generator card closes the modal
  and loads that generator into the main preview view with default params.

- **Back navigation**: Backspace or clicking a "Back to categories" link
  returns from category detail to the category grid. Escape from the
  category grid closes the modal entirely.

- **Search mode**: When typing in the search bar, the category grid is
  replaced by a flat filtered list of matching generators across all
  categories, each with its thumbnail. Clicking a result loads it.

**Suggested path:**

1. Replace `#generator-search` input click behavior to open the browser
   modal instead of showing the old dropdown. Keep the input visible as
   a display of the currently selected generator.

2. Build the modal as `<div id="generator-browser">` overlay, structured
   like `#snapshot-gallery` but larger (95% viewport).

3. Populate dynamically from `GET /api/generators` grouped by category.
   Sort categories alphabetically, generators alphabetically within each.

4. Lazy-load thumbnail `<img>` tags from the Task 123 thumbnail endpoint.
   Show a CSS placeholder (gray box with spinner) until loaded.

5. Implement a two-level navigation state machine in JS:
   `browsing_categories` → `browsing_generators_in_category`.

6. Search filtering: on input, fetch all generators (cached client-side),
   filter by name/category substring match, display flat grid.

**Files:**

- `src/mathviz/static/index.html`

**Tests:** `tests/test_preview/test_generator_browser.py`

- Cmd+K opens the browser modal
- Browser shows all categories in a grid
- Each category card shows name, count, and thumbnail previews
- Clicking a category shows its generators in a sub-grid
- Clicking a generator loads it and closes the modal
- Search input filters generators across all categories
- Escape closes the modal from category grid view
- Back navigation returns from category detail to category grid
- Currently selected generator is highlighted if visible
- Modal populates dynamically from the generators API

---

## Task 125: Generator browser keyboard navigation and shortcuts

**Objective:**

Add full keyboard navigation to the generator browser modal (Task 124).
Users should be able to open, browse, select, and load generators entirely
via keyboard. Depends on Task 124.

**Keyboard shortcuts:**

- `Cmd+K`: Open browser (global shortcut, works when modal is closed)
- `Escape`: Close modal (from category grid) or go back (from category
  detail to category grid)
- `1`–`9`, `0` (for 10th item): Select category or generator by position
  number. When browsing categories, pressing `3` opens the 3rd category.
  When browsing generators within a category, pressing `5` selects the
  5th generator.
- Items beyond 10: type two digits quickly (e.g. `1` then `2` within
  500ms selects the 12th item). If only one digit is typed and 500ms
  elapses, select that single-digit item.
- `Arrow keys`: Move focus/highlight through the grid (left, right, up,
  down wrap at grid edges)
- `Enter`: Open the focused category or load the focused generator
- `Backspace`: Go back from category detail to category grid

**Suggested path:**

1. Track focus state: `focusedIndex` (which card is highlighted) and
   `browserLevel` (categories vs generators).

2. Add a visible focus indicator (border highlight or glow) on the
   currently focused card. Update on arrow key navigation.

3. Implement number key handler with a 500ms timeout for two-digit
   input. First digit starts a timer; if a second digit arrives within
   500ms, combine them (e.g. `1` + `2` = 12). Otherwise, use the
   single digit.

4. `Enter` on a focused category card transitions to that category's
   generator grid. `Enter` on a focused generator card loads it.

5. Arrow key grid navigation: calculate row/column from `focusedIndex`
   and grid width, move accordingly, wrap at edges.

6. Ensure keyboard navigation works alongside mouse — clicking a card
   also updates `focusedIndex`. Search input typing should not trigger
   number shortcuts (only activate shortcuts when search input is not
   focused, or use a modifier).

**Files:**

- `src/mathviz/static/index.html`

**Tests:** `tests/test_preview/test_browser_keyboard.py`

- Cmd+K opens the browser modal
- Number keys select categories by position
- Number keys select generators within a category
- Two-digit input (e.g. `1` `2` within 500ms) selects item 12
- Arrow keys move focus through the grid
- Enter opens focused category or loads focused generator
- Escape goes back from category detail, closes from category grid
- Backspace goes back from category detail to category grid
- Keyboard navigation does not trigger while search input is focused
- Focus indicator is visible on the currently highlighted card

---

## Task 126: Point cloud density slider — real-time thinning without regeneration

**Objective:**

Add a density slider to the preview UI that controls how many points are
displayed in point cloud mode. This is a client-side display filter — it
thins or thickens the visible points without re-running the generation
pipeline. Useful for performance tuning and visual clarity on dense clouds.

**Suggested path:**

1. Add a range slider labeled "Density" (0.01 to 1.0, default 1.0, step
   0.01) in the Options panel, visible only when view mode is "points".

2. When the slider value changes, update the point cloud geometry in
   real-time by randomly sampling a fraction of the full point set. Store
   the full point array in `state` and create a display subset based on
   the density value.

3. Use a deterministic sampling approach (e.g. every Nth point, or a
   seeded random subset) so the same density value always shows the same
   points. This avoids visual flickering when dragging the slider.

4. Update the info panel to show "Points: 5,000 / 50,000 (10%)" so the
   user knows how many points are displayed vs total.

5. The slider should update in real-time as you drag — no need to click
   Apply. Use a `requestAnimationFrame` debounce to keep it smooth.

6. Density value persists across regeneration (if you set density to 0.5
   and regenerate, the new cloud is also shown at 50%).

**Files:**

- `src/mathviz/static/index.html`

**Tests:** `tests/test_preview/test_density_slider.py`

- Preview HTML contains a density slider input
- Slider defaults to 1.0 (full density)
- Setting density to 0.5 shows approximately half the points
- Setting density to 0.01 shows approximately 1% of points
- Density persists across regeneration
- Slider is hidden when view mode is not "points"
- Info panel shows filtered vs total point count

---

## Task 127: Turntable animation with GIF/MP4 export

**Objective:**

Add an auto-rotate (turntable) mode and the ability to export the
animation as a GIF or MP4 for sharing.

**Suggested path:**

1. **Turntable toggle**: Add a "Turntable" button or checkbox in the
   Options panel. When enabled, the scene auto-rotates around the Y axis
   at a configurable speed (default ~10 degrees/second). Uses
   `OrbitControls.autoRotate` and `autoRotateSpeed`.

2. **Speed control**: A small slider or number input next to the toggle
   to control rotation speed (0.5x to 5x).

3. **Export button**: When turntable is active, show an "Export" button
   with options for GIF and MP4. Clicking it captures one full 360-degree
   rotation.

4. **Capture mechanism**: Use `renderer.domElement.toDataURL()` to capture
   frames at a fixed interval (e.g. 30fps, 360 frames for 12 seconds at
   30fps). For GIF, use a client-side library like `gif.js` or similar.
   For MP4, use `MediaRecorder` API with `canvas.captureStream()`.

5. **Progress indicator**: Show a progress bar during export ("Capturing
   frame 120/360...") since GIF encoding can be slow.

6. **Output**: Download the file automatically when export completes.
   Filename: `<generator_name>_turntable.<ext>`.

7. **Resolution**: Export at the current canvas resolution. Optionally
   allow a resolution multiplier (1x, 2x) for higher quality exports.

**Files:**

- `src/mathviz/static/index.html`

**Tests:** `tests/test_preview/test_turntable.py`

- Turntable toggle exists and defaults to off
- Enabling turntable sets `controls.autoRotate` to true
- Speed slider adjusts `controls.autoRotateSpeed`
- Export button is visible only when turntable is active
- Disabling turntable stops rotation
- Turntable works with all view modes (points, shaded, wireframe)

---

## Task 128: Color mapping view mode — vertex coloring by curvature, height, or distance

**Objective:**

Add a fifth view mode ("Color Map") that colors vertices using a gradient
based on a selectable metric — curvature, height (Z), distance from
center, or a custom gradient. This transforms monochrome geometry into
visually striking heatmap-style renders. A Schwarz surface colored by
curvature or a Lorenz attractor colored by velocity would look stunning.

**Suggested path:**

1. **New view mode**: Add "Color Map" to the view mode dropdown alongside
   points, shaded, wireframe, and (future) crystal.

2. **Metric selector**: When Color Map mode is active, show a dropdown
   to select the coloring metric:
   - **Height (Z)**: Color by Z coordinate (low=cool, high=warm)
   - **Distance from center**: Color by distance from origin
   - **Curvature** (mesh only): Estimate mean curvature per vertex
   - **Velocity/speed** (attractors): Color by distance between
     consecutive points
   - **Custom gradient**: Let user pick two endpoint colors

3. **Gradient presets**: Provide several built-in color gradients:
   - Viridis (blue-green-yellow)
   - Inferno (black-red-yellow-white)
   - Coolwarm (blue-white-red)
   - Rainbow
   - User-defined (two color pickers for start/end)

4. **Implementation**: Compute a scalar value per vertex based on the
   selected metric. Normalize to [0, 1] range. Map through the gradient
   to produce per-vertex RGB colors. Apply via `vertexColors` on
   `BufferGeometry` with a `Float32Array` color attribute. Use
   `MeshStandardMaterial({vertexColors: true})` for meshes or
   `PointsMaterial({vertexColors: true})` for point clouds.

5. **Real-time updates**: Changing the metric or gradient should update
   colors immediately without regenerating geometry — it's a client-side
   recoloring of existing vertex data.

6. **Works with all geometry types**: Point clouds, meshes, and curves
   should all support color mapping. For curves rendered as points, color
   by arc length or the selected metric.

**Files:**

- `src/mathviz/static/index.html`

**Tests:** `tests/test_preview/test_color_map.py`

- "Color Map" appears as a view mode option
- Selecting Color Map mode shows the metric and gradient selectors
- Height metric colors vertices by Z coordinate
- Distance metric colors vertices by distance from origin
- Changing metric updates colors without regenerating geometry
- Gradient presets produce visually distinct color mappings
- Color mapping works with both point clouds and meshes
- Switching away from Color Map mode restores original material

---

## Task 129: Comprehensive generator documentation with examples and thumbnails

**Objective:**

Create thorough documentation for every generator in the project. Each
generator should have a dedicated section covering what it is, what it
looks like, its parameters and their effects, and example configurations
that produce interesting results.

**Suggested path:**

1. **`docs/generators.md`**: One section per generator, organized by
   category. Each section includes:
   - Generator name, category, and one-line description
   - Brief mathematical/conceptual explanation (2–3 sentences)
   - Thumbnail image of the default output
   - Full parameter table: name, type, default, range, description
   - 2–3 example parameter sets with descriptions of what they produce
     (e.g. "Set `p=11, q=1` for a wide pretzel shape")
   - Notes on performance (e.g. "O(N³) in voxel_resolution")
   - Seed behavior (deterministic vs seed-dependent)

2. **Auto-generation**: Write a script `scripts/generate_generator_docs.py`
   that introspects the registry, reads defaults, param ranges, and
   descriptions, and produces the markdown. Manual descriptions and
   examples can be added via a `docs/generator_notes.yaml` file that the
   script merges in.

3. **Thumbnails in docs**: Reference the thumbnail images generated by
   Task 123's endpoint or pre-render them with a script. Store in
   `docs/images/generators/`.

4. **Index by category**: Include a table of contents at the top grouped
   by category with links to each generator section.

**Files:**

- `docs/generators.md`
- `scripts/generate_generator_docs.py`
- `docs/images/generators/` (thumbnail images)

**Tests:**

- Script runs without errors and produces `docs/generators.md`
- Every registered generator has a section in the output
- Parameter tables match actual defaults from the registry
- Thumbnail images exist for all generators

---

## Task 130: Document all preview UI features and controls

**Objective:**

Create comprehensive user-facing documentation covering every feature and
control in the preview UI. This should serve as both a reference manual
and a quick-start guide for new users.

**Suggested path:**

1. **`docs/preview-ui.md`**: Document every UI feature, organized by
   panel/area:
   - **Generator browser**: Cmd+K, category grid, search, keyboard
     shortcuts, thumbnail previews
   - **Parameters panel**: Editing params, auto-apply toggle, randomize
     button (dice), editable min/max ranges, Enter to apply
   - **Resolution controls**: Resolution inputs, performance warnings
   - **Container panel**: Dimensions, margins, uniform toggle,
     collapsible behavior
   - **View modes**: Points, shaded, wireframe, color map, crystal
     (once implemented). What each looks like and when to use it
   - **Camera controls**: Orbit/pan/zoom, lock camera modes (render lock
     vs full lock), reset view
   - **Display options**: Bounding box toggle, axes toggle (colored XYZ),
     light/dark background, per-axis stretch controls
   - **Point cloud density slider**: Real-time thinning
   - **Turntable and export**: Auto-rotate, GIF/MP4 capture
   - **Compare mode**: 2×2 and 3×3 grid, per-panel generators
   - **Save/Load snapshots**: What gets saved (full UI state), gallery
     browsing, parameter display in gallery
   - **Disk cache**: Cache indicator, force regenerate button
   - **Keyboard shortcuts**: Complete reference table of all shortcuts

2. **Screenshots**: Include annotated screenshots of each panel/feature.
   Store in `docs/images/ui/`.

3. **Update README**: Add a "Preview UI" section to README.md with a
   brief overview and link to the full docs.

**Files:**

- `docs/preview-ui.md`
- `docs/images/ui/` (screenshots)
- `README.md` (add link to preview UI docs)

**Tests:**

- `docs/preview-ui.md` exists and covers all documented features
- Every keyboard shortcut in the code has a corresponding docs entry
- README links to the preview UI documentation

---

## Task 131: Fix Möbius strip u-seam wrapping — v-reversal for half-twist

**Objective:**

Fix the Möbius strip mesh so it renders as a continuous, correctly-connected
surface. The u-seam (where u wraps from 2π back to 0) has a half-twist that
reverses the v-direction, but `_build_u_wrapped_grid_faces` naively connects
`(n-1, v)` to `(0, v)` without the reversal. This creates 254 seam faces
with edges up to **18× the median** edge length, producing a visible "><"
cross pattern in wireframe and shaded views.

**Root cause:**

The Möbius strip half-twist means vertex `(n-1, v_idx)` is spatially near
`(0, n_v-1-v_idx)`, not `(0, v_idx)`. The left edge of the strip (v=-hw)
at u=2π should connect to the right edge (v=+hw) at u=0, and vice versa.

Measured distances at the seam:
- `(127, 0)` → `(0, 0)`: **0.80 units** (wrong — stretches full strip width)
- `(127, 0)` → `(0, 127)`: **0.07 units** (correct — adjacent on surface)

The naive wrapping creates the "><" artifact: left-to-left and right-to-right
connections instead of left-to-right crossover.

**Suggested path:**

1. **New face builder**: Add `build_mobius_wrapped_faces(n_u, n_v)` to
   `_mesh_utils.py`. Interior faces: rows 0 to n-2, open in v (same as
   current). Seam faces: connect `(n-1, v)` to `(0, n_v-1-v)` with
   v-reversal. Note: v is open (not periodic) on the Möbius strip, so
   use `n_v-1-v`, not `(n_v-v) % n_v`.

2. **Vertex duplication at seam**: Same as Klein bottle (Task 96) —
   duplicate the u=0 row so seam faces and interior faces compute normals
   independently. This prevents the dark seam line in shaded view.

3. **Update `mobius_strip.py`**: Use `build_mobius_wrapped_faces` and
   append duplicate row vertices.

**Files:**

- `src/mathviz/generators/parametric/_mesh_utils.py`
- `src/mathviz/generators/parametric/mobius_strip.py`
- `tests/test_generators/test_mobius_strip.py`

**Tests:**

- No seam edge > 3× median edge length (eliminates "><" artifact)
- Seam face areas within 5× of interior face areas
- Vertex normals at u=0 have magnitude > 50% of interior mean

## Task 132: Fix invisible point cloud rendering — sub-pixel point size for cloud-only generators

**Objective:**

Fix point cloud rendering so cloud-only generators (sacks_spiral, prime_gaps,
ulam_spiral, digit_encoding) produce visible points in the preview UI. Currently
these generators show a blank scene because the THREE.js point size is sub-pixel.

**Root cause:**

`loadCloudFromPLY` creates `THREE.PointsMaterial` with
`size: state.pointSize * 0.01` (= 0.02 world units) and `sizeAttenuation: true`
(default). After the pipeline transforms points into the 100mm container, the
camera sits ~162 units away. THREE.js attenuated point size formula:
`pixelSize = size * (canvasHeight / 2) / distance` = 0.02 * 300 / 162 = **0.037 pixels**.
This is sub-pixel and invisible.

Mesh-based generators are unaffected because they render surfaces; the tiny
points in their "points" view mode are a secondary issue. But cloud-only
generators have NO mesh fallback, so nothing renders at all.

Affected generators: sacks_spiral, prime_gaps, ulam_spiral, digit_encoding,
and any future point-cloud-only generator.

**Suggested path:**

1. **Adaptive point size**: In `loadCloudFromPLY` (or `displayCloud`), compute
   a point size based on the scene extent. For example:
   `size = maxExtent * 0.02` where maxExtent is computed from the point cloud
   bounding box. This keeps points at ~2% of the scene, visible at any scale.

2. **Alternative**: Set `sizeAttenuation: false` on the PointsMaterial and use
   a fixed pixel size (e.g. 2-4 pixels). This is simpler but ignores depth.

3. **Point size slider**: The existing point size slider should also scale
   adaptively. Currently `updatePointSize` sets `size * 0.01` which has the
   same sub-pixel problem.

**Files:**

- `src/mathviz/static/index.html` (loadCloudFromPLY, displayCloud, updatePointSize)

**Tests:**

- Manual: generate sacks_spiral and verify points are visible
- Manual: generate prime_gaps, ulam_spiral, digit_encoding — all visible
- Manual: point size slider changes visible point density/size
- Manual: mesh generators still render correctly in all view modes

## Task 133: Fix density slider for mesh-derived point clouds

**Objective:**

Make the point density slider work for mesh-based generators, not just PLY
point cloud generators. Currently the density slider only affects generators
that go through the `displayCloud` path (cloud-only generators). For mesh
generators, the "points" view mode creates THREE.Points from mesh vertices
but never sets `state.cloudPoints` or `state.fullCloudPositions`, so
`applyDensityFilter()` early-returns on its null check.

**Root cause:**

In `loadMeshFromGLB`, points are created as children of `meshGroup`:
```javascript
const pts = new THREE.Points(geom.clone(), pointsMat);
pts.name = 'points';
group.add(shadedMesh, wireMesh, pts);
```

But `state.cloudPoints` is only set in `displayCloud`, which is only called
for generators with a `cloud_url` (point-cloud-only generators). The density
slider handler calls `applyDensityFilter()` which early-returns when
`state.cloudPoints` is null.

**Suggested path:**

1. After loading a mesh, extract the points geometry and set up the density
   filtering infrastructure (state.cloudPoints, state.fullCloudPositions,
   state.fullCloudCount) the same way `displayCloud` does.

2. Alternatively, make `applyDensityFilter` work on mesh group points by
   finding the `points` child in `state.meshGroup` and filtering its geometry.

3. Update `updateDensitySliderVisibility` to show the slider when mesh points
   are available, not just when `state.cloudPoints` is set.

**Files:**

- `src/mathviz/static/index.html` (displayMesh, applyDensityFilter, updateDensitySliderVisibility)

**Tests:**

- Manual: load a mesh generator (e.g. torus), switch to points view, adjust density slider — point count should change
- Manual: density slider still works for cloud-only generators (sacks_spiral)
- Manual: density percentage label updates correctly
- Max edge length < 3× median across entire mesh
- Existing Möbius strip tests still pass
