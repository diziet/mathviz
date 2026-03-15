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
