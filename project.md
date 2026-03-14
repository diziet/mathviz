# project.md

MathViz is a pipeline for programmatically generating 3D mathematical forms
(strange attractors, fractals, knots, minimal surfaces, etc.) and preparing them
for subsurface laser engraving in crystal glass blocks. The output is a
wall-mounted installation of 100–256 blocks arranged in a grid, each containing
a unique form rendered as a monochrome point cloud of micro-fractures inside
glass.

The operator (a single human artist) uses the CLI and a browser-based preview
viewer to explore generators, tune parameters, compare representations, and
export production files. The Tier 1 preview is a single HTML file with vanilla
Three.js served by FastAPI; a React-based Studio with parameter controls,
presets, and gallery is a separate future tier. Agents use the CLI's `--json`
and `--dry-run` modes to implement generators and automate batch work. A grid
manifest tracks which preset is assigned to each physical block position.

The pipeline is a linear chain of independently callable stages: **Generate** →
**Represent** (fabrication policy) → **Transform** (fit to glass block) →
**Sample** (mesh → point cloud) → **Engraving Optimize** → **Validate** →
**Export**. Each stage operates on a `MathObject` dataclass and calls
`validate_or_raise()` at every boundary.

Key constraints that affect everything:

- Glass block dimensions are configurable (Container model) — the operator
  may use cubes, shallow blocks, or other shapes. Nothing in the codebase
  assumes a specific aspect ratio. Depth bias tuning matters most when one
  axis is shallow relative to the others.
- The physical output is always a point cloud of laser fracture points, even
  when the pipeline works with meshes internally. Mesh and cloud are parallel
  representations, never silently converted.
- Every generator must be deterministic given a seed (`default_rng(seed)`,
  never `np.random.seed()`), so any block can be reproduced from its metadata.
- The Representation layer separates "what the math produces" from "how it
  looks engraved in glass," keeping generators free of fabrication concerns.
