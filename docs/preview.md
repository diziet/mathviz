# Preview UI

`mathviz preview` starts an interactive 3D preview: a Three.js viewer served by
FastAPI.

## Starting the Preview

```bash
# Preview a generator
mathviz preview lorenz

# Preview with custom parameters
mathviz preview lorenz --param sigma=12 --seed 7

# Preview an exported file
mathviz preview output.ply

# Custom port, no auto-open
mathviz preview gyroid --port 9000 --no-open
```

## Generator Switcher

A searchable dropdown at the top of the right panel switches the generator
without restarting the server. Typing filters the list by name. Selecting a
generator loads its parameter schema and regenerates the preview.

## Container / Dimensions Editor

The left-column **Container** panel sets the glass block dimensions and
margins that the Transform pipeline stage uses.

| Control | Default | Description |
|---|---|---|
| Width (W) | 100 mm | Block width |
| Height (H) | 100 mm | Block height |
| Depth (D) | 100 mm | Block depth |
| Margin X | 5 mm | Horizontal margin |
| Margin Y | 5 mm | Vertical margin |
| Margin Z | 5 mm | Depth margin |
| Uniform Margin | on | Lock all margins to the same value |

The panel shows the calculated usable volume. **Apply** regenerates with the
new dimensions. **Reset** restores the defaults.

## Parameter Editor

The left-column **Parameter** panel shows an editable field for each parameter
in the selected generator's schema. Each field is a text, number, or checkbox
input, chosen from the parameter's detected type.

- **Apply** — regenerate with the current parameter values
- **Reset** — restore the generator's default parameters
- **Randomize** (dice button) — set every parameter to a random value within
  its valid range. The **R** keyboard shortcut does the same.
- **Resolution fields** — generators with resolution parameters (e.g.
  `integration_steps`, `grid_resolution`) show an additional section.

### Auto-Apply

With the **Auto-Apply** checkbox on, the preview regenerates whenever a
parameter value changes. A 300 ms debounce limits how often it regenerates.

### Seed

The seed input at the top of the right panel sets the seed for generation.
The random button next to it picks a random seed.

## View Modes

~~The **View Mode** dropdown switches between three rendering styles:~~ The
**View Mode** dropdown has eight modes: Shaded Mesh, Wireframe, Vertex Cloud
(the default, called Point Cloud below), Dense Cloud, Edge Cloud, Surface
Cloud, Crystal Preview, and Color Map (checked 2026-09-23 in
`src/mathviz/static/index.html`). [preview-ui.md](preview-ui.md) describes five
of them. This table describes three:

| Mode | Description |
|---|---|
| Shaded Mesh | Physically-based material with shadows and lighting |
| Wireframe | Edge-only display (blue lines) |
| Point Cloud | Individual vertex points (default) |

A **Point Size** slider (0.5–10) sets the size of point cloud dots in all views.

## Display Options

| Control | Default | Description |
|---|---|---|
| Show Bounding Box | on | Show the axis-aligned bounding box |
| Light Background | off | Switch the canvas background from dark to light |
| Lock Camera | off | Disable orbit controls to prevent accidental rotation |

## Camera Controls

- **Mouse drag** — rotate the view (orbit)
- **Scroll wheel** — zoom in/out
- **Right-click drag** — pan
- **Reset View** button (or **Home** key) — fit all geometry in view

Camera controls are disabled when **Lock Camera** is checked.

## Save / Load Snapshots

### Save

**Save** stores the current state:

- Generated geometry (GLB mesh + PLY point cloud)
- Metadata (generator name, parameters, seed, container, timestamp)
- Thumbnail (256×256 PNG captured from the canvas)

Snapshots are stored in the directory configured by `MATHVIZ_SNAPSHOTS_DIR`
(default: `~/.mathviz/snapshots`). See
[configuration.md](configuration.md#environment-variables) for details.

### Load

**Load** opens the snapshot gallery. Each snapshot card shows a thumbnail,
generator name, seed, and timestamp. Clicking a card restores the snapshot's
geometry, parameters, and view state. The gallery can also delete a snapshot,
after a confirmation dialog.

## Comparison Mode

The **Compare Mode** dropdown shows several panels side by side:

| Mode | Layout |
|---|---|
| Single View | Full canvas (default) |
| 2×2 Grid | 4 panels (A–D) |
| 3×3 Grid | 9 panels (A–I) |

### How it works

- Panel A (top-left) copies the current single-view geometry.
- The other panels use the same generator and parameters as panel A, with
  incremented seeds.
- Each panel has a collapsible overlay at the bottom for editing its seed and
  parameter overrides. Clicking the overlay summary expands or collapses it.
- Panels generate in parallel through the `/api/generate-batch` endpoint.
- All panels share the global view mode and point size settings.

### Exiting comparison mode

Switching back to Single View restores panel A's geometry to the main viewport
and frees the GPU resources of the comparison panels.

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| **Home** | Reset view (fit geometry) |
| **R** | Randomize parameters (ignored when typing in an input) |
| **Enter** | Apply parameters (when an input field is focused) |
| **Escape** | Close expanded panel overlays in comparison mode |

## Info Panel

The bottom-left info panel shows live statistics:

- Current generator name
- Vertex count / face count (mesh)
- Point count (point cloud)
- FPS (frames per second)

## Loading and Cancellation

A centered loading indicator shows the elapsed time during generation.
**Cancel** stops a long-running generation. In comparison mode, the progress
updates as each panel completes.

## Screenshot

The **Screenshot** button downloads the current canvas as
`mathviz-screenshot.png` at the current viewport resolution.
