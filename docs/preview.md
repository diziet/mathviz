# Preview UI

MathViz includes an interactive 3D preview served by FastAPI with a Three.js
viewer. Launch it with `mathviz preview`.

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

A searchable dropdown at the top of the right panel lets you switch generators
without restarting the server. Type to filter the list by name. Selecting a
generator loads its parameter schema and regenerates the preview.

## Container / Dimensions Editor

The left-column **Container** panel controls the glass block dimensions and
margins used by the Transform pipeline stage.

| Control | Default | Description |
|---|---|---|
| Width (W) | 100 mm | Block width |
| Height (H) | 100 mm | Block height |
| Depth (D) | 100 mm | Block depth |
| Margin X | 5 mm | Horizontal margin |
| Margin Y | 5 mm | Vertical margin |
| Margin Z | 5 mm | Depth margin |
| Uniform Margin | on | Lock all margins to the same value |

The panel shows the calculated usable volume. Click **Apply** to regenerate
with the new dimensions, or **Reset** to revert to defaults.

## Parameter Editor

The left-column **Parameter** panel displays editable fields for the selected
generator's parameters, populated from the generator's schema. Fields support
text, number, and checkbox inputs with type detection.

- **Apply** — regenerate with current parameter values
- **Reset** — revert parameters to generator defaults
- **Randomize** (dice button) — randomize all parameters within their valid
  ranges. Also available via the **R** keyboard shortcut.
- **Resolution fields** — generators with resolution parameters (e.g.
  `integration_steps`, `grid_resolution`) show an additional section.

### Auto-Apply

Enable the **Auto-Apply** checkbox to regenerate automatically whenever a
parameter value changes. A 300 ms debounce prevents excessive regeneration.

### Seed

The seed input (top of the right panel) controls deterministic generation.
Click the random button next to it for a random seed.

## View Modes

The **View Mode** dropdown switches between three rendering styles:

| Mode | Description |
|---|---|
| Shaded Mesh | Physically-based material with shadows and lighting |
| Wireframe | Edge-only display (blue lines) |
| Point Cloud | Individual vertex points (default) |

A **Point Size** slider (0.5–10) controls point cloud dot size across all views.

## Display Options

| Control | Default | Description |
|---|---|---|
| Show Bounding Box | on | Render the axis-aligned bounding box helper |
| Light Background | off | Toggle between dark and light canvas background |
| Lock Camera | off | Disable orbit controls to prevent accidental rotation |

## Camera Controls

- **Mouse drag** — rotate the view (orbit)
- **Scroll wheel** — zoom in/out
- **Right-click drag** — pan
- **Reset View** button (or **Home** key) — fit all geometry in view

Camera controls are disabled when **Lock Camera** is checked.

## Save / Load Snapshots

### Save

Click **Save** to persist the current state:

- Generated geometry (GLB mesh + PLY point cloud)
- Metadata (generator name, parameters, seed, container, timestamp)
- Thumbnail (256×256 PNG captured from the canvas)

Snapshots are stored in the directory configured by `MATHVIZ_SNAPSHOTS_DIR`
(default: `~/.mathviz/snapshots`). See
[configuration.md](configuration.md#environment-variables) for details.

### Load

Click **Load** to open the snapshot gallery. Each snapshot card shows a
thumbnail, generator name, seed, and timestamp. Click a card to restore the
snapshot's geometry, parameters, and view state. Snapshots can be deleted from
the gallery with a confirmation dialog.

## Comparison Mode

The **Compare Mode** dropdown offers side-by-side viewing:

| Mode | Layout |
|---|---|
| Single View | Full canvas (default) |
| 2×2 Grid | 4 panels (A–D) |
| 3×3 Grid | 9 panels (A–I) |

### How it works

- Panel A (top-left) copies the current single-view geometry.
- Remaining panels generate with incremented seeds using the same generator
  and parameters as panel A.
- Each panel has a collapsible overlay at the bottom for editing its seed and
  parameter overrides. Click the overlay summary to expand/collapse.
- Panels generate in parallel via the `/api/generate-batch` endpoint.
- All panels share the global view mode and point size settings.

### Exiting comparison mode

Switching back to Single View restores panel A's geometry to the main viewport
and frees GPU resources for other panels.

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| **Home** | Reset view (fit geometry) |
| **R** | Randomize parameters (ignored when typing in an input) |
| **Enter** | Apply parameters (when an input field is focused) |
| **Escape** | Close expanded panel overlays in comparison mode |

## Info Panel

The bottom-left info panel displays real-time statistics:

- Current generator name
- Vertex count / face count (mesh)
- Point count (point cloud)
- FPS (frames per second)

## Loading and Cancellation

A centered loading indicator shows elapsed time during generation. Click
**Cancel** to abort a long-running generation. In comparison mode, progress
updates as each panel completes.

## Screenshot

Click the **Screenshot** button to download the current canvas as
`mathviz-screenshot.png` at the current viewport resolution.
