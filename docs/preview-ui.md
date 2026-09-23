# Preview UI Reference

This page describes every feature and control of the MathViz interactive 3D
preview. `mathviz preview` starts it.

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

The preview opens in a browser. It shows a full-screen Three.js canvas with
control panels on the left and right.

---

## Generator Browser

**Cmd+K** (macOS) or **Ctrl+K** (other platforms) opens the generator browser.
The browser is a full-screen modal that lists all generators by category.

<!-- Screenshot: generator-browser.png — full-screen modal with category cards -->

### Category Grid

The first view is a grid of category cards. Each card shows the category name,
the generator count, and a row of thumbnails from generators in that category.
Clicking a category opens its generators.

### Generator Grid

Inside a category, each generator is a card with a thumbnail, its name, and a
short description. Clicking a generator loads it in the preview. The **back
arrow** or **Backspace** returns to the category view.

### Search

A search input at the top of the browser filters generators by name across
all categories. The results update as you type. **Escape** clears the search,
and a second **Escape** closes the browser.

### Keyboard Navigation

The generator browser can be used entirely from the keyboard:

| Key | Action |
|---|---|
| **Cmd+K** / **Ctrl+K** | Open / close the browser |
| **Arrow keys** | Navigate the grid (wraps at edges) |
| **Enter** | Activate the focused item |
| **Number keys (1–9)** | Jump to an item by position (two digits for 10 and above) |
| **Backspace** | Go back to category view (when not in search) |
| **Escape** | Clear search → go back to categories → close browser |

Each card shows a badge with its number key.

### Quick Selector

A dropdown at the top of the right panel switches the generator without
opening the browser. Typing filters the list.

---

## Parameters Panel

The left-column **Parameters** panel shows an editable field for each
parameter in the selected generator's schema.

<!-- Screenshot: parameters-panel.png — left-column parameter editor -->

### Parameter Fields

Each parameter is a text, number, or checkbox input, chosen from the
parameter's detected type. A field shows the parameter name, the current
value, and, where one exists, the valid range (min–max).

### Editable Min/Max Ranges

Numeric parameters show their allowed bounds next to the field. The bounds
come from the generator schema.

### Auto-Apply

With the **Auto-Apply** checkbox on, the preview regenerates whenever a
parameter value changes. A 300 ms debounce limits how often it regenerates
while you type.

### Randomize (Dice Button)

The **dice button** or the **R** key sets every parameter to a random value
within its valid range. The **R** key does nothing while an input, textarea,
or select element has focus.

### Enter to Apply

**Enter** in any parameter input regenerates immediately, whether auto-apply
is on or off.

### Apply and Reset

- **Apply** — regenerate with the current parameter values
- **Reset** — restore all parameters to the generator's defaults

### Seed

The seed input at the top of the right panel sets the seed for generation.
The default seed is 42. The **random button** next to it picks a random seed.
Changing the seed and pressing Enter regenerates the preview.

---

## Resolution Controls

Generators with resolution parameters (such as `integration_steps` or
`grid_resolution`) show a **Resolution** subsection in the parameters panel.
Higher resolutions produce finer detail but take longer to generate.

---

## Container Panel

The left-column **Container** panel sets the glass block dimensions and
margins that the Transform pipeline stage uses. The panel is collapsible:
clicking its header expands or collapses it. The collapsed state is stored in
localStorage and restored in later sessions.

<!-- Screenshot: container-panel.png — dimensions and margin controls -->

| Control | Default | Description |
|---|---|---|
| Width (W) | 100 mm | Block width |
| Height (H) | 100 mm | Block height |
| Depth (D) | 100 mm | Block depth |
| Margin X | 5 mm | Horizontal inset |
| Margin Y | 5 mm | Vertical inset |
| Margin Z | 5 mm | Depth inset |
| Uniform Margin | on | Lock all three margins to the same value |

The panel shows the calculated usable volume (dimensions minus margins).
**Apply** regenerates with the new dimensions. **Reset** restores the
defaults.

---

## View Modes

The **View Mode** dropdown in `src/mathviz/static/index.html`
has <!-- fact:preview-view-mode-count -->8<!-- /fact --> modes: <!-- fact:preview-view-modes -->Shaded Mesh, Wireframe, Vertex Cloud, Dense Cloud, Edge Cloud, Surface Cloud, Crystal Preview, Color Map<!-- /fact -->.
Vertex Cloud is the default, called Point Cloud below. The table describes five
of them:

| Mode | Description | When to Use |
|---|---|---|
| **Point Cloud** | Individual vertex points (default) | Closest to the final engraved result |
| **Shaded Mesh** | Physically-based material with shadows | Inspecting surface shape and lighting |
| **Wireframe** | Edge-only display (blue lines) | Checking mesh topology and density |
| **Crystal Preview** | Simulated glass block with bloom effects | Previewing how the form looks engraved |
| **Color Map** | Vertex coloring by a computed metric | Analyzing geometric properties |

### Point Size

A **Point Size** slider (0.5–10) sets the dot size in every view mode that
shows points.

### Crystal Preview Settings

In Crystal Preview mode, these extra controls appear:

| Control | Range | Description |
|---|---|---|
| Glass Tint | Color picker | Tint color applied to the glass material |
| Bloom | 0–1 | Strength of the glow (bloom) post-processing effect |
| Point Brightness | 0–1 | Brightness of the point cloud inside the glass |
| LED Base | Checkbox | Turn a simulated illumination base on or off |
| LED Color | Color picker | Color of the LED base illumination |

### Color Map Settings

In Color Map mode, controls set the coloring metric and the gradient:

**Metrics:**

| Metric | Description |
|---|---|
| Height (Z) | Colors vertices by their Z-axis position |
| Distance from Center | Colors by Euclidean distance from the origin |
| Curvature (curves) | Colors by local curvature (curve generators) |
| Velocity (curves) | Colors by parametric velocity (curve generators) |

**Gradients:** Viridis, Inferno, Coolwarm, Rainbow, or Custom. Selecting
Custom shows start and end color pickers that define the gradient.

---

## Camera Controls

| Input | Action |
|---|---|
| **Mouse drag** | Rotate the view (orbit) |
| **Scroll wheel** | Zoom in/out |
| **Right-click drag** | Pan |
| **Reset View** button / **Home** key | Fit all geometry in view |

The orbit controls are damped, with a damping factor of 0.08.

### Camera Lock Modes

The **lock button** cycles through three modes:

| Mode | Icon | Behavior |
|---|---|---|
| **Render Lock** | Lock icon | Orbit and zoom are disabled; pan still works. Use it for consistent screenshots. |
| **Full Lock** | Double-lock icon | All camera interaction disabled. Cursor shows "not-allowed". |
| **Free** | Unlock icon | Default — full orbit, pan, and zoom. |

Each click of the lock button moves to the next mode: Render Lock → Full
Lock → Free → Render Lock.

---

## Display Options

| Control | Default | Description |
|---|---|---|
| **Bounding Box** | on | Show the axis-aligned bounding box wireframe |
| **Axes** | off | Show colored coordinate axes (red = X, green = Y, blue = Z) |
| **Light Background** | off | Switch the canvas background from dark (#1a1a2e) to light |

---

## Stretch Controls

Stretch controls scale the geometry along each axis without regenerating it.
Each axis (X, Y, Z) has a **slider** and a **numeric input** that stay in
sync.

| Control | Range | Default |
|---|---|---|
| Stretch X | 0.1–3.0 | 1.0 |
| Stretch Y | 0.1–3.0 | 1.0 |
| Stretch Z | 0.1–3.0 | 1.0 |

**Reset Scale** sets all axes back to 1.0. Changes apply immediately and
affect only the displayed transform; the geometry is not regenerated.

---

## Point Cloud Density Slider

In **Point Cloud** view mode, a **Density** slider appears (range 0.01–1.0,
default 1.0). Moving the slider thins the displayed point cloud immediately by
subsampling it uniformly. Use it for:

- Reducing visual clutter on dense point clouds
- Raising the frame rate while you explore
- Previewing how the form looks at lower point counts

The slider does not regenerate geometry. It filters the existing point cloud
on the GPU, so the change shows at once. Snapshots save the density setting
and restore it on load.

---

## Turntable and Export

### Auto-Rotate

The **Auto-Rotate** checkbox starts a continuous turntable rotation. A speed
slider (0.5x–5x) sets the rotation speed.

### Export

When the turntable is on, export controls appear:

| Control | Options | Description |
|---|---|---|
| Format | GIF, WebM | Output animation format |
| Resolution | 1x, 2x | Capture resolution multiplier |
| Export button | — | Start capturing a full 360-degree rotation |

During export, a progress overlay shows the current frame (out of 360) and
a progress bar. The export captures one frame per degree of rotation and
downloads the file when it finishes.

---

## Compare Mode

The **Compare Mode** dropdown shows several panels side by side:

| Mode | Layout |
|---|---|
| **Single View** | Full canvas (default) |
| **2x2 Grid** | 4 panels labeled A–D |
| **3x3 Grid** | 9 panels labeled A–I |

### How It Works

- Panel A (top-left) copies the current single-view geometry and settings.
- The other panels use the same generator and parameters as panel A, with
  incremented seeds.
- Panels generate in parallel through the `/api/generate-batch` endpoint.
- All panels share the global view mode, point size, and density settings.

### Per-Panel Controls

Each panel has a collapsible overlay at the bottom showing its generator
name, seed, and parameter summary. Clicking the overlay expands it for editing
the panel's seed or parameter overrides. **Apply** then regenerates only that
panel.

### Exiting Compare Mode

Switching back to Single View restores panel A's geometry to the main
viewport and frees the GPU resources of the comparison panels.

---

## Save/Load Snapshots

### Save

**Save** (at the bottom of the canvas) stores the complete UI state:

- Generated geometry (GLB mesh + PLY point cloud)
- Metadata (generator name, parameters, seed, container, timestamp)
- Thumbnail (256x256 PNG captured from the canvas)
- View settings (view mode, stretch, camera lock, background, point size,
  density, turntable speed, axes, bounding box)

Snapshots are stored in `MATHVIZ_SNAPSHOTS_DIR` (default:
`~/.mathviz/snapshots`). See
[configuration.md](configuration.md#environment-variables) for details.

### Load / Gallery

**Load** opens the snapshot gallery. Each snapshot card shows:

- Thumbnail preview
- Generator name
- Seed value
- Timestamp
- Parameter summary

Clicking a card restores the snapshot's geometry, parameters, and all view
settings. Each card also has a **Delete** button, which asks for
confirmation. Clicking outside the gallery or pressing **Escape** closes it.

---

## Disk Cache

The preview server caches generated geometry and does not recompute geometry
that is already in the cache.

### Cache Indicator

A green **Cached** badge appears next to the generator name in the info
panel when the displayed geometry came from the cache instead of a new
generation.

### Force Regenerate

The **Regenerate** button skips the cache and generates again with the
current parameters. Use it after a code change, or to check that generation
is deterministic.

---

## Info Panel

The bottom-left info panel shows live statistics:

| Field | Description |
|---|---|
| Generator | Current generator name (with cache badge when applicable) |
| Vertices | Vertex count of the loaded mesh |
| Faces | Face count of the loaded mesh |
| Points | Point count of the point cloud |
| FPS | Frames per second of the 3D renderer |

---

## Loading and Cancellation

A centered loading overlay shows the elapsed time during generation.
**Cancel** stops a long-running generation. In comparison mode, the progress
updates as each panel completes.

---

## Screenshot

The **Screenshot** button downloads the current canvas as
`mathviz-screenshot.png` at the current viewport resolution.

---

## Keyboard Shortcuts

All keyboard shortcuts:

| Shortcut | Context | Action |
|---|---|---|
| **Cmd+K** / **Ctrl+K** | Global | Open / close the generator browser |
| **Home** | Global | Reset view (fit geometry in viewport) |
| **R** | Global (not in input) | Randomize all parameters |
| **Enter** | Input field focused | Apply parameters / trigger regeneration |
| **Escape** | Generator browser | Clear search → back to categories → close |
| **Escape** | Compare mode | Collapse all panel overlays |
| **Arrow Up/Down/Left/Right** | Generator browser | Navigate the grid |
| **Backspace** | Generator browser (not in search) | Go back to category view |
| **1–9** | Generator browser (no search) | Select item by position number |
| **Enter** | Generator browser | Activate focused item |
