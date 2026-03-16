# Preview UI Reference

Comprehensive guide to every feature and control in the MathViz interactive
3D preview. Launch it with `mathviz preview`.

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

The preview opens in a browser and consists of a full-screen Three.js canvas
with control panels on the left and right sides.

---

## Generator Browser

Open the generator browser with **Cmd+K** (macOS) or **Ctrl+K** (other
platforms). The browser is a full-screen modal that lets you explore all
available generators organized by category.

<!-- Screenshot: generator-browser.png — full-screen modal with category cards -->

### Category Grid

The initial view shows a grid of category cards. Each card displays the
category name, generator count, and a strip of thumbnail previews from
generators in that category. Click a category to browse its generators.

### Generator Grid

Inside a category, generators appear as cards with thumbnail previews,
names, and short descriptions. Click a generator to load it in the preview.
Use the **back arrow** or press **Backspace** to return to the category view.

### Search

A search input at the top of the browser filters generators by name across
all categories. Results update as you type. Press **Escape** to clear the
search, or press **Escape** again to close the browser.

### Keyboard Navigation

The generator browser supports full keyboard navigation:

| Key | Action |
|---|---|
| **Cmd+K** / **Ctrl+K** | Open / close the browser |
| **Arrow keys** | Navigate the grid (wraps at edges) |
| **Enter** | Activate the focused item |
| **Number keys (1–9)** | Jump to item by position (two-digit support for 10+) |
| **Backspace** | Go back to category view (when not in search) |
| **Escape** | Clear search → go back to categories → close browser |

Number keys show badges on each card indicating the shortcut digit.

### Quick Selector

A dropdown at the top of the right panel provides a quick way to switch
generators without opening the full browser. Type to filter the list.

---

## Parameters Panel

The left-column **Parameters** panel displays editable fields for the
selected generator's parameters, populated from the generator's schema.

<!-- Screenshot: parameters-panel.png — left-column parameter editor -->

### Parameter Fields

Each parameter renders as a text, number, or checkbox input with automatic
type detection. Fields show the parameter name, current value, and valid
range hint (min–max) where applicable.

### Editable Min/Max Ranges

Range hints displayed next to numeric parameters show the allowed bounds.
These ranges come from the generator schema and help guide valid input.

### Auto-Apply

Enable the **Auto-Apply** checkbox to regenerate automatically whenever a
parameter value changes. A 300 ms debounce prevents excessive regeneration
while typing.

### Randomize (Dice Button)

Click the **dice button** or press **R** to randomize all parameters within
their valid ranges. The keyboard shortcut is ignored when focus is on an
input, textarea, or select element.

### Enter to Apply

Press **Enter** while focused on any parameter input to trigger regeneration
immediately, regardless of the auto-apply setting.

### Apply and Reset

- **Apply** — regenerate with the current parameter values
- **Reset** — revert all parameters to the generator's defaults

### Seed

The seed input (top of the right panel) controls deterministic generation.
The default seed is 42. Click the **random button** next to it for a random
seed. Changing the seed and pressing Enter regenerates the preview.

---

## Resolution Controls

Generators with resolution parameters (such as `integration_steps` or
`grid_resolution`) display a dedicated **Resolution** subsection within the
parameters panel. Higher resolutions produce finer detail but take longer
to generate.

---

## Container Panel

The left-column **Container** panel controls the glass block dimensions and
margins used by the Transform pipeline stage. The panel is collapsible —
click its header to expand or collapse. Its collapsed state persists across
sessions via localStorage.

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
Click **Apply** to regenerate with the new dimensions, or **Reset** to
revert to defaults.

---

## View Modes

The **View Mode** dropdown switches between five rendering styles:

| Mode | Description | When to Use |
|---|---|---|
| **Point Cloud** | Individual vertex points (default) | Closest to the final engraved result |
| **Shaded Mesh** | Physically-based material with shadows | Inspecting surface shape and lighting |
| **Wireframe** | Edge-only display (blue lines) | Checking mesh topology and density |
| **Crystal Preview** | Simulated glass block with bloom effects | Previewing how the form looks engraved |
| **Color Map** | Vertex coloring by a computed metric | Analyzing geometric properties |

### Point Size

A **Point Size** slider (0.5–10) controls the dot size in all view modes
that display points.

### Crystal Preview Settings

When Crystal Preview mode is active, additional controls appear:

| Control | Range | Description |
|---|---|---|
| Glass Tint | Color picker | Tint color applied to the glass material |
| Bloom | 0–1 | Intensity of the glow/bloom post-processing effect |
| Point Brightness | 0–1 | Brightness of the point cloud inside the glass |
| LED Base | Checkbox | Toggle a simulated illumination base |
| LED Color | Color picker | Color of the LED base illumination |

### Color Map Settings

When Color Map mode is active, controls let you choose the coloring metric
and gradient:

**Metrics:**

| Metric | Description |
|---|---|
| Height (Z) | Colors vertices by their Z-axis position |
| Distance from Center | Colors by Euclidean distance from the origin |
| Curvature (curves) | Colors by local curvature (curve generators) |
| Velocity (curves) | Colors by parametric velocity (curve generators) |

**Gradients:** Viridis, Inferno, Coolwarm, Rainbow, or Custom. When Custom
is selected, start and end color pickers appear for defining your own
gradient.

---

## Camera Controls

| Input | Action |
|---|---|
| **Mouse drag** | Rotate the view (orbit) |
| **Scroll wheel** | Zoom in/out |
| **Right-click drag** | Pan |
| **Reset View** button / **Home** key | Fit all geometry in view |

Camera controls use damped orbit with a 0.08 damping factor for smooth
interaction.

### Camera Lock Modes

The **lock button** cycles through three modes:

| Mode | Icon | Behavior |
|---|---|---|
| **Render Lock** | Lock icon | Orbit/zoom disabled; pan still works. Use for consistent screenshots. |
| **Full Lock** | Double-lock icon | All camera interaction disabled. Cursor shows "not-allowed". |
| **Free** | Unlock icon | Default — full orbit, pan, and zoom. |

Click the lock button to cycle: Render Lock → Full Lock → Free → Render
Lock.

---

## Display Options

| Control | Default | Description |
|---|---|---|
| **Bounding Box** | on | Show the axis-aligned bounding box wireframe |
| **Axes** | off | Show colored coordinate axes (red = X, green = Y, blue = Z) |
| **Light Background** | off | Toggle between dark (#1a1a2e) and light canvas background |

---

## Stretch Controls

Per-axis stretch controls let you scale geometry along individual axes
without regenerating. Each axis (X, Y, Z) has a synchronized **slider** and
**numeric input** field.

| Control | Range | Default |
|---|---|---|
| Stretch X | 0.1–3.0 | 1.0 |
| Stretch Y | 0.1–3.0 | 1.0 |
| Stretch Z | 0.1–3.0 | 1.0 |

Click **Reset Scale** to return all axes to 1.0. Changes apply in real time
without regeneration — only the visual transform is affected.

---

## Point Cloud Density Slider

When in **Point Cloud** view mode, a **Density** slider appears (range
0.01–1.0, default 1.0). Dragging the slider thins the point cloud in real
time by uniformly subsampling the displayed points. This is useful for:

- Reducing visual clutter on dense point clouds
- Improving frame rate during interactive exploration
- Previewing how the form looks at lower point counts

The slider does not regenerate geometry — it filters the existing point
cloud on the GPU for instant feedback. The density setting is saved in
snapshots and restored on load.

---

## Turntable and Export

### Auto-Rotate

Enable the **Auto-Rotate** checkbox to start a continuous turntable
rotation. A speed slider (0.5x–5x) controls the rotation speed.

### Export

When turntable is enabled, export controls appear:

| Control | Options | Description |
|---|---|---|
| Format | GIF, WebM | Output animation format |
| Resolution | 1x, 2x | Capture resolution multiplier |
| Export button | — | Start capturing a full 360-degree rotation |

During export, a progress overlay shows the current frame (out of 360) and
a progress bar. The animation captures one frame per degree of rotation and
downloads the result automatically when complete.

---

## Compare Mode

The **Compare Mode** dropdown provides side-by-side viewing:

| Mode | Layout |
|---|---|
| **Single View** | Full canvas (default) |
| **2x2 Grid** | 4 panels labeled A–D |
| **3x3 Grid** | 9 panels labeled A–I |

### How It Works

- Panel A (top-left) copies the current single-view geometry and settings.
- Remaining panels generate with incremented seeds, using the same generator
  and parameters as panel A.
- Panels generate in parallel via the `/api/generate-batch` endpoint.
- All panels share the global view mode, point size, and density settings.

### Per-Panel Controls

Each panel has a collapsible overlay at the bottom showing its generator
name, seed, and parameter summary. Click the overlay to expand it and edit
the panel's seed or parameter overrides, then click **Apply** to regenerate
that panel individually.

### Exiting Compare Mode

Switching back to Single View restores panel A's geometry to the main
viewport and frees GPU resources for the other panels.

---

## Save/Load Snapshots

### Save

Click **Save** (bottom of canvas) to persist the complete UI state:

- Generated geometry (GLB mesh + PLY point cloud)
- Metadata (generator name, parameters, seed, container, timestamp)
- Thumbnail (256x256 PNG captured from the canvas)
- View settings (view mode, stretch, camera lock, background, point size,
  density, turntable speed, axes, bounding box)

Snapshots are stored in `MATHVIZ_SNAPSHOTS_DIR` (default:
`~/.mathviz/snapshots`). See
[configuration.md](configuration.md#environment-variables) for details.

### Load / Gallery

Click **Load** to open the snapshot gallery. Each snapshot card shows:

- Thumbnail preview
- Generator name
- Seed value
- Timestamp
- Parameter summary

Click a card to restore the snapshot's geometry, parameters, and all view
settings. Each card also has a **Delete** button with a confirmation dialog.
Click outside the gallery or press **Escape** to close it.

---

## Disk Cache

The preview server caches generated geometry to avoid redundant computation.

### Cache Indicator

A green **Cached** badge appears next to the generator name in the info
panel when the displayed geometry was served from cache rather than freshly
generated.

### Force Regenerate

Click the **Regenerate** button to bypass the cache and force a fresh
generation with the current parameters. This is useful after changing code
or when you want to verify generation is deterministic.

---

## Info Panel

The bottom-left info panel displays real-time statistics:

| Field | Description |
|---|---|
| Generator | Current generator name (with cache badge when applicable) |
| Vertices | Vertex count of the loaded mesh |
| Faces | Face count of the loaded mesh |
| Points | Point count of the point cloud |
| FPS | Frames per second of the 3D renderer |

---

## Loading and Cancellation

A centered loading overlay shows elapsed time during generation. Click
**Cancel** to abort a long-running generation. In comparison mode, progress
updates as each panel completes.

---

## Screenshot

Click the **Screenshot** button to download the current canvas as
`mathviz-screenshot.png` at the current viewport resolution.

---

## Keyboard Shortcuts

Complete reference of all keyboard shortcuts:

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
