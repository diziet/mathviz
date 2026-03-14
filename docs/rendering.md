# Rendering

MathViz can render generators to high-resolution PNG images for preview and
documentation purposes. Rendering requires the optional `[render]` dependency.

## Installation

Rendering uses PyVista for 3D visualization. Install it with:

```bash
pip install ".[render]"
```

This installs PyVista (>= 0.43.0) and its dependencies including VTK.

## 3D Rendering

The `render` command produces a high-resolution 3D PNG:

```bash
mathviz render lorenz -o lorenz.png
```

### Options

| Option | Type | Default | Description |
|---|---|---|---|
| `--output`, `-o` | path | (required) | Output PNG file path |
| `--param` | key=value | | Generator parameter (repeatable) |
| `--seed` | int | 42 | Random seed |
| `--width` | int | 1920 | Image width in pixels |
| `--height` | int | 1080 | Image height in pixels |

### Examples

```bash
# Default resolution (1920x1080)
mathviz render gyroid -o gyroid.png

# 4K resolution
mathviz render mandelbulb -o mandelbulb_4k.png --width 3840 --height 2160

# Custom parameters
mathviz render lorenz -o lorenz.png --param sigma=12 --seed 7
```

## 2D Projection Rendering

The `render-2d` command produces orthographic 2D projections:

```bash
mathviz render-2d lorenz -o lorenz_top.png --view top
```

### Projection Views

| View | Description |
|---|---|
| `top` | Looking down the Z axis (XY plane) |
| `front` | Looking along the Y axis (XZ plane) |
| `side` | Looking along the X axis (YZ plane) |
| `angle` | Isometric-style angled view |

### Examples

```bash
# Top-down view
mathviz render-2d gyroid -o gyroid_top.png --view top

# Front view
mathviz render-2d torus_knot -o knot_front.png --view front

# Side view
mathviz render-2d klein_bottle -o klein_side.png --view side

# Angled view at 4K
mathviz render-2d mandelbulb -o mandelbulb.png --view angle --width 3840 --height 2160
```

## Output Format

Both commands produce PNG images. The output path must end with `.png`.

The renderer runs the full pipeline (Generate → Represent → Transform) before
rendering, so rendered images reflect the same geometry that would be exported.

## Headless Rendering

PyVista supports headless rendering on servers without a display by using the
OSMesa or EGL backends. Set the environment variable before running:

```bash
export PYVISTA_OFF_SCREEN=true
mathviz render lorenz -o lorenz.png
```
