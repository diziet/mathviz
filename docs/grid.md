# Grid Layout

MathViz manages multi-block installations through a grid manifest. The grid
tracks which generator preset is assigned to each physical block position and
its export status.

## Manifest Format

The grid manifest is stored as a TOML file (default: `grid.toml`). It contains
metadata about the grid dimensions and a section for each block.

```toml
rows = 10
cols = 10

[blocks."0,0"]
preset = "lorenz"
status = "assigned"
config_path = ""

[blocks."0,1"]
preset = "gyroid"
status = "exported"
config_path = "configs/gyroid_custom.toml"
```

## Block Status

Each block has one of four statuses:

| Status | Description |
|---|---|
| `empty` | No preset assigned |
| `assigned` | Preset assigned, not yet exported |
| `exported` | Successfully exported |
| `error` | Export failed |

## CLI Commands

### Create a Grid

```bash
mathviz grid init <rows> <cols> [--path grid.toml]
```

Creates a new empty grid manifest with all blocks in `empty` status.

```bash
mathviz grid init 10 10
mathviz grid init 16 16 --path installation.toml
```

### View the Grid

```bash
mathviz grid show [--path grid.toml]
```

Displays an ASCII table showing all blocks with color-coded status. Use
`--json` for machine-readable output.

### Assign Presets

```bash
mathviz grid assign <row> <col> <preset> [--config path] [--path grid.toml]
```

Assigns a generator preset name to a grid position. Optionally link a
per-block config file.

```bash
mathviz grid assign 0 0 lorenz
mathviz grid assign 0 1 gyroid --config configs/gyroid_custom.toml
mathviz grid assign 1 0 mandelbulb
```

### Check Block Status

```bash
mathviz grid status <row> <col> [--set <status>] [--path grid.toml]
```

View or update a block's status.

```bash
# View status
mathviz grid status 0 0

# Mark as exported
mathviz grid status 0 0 --set exported

# Reset to assigned (for re-export)
mathviz grid status 0 0 --set assigned
```

### View Neighbors

```bash
mathviz grid neighbors <row> <col> [--path grid.toml]
```

Shows the 8 surrounding blocks for a position. Useful for checking visual
coherence between adjacent blocks.

```bash
mathviz grid neighbors 5 5
```

### Grid Summary

```bash
mathviz grid summary [--path grid.toml]
```

Shows counts of blocks by status.

```bash
mathviz grid summary
# Output:
#   empty: 90
#   assigned: 8
#   exported: 2
#   total: 100
```

### Batch Export

```bash
mathviz grid export-all [--path grid.toml] [--output-dir export] [--format ply]
```

Exports all assigned blocks sequentially through the full pipeline. Each
block's status is updated to `exported` on success or `error` on failure.

```bash
# Export all assigned blocks to PLY files
mathviz grid export-all --output-dir output/

# Export with specific format
mathviz grid export-all --format stl --output-dir output/
```

Output files are named `block_<row>_<col>.<format>`.

## Workflow Example

A typical workflow for managing an installation:

```bash
# 1. Create the grid
mathviz grid init 10 10

# 2. Assign generators to positions
mathviz grid assign 0 0 lorenz
mathviz grid assign 0 1 rossler
mathviz grid assign 0 2 chen
mathviz grid assign 1 0 gyroid
mathviz grid assign 1 1 schwarz_d
mathviz grid assign 1 2 mandelbulb

# 3. Preview assignments
mathviz grid show

# 4. Check summary
mathviz grid summary

# 5. Export all blocks
mathviz grid export-all --output-dir blocks/

# 6. Check for errors
mathviz grid summary

# 7. Re-export failed blocks after fixing
mathviz grid status 1 2 --set assigned
mathviz grid export-all --output-dir blocks/
```
