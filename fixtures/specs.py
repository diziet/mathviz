"""Shared fixture specs used by both the generator script and tests.

Single source of truth for resolution kwargs and representation configs
so that fixture generation and test comparison never drift apart.
"""

from typing import Any

from mathviz.core.representation import RepresentationConfig, RepresentationType

# Each entry maps a generator name to its low-resolution settings and
# optional explicit representation config. The generator script layers
# on export-specific fields (export_type, export_ext, category).
FIXTURE_SPECS: dict[str, dict[str, Any]] = {
    "torus": {
        "resolution_kwargs": {"grid_resolution": 16},
    },
    "gyroid": {
        "resolution_kwargs": {"voxel_resolution": 16},
    },
    "lorenz": {
        "resolution_kwargs": {"integration_steps": 2000},
    },
    "mandelbulb": {
        "resolution_kwargs": {"voxel_resolution": 16},
    },
    "torus_knot": {
        "resolution_kwargs": {"curve_points": 64},
    },
    "ulam_spiral": {
        "resolution_kwargs": {"num_points": 200},
        "representation": RepresentationConfig(
            type=RepresentationType.WEIGHTED_CLOUD,
        ),
    },
    "lissajous_curve": {
        "resolution_kwargs": {"curve_points": 64},
        "representation": RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=0.05,
        ),
    },
}

SEED = 42
