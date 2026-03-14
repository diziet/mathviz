"""Generate reference fixture files for end-to-end regression tests.

Run this script once to create reference outputs. Tests compare regenerated
outputs against these references. Each generator runs at low resolution
with seed=42 to keep fixture files small and generation fast.
"""

import json
import logging
from pathlib import Path

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.pipeline.runner import ExportConfig, run

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent
SEED = 42

# One generator per category with low-resolution settings
FIXTURE_SPECS: list[dict] = [
    {
        "name": "torus",
        "category": "parametric",
        "resolution_kwargs": {"grid_resolution": 16},
        "export_type": "mesh",
        "export_ext": ".stl",
    },
    {
        "name": "gyroid",
        "category": "implicit",
        "resolution_kwargs": {"voxel_resolution": 16},
        "export_type": "mesh",
        "export_ext": ".stl",
    },
    {
        "name": "lorenz",
        "category": "attractors",
        "resolution_kwargs": {"integration_steps": 2000},
        "export_type": "mesh",
        "export_ext": ".stl",
    },
    {
        "name": "mandelbulb",
        "category": "fractals",
        "resolution_kwargs": {"voxel_resolution": 16},
        "export_type": "mesh",
        "export_ext": ".stl",
    },
    {
        "name": "torus_knot",
        "category": "knots",
        "resolution_kwargs": {"curve_points": 64},
        "export_type": "mesh",
        "export_ext": ".stl",
    },
    {
        "name": "ulam_spiral",
        "category": "number_theory",
        "resolution_kwargs": {"num_points": 200},
        "export_type": "point_cloud",
        "export_ext": ".ply",
        "representation": RepresentationConfig(type=RepresentationType.WEIGHTED_CLOUD),
    },
    {
        "name": "lissajous_curve",
        "category": "curves",
        "resolution_kwargs": {"curve_points": 64},
        "export_type": "mesh",
        "export_ext": ".stl",
        "representation": RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=0.05,
        ),
    },
]


def _build_summary(result, spec: dict) -> dict:
    """Build a summary dict with counts and bounding box for a pipeline result."""
    obj = result.math_object
    summary: dict = {
        "generator_name": obj.generator_name,
        "category": obj.category,
        "seed": obj.seed,
        "parameters": obj.parameters,
    }

    if obj.mesh is not None:
        summary["vertex_count"] = len(obj.mesh.vertices)
        summary["face_count"] = len(obj.mesh.faces)
    if obj.point_cloud is not None:
        summary["point_count"] = len(obj.point_cloud.points)

    if obj.bounding_box is not None:
        summary["bounding_box"] = {
            "min_corner": list(obj.bounding_box.min_corner),
            "max_corner": list(obj.bounding_box.max_corner),
        }

    return summary


def generate_all() -> None:
    """Generate reference fixture files for all specs."""
    container = Container.with_uniform_margin()
    placement = PlacementPolicy()
    all_summaries: dict[str, dict] = {}

    for spec in FIXTURE_SPECS:
        name = spec["name"]
        logger.info("Generating fixture: %s", name)

        export_path = FIXTURES_DIR / f"{name}{spec['export_ext']}"
        export_config = ExportConfig(
            path=export_path,
            export_type=spec["export_type"],
        )

        rep_config = spec.get("representation")

        result = run(
            generator=name,
            seed=SEED,
            resolution_kwargs=spec["resolution_kwargs"],
            container=container,
            placement=placement,
            representation_config=rep_config,
            export_config=export_config,
        )

        all_summaries[name] = _build_summary(result, spec)
        logger.info("  -> %s written", export_path)

    summary_path = FIXTURES_DIR / "reference_summary.json"
    summary_path.write_text(
        json.dumps(all_summaries, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Reference summary written to %s", summary_path)


if __name__ == "__main__":
    generate_all()
