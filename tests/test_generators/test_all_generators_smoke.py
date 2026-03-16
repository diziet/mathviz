"""Full-pipeline smoke test for every registered generator."""

import numpy as np
import pytest

from mathviz.core import (
    CoordSpace,
    list_generators,
)
from mathviz.core.container import Container, PlacementPolicy
from mathviz.pipeline.runner import run


def _generator_names() -> list[str]:
    """Discover all registered generator names for parametrization."""
    return [meta.name for meta in list_generators()]


class _AccessTrackingDict(dict):
    """Dict subclass that records which keys are accessed."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.accessed_keys: set[str] = set()

    def __getitem__(self, key: str) -> object:
        self.accessed_keys.add(key)
        return super().__getitem__(key)

    def get(self, key: str, default: object = None) -> object:
        """Track key access via .get() calls."""
        self.accessed_keys.add(key)
        return super().get(key, default)

    def pop(self, key: str, *args: object) -> object:
        """Track key access via .pop() calls."""
        self.accessed_keys.add(key)
        return super().pop(key, *args)


def _check_no_nan_inf(arr: np.ndarray, label: str) -> None:
    """Assert array has no NaN or inf values."""
    assert not np.any(np.isnan(arr)), f"{label} contains NaN values"
    assert not np.any(np.isinf(arr)), f"{label} contains inf values"


@pytest.mark.slow
@pytest.mark.parametrize("generator_name", _generator_names())
def test_generator_full_pipeline(generator_name: str) -> None:
    """Run a generator through the full pipeline and validate output."""
    container = Container.with_uniform_margin()
    placement = PlacementPolicy()

    result = run(
        generator=generator_name,
        seed=42,
        container=container,
        placement=placement,
    )

    obj = result.math_object

    # Must have at least one geometry type
    has_mesh = obj.mesh is not None
    has_point_cloud = obj.point_cloud is not None
    has_curves = obj.curves is not None and len(obj.curves) > 0
    assert has_mesh or has_point_cloud or has_curves, (
        f"{generator_name} produced no geometry (no mesh, point_cloud, or curves)"
    )

    # Validate mesh if present
    if obj.mesh is not None:
        assert obj.mesh.vertices.shape[0] > 0, f"{generator_name}: mesh has no vertices"
        assert obj.mesh.faces.shape[0] > 0, f"{generator_name}: mesh has no faces"
        _check_no_nan_inf(obj.mesh.vertices, f"{generator_name} mesh vertices")

    # Validate point cloud if present
    if obj.point_cloud is not None:
        assert obj.point_cloud.points.shape[0] > 0, (
            f"{generator_name}: point_cloud has no points"
        )
        _check_no_nan_inf(obj.point_cloud.points, f"{generator_name} point_cloud points")

    # Validate curves if present
    if obj.curves:
        for i, curve in enumerate(obj.curves):
            assert curve.points.shape[0] > 0, (
                f"{generator_name}: curve[{i}] has no points"
            )
            _check_no_nan_inf(curve.points, f"{generator_name} curve[{i}] points")

    # validate_or_raise must pass
    obj.validate_or_raise()

    # Coordinate space must be PHYSICAL after transform
    assert obj.coord_space == CoordSpace.PHYSICAL, (
        f"{generator_name}: expected PHYSICAL coord_space after transform, "
        f"got {obj.coord_space}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("generator_name", _generator_names())
def test_generator_default_params_consistency(generator_name: str) -> None:
    """Verify get_default_params() keys match what generate() actually reads."""
    from mathviz.core import get_generator

    gen_class = get_generator(generator_name)
    gen = gen_class()

    default_params = gen.get_default_params()
    if not default_params:
        return  # No params to check

    # Monkey-patch get_default_params to return a tracking dict.
    # Generators do `merged = self.get_default_params()` then read from merged,
    # so we must intercept at this level, not via the params argument.
    tracking = _AccessTrackingDict(default_params)
    gen.get_default_params = lambda: tracking  # type: ignore[assignment]
    gen.generate(seed=42)

    unused_keys = set(default_params.keys()) - tracking.accessed_keys
    assert not unused_keys, (
        f"{generator_name}: get_default_params() declares keys that generate() "
        f"never reads: {unused_keys}"
    )
