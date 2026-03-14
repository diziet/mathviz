"""Tests for the pipeline runner with timing and validation at stage boundaries."""

from typing import Any

import numpy as np
import pytest

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import GeneratorBase, clear_registry, register
from mathviz.core.math_object import MathObject, Mesh, PointCloud
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.core.validator import ValidationResult
from mathviz.pipeline.runner import PipelineResult, run
from mathviz.pipeline.sampler import SamplerConfig
from mathviz.pipeline.timer import PipelineTimer


# --- Helpers ---


def _make_cube_mesh() -> Mesh:
    """Create a simple cube mesh centered at origin."""
    vertices = np.array(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [2, 6, 7], [2, 7, 3],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ],
        dtype=np.int64,
    )
    return Mesh(vertices=vertices, faces=faces)


def _default_container() -> Container:
    """Create a default container."""
    return Container.with_uniform_margin()


def _default_placement() -> PlacementPolicy:
    """Create a default placement policy."""
    return PlacementPolicy()


# --- Placeholder generators ---


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Clear and suppress registry auto-discovery for each test."""
    clear_registry(suppress_discovery=True)


class _CubeGenerator(GeneratorBase):
    """Placeholder generator that produces a cube mesh."""

    name = "test_cube"
    category = "test"
    description = "Test cube generator"

    def get_default_params(self) -> dict[str, Any]:
        return {"size": 1.0}

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        return MathObject(
            mesh=_make_cube_mesh(),
            generator_name="test_cube",
            category="test",
            parameters=params or self.get_default_params(),
            seed=seed,
        )


class _InvalidGenerator(GeneratorBase):
    """Generator that produces invalid geometry (NaN vertices)."""

    name = "test_invalid"
    category = "test"
    description = "Generator that produces invalid geometry"

    def get_default_params(self) -> dict[str, Any]:
        return {}

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        vertices = np.array(
            [[float("nan"), 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        return MathObject(
            mesh=Mesh(vertices=vertices, faces=faces),
            generator_name="test_invalid",
        )


class _EmptyGenerator(GeneratorBase):
    """Generator that produces a MathObject with no geometry."""

    name = "test_empty"
    category = "test"
    description = "Generator that produces empty geometry"

    def get_default_params(self) -> dict[str, Any]:
        return {}

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        return MathObject(generator_name="test_empty")


# --- PipelineTimer tests ---


class TestPipelineTimer:
    """Test PipelineTimer context manager."""

    def test_records_stage_timing(self) -> None:
        """Timer records wall-clock time for each stage."""
        timer = PipelineTimer()
        with timer.stage("generate"):
            _ = sum(range(1000))
        timings = timer.timings
        assert "generate" in timings
        assert timings["generate"] >= 0.0

    def test_multiple_stages(self) -> None:
        """Timer records timing for multiple stages independently."""
        timer = PipelineTimer()
        with timer.stage("a"):
            pass
        with timer.stage("b"):
            pass
        timings = timer.timings
        assert "a" in timings
        assert "b" in timings

    def test_timings_returns_copy(self) -> None:
        """Timings property returns a copy, not a reference."""
        timer = PipelineTimer()
        with timer.stage("x"):
            pass
        t1 = timer.timings
        t2 = timer.timings
        assert t1 is not t2
        assert t1 == t2


# --- Pipeline runner tests ---


class TestPipelineRunner:
    """Test the pipeline runner."""

    def test_full_pipeline_produces_valid_output_and_timing(self) -> None:
        """Full pipeline with all stages produces valid output and timing."""
        register(_CubeGenerator)
        result = run(
            "test_cube",
            container=_default_container(),
            placement=_default_placement(),
            representation_config=RepresentationConfig(
                type=RepresentationType.SURFACE_SHELL,
            ),
            sampler_config=SamplerConfig(num_points=100, seed=42),
        )
        assert isinstance(result, PipelineResult)
        assert result.math_object is not None
        assert result.math_object.point_cloud is not None
        assert isinstance(result.validation, ValidationResult)
        # Timing should have entries for all stages that ran
        for stage in ("generate", "represent", "transform", "sample", "validate"):
            assert stage in result.timings, f"Missing timing for stage: {stage}"
            assert result.timings[stage] >= 0.0

    def test_pipeline_skips_sampling_when_no_config(self) -> None:
        """Pipeline with no sampling config skips sampling stage."""
        register(_CubeGenerator)
        result = run(
            "test_cube",
            container=_default_container(),
            placement=_default_placement(),
            representation_config=RepresentationConfig(
                type=RepresentationType.SURFACE_SHELL,
            ),
        )
        assert "sample" not in result.timings
        # Should still have mesh from generate
        assert result.math_object.mesh is not None

    def test_validate_or_raise_fires_between_stages(self) -> None:
        """Invalid geometry from generator fails before reaching transformer."""
        register(_EmptyGenerator)
        with pytest.raises(ValueError, match="no geometry"):
            run(
                "test_empty",
                container=_default_container(),
                placement=_default_placement(),
            )

    def test_timing_has_entries_for_ran_stages_only(self) -> None:
        """Timing dict has entries for every stage that ran, none for skipped."""
        register(_CubeGenerator)
        result = run(
            "test_cube",
            container=_default_container(),
            placement=_default_placement(),
            representation_config=RepresentationConfig(
                type=RepresentationType.SURFACE_SHELL,
            ),
        )
        # Without sampler_config, sample should not appear
        assert "sample" not in result.timings
        # Without export_config, export should not appear
        assert "export" not in result.timings
        # These should always be present
        assert "generate" in result.timings
        assert "represent" in result.timings
        assert "transform" in result.timings
        assert "validate" in result.timings

    def test_pipeline_result_includes_math_object_and_validation(self) -> None:
        """Pipeline result includes final MathObject and ValidationResult."""
        register(_CubeGenerator)
        result = run(
            "test_cube",
            container=_default_container(),
            placement=_default_placement(),
        )
        assert isinstance(result.math_object, MathObject)
        assert isinstance(result.validation, ValidationResult)

    def test_pipeline_accepts_generator_instance(self) -> None:
        """Pipeline accepts a GeneratorBase instance, not just a name."""
        gen = _CubeGenerator()
        result = run(
            gen,
            container=_default_container(),
            placement=_default_placement(),
        )
        assert result.math_object.generator_name == "test_cube"

    def test_pipeline_uses_default_representation_when_none(self) -> None:
        """Pipeline uses get_default() when no representation_config given."""
        register(_CubeGenerator)
        # No representation_config — should use default (surface_shell fallback)
        result = run(
            "test_cube",
            container=_default_container(),
            placement=_default_placement(),
        )
        assert result.math_object is not None
        assert "represent" in result.timings
