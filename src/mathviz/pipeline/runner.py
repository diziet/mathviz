"""Pipeline runner: chains stages with timing and validation at every boundary."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.engraving import EngravingProfile
from mathviz.core.generator import GeneratorBase, get_generator
from mathviz.core.math_object import MathObject
from mathviz.core.representation import RepresentationConfig
from mathviz.core.validator import ValidationResult, validate_engraving, validate_mesh
from mathviz.pipeline import representation_strategy, sampler, transformer
from mathviz.pipeline.mesh_exporter import export_mesh
from mathviz.pipeline.point_cloud_exporter import export_point_cloud
from mathviz.pipeline.sampler import SamplerConfig
from mathviz.pipeline.timer import PipelineTimer

logger = logging.getLogger(__name__)


_VALID_EXPORT_TYPES = frozenset({"mesh", "point_cloud"})


@dataclass
class ExportConfig:
    """Configuration for the export stage."""

    path: Path
    fmt: str | None = None
    export_type: str = "point_cloud"

    def __post_init__(self) -> None:
        """Validate export_type."""
        if self.export_type not in _VALID_EXPORT_TYPES:
            raise ValueError(
                f"export_type must be one of {sorted(_VALID_EXPORT_TYPES)}, "
                f"got {self.export_type!r}"
            )


@dataclass
class PipelineResult:
    """Result of a complete pipeline run."""

    math_object: MathObject
    validation: ValidationResult
    timings: dict[str, float] = field(default_factory=dict)
    export_path: Path | None = None


def run(
    generator: str | GeneratorBase,
    *,
    params: dict[str, Any] | None = None,
    seed: int = 42,
    resolution_kwargs: dict[str, Any] | None = None,
    container: Container,
    placement: PlacementPolicy,
    representation_config: RepresentationConfig | None = None,
    sampler_config: SamplerConfig | None = None,
    engraving_profile: EngravingProfile | None = None,
    export_config: ExportConfig | None = None,
) -> PipelineResult:
    """Execute the full or partial pipeline.

    Chains: generate -> represent -> transform -> sample -> validate -> export.
    Calls validate_or_raise() at every stage boundary.
    """
    timer = PipelineTimer()

    gen_instance = _resolve_generator(generator)

    # --- Generate ---
    with timer.stage("generate"):
        obj = gen_instance.generate(
            params=params,
            seed=seed,
            **(resolution_kwargs or {}),
        )
    obj.validate_or_raise()

    # --- Represent ---
    with timer.stage("represent"):
        rep_config = representation_config or representation_strategy.get_default(
            obj.generator_name
        )
        obj = representation_strategy.apply(obj, rep_config)
    obj.validate_or_raise()

    # --- Transform ---
    with timer.stage("transform"):
        obj = transformer.fit(obj, container, placement)
    # transformer.fit already calls validate_or_raise internally

    # --- Sample (optional) ---
    if sampler_config is not None:
        with timer.stage("sample"):
            obj = sampler.sample(obj, sampler_config)
        obj.validate_or_raise()

    # --- Validate ---
    with timer.stage("validate"):
        validation = _run_validation(obj, container, engraving_profile)

    # --- Export (optional) ---
    export_path: Path | None = None
    if export_config is not None:
        with timer.stage("export"):
            export_path = _run_export(obj, export_config)

    return PipelineResult(
        math_object=obj,
        validation=validation,
        timings=timer.timings,
        export_path=export_path,
    )


def _resolve_generator(generator: str | GeneratorBase) -> GeneratorBase:
    """Resolve a generator name or instance to a GeneratorBase instance."""
    if isinstance(generator, GeneratorBase):
        return generator
    gen_class = get_generator(generator)
    return gen_class()


def _run_validation(
    obj: MathObject,
    container: Container,
    engraving_profile: EngravingProfile | None,
) -> ValidationResult:
    """Run mesh and/or engraving validation on the final object."""
    result = ValidationResult()

    if obj.mesh is not None:
        mesh_result = validate_mesh(obj.mesh, container=container)
        result.checks.extend(mesh_result.checks)

    if obj.point_cloud is not None and engraving_profile is not None:
        engraving_result = validate_engraving(
            obj.point_cloud, engraving_profile, container=container
        )
        result.checks.extend(engraving_result.checks)
    elif obj.point_cloud is not None:
        logger.warning(
            "Point cloud present but no engraving profile provided; "
            "skipping engraving validation"
        )

    return result


def _run_export(obj: MathObject, config: ExportConfig) -> Path:
    """Run the export stage based on config."""
    if config.export_type == "mesh":
        return export_mesh(obj, config.path, fmt=config.fmt)
    return export_point_cloud(obj, config.path, fmt=config.fmt)
