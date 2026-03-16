"""Pipeline runner: chains stages with timing and validation at every boundary."""

import logging
import threading
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.engraving import EngravingProfile
from mathviz.core.generator import GeneratorBase, get_generator
from mathviz.core.math_object import MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.core.validator import ValidationResult, validate_engraving, validate_mesh
from mathviz.pipeline import representation_strategy, sampler, transformer
from mathviz.pipeline.dense_sampling import (
    SamplingMode,
    apply_edge_sampling,
    apply_post_transform_sampling,
    apply_resolution_scaled_sampling,
)
from mathviz.pipeline.mesh_exporter import export_mesh
from mathviz.pipeline.point_cloud_exporter import export_point_cloud
from mathviz.pipeline.sampler import SamplerConfig
from mathviz.pipeline.timer import PipelineTimer

logger = logging.getLogger(__name__)


_VALID_EXPORT_TYPES = frozenset({"auto", "mesh", "point_cloud"})

# File extensions that unambiguously map to one exporter.
_MESH_ONLY_EXTENSIONS = frozenset({".stl", ".obj"})
_CLOUD_ONLY_EXTENSIONS = frozenset({".xyz", ".pcd"})


@dataclass
class ExportConfig:
    """Configuration for the export stage."""

    path: Path
    fmt: str | None = None
    export_type: str = "auto"

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


def _check_cancelled(cancel_event: threading.Event | None) -> None:
    """Raise CancelledError if the cancel event is set."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError("Generation cancelled")


def _apply_sampling(
    obj: MathObject,
    mode: SamplingMode,
    timer: PipelineTimer,
    *,
    max_samples: int | None,
    resolution_kwargs: dict[str, Any] | None,
    gen_instance: GeneratorBase,
) -> MathObject:
    """Run the appropriate post-transform sampling and validate."""
    kwargs: dict[str, Any] = {}
    if max_samples is not None:
        kwargs["max_samples"] = max_samples

    if mode == "resolution_scaled":
        kwargs["resolution_kwargs"] = resolution_kwargs or {}
        kwargs["default_resolution"] = gen_instance.get_default_resolution()
        sample_fn = apply_resolution_scaled_sampling
    elif mode == "edge":
        sample_fn = apply_edge_sampling
    else:
        sample_fn = apply_post_transform_sampling

    with timer.stage("dense_sample"):
        obj = sample_fn(obj, **kwargs)
    obj.validate_or_raise()
    return obj


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
    cancel_event: threading.Event | None = None,
    sampling_mode: SamplingMode = "default",
    max_samples: int | None = None,
) -> PipelineResult:
    """Execute the full or partial pipeline.

    Chains: generate -> represent -> transform -> sample -> validate -> export.
    Calls validate_or_raise() at every stage boundary.

    If *cancel_event* is provided, it is checked between stages. Cancellation
    is cooperative: a long-running stage (e.g. generate) will not be interrupted
    mid-execution — the event is only checked at stage boundaries.

    The *sampling_mode* parameter selects the post-transform sampling strategy:
    - ``"default"``: no post-transform sampling
    - ``"post_transform"``: dense surface + edge combined sampling
    - ``"edge"``: edge-only wireframe skeleton
    - ``"resolution_scaled"``: density scaled by (resolution / default)²
    """
    timer = PipelineTimer()

    gen_instance = _resolve_generator(generator)

    # --- Validate param keys ---
    gen_instance.validate_param_keys(params)

    # --- Generate ---
    _check_cancelled(cancel_event)
    with timer.stage("generate"):
        obj = gen_instance.generate(
            params=params,
            seed=seed,
            **(resolution_kwargs or {}),
        )
    obj.validate_or_raise()

    # --- Represent ---
    _check_cancelled(cancel_event)
    _needs_mesh = sampling_mode != "default"
    with timer.stage("represent"):
        if _needs_mesh and obj.mesh is not None:
            if representation_config is not None:
                logger.warning(
                    "%s overrides representation_config "
                    "(%s) with SURFACE_SHELL",
                    sampling_mode,
                    representation_config.type.value,
                )
            rep_config = RepresentationConfig(
                type=RepresentationType.SURFACE_SHELL,
            )
        elif _needs_mesh and obj.mesh is None:
            logger.warning(
                "%s requested but %s has no mesh; "
                "falling back to normal pipeline",
                sampling_mode,
                obj.generator_name or "object",
            )
            rep_config = representation_config or representation_strategy.get_default(
                obj.generator_name, obj=obj
            )
        else:
            rep_config = representation_config or representation_strategy.get_default(
                obj.generator_name, obj=obj
            )
        obj = representation_strategy.apply(obj, rep_config)
    obj.validate_or_raise()

    # --- Transform ---
    _check_cancelled(cancel_event)
    with timer.stage("transform"):
        obj = transformer.fit(obj, container, placement)
    # transformer.fit already calls validate_or_raise internally

    # --- Post-transform sampling (dense / edge / resolution-scaled) ---
    if sampling_mode != "default" and obj.mesh is not None:
        _check_cancelled(cancel_event)
        obj = _apply_sampling(
            obj, sampling_mode, timer,
            max_samples=max_samples,
            resolution_kwargs=resolution_kwargs,
            gen_instance=gen_instance,
        )

    # --- Sample (optional) ---
    if sampler_config is not None:
        _check_cancelled(cancel_event)
        with timer.stage("sample"):
            obj = sampler.sample(obj, sampler_config)
        obj.validate_or_raise()

    # --- Validate ---
    _check_cancelled(cancel_event)
    with timer.stage("validate"):
        validation = _run_validation(obj, container, engraving_profile)

    # --- Export (optional) ---
    export_path: Path | None = None
    if export_config is not None:
        _check_cancelled(cancel_event)
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
    return gen_class.create(resolved_name=generator)


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
    """Run the export stage, auto-detecting geometry type when export_type is 'auto'."""
    export_type = config.export_type
    if export_type == "auto":
        export_type = _detect_export_type(obj, config.path)
    if export_type == "mesh":
        return export_mesh(obj, config.path, fmt=config.fmt)
    return export_point_cloud(obj, config.path, fmt=config.fmt)


def _detect_export_type(obj: MathObject, path: Path) -> str:
    """Decide whether to export as mesh or point_cloud based on geometry and extension."""
    has_mesh = obj.mesh is not None
    has_cloud = obj.point_cloud is not None

    if not has_mesh and not has_cloud:
        raise ValueError(
            f"MathObject '{obj.generator_name}' has no mesh or point_cloud to export. "
            "Run a representation stage to convert curves to mesh before exporting."
        )

    if has_mesh and not has_cloud:
        return "mesh"
    if has_cloud and not has_mesh:
        return "point_cloud"

    # Both present — use file extension to disambiguate.
    suffix = path.suffix.lower()
    if suffix in _MESH_ONLY_EXTENSIONS:
        return "mesh"
    if suffix in _CLOUD_ONLY_EXTENSIONS:
        return "point_cloud"

    # Ambiguous extension (e.g. .ply) with both geometries — prefer mesh.
    return "mesh"
