"""Core geometry containers, enums, and configuration models."""

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.engraving import EngravingProfile
from mathviz.core.generator import (
    DuplicateRegistrationError,
    GeneratorBase,
    GeneratorMeta,
    clear_registry,
    get_generator,
    get_generator_meta,
    list_generators,
    register,
)
from mathviz.core.math_object import (
    BoundingBox,
    CoordSpace,
    Curve,
    MathObject,
    Mesh,
    PointCloud,
)
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.core.validator import (
    CheckResult,
    Severity,
    ValidationResult,
    validate_engraving,
    validate_mesh,
)

__all__ = [
    "BoundingBox",
    "CheckResult",
    "Container",
    "CoordSpace",
    "Curve",
    "DuplicateRegistrationError",
    "EngravingProfile",
    "GeneratorBase",
    "GeneratorMeta",
    "MathObject",
    "Mesh",
    "PlacementPolicy",
    "PointCloud",
    "RepresentationConfig",
    "RepresentationType",
    "Severity",
    "ValidationResult",
    "clear_registry",
    "get_generator",
    "get_generator_meta",
    "list_generators",
    "register",
    "validate_engraving",
    "validate_mesh",
]
