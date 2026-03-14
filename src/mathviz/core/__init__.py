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

__all__ = [
    "BoundingBox",
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
    "clear_registry",
    "get_generator",
    "get_generator_meta",
    "list_generators",
    "register",
]
