"""Generator base class and registry for mathematical object generators."""

import importlib
import logging
import pkgutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from mathviz.core.math_object import MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)


@dataclass
class GeneratorMeta:
    """Metadata about a registered generator."""

    name: str
    category: str
    aliases: list[str]
    description: str
    resolution_params: dict[str, str]
    generator_class: type


class GeneratorBase(ABC):
    """Base class for all mathematical object generators."""

    name: str = ""
    category: str = ""
    aliases: list[str] = []
    description: str = ""
    resolution_params: dict[str, str] = {}

    @abstractmethod
    def get_default_params(self) -> dict[str, Any]:
        """Return default parameter dict with descriptions and valid ranges."""
        ...

    @abstractmethod
    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate geometry in abstract coordinate space."""
        ...

    def get_param_schema(self) -> dict[str, Any]:
        """Return JSON-schema-like description of all parameters for CLI/UI."""
        return {}

    def get_default_representation(self) -> RepresentationConfig:
        """Return the recommended representation for engraving."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_registry: dict[str, GeneratorMeta] = {}
_alias_map: dict[str, str] = {}
_discovered = False


class DuplicateRegistrationError(Exception):
    """Raised when a generator name or alias is already registered."""


def register(
    cls: type[GeneratorBase] | None = None,
    *,
    aliases: list[str] | None = None,
) -> type[GeneratorBase] | Any:
    """Register a generator class in the global registry.

    Can be used as a decorator with or without arguments:
        @register
        class MyGen(GeneratorBase): ...

        @register(aliases=["alt_name"])
        class MyGen(GeneratorBase): ...
    """
    def _do_register(gen_cls: type[GeneratorBase]) -> type[GeneratorBase]:
        canonical = gen_cls.name
        if not canonical:
            raise ValueError(f"{gen_cls.__name__} must define a non-empty 'name'")

        resolved_aliases = aliases if aliases is not None else list(gen_cls.aliases)

        _check_name_available(canonical, gen_cls)
        for alias in resolved_aliases:
            _check_name_available(alias, gen_cls)

        meta = GeneratorMeta(
            name=canonical,
            category=gen_cls.category,
            aliases=resolved_aliases,
            description=gen_cls.description,
            resolution_params=dict(gen_cls.resolution_params),
            generator_class=gen_cls,
        )

        _registry[canonical] = meta
        _alias_map[canonical] = canonical
        for alias in resolved_aliases:
            _alias_map[alias] = canonical

        logger.debug("Registered generator %r (aliases=%s)", canonical, resolved_aliases)
        return gen_cls

    if cls is not None:
        return _do_register(cls)
    return _do_register


def _check_name_available(name: str, gen_cls: type[GeneratorBase]) -> None:
    """Raise DuplicateRegistrationError if name is already taken."""
    if name in _alias_map:
        existing = _registry[_alias_map[name]].generator_class
        raise DuplicateRegistrationError(
            f"Name {name!r} is already registered to {existing.__name__}, "
            f"cannot register {gen_cls.__name__}"
        )


def get_generator(name: str) -> type[GeneratorBase]:
    """Look up a generator class by canonical name or alias."""
    _ensure_discovered()
    canonical = _alias_map.get(name)
    if canonical is None:
        raise KeyError(f"No generator registered with name or alias {name!r}")
    return _registry[canonical].generator_class


def get_generator_meta(name: str) -> GeneratorMeta:
    """Look up generator metadata by canonical name or alias."""
    _ensure_discovered()
    canonical = _alias_map.get(name)
    if canonical is None:
        raise KeyError(f"No generator registered with name or alias {name!r}")
    return _registry[canonical]


def list_generators() -> list[GeneratorMeta]:
    """Return metadata for all registered generators."""
    _ensure_discovered()
    return list(_registry.values())


def clear_registry() -> None:
    """Clear all registrations. Intended for testing only."""
    global _discovered
    _registry.clear()
    _alias_map.clear()
    _discovered = False


def _ensure_discovered() -> None:
    """Auto-discover generators from the generators package on first access."""
    global _discovered
    if _discovered:
        return
    _discovered = True
    _discover_generators()


def _discover_generators() -> None:
    """Walk the mathviz.generators package and import all submodules."""
    try:
        import mathviz.generators as gen_pkg
    except ImportError:
        logger.warning("mathviz.generators package not found, skipping discovery")
        return

    if gen_pkg.__path__ is None:
        return

    for importer, modname, is_pkg in pkgutil.walk_packages(
        gen_pkg.__path__, prefix="mathviz.generators."
    ):
        try:
            importlib.import_module(modname)
        except ImportError:
            logger.warning("Failed to import generator module %s", modname)
