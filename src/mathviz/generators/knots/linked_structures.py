"""Multi-component linked structure generators: Borromean rings and chain links.

These generators produce multiple closed curves that are topologically
linked — they look particularly striking as glass tubes.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Curve, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.knots._knot_utils import (
    DEFAULT_TUBE_RADIUS,
    extract_curve_points,
    validate_curve_points,
)

logger = logging.getLogger(__name__)


def _compute_borromean_ring(
    axis: int, radius: float, num_points: int,
) -> np.ndarray:
    """Compute a single Borromean ring in an orthogonal plane.

    Each ring is an ellipse slightly deformed to create mutual linking.
    axis: 0=xy-plane, 1=xz-plane, 2=yz-plane.
    """
    t = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    # Slight deformation amplitude to create the linking
    deform = 0.3 * radius

    if axis == 0:  # Ring in xy-plane, deformed along z
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = deform * np.sin(2.0 * t)
    elif axis == 1:  # Ring in xz-plane, deformed along y
        x = radius * np.cos(t)
        y = deform * np.sin(2.0 * t)
        z = radius * np.sin(t)
    else:  # Ring in yz-plane, deformed along x
        x = deform * np.sin(2.0 * t)
        y = radius * np.cos(t)
        z = radius * np.sin(t)

    return np.column_stack([x, y, z]).astype(np.float64)


def _compute_chain_link(
    index: int, link_radius: float, num_points: int,
) -> np.ndarray:
    """Compute a single chain link as a torus curve.

    Links alternate orientation (xy vs xz plane) and are offset along x.
    """
    t = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
    inner_r = link_radius * 0.3  # Cross-section radius for linking

    x_offset = index * link_radius * 1.5

    if index % 2 == 0:
        # Ring in xy-plane
        x = link_radius * np.cos(t) + x_offset
        y = link_radius * np.sin(t)
        z = inner_r * np.sin(t)
    else:
        # Ring in xz-plane
        x = link_radius * np.cos(t) + x_offset
        y = inner_r * np.sin(t)
        z = link_radius * np.sin(t)

    return np.column_stack([x, y, z]).astype(np.float64)


@register
class BorromeanRingsGenerator(GeneratorBase):
    """Three mutually linked rings where no two are directly linked."""

    name = "borromean_rings"
    category = "knots"
    aliases = ()
    description = (
        "Three mutually linked rings — removing any one frees the other two"
    )
    resolution_params = {
        "curve_points": "Number of sample points per ring",
    }
    _resolution_defaults = {"curve_points": 512}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for Borromean rings."""
        return {
            "ring_radius": 1.0,
            "ring_thickness": 0.08,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate three Borromean rings as closed curves."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        merged, curve_points = extract_curve_points(
            merged, resolution_kwargs, 512,
        )
        validate_curve_points(curve_points)

        ring_radius = float(merged["ring_radius"])
        if ring_radius <= 0:
            raise ValueError(f"ring_radius must be positive, got {ring_radius}")

        merged["curve_points"] = curve_points

        curves = []
        all_points = []
        for axis in range(3):
            points = _compute_borromean_ring(axis, ring_radius, curve_points)
            curves.append(Curve(points=points, closed=True))
            all_points.append(points)

        combined = np.vstack(all_points)
        bbox = BoundingBox.from_points(combined)

        logger.info(
            "Generated Borromean rings: radius=%.2f, points_per_ring=%d",
            ring_radius, curve_points,
        )

        return MathObject(
            curves=curves,
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return the recommended representation for Borromean rings."""
        return RepresentationConfig(
            type=RepresentationType.TUBE,
            tube_radius=self.get_default_params()["ring_thickness"],
        )


@register
class ChainLinksGenerator(GeneratorBase):
    """A chain of N interlocking torus links."""

    name = "chain_links"
    category = "knots"
    aliases = ()
    description = "Chain of interlocking torus links with alternating orientation"
    resolution_params = {
        "curve_points": "Number of sample points per link",
    }
    _resolution_defaults = {"curve_points": 256}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for chain links."""
        return {
            "num_links": 5,
            "link_radius": 0.5,
            "link_thickness": 0.1,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a chain of interlocking links as closed curves."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        merged, curve_points = extract_curve_points(
            merged, resolution_kwargs, 256,
        )
        validate_curve_points(curve_points)

        num_links = int(merged["num_links"])
        link_radius = float(merged["link_radius"])

        if num_links < 1:
            raise ValueError(f"num_links must be >= 1, got {num_links}")
        if link_radius <= 0:
            raise ValueError(
                f"link_radius must be positive, got {link_radius}"
            )

        merged["curve_points"] = curve_points

        curves = []
        all_points = []
        for i in range(num_links):
            points = _compute_chain_link(i, link_radius, curve_points)
            curves.append(Curve(points=points, closed=True))
            all_points.append(points)

        combined = np.vstack(all_points)
        bbox = BoundingBox.from_points(combined)

        logger.info(
            "Generated chain links: num=%d, radius=%.2f, points_per_link=%d",
            num_links, link_radius, curve_points,
        )

        return MathObject(
            curves=curves,
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return the recommended representation for chain links."""
        return RepresentationConfig(
            type=RepresentationType.TUBE,
            tube_radius=self.get_default_params()["link_thickness"],
        )
