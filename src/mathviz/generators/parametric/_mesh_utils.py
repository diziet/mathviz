"""Shared mesh face-building utilities for parametric surface generators."""

import numpy as np


def build_wrapped_grid_faces(n_u: int, n_v: int) -> np.ndarray:
    """Build triangle faces for a grid periodic in both u and v."""
    rows = np.arange(n_u)
    cols = np.arange(n_v)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()

    i00 = rr * n_v + cc
    i10 = ((rr + 1) % n_u) * n_v + cc
    i01 = rr * n_v + ((cc + 1) % n_v)
    i11 = ((rr + 1) % n_u) * n_v + ((cc + 1) % n_v)

    tri1 = np.stack([i00, i10, i11], axis=-1)
    tri2 = np.stack([i00, i11, i01], axis=-1)
    return np.concatenate([tri1, tri2], axis=0).astype(np.int64)


def build_open_grid_faces(n_u: int, n_v: int) -> np.ndarray:
    """Build triangle faces for an open grid (no wrapping)."""
    rows = np.arange(n_u - 1)
    cols = np.arange(n_v - 1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()

    i00 = rr * n_v + cc
    i10 = (rr + 1) * n_v + cc
    i01 = rr * n_v + (cc + 1)
    i11 = (rr + 1) * n_v + (cc + 1)

    tri1 = np.stack([i00, i10, i11], axis=-1)
    tri2 = np.stack([i00, i11, i01], axis=-1)
    return np.concatenate([tri1, tri2], axis=0).astype(np.int64)


def build_sphere_faces(n_lat: int, n_lon: int) -> np.ndarray:
    """Build triangle faces for a sphere-like grid with poles.

    The grid has n_lat latitude rows (excluding poles) and n_lon longitude
    columns. Two extra vertices are appended: south pole at index n_lat*n_lon,
    north pole at n_lat*n_lon + 1.
    """
    n_body = n_lat * n_lon
    south_pole = n_body
    north_pole = n_body + 1

    # South pole fan
    j_range = np.arange(n_lon)
    j_next = (j_range + 1) % n_lon
    south_fan = np.column_stack([
        np.full(n_lon, south_pole), j_range, j_next,
    ])

    # Body quads (vectorized)
    rows = np.arange(n_lat - 1)
    cols = np.arange(n_lon)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    cc_next = (cc + 1) % n_lon

    i00 = rr * n_lon + cc
    i10 = (rr + 1) * n_lon + cc
    i01 = rr * n_lon + cc_next
    i11 = (rr + 1) * n_lon + cc_next

    body_tri1 = np.column_stack([i00, i10, i11])
    body_tri2 = np.column_stack([i00, i11, i01])

    # North pole fan
    last_row = (n_lat - 1) * n_lon
    north_fan = np.column_stack([
        last_row + j_range, np.full(n_lon, north_pole), last_row + j_next,
    ])

    return np.concatenate([
        south_fan, body_tri1, body_tri2, north_fan,
    ], axis=0).astype(np.int64)
