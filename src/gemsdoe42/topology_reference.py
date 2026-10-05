"""Small, dependency-free reference implementation of 0D superlevel persistence.

This is deliberately simple and intended for tests/small rasters. Production-sized
rasters use the Numba implementation in :mod:`gemsdoe42.topology`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def h0_superlevel_persistence(
    surface: Sequence[Sequence[float]],
    valid: Sequence[Sequence[bool]] | None = None,
) -> list[list[float]]:
    """Return the birth-minus-death lifetime at each finite local maximum.

    Pixels enter a superlevel filtration in descending scalar-value order. Four-
    neighbor components merge when the next pixel is activated. At a merge, the
    component with the highest birth value survives (elder rule); younger peaks die
    at the current level. The essential H0 component of each connected valid-mask
    component is omitted. The function therefore returns zero for each essential
    peak and for zero-lifetime peaks. This is a graph/cubical 0D filtration, not a full
    H1 cubical computation.

    Args:
        surface: Rectangular 2D scalar field (e.g. a smoothed gradient-magnitude map).
        valid: Optional rectangular mask. Invalid or non-finite pixels are excluded.

    Returns:
        A nested list with the persistence lifetime at each finite peak's birth pixel.
    """
    height = len(surface)
    if height == 0:
        raise ValueError("surface must have at least one row")
    width = len(surface[0])
    if width == 0 or any(len(row) != width for row in surface):
        raise ValueError("surface must be a non-empty rectangular 2D array")
    if valid is not None and (
        len(valid) != height or any(len(row) != width for row in valid)
    ):
        raise ValueError("valid mask shape must match surface")

    n = height * width
    flat_values = [float(surface[y][x]) for y in range(height) for x in range(width)]
    flat_valid = [
        math.isfinite(flat_values[index])
        and (valid is None or bool(valid[index // width][index % width]))
        for index in range(n)
    ]
    order = sorted(
        (index for index, keep in enumerate(flat_valid) if keep),
        key=lambda index: (-flat_values[index], index),
    )

    parent = [-1] * n
    birth = [0.0] * n
    birth_pixel = [-1] * n
    persistence = [0.0] * n

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != index:
            following = parent[index]
            parent[index] = root
            index = following
        return root

    for index in order:
        value = flat_values[index]
        parent[index] = index
        birth[index] = value
        birth_pixel[index] = index
        roots = [index]
        y, x = divmod(index, width)
        neighbors = []
        if x > 0:
            neighbors.append(index - 1)
        if x + 1 < width:
            neighbors.append(index + 1)
        if y > 0:
            neighbors.append(index - width)
        if y + 1 < height:
            neighbors.append(index + width)

        for neighbor in neighbors:
            if parent[neighbor] < 0:
                continue
            root = find(neighbor)
            if root not in roots:
                roots.append(root)

        winner = max(roots, key=lambda root: (birth[root], -birth_pixel[root]))
        for root in roots:
            if root == winner:
                continue
            lifetime = birth[root] - value
            if lifetime > 0:
                persistence[birth_pixel[root]] = max(
                    persistence[birth_pixel[root]], lifetime
                )
            parent[root] = winner
        parent[index] = winner

    return [
        persistence[row * width : (row + 1) * width]
        for row in range(height)
    ]
