"""Wavefront-style frontier detection on an int8 occupancy grid.

Frontiers are FREE cells (0) adjacent to UNKNOWN cells (-1). They are clustered
by connectivity into regions; each cluster reports a centroid and size. Pure
numpy/python so it is unit-testable without ROS.
"""
import math
from collections import deque

import numpy as np


class Frontier:
    __slots__ = ('cells', 'centroid_cell', 'size')

    def __init__(self, cells):
        self.cells = cells                       # list of (col, row)
        self.size = len(cells)
        cs = np.asarray(cells, dtype=np.float64)
        c = cs.mean(axis=0)
        self.centroid_cell = (float(c[0]), float(c[1]))  # (col, row), fractional


def _frontier_mask(occ):
    """Boolean mask of FREE cells that touch an UNKNOWN cell (4-connectivity)."""
    free = occ == 0
    unknown = occ == -1
    nbr_unknown = np.zeros_like(unknown)
    nbr_unknown[:-1, :] |= unknown[1:, :]
    nbr_unknown[1:, :] |= unknown[:-1, :]
    nbr_unknown[:, :-1] |= unknown[:, 1:]
    nbr_unknown[:, 1:] |= unknown[:, :-1]
    return free & nbr_unknown


def information_gain(occ, ci, cj, radius_cells):
    """Estimate the unmapped area a frontier reveals: count UNKNOWN (-1) cells
    in a window of ``radius_cells`` around cell (ci, cj). A better exploration
    signal than raw cluster size — it favours frontiers opening into large
    unexplored regions rather than slivers along a known wall.
    """
    h, w = occ.shape
    r = int(radius_cells)
    i0, i1 = max(0, ci - r), min(w, ci + r + 1)
    j0, j1 = max(0, cj - r), min(h, cj + r + 1)
    if i1 <= i0 or j1 <= j0:
        return 0
    return int(np.count_nonzero(occ[j0:j1, i0:i1] == -1))


def camera_visibility(occ, res, ox, oy, rx, ry, heading, fov_half, range_m,
                      n_rays=9):
    """Fraction of the camera FOV that would face a wall (a sign-bearing surface).

    Casts ``n_rays`` rays spanning ``heading +/- fov_half`` from world point
    (rx, ry), each up to ``range_m``. A ray that strikes an OCCUPIED (100) cell
    before leaving the map (or exceeding range) counts as "sees a wall"; a ray
    that runs into UNKNOWN/free and off the edge does not. Returns hits / n_rays
    in [0, 1].

    Room-number signs live on walls, and the Ridgeback's camera is forward-
    facing, so a frontier whose *approach heading* points the camera at walls is
    a better place to actually read signs than one that faces open space. This
    is the camera-guided exploration bias: scored from the robot's current pose
    looking toward the candidate (i.e. "if I drive at this frontier, will the
    camera be aimed at walls along the way?").
    """
    h, w = occ.shape
    if n_rays < 1 or range_m <= 0.0 or res <= 0.0:
        return 0.0
    step = res
    n_steps = max(1, int(range_m / step))
    hits = 0
    for k in range(n_rays):
        frac = 0.5 if n_rays == 1 else k / (n_rays - 1)
        a = heading - fov_half + 2.0 * fov_half * frac
        ca, sa = math.cos(a), math.sin(a)
        for s in range(1, n_steps + 1):
            d = s * step
            i = int((rx + d * ca - ox) / res)
            j = int((ry + d * sa - oy) / res)
            if not (0 <= i < w and 0 <= j < h):
                break                 # left the map: no wall in this direction
            if occ[j, i] == 100:
                hits += 1             # wall in view along this ray
                break
    return hits / n_rays


def detect_frontiers(occ, min_size=1, cluster_8conn=True):
    """Return a list of Frontier clusters from an int8 occupancy grid.

    Args:
        occ: (H, W) int8 grid, values in {-1 unknown, 0 free, 100 occupied}.
        min_size: drop clusters smaller than this many cells.
        cluster_8conn: 8-connectivity clustering if True, else 4.
    """
    mask = _frontier_mask(occ)
    h, w = occ.shape
    visited = np.zeros_like(mask)
    if cluster_8conn:
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1),
                 (1, 1), (1, -1), (-1, 1), (-1, -1))
    else:
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1))

    frontiers = []
    ys, xs = np.where(mask)
    for sy, sx in zip(ys, xs):
        if visited[sy, sx]:
            continue
        cells = []
        q = deque([(sy, sx)])
        visited[sy, sx] = True
        while q:
            cy, cx = q.popleft()
            cells.append((cx, cy))  # (col, row)
            for dy, dx in steps:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))
        if len(cells) >= min_size:
            frontiers.append(Frontier(cells))
    return frontiers
