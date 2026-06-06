"""Polygon helpers for no-go zones (pure numpy/python, unit-testable).

A no-go zone is a polygon in map coordinates. We stamp it as OCCUPIED into the
int8 occupancy grid *before* costmap inflation, so the planner both refuses to
enter it and keeps the normal inflation standoff around it. We also sample
points along the polygon edges so the local controller treats the (LIDAR-
invisible, e.g. glass) boundary as a real obstacle.
"""
import numpy as np


def parse_zone_strings(strings):
    """Parse ['x1 y1 x2 y2 ...', ...] into a list of (N,2) float arrays."""
    zones = []
    for s in strings or []:
        try:
            vals = [float(v) for v in s.replace(',', ' ').split()]
        except (ValueError, AttributeError):
            continue
        if len(vals) >= 6 and len(vals) % 2 == 0:   # >= 3 vertices
            zones.append(np.array(vals, dtype=float).reshape(-1, 2))
    return zones


def point_in_polygon(x, y, poly):
    """Ray-casting point-in-polygon test. poly is (N,2)."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def stamp_polygons_occupied(occ_int8, polygons, origin_x, origin_y, res, value=100):
    """Set cells whose centre lies inside any polygon to ``value`` (in place)."""
    if not polygons:
        return occ_int8
    h, w = occ_int8.shape
    for poly in polygons:
        xs, ys = poly[:, 0], poly[:, 1]
        i0 = max(0, int((xs.min() - origin_x) / res))
        i1 = min(w, int((xs.max() - origin_x) / res) + 1)
        j0 = max(0, int((ys.min() - origin_y) / res))
        j1 = min(h, int((ys.max() - origin_y) / res) + 1)
        for j in range(j0, j1):
            wy = origin_y + (j + 0.5) * res
            for i in range(i0, i1):
                wx = origin_x + (i + 0.5) * res
                if point_in_polygon(wx, wy, poly):
                    occ_int8[j, i] = value
    return occ_int8


def polygon_edge_points(polygons, spacing=0.1):
    """Sample (x, y) map points along the edges of each polygon (closed)."""
    pts = []
    for poly in polygons:
        n = len(poly)
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            steps = max(1, int(length / spacing))
            for k in range(steps):
                t = k / steps
                pts.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return np.array(pts, dtype=float).reshape(-1, 2) if pts else np.zeros((0, 2))
