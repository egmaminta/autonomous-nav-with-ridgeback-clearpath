"""Path smoothing so the controller follows a smooth curve, not a grid staircase.

A* on a grid produces zig-zag (8-connected) paths; following them makes a robot
weave. Chaikin corner-cutting rounds the corners into a smooth path (a quadratic
B-spline in the limit), which is what makes motion look natural. We keep the
smoothed path only if it stays collision-free (it normally does, since A* already
runs inside an inflated costmap).
"""


def chaikin_smooth(points, iterations=2, ratio=0.25):
    """Chaikin corner-cutting on a list of (x, y) points (endpoints preserved)."""
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) < 3:
        return pts
    r = ratio
    for _ in range(max(0, iterations)):
        new = [pts[0]]
        for i in range(len(pts) - 1):
            p, q = pts[i], pts[i + 1]
            new.append(((1 - r) * p[0] + r * q[0], (1 - r) * p[1] + r * q[1]))
            new.append((r * p[0] + (1 - r) * q[0], r * p[1] + (1 - r) * q[1]))
        new.append(pts[-1])
        pts = new
    return pts


def smooth_path_safe(points, is_free, iterations=2, ratio=0.25):
    """Chaikin-smooth a path, falling back to the original if smoothing would
    clip an obstacle.

    Args:
        points: list of (x, y) world points.
        is_free: callable (x, y) -> bool, True if the point is traversable.
    """
    if len(points) < 3:
        return [(float(p[0]), float(p[1])) for p in points]
    sm = chaikin_smooth(points, iterations, ratio)
    if all(is_free(x, y) for (x, y) in sm):
        return sm
    return [(float(p[0]), float(p[1])) for p in points]
