"""Vectorized 2D LIDAR raycasting against a boolean occupancy grid (pure numpy)."""
import numpy as np


def raycast(occ, res, origin_x, origin_y, lx, ly, ltheta, angles,
            max_range=10.0, min_range=0.06):
    """Cast one scan from laser pose (lx, ly, ltheta).

    Args:
        occ: (H, W) bool grid, True == wall.
        res, origin_x, origin_y: grid geometry.
        lx, ly, ltheta: laser pose in world metres / radians.
        angles: (N,) per-beam bearings in the laser frame.
        max_range, min_range: sensor limits (m).
    Returns:
        (N,) ranges; a beam with no hit returns ``max_range``.
    """
    angles = np.asarray(angles, dtype=np.float64)
    h, w = occ.shape
    step = res * 0.5
    n_steps = int(np.ceil(max_range / step)) + 1
    t = np.arange(n_steps, dtype=np.float64) * step           # (S,)
    t = np.maximum(t, min_range * (t > 0))                    # keep near-field sane

    beam = ltheta + angles
    cos_b = np.cos(beam)[:, None]                             # (N,1)
    sin_b = np.sin(beam)[:, None]
    xs = lx + t[None, :] * cos_b                              # (N,S)
    ys = ly + t[None, :] * sin_b
    ci = ((xs - origin_x) / res).astype(np.int64)
    cj = ((ys - origin_y) / res).astype(np.int64)

    inside = (ci >= 0) & (ci < w) & (cj >= 0) & (cj < h)
    hit = np.zeros(xs.shape, dtype=bool)
    ii, jj = ci[inside], cj[inside]
    hit[inside] = occ[jj, ii]
    # Anything that leaves the grid counts as a hit (world boundary).
    hit |= ~inside & (t[None, :] > min_range)

    first = np.argmax(hit, axis=1)                            # first True per beam
    any_hit = hit[np.arange(hit.shape[0]), first]
    ranges = np.where(any_hit, t[first], max_range)
    return np.clip(ranges, 0.0, max_range)
