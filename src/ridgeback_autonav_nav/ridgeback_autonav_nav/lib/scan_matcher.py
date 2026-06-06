"""Correlative scan matcher (our own, no SLAM library).

Refines a predicted laser pose by searching a small (dx, dy, dtheta) window for
the offset that best aligns the scan's hit endpoints with the accumulated
occupancy grid's log-odds. This bounds odometry drift so walls stay crisp and
return-home stays accurate.
"""
import math

import numpy as np


def match(grid, pred_x, pred_y, pred_theta, ranges, angles,
          min_range=0.1, max_range=10.0, max_beams=180,
          window_xy=0.10, step_xy=0.025, window_theta=0.08, step_theta=0.02,
          prior_weight=4.0, prior_weight_theta=2.0):
    """Search for the pose offset that maximizes scan/grid agreement.

    A prior term penalizes deviation from the odom prediction so that in
    featureless or symmetric areas (where overlap scores are ambiguous) the
    matcher stays anchored to odometry instead of snapping to a spurious offset.

    Args:
        grid: OccupancyGrid2D (uses .logodds, .res, origin, l_max).
        pred_x/y/theta: predicted laser pose in the map frame (from odom).
        ranges, angles: laser scan.
        prior_weight(_theta): regularization strength toward the prediction.
    Returns:
        (best_x, best_y, best_theta, score) where score is in roughly [0, 1].
        Falls back to the predicted pose with score 0 if there are too few hits.
    """
    ranges = np.asarray(ranges, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    valid = np.isfinite(ranges) & (ranges >= min_range) & (ranges <= max_range)
    idx = np.where(valid)[0]
    if idx.size < 20:
        return pred_x, pred_y, pred_theta, 0.0
    if idx.size > max_beams:
        idx = idx[np.linspace(0, idx.size - 1, max_beams).astype(np.int64)]

    r = ranges[idx]
    a = angles[idx]
    # Hit points in the laser frame.
    px = r * np.cos(a)
    py = r * np.sin(a)

    lo = grid.logodds
    h, w = lo.shape
    inv_res = 1.0 / grid.res
    ox, oy = grid.origin_x, grid.origin_y
    occ_field = np.maximum(lo, 0.0)  # reward overlap with confidently-occupied cells

    dxs = np.arange(-window_xy, window_xy + 1e-9, step_xy)
    dys = np.arange(-window_xy, window_xy + 1e-9, step_xy)
    dths = np.arange(-window_theta, window_theta + 1e-9, step_theta)

    norm = max(grid.l_max, 1e-6)
    best = (pred_x, pred_y, pred_theta)
    best_overlap = 0.0
    best_score = -math.inf
    for dth in dths:
        th = pred_theta + dth
        c, s = np.cos(th), np.sin(th)
        # Rotate points once per theta candidate.
        wx0 = c * px - s * py
        wy0 = s * px + c * py
        for dx in dxs:
            bx = pred_x + dx
            ci = ((wx0 + bx - ox) * inv_res).astype(np.int64)
            for dy in dys:
                by = pred_y + dy
                cj = ((wy0 + by - oy) * inv_res).astype(np.int64)
                m = (ci >= 0) & (ci < w) & (cj >= 0) & (cj < h)
                if not np.any(m):
                    continue
                overlap = occ_field[cj[m], ci[m]].sum() / (idx.size * norm)
                # Penalize deviation from the odom prediction (anchors drift).
                prior = prior_weight * (dx * dx + dy * dy) + prior_weight_theta * dth * dth
                score = overlap - prior
                if score > best_score:
                    best_score = score
                    best_overlap = overlap
                    best = (bx, by, th)

    return best[0], best[1], best[2], float(max(best_overlap, 0.0))
