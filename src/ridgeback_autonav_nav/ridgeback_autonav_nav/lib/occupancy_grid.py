"""Log-odds occupancy grid built from 2D LIDAR + Bresenham-style ray casting.

Pure numpy (no ROS), so it is unit-testable without hardware. The grid is
stored row-major as ``self.logodds[row, col]`` where ``col`` indexes +x (world)
and ``row`` indexes +y. The world coordinate of cell ``(col=i, row=j)`` is
``(origin_x + i*res, origin_y + j*res)`` — i.e. the cell *corner* convention used
by ``nav_msgs/OccupancyGrid``. The grid grows automatically when a scan falls
outside the current bounds.
"""
import math
import os

import numpy as np
import yaml


class OccupancyGrid2D:
    def __init__(self, resolution=0.05, size_m=30.0,
                 origin_x=None, origin_y=None,
                 log_odds_occ=1.4, log_odds_free=-0.35,
                 log_odds_min=-5.0, log_odds_max=5.0,
                 occ_threshold=0.65, free_threshold=0.25):
        self.res = float(resolution)
        n = int(round(size_m / self.res))
        # Center the map on (0,0) by default so the start pose sits at the middle.
        self.origin_x = float(origin_x) if origin_x is not None else -0.5 * n * self.res
        self.origin_y = float(origin_y) if origin_y is not None else -0.5 * n * self.res
        self.logodds = np.zeros((n, n), dtype=np.float32)
        self.l_occ = float(log_odds_occ)
        self.l_free = float(log_odds_free)
        self.l_min = float(log_odds_min)
        self.l_max = float(log_odds_max)
        self.occ_threshold = float(occ_threshold)
        self.free_threshold = float(free_threshold)

    # ---- shape -------------------------------------------------------------
    @property
    def height(self):
        return self.logodds.shape[0]

    @property
    def width(self):
        return self.logodds.shape[1]

    # ---- coordinate transforms --------------------------------------------
    def world_to_grid(self, x, y):
        """World -> (col, row), float arrays or scalars (not clamped)."""
        i = (np.asarray(x) - self.origin_x) / self.res
        j = (np.asarray(y) - self.origin_y) / self.res
        return i, j

    def grid_to_world(self, i, j):
        """(col, row) cell center -> world coordinates."""
        x = self.origin_x + (np.asarray(i) + 0.5) * self.res
        y = self.origin_y + (np.asarray(j) + 0.5) * self.res
        return x, y

    def in_bounds(self, i, j):
        return 0 <= i < self.width and 0 <= j < self.height

    # ---- dynamic expansion -------------------------------------------------
    def _ensure_bounds(self, min_x, min_y, max_x, max_y, pad_cells=20):
        """Grow the grid (preserving data) so the world box fits."""
        i0, j0 = self.world_to_grid(min_x, min_y)
        i1, j1 = self.world_to_grid(max_x, max_y)
        lo_i = int(math.floor(min(i0, i1))) - pad_cells
        lo_j = int(math.floor(min(j0, j1))) - pad_cells
        hi_i = int(math.ceil(max(i0, i1))) + pad_cells
        hi_j = int(math.ceil(max(j0, j1))) + pad_cells

        add_left = max(0, -lo_i)
        add_bottom = max(0, -lo_j)
        add_right = max(0, hi_i - self.width)
        add_top = max(0, hi_j - self.height)
        if add_left == add_bottom == add_right == add_top == 0:
            return

        new_h = self.height + add_bottom + add_top
        new_w = self.width + add_left + add_right
        new = np.zeros((new_h, new_w), dtype=np.float32)
        new[add_bottom:add_bottom + self.height,
            add_left:add_left + self.width] = self.logodds
        self.logodds = new
        # Cell (0,0) of the new grid sits add_left/add_bottom cells to the
        # lower-left of the old origin.
        self.origin_x -= add_left * self.res
        self.origin_y -= add_bottom * self.res

    # ---- scan integration --------------------------------------------------
    def integrate_scan(self, lx, ly, ltheta, ranges, angles,
                       min_range=0.1, max_range=10.0,
                       blind_mask=None, free_step_frac=0.5):
        """Integrate one laser scan taken from laser pose (lx, ly, ltheta).

        ``angles`` are per-beam bearings in the laser frame. ``blind_mask`` is an
        optional boolean array (True == beam in rear blind arc -> skip entirely,
        leaving those cells UNKNOWN). No-return beams (range > max_range or inf)
        clear free space out to ``max_range`` but mark no endpoint.
        """
        ranges = np.asarray(ranges, dtype=np.float64)
        angles = np.asarray(angles, dtype=np.float64)
        n = ranges.shape[0]
        if angles.shape[0] != n:
            raise ValueError('ranges and angles length mismatch')

        finite = np.isfinite(ranges)
        valid = finite & (ranges >= min_range)
        if blind_mask is not None:
            valid &= ~np.asarray(blind_mask, dtype=bool)
        if not np.any(valid):
            return

        hit = valid & (ranges <= max_range)
        # Ray endpoint distance for the free-space sweep (capped at max_range).
        end_r = np.where(hit, ranges, max_range)
        end_r = np.clip(end_r, 0.0, max_range)

        beam_theta = ltheta + angles
        cos_t = np.cos(beam_theta)
        sin_t = np.sin(beam_theta)

        # World endpoints (used for occupied marks and bounds).
        ex = lx + ranges * cos_t
        ey = ly + ranges * sin_t

        # Bounds: include laser origin and all ray endpoints (capped).
        sweep_ex = lx + end_r * cos_t
        sweep_ey = ly + end_r * sin_t
        min_x = min(lx, float(np.min(sweep_ex[valid])))
        max_x = max(lx, float(np.max(sweep_ex[valid])))
        min_y = min(ly, float(np.min(sweep_ey[valid])))
        max_y = max(ly, float(np.max(sweep_ey[valid])))
        self._ensure_bounds(min_x, min_y, max_x, max_y)

        step = self.res * free_step_frac
        max_steps = int(math.ceil(max_range / step)) + 1
        t = np.arange(max_steps, dtype=np.float64) * step  # (S,)

        vidx = np.where(valid)[0]
        # Free-space samples: (B, S) world coords.
        cos_v = cos_t[vidx][:, None]
        sin_v = sin_t[vidx][:, None]
        dist = t[None, :]
        # Only sample up to just before the endpoint so the hit cell stays occupied.
        keep = dist <= (end_r[vidx][:, None] - 0.5 * self.res)
        xs = lx + dist * cos_v
        ys = ly + dist * sin_v
        fi = ((xs - self.origin_x) / self.res).astype(np.int64)
        fj = ((ys - self.origin_y) / self.res).astype(np.int64)
        keep &= (fi >= 0) & (fi < self.width) & (fj >= 0) & (fj < self.height)
        fi = fi[keep]
        fj = fj[keep]
        if fi.size:
            np.add.at(self.logodds, (fj, fi), self.l_free)

        # Occupied endpoints (only true hits).
        hidx = np.where(hit)[0]
        if hidx.size:
            oi = ((ex[hidx] - self.origin_x) / self.res).astype(np.int64)
            oj = ((ey[hidx] - self.origin_y) / self.res).astype(np.int64)
            ok = (oi >= 0) & (oi < self.width) & (oj >= 0) & (oj < self.height)
            oi, oj = oi[ok], oj[ok]
            if oi.size:
                np.add.at(self.logodds, (oj, oi), self.l_occ)

        np.clip(self.logodds, self.l_min, self.l_max, out=self.logodds)

    # ---- queries -----------------------------------------------------------
    def prob(self):
        """P(occupied) for every cell."""
        return 1.0 - 1.0 / (1.0 + np.exp(self.logodds))

    def to_int8(self):
        """nav_msgs/OccupancyGrid data: -1 unknown, 0 free, 100 occupied."""
        p = self.prob()
        out = np.full(self.logodds.shape, -1, dtype=np.int8)
        out[p >= self.occ_threshold] = 100
        out[p <= self.free_threshold] = 0
        return out

    def occupancy_classes(self):
        """Return (-1/0/100) int8 grid; alias of to_int8 for clarity."""
        return self.to_int8()

    # ---- costmap for the planner ------------------------------------------
    def inflate(self, robot_radius_m, inflation_radius_m,
                unknown_cost=100.0, lethal_cost=254.0, decay=3.0):
        """Build a float cost grid from this grid's occupancy classes."""
        return costmap_from_occupancy(
            self.to_int8(), self.res, robot_radius_m, inflation_radius_m,
            unknown_cost=unknown_cost, lethal_cost=lethal_cost, decay=decay)

    # ---- persistence -------------------------------------------------------
    def save(self, path_stem):
        path_stem = os.path.expanduser(path_stem)
        os.makedirs(os.path.dirname(path_stem) or '.', exist_ok=True)
        np.save(path_stem + '.npy', self.logodds)
        meta = {
            'resolution': self.res,
            'origin_x': self.origin_x,
            'origin_y': self.origin_y,
            'width': self.width,
            'height': self.height,
            'log_odds_occ': self.l_occ,
            'log_odds_free': self.l_free,
            'occ_threshold': self.occ_threshold,
            'free_threshold': self.free_threshold,
        }
        with open(path_stem + '.yaml', 'w') as fh:
            yaml.safe_dump(meta, fh)

    @classmethod
    def load(cls, path_stem):
        path_stem = os.path.expanduser(path_stem)
        with open(path_stem + '.yaml') as fh:
            meta = yaml.safe_load(fh)
        g = cls(resolution=meta['resolution'],
                origin_x=meta['origin_x'], origin_y=meta['origin_y'],
                log_odds_occ=meta['log_odds_occ'],
                log_odds_free=meta['log_odds_free'],
                occ_threshold=meta['occ_threshold'],
                free_threshold=meta['free_threshold'])
        g.logodds = np.load(path_stem + '.npy').astype(np.float32)
        return g


def costmap_from_occupancy(occ_int8, res, robot_radius_m, inflation_radius_m,
                           unknown_cost=100.0, lethal_cost=254.0, decay=3.0):
    """Build a float cost grid from an int8 occupancy array (-1/0/100).

    Works on the published ``/map`` directly (no log-odds needed), so the nav
    server can reuse it. Occupied cells and everything within ``robot_radius``
    are lethal; cost decays out to ``inflation_radius``; unknown cells get a flat
    ``unknown_cost`` (low in exploration mode, high for missions).
    """
    cls = np.asarray(occ_int8)
    occ = cls == 100
    cost = np.zeros(cls.shape, dtype=np.float32)
    cost[cls == -1] = float(unknown_cost)
    if np.any(occ):
        dist = _edt(~occ) * res  # metres to nearest occupied cell
        lethal = dist <= robot_radius_m
        infl = (dist > robot_radius_m) & (dist <= inflation_radius_m)
        cost[lethal] = float(lethal_cost)
        span = max(inflation_radius_m - robot_radius_m, 1e-6)
        d = dist[infl] - robot_radius_m
        cost[infl] = np.maximum(
            cost[infl],
            (lethal_cost - 1.0) * np.exp(-decay * d / span)).astype(np.float32)
    return cost


def _edt(mask):
    """Euclidean distance transform (in cells) of the True region of ``mask``.

    Distance is 0 on the False cells (the 'sources', e.g. occupied) and grows
    into the True region. Uses OpenCV (fast C) when available, then SciPy, then
    a pure-python two-pass Felzenszwalb fallback.
    """
    try:
        import cv2
        src = mask.astype(np.uint8)  # nonzero where we measure, 0 at sources
        return cv2.distanceTransform(src, cv2.DIST_L2, 5).astype(np.float64)
    except Exception:
        pass
    try:
        from scipy import ndimage
        return ndimage.distance_transform_edt(mask)
    except Exception:
        pass
    INF = 1e20
    f = np.where(mask, INF, 0.0).astype(np.float64)
    d = f.copy()
    for axis in (0, 1):
        d = np.apply_along_axis(_edt_1d, axis, d)
    return np.sqrt(d)


def _edt_1d(f):
    n = f.shape[0]
    d = np.empty(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.int64)
    z = np.empty(n + 1, dtype=np.float64)
    k = 0
    v[0] = 0
    z[0] = -1e20
    z[1] = 1e20
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = 1e20
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        dq = q - v[k]
        d[q] = dq * dq + f[v[k]]
    return d
