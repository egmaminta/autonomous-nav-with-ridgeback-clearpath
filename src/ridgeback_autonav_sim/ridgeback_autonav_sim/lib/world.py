"""Ground-truth world: an occupancy grid built from a wall spec, plus signs.

Pure numpy so it is unit-testable. A world is defined by a YAML spec:

    resolution: 0.05
    bounds: [xmin, ymin, xmax, ymax]      # metres
    border: true                          # occupy the 1-cell outer ring
    wall_thickness: 0.10                  # metres (segment rasterization)
    walls:                                # axis-aligned or diagonal segments
      - [x1, y1, x2, y2]
    signs:
      - {text: "206", x: 11.0, y: 5.6, z: 1.4}
    start: {x: 0.6, y: 1.0, theta: 0.0}
"""
import math
from collections import deque

import numpy as np

# Base colours (BGR, the camera's convention) for the per-cell "colour grid":
# every occupied cell carries the base colour the camera shades and draws, so
# free-standing objects can look different from walls. WALL_BGR is the single
# source of truth for the wall colour (the camera imports it from here).
WALL_BGR = (210, 214, 218)
OBSTACLE_BGR = (70, 120, 200)  # default free-standing object colour
OBSTACLE_PALETTE = [(70, 120, 200), (90, 165, 90), (180, 120, 60),
                    (60, 80, 205), (150, 150, 70), (130, 90, 175)]

# Vertical heights (metres) for the 2.5D camera. Walls are full height; objects
# default shorter so tables and chairs render as low blocks, not full pillars.
WALL_HEIGHT_M = 2.5
OBSTACLE_HEIGHT_M = 0.8


class Sign:
    """A wall-mounted plaque. ``yaw`` is the heading of the outward wall normal
    (radians): the plaque faces that way and the renderer perspective-warps it
    from its 3D pose, so off-axis views foreshorten. ``w``/``h`` are metres."""
    __slots__ = ('text', 'x', 'y', 'z', 'yaw', 'w', 'h')

    def __init__(self, text, x, y, z=1.4, yaw=0.0, w=0.5, h=0.28):
        self.text = str(text)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.yaw = float(yaw)
        self.w = float(w)
        self.h = float(h)


class World:

    def __init__(self,
                 resolution,
                 bounds,
                 occupied,
                 signs,
                 start,
                 color_grid=None,
                 height_grid=None,
                 obstacles=None,
                 spawn_regions=None):
        self.res = float(resolution)
        self.xmin, self.ymin, self.xmax, self.ymax = [float(b) for b in bounds]
        self.occ = occupied  # bool (H, W), True == wall/obstacle
        self.signs = signs  # list[Sign]
        self.start = start  # (x, y, theta)
        self.origin_x = self.xmin
        self.origin_y = self.ymin
        # Per-cell base colour (HxWx3 uint8 BGR) and height (HxW float32 metres)
        # for the camera, plus the immutable base scene (walls + border + static
        # obstacles) so an episode that scatters random obstacles can cheaply
        # restore it via occ_base/color_base/height_base.
        self.color_grid = color_grid
        self.height_grid = height_grid
        self.obstacles = list(obstacles) if obstacles else []
        # Optional [xlo, ylo, xhi, yhi] rects the RL env may spawn the robot in
        # (e.g. the room interiors, so the agent has to drive out to a sign).
        self.spawn_regions = [
            [float(v) for v in r] for r in (spawn_regions or [])
        ]
        self.occ_base = occupied.copy()
        self.color_base = None if color_grid is None else color_grid.copy()
        self.height_base = None if height_grid is None else height_grid.copy()

    @property
    def width(self):
        return self.occ.shape[1]

    @property
    def height(self):
        return self.occ.shape[0]

    @classmethod
    def from_spec(cls, spec):
        res = float(spec.get('resolution', 0.05))
        bounds = spec['bounds']
        xmin, ymin, xmax, ymax = [float(b) for b in bounds]
        w = int(math.ceil((xmax - xmin) / res))
        h = int(math.ceil((ymax - ymin) / res))
        occ = np.zeros((h, w), dtype=bool)

        thick = float(spec.get('wall_thickness', 0.10))
        for seg in spec.get('walls', []):
            _rasterize_segment(occ, res, xmin, ymin, seg, thick)

        if spec.get('border', True):
            occ[0, :] = True
            occ[-1, :] = True
            occ[:, 0] = True
            occ[:, -1] = True

        # Colour + height grids and static obstacles/objects inside rooms.
        color_grid = np.empty((h, w, 3), dtype=np.uint8)
        color_grid[:] = WALL_BGR
        height_grid = np.full((h, w), WALL_HEIGHT_M, dtype=np.float32)
        obstacles = list(spec.get('obstacles', []))
        for obs in obstacles:
            rasterize_obstacle(occ, color_grid, height_grid, res, xmin, ymin,
                               obs)

        signs = [
            Sign(s['text'], s['x'], s['y'], s.get('z', 1.4),
                 math.radians(float(s.get('yaw_deg', 0.0))),
                 float(s.get('w', 0.5)), float(s.get('h', 0.28)))
            for s in spec.get('signs', [])
        ]
        st = spec.get('start', {'x': 0.0, 'y': 0.0, 'theta': 0.0})
        start = (float(st.get('x', 0.0)), float(st.get('y', 0.0)),
                 float(st.get('theta', 0.0)))
        return cls(res,
                   bounds,
                   occ,
                   signs,
                   start,
                   color_grid=color_grid,
                   height_grid=height_grid,
                   obstacles=obstacles,
                   spawn_regions=spec.get('spawn_regions', []))

    def world_to_cell(self, x, y):
        return (int(
            (x - self.origin_x) / self.res), int(
                (y - self.origin_y) / self.res))

    def is_occupied(self, x, y):
        i, j = self.world_to_cell(x, y)
        if 0 <= i < self.width and 0 <= j < self.height:
            return bool(self.occ[j, i])
        return True  # outside the world == blocked

    def to_int8(self):
        """Ground-truth occupancy as nav_msgs data (0 free, 100 occupied)."""
        out = np.zeros(self.occ.shape, dtype=np.int8)
        out[self.occ] = 100
        return out


def _rasterize_segment(occ, res, ox, oy, seg, thickness):
    x1, y1, x2, y2 = [float(v) for v in seg[:4]]
    h, w = occ.shape
    half = max(1, int(round(0.5 * thickness / res)))
    n = max(2, int(math.hypot(x2 - x1, y2 - y1) / (res * 0.5)) + 1)
    for t in np.linspace(0.0, 1.0, n):
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        ci = int((x - ox) / res)
        cj = int((y - oy) / res)
        i0, i1 = max(0, ci - half), min(w, ci + half + 1)
        j0, j1 = max(0, cj - half), min(h, cj + half + 1)
        occ[j0:j1, i0:i1] = True


def _rasterize_rect(grid, res, ox, oy, cx, cy, w, h, yaw=0.0, value=True):
    """Mark an oriented rectangle into a 2D grid: centre (cx, cy), size w x h
    metres, rotation ``yaw`` radians. Cell centres inside the rectangle are set
    to ``value``. Works on any 2D array (bool occupancy or an int class grid),
    so the same call rasterizes an obstacle into ``occ`` and into a colour grid.
    Same cell-index convention as ``_rasterize_segment`` / ``world_to_cell``."""
    hh, ww = grid.shape[:2]  # 2D occ or 3D colour grid
    hw, hd = 0.5 * float(w), 0.5 * float(h)
    rad = math.hypot(hw, hd)  # bounding circle for the bbox
    i0 = max(0, int((cx - rad - ox) / res))
    i1 = min(ww, int((cx + rad - ox) / res) + 1)
    j0 = max(0, int((cy - rad - oy) / res))
    j1 = min(hh, int((cy + rad - oy) / res) + 1)
    if i0 >= i1 or j0 >= j1:
        return
    xs = ox + (np.arange(i0, i1) + 0.5) * res  # cell-centre world x (cols)
    ys = oy + (np.arange(j0, j1) + 0.5) * res  # cell-centre world y (rows)
    dx = xs[None, :] - cx
    dy = ys[:, None] - cy
    c, s = math.cos(yaw), math.sin(yaw)
    xl = c * dx + s * dy  # rotate into the rect's frame
    yl = -s * dx + c * dy
    mask = (np.abs(xl) <= hw) & (np.abs(yl) <= hd)
    grid[j0:j1, i0:i1][mask] = value  # basic slice -> view, writes through


def _rasterize_circle(grid, res, ox, oy, cx, cy, r, value=True):
    """Mark a disc (centre cx, cy; radius r metres) into a 2D grid (see
    ``_rasterize_rect`` for the grid/value convention)."""
    hh, ww = grid.shape[:2]  # 2D occ or 3D colour grid
    r = float(r)
    i0 = max(0, int((cx - r - ox) / res))
    i1 = min(ww, int((cx + r - ox) / res) + 1)
    j0 = max(0, int((cy - r - oy) / res))
    j1 = min(hh, int((cy + r - oy) / res) + 1)
    if i0 >= i1 or j0 >= j1:
        return
    xs = ox + (np.arange(i0, i1) + 0.5) * res
    ys = oy + (np.arange(j0, j1) + 0.5) * res
    dx = xs[None, :] - cx
    dy = ys[:, None] - cy
    mask = (dx * dx + dy * dy) <= r * r
    grid[j0:j1, i0:i1][mask] = value


def _obstacle_bgr(obs):
    """Resolves an obstacle's base colour (BGR) from its dict.

    Uses an explicit ``color_bgr`` (BGR), else ``color`` ([r, g, b] in RGB),
    else the default object colour.
    """
    if obs.get('color_bgr') is not None:
        return tuple(int(c) for c in obs['color_bgr'][:3])
    c = obs.get('color')
    if c is None:
        return OBSTACLE_BGR
    r, g, b = [int(v) for v in c[:3]]
    return (b, g, r)


def _obstacle_height(obs):
    """Vertical height (m) of an obstacle, default OBSTACLE_HEIGHT_M."""
    return float(obs.get('height', OBSTACLE_HEIGHT_M))


def rasterize_obstacle(occ, color_grid, height_grid, res, ox, oy, obs):
    """Rasterize one obstacle into ``occ`` (and ``color_grid``/``height_grid``).

    obs = {type: box|rect|circle, x, y, ...}: a box/rect needs w, h and optional
    yaw_deg; a circle needs r; both take optional ``color`` and ``height``.
    Shared by ``World.from_spec`` (static obstacles) and the RL env (per-episode
    random obstacles). Pass None for grids not needed (e.g. reachability)."""
    t = str(obs.get('type', 'box')).lower()
    col = _obstacle_bgr(obs)
    ht = _obstacle_height(obs)
    if t in ('circle', 'disc', 'cylinder'):
        r = float(obs['r'])
        _rasterize_circle(occ, res, ox, oy, obs['x'], obs['y'], r, True)
        if color_grid is not None:
            _rasterize_circle(color_grid, res, ox, oy, obs['x'], obs['y'], r,
                              col)
        if height_grid is not None:
            _rasterize_circle(height_grid, res, ox, oy, obs['x'], obs['y'], r,
                              ht)
    else:  # box / rect
        w, h = float(obs['w']), float(obs['h'])
        yaw = math.radians(float(obs.get('yaw_deg', 0.0)))
        _rasterize_rect(occ, res, ox, oy, obs['x'], obs['y'], w, h, yaw, True)
        if color_grid is not None:
            _rasterize_rect(color_grid, res, ox, oy, obs['x'], obs['y'], w, h,
                            yaw, col)
        if height_grid is not None:
            _rasterize_rect(height_grid, res, ox, oy, obs['x'], obs['y'], w, h,
                            yaw, ht)


def inflate_occ(occ, cells):
    """Square (Chebyshev) dilation of a bool occupancy grid by ``cells`` cells.

    The configuration-space trick for a finite-radius robot: inflate the walls
    by the footprint radius (in cells) and treat the robot as a point for
    reachability / geodesic planning. Square dilation over-inflates corners
    slightly, which is the safe direction (it never reports a too-narrow gap as
    passable). Pure numpy, O(cells * grid), reset-time only."""
    cells = int(cells)
    if cells <= 0:
        return occ.copy()
    horiz = occ.copy()
    for s in range(1, cells + 1):
        horiz[:, s:] |= occ[:, :-s]
        horiz[:, :-s] |= occ[:, s:]
    out = horiz.copy()
    for s in range(1, cells + 1):
        out[s:, :] |= horiz[:-s, :]
        out[:-s, :] |= horiz[s:, :]
    return out


def bfs_reachable(occ, start_cell, goal_cell):
    """True if goal is reachable from start over free (~occ) cells, 4-connected.
    Cells are (i, j) = (col, row), matching ``World.world_to_cell``."""
    h, w = occ.shape
    si, sj = int(start_cell[0]), int(start_cell[1])
    gi, gj = int(goal_cell[0]), int(goal_cell[1])
    if not (0 <= si < w and 0 <= sj < h and 0 <= gi < w and 0 <= gj < h):
        return False
    if occ[sj, si] or occ[gj, gi]:
        return False
    seen = np.zeros((h, w), dtype=bool)
    seen[sj, si] = True
    q = deque([(si, sj)])
    while q:
        i, j = q.popleft()
        if i == gi and j == gj:
            return True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < w and 0 <= nj < h and not occ[nj, ni] and not seen[nj,
                                                                            ni]:
                seen[nj, ni] = True
                q.append((ni, nj))
    return False


def nearest_free_cell(occ, cell, max_radius=8):
    """Nearest free (~occ) cell to ``cell`` (i, j), spiralling outward up to a
    Chebyshev radius. Returns ``cell`` if already free, else the closest free
    cell, else None. Used to snap a footprint-valid pose onto the inflated
    planning grid (where it can sit one cell inside the dilated wall)."""
    i0, j0 = int(cell[0]), int(cell[1])
    h, w = occ.shape
    if 0 <= i0 < w and 0 <= j0 < h and not occ[j0, i0]:
        return (i0, j0)
    for r in range(1, max_radius + 1):
        for dj in range(-r, r + 1):
            for di in range(-r, r + 1):
                if max(abs(di), abs(dj)) != r:
                    continue  # ring at Chebyshev radius r only
                i, j = i0 + di, j0 + dj
                if 0 <= i < w and 0 <= j < h and not occ[j, i]:
                    return (i, j)
    return None


def bfs_distance_field(occ, goal_cell):
    """Integer cell-distance from goal over free (~occ) cells, 4-connected.
    Occupied or unreachable cells are -1; multiply by ``World.res`` for metres.
    One field per reset serves the geodesic reward, spawn distance gating, and
    the reachability guard. Cells are (i, j) = (col, row)."""
    h, w = occ.shape
    dist = np.full((h, w), -1, dtype=np.int32)
    gi, gj = int(goal_cell[0]), int(goal_cell[1])
    if not (0 <= gi < w and 0 <= gj < h) or occ[gj, gi]:
        return dist
    dist[gj, gi] = 0
    q = deque([(gi, gj)])
    while q:
        i, j = q.popleft()
        d = dist[j, i]
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < w and 0 <= nj < h and not occ[nj, ni] and dist[nj,
                                                                        ni] < 0:
                dist[nj, ni] = d + 1
                q.append((ni, nj))
    return dist
