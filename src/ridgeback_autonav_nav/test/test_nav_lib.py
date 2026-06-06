"""Unit tests for the ridgeback_autonav_nav pure-math library (no ROS, no hardware)."""
import math

import numpy as np

from ridgeback_autonav_nav.lib import astar, scan_matcher as sm
from ridgeback_autonav_nav.lib.dwa_controller import DWAConfig, compute_cmd
from ridgeback_autonav_nav.lib.frontier import detect_frontiers
from ridgeback_autonav_nav.lib.occupancy_grid import OccupancyGrid2D, costmap_from_occupancy
from ridgeback_autonav_nav.lib import tf_utils as tf


def _box_scan(n=360, walls=(2.0, -2.0, 2.0, -2.0)):
    angles = np.linspace(-math.pi, math.pi, n, endpoint=False)
    xp, xn, yp, yn = walls
    ranges = []
    for a in angles:
        cx, cy = math.cos(a), math.sin(a)
        ts = []
        for wall, comp in ((xp, cx), (xn, cx), (yp, cy), (yn, cy)):
            if abs(comp) > 1e-6:
                t = wall / comp
                if t > 0:
                    ts.append(t)
        ranges.append(min(ts) if ts else float('inf'))
    return np.array(ranges), angles


# ---- tf_utils --------------------------------------------------------------
def test_compose_invert_identity():
    a = (1.0, 2.0, 0.5)
    inv = tf.invert_2d(*a)
    comp = tf.compose_2d(*a, *inv)
    assert all(abs(c) < 1e-9 for c in comp)


def test_point_to_robot_frame():
    # robot at (1,0) facing +90deg; map point (1,1) is straight ahead (x=1,y=0)
    x, y = tf.point_to_robot_frame(1.0, 1.0, 1.0, 0.0, math.pi / 2)
    assert abs(x - 1.0) < 1e-9 and abs(y) < 1e-9


def test_world_cell_roundtrip():
    i, j = tf.world_to_cell(1.23, -4.56, -10.0, -10.0, 0.05)
    x, y = tf.cell_to_world(i, j, -10.0, -10.0, 0.05)
    assert abs(x - 1.23) <= 0.05 and abs(y + 4.56) <= 0.05


# ---- occupancy grid --------------------------------------------------------
def test_grid_integrate_marks_free_and_occupied():
    g = OccupancyGrid2D(resolution=0.05, size_m=10.0)
    ranges, angles = _box_scan()
    g.integrate_scan(0.0, 0.0, 0.0, ranges, angles, max_range=10.0)
    cls = g.to_int8()
    assert (cls == 0).sum() > 1000     # interior free
    assert (cls == 100).sum() > 50     # walls occupied
    # cell at origin should be free
    i, j = g.world_to_grid(0.0, 0.0)
    assert cls[int(j), int(i)] == 0


def test_costmap_lethal_near_obstacle():
    g = OccupancyGrid2D(resolution=0.05, size_m=10.0)
    ranges, angles = _box_scan()
    g.integrate_scan(0.0, 0.0, 0.0, ranges, angles, max_range=10.0)
    cost = costmap_from_occupancy(g.to_int8(), 0.05, 0.2, 0.4)
    assert cost.max() >= 254.0          # some lethal cells exist
    i, j = g.world_to_grid(0.0, 0.0)
    assert cost[int(j), int(i)] < 254.0  # robot's own free cell is traversable


# ---- A* --------------------------------------------------------------------
def test_astar_finds_path_open_grid():
    cost = np.zeros((50, 50), dtype=np.float32)
    path = astar.plan(cost, (2, 2), (45, 40))
    assert path and path[0] == (2, 2) and path[-1] == (45, 40)


def test_astar_no_path_through_wall():
    cost = np.zeros((50, 50), dtype=np.float32)
    cost[:, 25] = 254.0  # full vertical wall
    assert astar.plan(cost, (2, 2), (45, 40)) == []


def test_nearest_free_cell():
    cost = np.zeros((20, 20), dtype=np.float32)
    cost[10, 10] = 254.0
    c = astar.nearest_free_cell(cost, (10, 10))
    assert c is not None and cost[c[1], c[0]] < 254.0


# ---- frontier --------------------------------------------------------------
def test_frontier_detects_free_unknown_border():
    occ = np.full((30, 30), -1, dtype=np.int8)
    occ[10:20, 10:20] = 0          # a free block surrounded by unknown
    occ[14:16, 14:16] = 100        # a little obstacle inside
    frs = detect_frontiers(occ, min_size=1)
    assert len(frs) >= 1
    assert max(f.size for f in frs) > 0


def test_frontier_none_when_fully_known():
    occ = np.zeros((20, 20), dtype=np.int8)   # all free, no unknown
    assert detect_frontiers(occ, min_size=1) == []


# ---- scan matcher ----------------------------------------------------------
def test_scan_matcher_recovers_offset():
    g = OccupancyGrid2D(resolution=0.05, size_m=12.0)
    ranges, angles = _box_scan(walls=(2.0, -3.0, 2.0, -2.0))  # asymmetric
    g.integrate_scan(0.0, 0.0, 0.0, ranges, angles, max_range=10.0)
    bx, by, bth, score = sm.match(g, -0.05, 0.05, -0.02, ranges, angles, max_range=10.0)
    assert math.hypot(bx, by) < 0.05 and abs(bth) < 0.05 and score > 0


# ---- DWA -------------------------------------------------------------------
def test_dwa_returns_command_toward_carrot():
    cfg = DWAConfig(robot_radius=0.2)
    cmd, score = compute_cmd(cfg, (0.0, 0.0, 0.0), carrot=(1.0, 0.0),
                             goal=(1.0, 0.0), goal_yaw=0.0,
                             obstacles=np.zeros((0, 2)), dist_to_goal=1.0)
    assert cmd is not None and cmd[0] > 0.0   # moves forward


def test_dwa_blocked_head_on_returns_none_or_turns():
    cfg = DWAConfig(robot_radius=0.3, max_vy=0.0)  # disable strafe
    obs = np.array([[d, 0.0] for d in np.linspace(0.1, 0.5, 9)])  # wall dead ahead
    cmd, _ = compute_cmd(cfg, (0.0, 0.0, 0.0), carrot=(1.0, 0.0), goal=(1.0, 0.0),
                         goal_yaw=0.0, obstacles=obs, dist_to_goal=1.0)
    # Either no safe straight-ahead command, or it must not drive forward into it.
    assert cmd is None or cmd[0] <= 1e-6


def test_dwa_no_strafe_when_disabled():
    # carrot to the LEFT; with strafe disabled the robot must NOT command vy
    # (it should turn instead) -> drive-like, no crabbing.
    cfg = DWAConfig(robot_radius=0.2, max_vy=0.1)
    cmd, _ = compute_cmd(cfg, (0.0, 0.0, 0.0), carrot=(0.2, 1.0), goal=(0.2, 1.0),
                         goal_yaw=0.0, obstacles=np.zeros((0, 2)), dist_to_goal=1.0,
                         allow_strafe=False)
    assert cmd is not None and abs(cmd[1]) < 1e-9    # vy == 0 (no crab)


def test_information_gain_counts_unknown():
    from ridgeback_autonav_nav.lib.frontier import information_gain
    occ = np.full((50, 50), -1, dtype=np.int8)   # all unknown
    occ[24:27, 24:27] = 0                          # 9 free cells at centre
    assert information_gain(occ, 25, 25, 5) == 121 - 9   # 11x11 window minus free


def test_camera_visibility_sees_wall_ahead():
    # 0.05 m grid, origin at (0,0). Robot at centre of a free room with a wall
    # to the EAST (occupied column). Facing +x -> high visibility; facing -x
    # (open) -> zero.
    from ridgeback_autonav_nav.lib.frontier import camera_visibility
    occ = np.zeros((40, 40), dtype=np.int8)        # all free
    occ[:, 30] = 100                                # vertical wall at col 30
    res, ox, oy = 0.05, 0.0, 0.0
    rx, ry = 20 * res, 20 * res                     # robot near col 20, row 20
    fov_half = math.radians(70.0) / 2.0
    facing_wall = camera_visibility(occ, res, ox, oy, rx, ry, 0.0, fov_half, 4.0, 9)
    facing_away = camera_visibility(occ, res, ox, oy, rx, ry, math.pi, fov_half, 4.0, 9)
    assert facing_wall > 0.5      # most of the FOV meets the wall
    assert facing_away == 0.0     # open space behind, no wall in range
