"""Unit tests for the ridgeback_autonav_sim pure-math libs (no ROS/torch)."""
import math

import numpy as np

from ridgeback_autonav_sim.lib import camera as cam
from ridgeback_autonav_sim.lib.raycast import raycast
from ridgeback_autonav_sim.lib.world import (World, _rasterize_rect,
                                             _rasterize_circle, inflate_occ,
                                             bfs_reachable, bfs_distance_field)


def _box_world(side=4.0, res=0.05):
    return World.from_spec({
        'resolution': res,
        'bounds': [0.0, 0.0, side, side],
        'border': True,
        'walls': [],
        'signs': [],
        'start': {
            'x': side / 2,
            'y': side / 2,
            'theta': 0.0
        }
    })


def test_world_border_occupied():
    w = _box_world()
    assert w.is_occupied(0.01, 2.0)  # left border
    assert w.is_occupied(3.99, 2.0)  # right border
    assert not w.is_occupied(2.0, 2.0)  # interior free


def test_world_wall_rasterized():
    w = World.from_spec({
        'resolution': 0.05,
        'bounds': [0, 0, 6, 6],
        'border': False,
        'walls': [[3.0, 0.0, 3.0, 6.0]],
        'signs': [],
        'start': {
            'x': 1,
            'y': 1,
            'theta': 0
        }
    })
    assert w.is_occupied(3.0, 2.5)  # on the wall
    assert not w.is_occupied(1.0, 2.5)  # off the wall


def test_raycast_hits_wall_ahead():
    w = _box_world(side=4.0)
    r = raycast(w.occ,
                w.res,
                w.origin_x,
                w.origin_y,
                2.0,
                2.0,
                0.0,
                np.array([0.0]),
                max_range=10.0)[0]
    assert 1.8 <= r <= 2.05  # right wall ~2 m ahead


def test_raycast_all_beams_bounded():
    w = _box_world(side=4.0)
    angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    r = raycast(w.occ,
                w.res,
                w.origin_x,
                w.origin_y,
                2.0,
                2.0,
                0.0,
                angles,
                max_range=10.0)
    assert np.all(r > 0.0) and np.all(
        r <= 2.95)  # inside a 4 m box, corners ~2*sqrt2


def test_optical_axes_convention():
    # A point 1 m directly in front (base +x) is optical +z (forward).
    opt = cam.map_point_to_optical((1.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                                   (0.0, 0.0, 0.0))
    assert np.allclose(opt, [0.0, 0.0, 1.0], atol=1e-9)


def test_optical_quaternion_unit():
    q = cam.optical_static_quaternion()
    assert abs(math.sqrt(sum(c * c for c in q)) - 1.0) < 1e-9


def test_project_unproject_round_trip():
    K = (525.0, 525.0, 320.0, 240.0)
    cam_off = (0.45, 0.0, 0.6)
    # (sign_map, robot_pose) chosen so the sign is genuinely in-frame (a 1.0 m
    # sign seen from ~3 m); the round trip itself is what we're proving.
    cases = [
        ((4.0, 2.0, 1.0), (1.0, 2.0, 0.0)),
        ((10.0, 3.15, 1.0), (6.5, 4.0, -0.15)),
        ((2.0, 4.0, 1.0), (2.0, 1.0, 1.4)),
    ]
    for sign_map, pose in cases:
        u, v, depth, vis = cam.project(sign_map, pose, cam_off, K, 640, 480)
        assert vis, f'sign should be visible from {pose}'
        rec = cam.unproject_to_map(u, v, depth, pose, cam_off, K)
        assert np.allclose(rec, sign_map, atol=1e-6), (rec, sign_map)


def test_project_behind_not_visible():
    K = (525.0, 525.0, 320.0, 240.0)
    # Sign behind the robot (robot faces +x, sign at -x).
    _, _, _, vis = cam.project((-2.0, 0.0, 1.4), (0.0, 0.0, 0.0),
                               (0.45, 0.0, 0.6), K, 640, 480)
    assert not vis


def test_rasterize_rect_in_occ_and_sensed():
    w = _box_world(side=6.0)
    _rasterize_rect(w.occ,
                    w.res,
                    w.origin_x,
                    w.origin_y,
                    3.0,
                    3.0,
                    0.8,
                    0.8,
                    yaw=0.0)
    assert w.is_occupied(3.0, 3.0)  # centre occupied
    assert not w.is_occupied(3.0, 4.2)  # 1.2 m away, outside the 0.8 m box
    # collision and LIDAR pick it up: a ray from x=1 toward the box hits
    # its near face (~x=2.6), i.e. ~1.6 m ahead.
    r = raycast(w.occ,
                w.res,
                w.origin_x,
                w.origin_y,
                1.0,
                3.0,
                0.0,
                np.array([0.0]),
                max_range=10.0)[0]
    assert 1.5 <= r <= 2.05


def test_rasterize_rect_rotated():
    w = _box_world(side=6.0)
    # long thin bar along the +45 deg diagonal (local x = long axis).
    _rasterize_rect(w.occ,
                    w.res,
                    w.origin_x,
                    w.origin_y,
                    3.0,
                    3.0,
                    2.0,
                    0.2,
                    yaw=math.pi / 4)
    assert w.is_occupied(3.0, 3.0)  # centre
    assert w.is_occupied(3.5, 3.5)  # along the long axis
    assert not w.is_occupied(3.5, 2.5)  # across the short axis


def test_rasterize_circle_in_occ():
    w = _box_world(side=6.0)
    _rasterize_circle(w.occ, w.res, w.origin_x, w.origin_y, 3.0, 3.0, 0.3)
    assert w.is_occupied(3.0, 3.0)
    assert w.is_occupied(3.0, 3.25)  # within r = 0.3
    assert not w.is_occupied(3.0, 3.45)  # outside r = 0.3


def test_bfs_reachable_open_vs_walled():
    w = _box_world(side=6.0)
    s = w.world_to_cell(1.0, 3.0)
    g = w.world_to_cell(5.0, 3.0)
    assert bfs_reachable(w.occ, s, g)
    w2 = World.from_spec({
        'resolution': 0.05,
        'bounds': [0, 0, 6, 6],
        'border': True,
        'walls': [[3.0, 0.0, 3.0, 6.0]],
        'signs': [],
        'start': {
            'x': 1,
            'y': 3,
            'theta': 0
        }
    })
    assert not bfs_reachable(w2.occ, w2.world_to_cell(1.0, 3.0),
                             w2.world_to_cell(5.0, 3.0))


def test_bfs_distance_field_monotonic():
    w = _box_world(side=6.0)
    g = w.world_to_cell(5.0, 3.0)
    field = bfs_distance_field(w.occ, g)
    gi, gj = g
    assert field[gj, gi] == 0
    prev = None
    for x in np.arange(1.0, 4.9, 0.2):  # step toward the goal along a free row
        i, j = w.world_to_cell(float(x), 3.0)
        d = field[j, i]
        assert d > 0
        if prev is not None:
            assert d < prev  # closer to goal -> smaller distance
        prev = d


def test_inflate_occ_grows_walls():
    w = _box_world(side=6.0)
    free_before = int((~w.occ).sum())
    inf = inflate_occ(w.occ, 3)
    assert int((~inf).sum()) < free_before  # walls grew inward
    i, j = w.world_to_cell(0.08, 3.0)  # ~ one cell inside the left border
    assert not w.occ[j, i]  # free before
    assert inf[j, i]  # covered after inflating by 3 cells


def test_object_height_renders_shorter_than_wall():
    # A short object fills fewer pixel rows than a full-height one at the same
    # spot, so furniture reads as low blocks rather than full pillars.
    def world_with(height_m):
        return World.from_spec({
            'resolution': 0.05,
            'bounds': [0, 0, 6, 6],
            'border': True,
            'obstacles': [{
                'type': 'box',
                'x': 3.0,
                'y': 4.0,
                'w': 1.2,
                'h': 1.2,
                'height': height_m,
                'color': [200, 80, 80]
            }],
            'signs': [{
                'text': 'X',
                'x': 5.9,
                'y': 3.0,
                'z': 1.0,
                'yaw_deg': 180
            }],
            'start': {
                'x': 1,
                'y': 3,
                'theta': 0
            }
        })

    k = (68.9, 68.9, 42.0, 42.0)

    def object_rows(world):
        color, _ = cam.render_scene(world, [3.0, 2.0, math.pi / 2],
                                    (0.45, 0.0, 0.6),
                                    k,
                                    84,
                                    84,
                                    max_depth=12.0)
        col = color[:, 42, :].astype(int)  # BGR centre column
        return int(((col[:, 2] - col[:, 0]) > 30).sum())  # reddish object rows

    assert 0 < object_rows(world_with(0.5)) < object_rows(world_with(2.5))
