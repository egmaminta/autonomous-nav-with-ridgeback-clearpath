"""Unit tests for ridgeback_autonav_perception pure-math library (no torch, no ROS)."""
import numpy as np

from ridgeback_autonav_perception.lib.projection import (intrinsics_from_k, pixel_to_camera,
                                            robust_depth)
from ridgeback_autonav_perception.lib.sign_registry import SignRegistry, normalize_text


# ---- projection ------------------------------------------------------------
def test_intrinsics_from_k():
    k = [200.0, 0.0, 320.0, 0.0, 210.0, 240.0, 0.0, 0.0, 1.0]
    fx, fy, cx, cy = intrinsics_from_k(k)
    assert (fx, fy, cx, cy) == (200.0, 210.0, 320.0, 240.0)


def test_pixel_to_camera_center_is_on_axis():
    x, y, z = pixel_to_camera(320, 240, 2.0, 200, 200, 320, 240)
    assert abs(x) < 1e-9 and abs(y) < 1e-9 and z == 2.0


def test_pixel_to_camera_offset():
    # one focal length right of centre at 2 m -> x == 2 m
    x, _, _ = pixel_to_camera(520, 240, 2.0, 200, 200, 320, 240)
    assert abs(x - 2.0) < 1e-9


def test_robust_depth_median_float_metres():
    d = np.full((100, 100), 3.0, dtype=np.float32)
    assert robust_depth(d, (40, 40, 60, 60)) == 3.0


def test_robust_depth_handles_mm_and_zeros():
    d = np.full((100, 100), 2500, dtype=np.uint16)  # 2.5 m in mm
    d[45:55, 45:55][::2] = 0                          # some invalid zeros
    z = robust_depth(d, (40, 40, 60, 60), is_mm=True)
    assert abs(z - 2.5) < 1e-6


def test_robust_depth_rejects_too_few_valid():
    d = np.zeros((100, 100), dtype=np.float32)
    assert robust_depth(d, (40, 40, 60, 60)) is None


# ---- sign registry ---------------------------------------------------------
def test_normalize_text():
    assert normalize_text(' #206 ') == '206'
    assert normalize_text('12b') == '12B'


def test_registry_confirms_after_min_observations():
    r = SignRegistry(min_observations=3, cluster_radius=0.75)
    for i in range(3):
        e = r.observe('206', 5.0 + 0.02 * i, 2.0, 0.9, now=i)
    assert e.confirmed
    assert r.find('206') is not None
    assert abs(e.x - 5.0) < 0.1 and abs(e.y - 2.0) < 0.1


def test_registry_not_confirmed_before_min():
    r = SignRegistry(min_observations=3)
    r.observe('301', 1.0, 1.0, 0.9, now=0)
    r.observe('301', 1.02, 1.0, 0.9, now=1)
    assert r.find('301') is None            # confirmed_only default
    assert r.find('301', confirmed_only=False) is not None


def test_registry_far_same_text_is_separate_sign():
    r = SignRegistry(min_observations=1, cluster_radius=0.75)
    a = r.observe('206', 0.0, 0.0, 0.9, now=0)
    b = r.observe('206', 50.0, 50.0, 0.9, now=1)  # 70 m away -> different sign
    assert a is not b
    assert len(r.entries) == 2


def test_registry_spread_gate_blocks_confirmation():
    # observations too spread out should not confirm even past the count.
    r = SignRegistry(min_observations=3, cluster_radius=0.75, max_spread=0.1)
    r.observe('400', 0.0, 0.0, 0.9, now=0)
    r.observe('400', 0.0, 0.4, 0.9, now=1)
    r.observe('400', 0.0, 0.0, 0.9, now=2)
    e = r.find('400', confirmed_only=False)
    assert e is not None and not e.confirmed
