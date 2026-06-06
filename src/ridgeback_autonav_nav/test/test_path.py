"""Unit tests for path smoothing."""
import math

from ridgeback_autonav_nav.lib import path_utils as pu


def test_chaikin_preserves_endpoints():
    pts = [(0, 0), (1, 0), (2, 1), (3, 1)]
    sm = pu.chaikin_smooth(pts, iterations=2)
    assert sm[0] == (0.0, 0.0)
    assert sm[-1] == (3.0, 1.0)
    assert len(sm) > len(pts)            # corner-cutting adds points


def test_chaikin_straight_line_stays_straight():
    pts = [(0, 0), (1, 0), (2, 0), (3, 0)]
    sm = pu.chaikin_smooth(pts, iterations=3)
    assert all(abs(y) < 1e-9 for (_, y) in sm)


def test_chaikin_rounds_corner():
    # An L: the sharp corner at (1,0) should be cut (no smoothed point sits
    # exactly on the original corner).
    pts = [(0, 0), (1, 0), (1, 1)]
    sm = pu.chaikin_smooth(pts, iterations=2)
    assert all(not (abs(x - 1.0) < 1e-6 and abs(y) < 1e-6) for (x, y) in sm)


def test_smooth_path_safe_fallback_on_collision():
    pts = [(0, 0), (1, 0), (2, 1), (3, 1)]
    # Pretend everything is blocked -> must fall back to the original points.
    out = pu.smooth_path_safe(pts, lambda x, y: False)
    assert out == [(float(a), float(b)) for (a, b) in pts]


def test_smooth_path_safe_smooths_when_free():
    pts = [(0, 0), (1, 0), (2, 1), (3, 1)]
    out = pu.smooth_path_safe(pts, lambda x, y: True, iterations=2)
    assert len(out) > len(pts)
    assert out[0] == (0.0, 0.0) and out[-1] == (3.0, 1.0)


def test_short_path_unchanged():
    assert pu.chaikin_smooth([(0, 0), (1, 1)]) == [(0.0, 0.0), (1.0, 1.0)]
