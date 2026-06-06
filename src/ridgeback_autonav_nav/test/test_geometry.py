"""Unit tests for no-go zone geometry."""
import numpy as np

from ridgeback_autonav_nav.lib import geometry as g


def test_parse_zone_strings():
    zones = g.parse_zone_strings(["9.5 3.0 10.5 3.0 10.5 5.0 9.5 5.0", "bad"])
    assert len(zones) == 1
    assert zones[0].shape == (4, 2)
    assert tuple(zones[0][0]) == (9.5, 3.0)


def test_parse_zone_comma_and_too_few():
    assert g.parse_zone_strings(["1,1 2,2"]) == []        # only 2 vertices -> dropped
    z = g.parse_zone_strings(["0,0 2,0 2,2 0,2"])
    assert len(z) == 1 and z[0].shape == (4, 2)


def test_point_in_polygon():
    sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
    assert g.point_in_polygon(1.0, 1.0, sq)
    assert not g.point_in_polygon(3.0, 1.0, sq)
    assert not g.point_in_polygon(-0.1, 1.0, sq)


def test_stamp_polygons_occupied():
    occ = np.zeros((40, 40), dtype=np.int8)   # 2x2 m grid at 0.05, origin 0,0
    poly = [np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]], float)]
    g.stamp_polygons_occupied(occ, poly, 0.0, 0.0, 0.05)
    # centre of the square should be occupied; a corner of the grid should not
    assert occ[20, 20] == 100
    assert occ[2, 2] == 0
    assert (occ == 100).sum() > 100


def test_polygon_edge_points():
    poly = [np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)]
    pts = g.polygon_edge_points(poly, spacing=0.1)
    assert pts.shape[0] >= 36                # ~4 m perimeter / 0.1
    # all points lie on the unit square boundary
    assert pts[:, 0].min() >= -1e-9 and pts[:, 0].max() <= 1.0 + 1e-9
