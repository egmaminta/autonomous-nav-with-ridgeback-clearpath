"""Pixel + depth -> 3D point projection (pure numpy, unit-testable)."""
import numpy as np


def robust_depth(depth_img, bbox, window=7, is_mm=False,
                 dmin=0.3, dmax=8.0, min_valid=10):
    """Robust depth (metres) at the centre of a bbox.

    Takes the median of finite, in-range depth samples in a ``window x window``
    patch at the bbox centre. Returns ``None`` if too few valid samples.

    Args:
        depth_img: (H, W) array; uint16 millimetres if ``is_mm`` else float metres.
        bbox: (x1, y1, x2, y2) in pixels.
        window: side length of the central sampling patch.
        is_mm: True if depth is uint16 millimetres.
    """
    h, w = depth_img.shape[:2]
    x1, y1, x2, y2 = bbox
    cx = int(round((x1 + x2) / 2.0))
    cy = int(round((y1 + y2) / 2.0))
    half = max(1, window // 2)
    x0, xe = max(0, cx - half), min(w, cx + half + 1)
    y0, ye = max(0, cy - half), min(h, cy + half + 1)
    patch = np.asarray(depth_img[y0:ye, x0:xe], dtype=np.float64)
    if is_mm:
        patch = patch / 1000.0
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    valid = valid[(valid >= dmin) & (valid <= dmax)]
    if valid.size < min_valid:
        return None
    return float(np.median(valid))


def pixel_to_camera(u, v, z, fx, fy, cx, cy):
    """Back-project a pixel + depth to a 3D point in the camera optical frame.

    Optical-frame convention: +x right, +y down, +z forward (REP 103 / OpenCV).
    """
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return (x, y, z)


def intrinsics_from_k(k):
    """Extract (fx, fy, cx, cy) from a row-major 3x3 K (len-9 sequence)."""
    return (k[0], k[4], k[2], k[5])
