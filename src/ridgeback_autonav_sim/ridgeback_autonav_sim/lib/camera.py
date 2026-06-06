"""Pinhole camera model: project map-frame signs to pixels + depth, render them.

The optical frame follows REP 103 (x right, y down, z forward). The base frame
is FLU (x forward, y left, z up). ``R_BO`` maps an optical-frame vector into the
base frame; the sim broadcasts the matching ``base_link -> optical`` static
TF so that perception_node's inverse (pixel -> camera -> TF -> map) recovers the
exact sign position. ``unproject_to_map`` mirrors that inverse and exists so a
unit test can prove the round trip.

Geometry is pure numpy; ``render`` additionally uses OpenCV (lazy import).
"""
import math

import numpy as np

from .raycast import raycast
from .world import WALL_BGR

# Columns are the optical axes expressed in the base frame:
#   optical x (right)   = base -y
#   optical y (down)    = base -z
#   optical z (forward) = base +x
R_BO = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
R_OB = R_BO.T


def _rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_to_quaternion(rm):
    """(x, y, z, w) quaternion from a 3x3 rotation matrix."""
    tr = rm[0, 0] + rm[1, 1] + rm[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (rm[2, 1] - rm[1, 2]) * s
        y = (rm[0, 2] - rm[2, 0]) * s
        z = (rm[1, 0] - rm[0, 1]) * s
    elif rm[0, 0] > rm[1, 1] and rm[0, 0] > rm[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rm[0, 0] - rm[1, 1] - rm[2, 2])
        w = (rm[2, 1] - rm[1, 2]) / s
        x = 0.25 * s
        y = (rm[0, 1] + rm[1, 0]) / s
        z = (rm[0, 2] + rm[2, 0]) / s
    elif rm[1, 1] > rm[2, 2]:
        s = 2.0 * math.sqrt(1.0 + rm[1, 1] - rm[0, 0] - rm[2, 2])
        w = (rm[0, 2] - rm[2, 0]) / s
        x = (rm[0, 1] + rm[1, 0]) / s
        y = 0.25 * s
        z = (rm[1, 2] + rm[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + rm[2, 2] - rm[0, 0] - rm[1, 1])
        w = (rm[1, 0] - rm[0, 1]) / s
        x = (rm[0, 2] + rm[2, 0]) / s
        y = (rm[1, 2] + rm[2, 1]) / s
        z = 0.25 * s
    return (x, y, z, w)


def optical_static_quaternion():
    """Quaternion for the base_link -> camera_optical static TF."""
    return rotation_to_quaternion(R_BO)


def map_point_to_optical(p_map, robot_pose, cam_offset):
    """Map-frame point -> camera optical frame (x,y,z)."""
    rx, ry, yaw = robot_pose
    p = np.asarray(p_map, dtype=np.float64) - np.array([rx, ry, 0.0])
    p_base = _rz(-yaw) @ p
    p_mount = p_base - np.asarray(cam_offset, dtype=np.float64)
    return R_OB @ p_mount


def project(p_map,
            robot_pose,
            cam_offset,
            K,
            width,
            height,
            min_range=0.3,
            max_range=8.0):
    """Project a map point to (u, v, depth, visible)."""
    fx, fy, cx, cy = K
    xo, yo, zo = map_point_to_optical(p_map, robot_pose, cam_offset)
    if zo <= 1e-3:
        return (0.0, 0.0, 0.0, False)
    u = fx * xo / zo + cx
    v = fy * yo / zo + cy
    visible = (min_range <= zo <= max_range and 0 <= u < width and
               0 <= v < height)
    return (float(u), float(v), float(zo), bool(visible))


def unproject_to_map(u, v, depth, robot_pose, cam_offset, K):
    """Inverse of project (mirrors perception's pixel->camera->TF->map)."""
    fx, fy, cx, cy = K
    p_opt = np.array([(u - cx) * depth / fx, (v - cy) * depth / fy, depth])
    p_mount = R_BO @ p_opt
    p_base = p_mount + np.asarray(cam_offset, dtype=np.float64)
    rx, ry, yaw = robot_pose
    p_map = np.array([rx, ry, 0.0]) + _rz(yaw) @ p_base
    return p_map


def render(visible_signs,
           width,
           height,
           K,
           plaque_w=0.5,
           plaque_h=0.28,
           bg_color=(120, 120, 120)):
    """Render a synthetic RGB image + aligned depth (metres) of the signs.

    ``visible_signs`` is a list of (text, u, v, depth). Each sign is drawn as a
    white plaque with black text, scaled by 1/depth; the depth image carries the
    sign distance over the plaque (0 elsewhere == invalid). Returns
    (bgr uint8 HxWx3, depth float32 HxW).
    """
    import cv2
    fx = K[0]
    color = np.zeros((height, width, 3), dtype=np.uint8)
    color[:] = bg_color
    depth = np.zeros((height, width), dtype=np.float32)
    for (text, u, v, z) in visible_signs:
        if z <= 1e-3:
            continue
        pw = max(8, int(fx * plaque_w / z))
        ph = max(5, int(fx * plaque_h / z))
        x1 = int(round(u - pw / 2))
        x2 = int(round(u + pw / 2))
        y1 = int(round(v - ph / 2))
        y2 = int(round(v + ph / 2))
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(width, x2), min(height, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        color[cy1:cy2, cx1:cx2] = (255, 255, 255)
        depth[cy1:cy2, cx1:cx2] = z
        scale = max(0.3, (cy2 - cy1) / 30.0)
        thick = max(1, int(scale))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale,
                                      thick)
        tx = int(cx1 + max(0, ((cx2 - cx1) - tw) / 2))
        ty = int(cy1 + min((cy2 - cy1) - 1, ((cy2 - cy1) + th) / 2))
        cv2.putText(color, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), thick, cv2.LINE_AA)
    return color, depth


# A software perspective ("raycasting") renderer in the spirit of the classic
# 2.5D engines: cast one ray per pixel column into the occupancy grid for the
# wall distance, then fill floor and ceiling with per-row depth gradients so the
# whole frame carries depth (not just the signs). Pure numpy + a little OpenCV
# for the sign text. Returns (bgr uint8 HxWx3, depth float32 HxW metres; 0 ==
# invalid). This is what makes the onboard depth image look like a real sensor.
FLOOR_BGR = (95, 98, 102)
CEIL_BGR = (170, 172, 176)
_LIGHT_DIR = math.radians(-35.0)  # fixed key light (world frame)


def render_scene(world,
                 robot_pose,
                 cam_offset,
                 K,
                 width,
                 height,
                 max_depth=10.0,
                 range_min=0.06,
                 wall_height_m=2.5,
                 ceiling_height_m=3.0):
    import cv2
    fx, fy, cx, cy = [float(v) for v in K]
    rx, ry, yaw = robot_pose
    camx, camy, camz = cam_offset
    # Camera world position (base x forward, y left), mount rotated by yaw.
    cam_x = rx + math.cos(yaw) * camx - math.sin(yaw) * camy
    cam_y = ry + math.sin(yaw) * camx + math.cos(yaw) * camy
    cam_z = float(camz)

    color = np.empty((height, width, 3), dtype=np.uint8)
    color[:] = CEIL_BGR
    depth_m = np.full((height, width), np.inf, dtype=np.float32)

    u = np.arange(width, dtype=np.float64)
    ray_local = np.arctan((u - cx) / fx)  # +right of optical axis
    # raycast casts at (ltheta + angles); world heading = yaw - ray_local
    d = raycast(world.occ, world.res, world.origin_x, world.origin_y, cam_x,
                cam_y, yaw, -ray_local, max_depth, range_min)
    world_head = yaw - ray_local

    hit = np.isfinite(d) & (d > 0.02) & (d < max_depth - 1e-3)
    # Hit cell per column, reused for per-object height and colour. Recomputing
    # it from d keeps raycast unchanged.
    hgrid = getattr(world, "height_grid", None)
    cgrid = getattr(world, "color_grid", None)
    hi = np.clip(((cam_x + d * np.cos(world_head) - world.origin_x) /
                  world.res).astype(np.int64), 0, world.width - 1)
    hj = np.clip(((cam_y + d * np.sin(world_head) - world.origin_y) /
                  world.res).astype(np.int64), 0, world.height - 1)
    # Wall/object top from the hit cell's height (shorter objects -> lower top).
    obj_h = hgrid[hj, hi] if hgrid is not None else wall_height_m
    z_top = obj_h - cam_z
    z_bot = -cam_z
    with np.errstate(divide='ignore', invalid='ignore'):
        v_top = np.round(cy - fy * (z_top / d)).astype(np.int64)
        v_bot = np.round(cy - fy * (z_bot / d)).astype(np.int64)
    wall_start = np.clip(np.minimum(v_top, v_bot), 0, height - 1)
    wall_end = np.clip(np.maximum(v_top, v_bot), 0, height - 1)
    # Per-column shade: ambient + directional diffuse + distance falloff.
    diffuse = np.maximum(0.12, np.cos(world_head - _LIGHT_DIR))
    dist_att = 1.0 / (1.0 + 0.055 * d * d)
    shade = np.clip((0.30 + 0.70 * diffuse) * dist_att, 0.18, 1.0)
    # Per-column base colour from the same hit cell (own colour per object).
    if cgrid is not None:
        base_cols = cgrid[hj, hi].astype(np.float32)
    else:
        base_cols = np.broadcast_to(np.array(WALL_BGR, np.float32), (width, 3))
    wall_col = np.clip(base_cols * shade[:, None], 0, 255).astype(np.uint8)
    for uu in range(width):
        if not hit[uu]:
            continue
        s, e = int(wall_start[uu]), int(wall_end[uu])
        if e >= s:
            color[s:e + 1, uu] = wall_col[uu]
            depth_m[s:e + 1, uu] = d[uu]

    rows = np.arange(height, dtype=np.float32)
    _fill_plane(color,
                depth_m,
                rows,
                cy,
                fy,
                cam_z,
                FLOOR_BGR,
                0.05,
                max_depth,
                below=True)
    _fill_plane(color,
                depth_m,
                rows,
                cy,
                fy,
                ceiling_height_m - cam_z,
                CEIL_BGR,
                0.03,
                max_depth,
                below=False)

    # Each sign is a real oriented quad in 3D; we project its corners and warp
    # the text texture onto that quad (with a depth z-buffer). Viewed off-axis
    # it foreshortens; only a head-on view looks "straightened".
    cam_world = np.array([cam_x, cam_y, cam_z], dtype=np.float64)
    for sign in world.signs:
        normal, corners, center = _sign_geometry(sign)
        # Back-face cull: the camera must be on the side the sign faces.
        if float(np.dot(normal, cam_world - center)) <= 0.0:
            continue
        cam_pts = [
            map_point_to_optical(c, robot_pose, cam_offset) for c in corners
        ]
        if any(p[2] <= 0.05 for p in cam_pts):  # behind / through camera
            continue
        dst = np.array(
            [[fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy] for p in cam_pts],
            dtype=np.float32)
        if (dst[:, 0].max() < -2 or dst[:, 0].min() > width + 2 or
                dst[:, 1].max() < -2 or dst[:, 1].min() > height + 2):
            continue  # fully off-screen
        tex, alpha, tw, th = _sign_texture(sign.text, sign.w, sign.h)
        src = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]],
                       dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(tex,
                                     M, (width, height),
                                     flags=cv2.INTER_LINEAR,
                                     borderValue=(0, 0, 0))
        walpha = cv2.warpPerspective(alpha,
                                     M, (width, height),
                                     flags=cv2.INTER_LINEAR,
                                     borderValue=0)
        # Per-pixel depth: interpolate the corners' camera-z, then warp.
        z_tl, z_tr, z_br, z_bl = [float(p[2]) for p in cam_pts]
        xs = np.linspace(0.0, 1.0, tw, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, th, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        z_src = ((1 - xx) * (1 - yy) * z_tl + xx * (1 - yy) * z_tr +
                 xx * yy * z_br + (1 - xx) * yy * z_bl).astype(np.float32)
        wz = cv2.warpPerspective(z_src,
                                 M, (width, height),
                                 flags=cv2.INTER_LINEAR,
                                 borderValue=float(max_depth + 1.0))
        mask = (walpha > 20) & (wz < depth_m + 0.06
                               )  # bias so it beats its wall
        if not mask.any():
            continue
        color[mask] = warped[mask]
        depth_m[mask] = wz[mask]

    # Finalize depth: invalid (no geometry / out of range) -> 0.
    valid = np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m <= max_depth)
    depth_out = np.where(valid, depth_m, 0.0).astype(np.float32)
    return color, depth_out


_SIGN_TEX = {}  # cache: (text, tw, th) -> (bgr, alpha)


def _sign_geometry(sign):
    """Return (outward normal, 4 corners TL/TR/BR/BL, centre) for a sign."""
    yaw = sign.yaw
    normal = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    right = np.array([-math.sin(yaw), math.cos(yaw), 0.0])  # along plaque width
    up = np.array([0.0, 0.0, 1.0])
    center = np.array([sign.x, sign.y, sign.z])
    hw, hh = sign.w * 0.5, sign.h * 0.5
    corners = [
        center - right * hw + up * hh,  # top-left
        center + right * hw + up * hh,  # top-right
        center + right * hw - up * hh,  # bottom-right
        center - right * hw - up * hh
    ]  # bottom-left
    return normal, corners, center


def _sign_texture(text, w_m, h_m):
    """White plaque with a dark border and centred black text. Cached."""
    import cv2
    th = 160
    tw = max(40, int(round(th * (w_m / max(1e-3, h_m)))))
    key = (text, tw, th)
    if key in _SIGN_TEX:
        img, alpha = _SIGN_TEX[key]
        return img, alpha, tw, th
    img = np.full((th, tw, 3), 245, dtype=np.uint8)  # off-white plaque
    cv2.rectangle(img, (2, 2), (tw - 3, th - 3), (40, 40, 40),
                  max(2, th // 36))  # dark frame
    scale = th / 70.0
    thick = max(2, int(round(scale * 1.6)))
    (twd, thd), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale,
                                    thick)
    tx = max(0, (tw - twd) // 2)
    ty = (th + thd) // 2
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (20, 20, 20), thick, cv2.LINE_AA)
    alpha = np.full((th, tw), 255, dtype=np.uint8)
    _SIGN_TEX[key] = (img, alpha)
    return img, alpha, tw, th


def _fill_plane(color, depth_m, rows, cy, fy, rel_height, base_bgr, atten,
                max_depth, below):
    """Fill floor (below) or ceiling (above) with per-row depth + shade."""
    h, w = depth_m.shape
    denom = (rows - cy) if below else (cy - rows)
    pd = np.full(h, np.inf, dtype=np.float32)
    m = denom > 1.0
    pd[m] = (abs(rel_height) * fy) / denom[m]
    pd = np.minimum(pd, max_depth)
    pd2 = np.broadcast_to(pd[:, None], (h, w))
    mask = np.isfinite(pd2) & (pd2 < depth_m)
    if not mask.any():
        return
    pshade = 1.0 / (1.0 + atten * pd)
    pcol = np.clip(
        np.array(base_bgr, dtype=np.float32)[None, :] * pshade[:, None], 0,
        255).astype(np.uint8)
    pcol2 = np.broadcast_to(pcol[:, None, :], (h, w, 3))
    color[mask] = pcol2[mask]
    depth_m[mask] = pd2[mask]
