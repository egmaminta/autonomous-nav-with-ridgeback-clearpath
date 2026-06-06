"""Small pose / quaternion helpers (tf_transformations is not installed)."""
import math


def yaw_from_quaternion(x, y, z, w):
    """Extract yaw (rotation about z) from a quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw):
    """Build a (x, y, z, w) quaternion from a yaw angle."""
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def normalize_angle(a):
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def angle_diff(a, b):
    """Smallest signed difference a - b, wrapped to (-pi, pi]."""
    return normalize_angle(a - b)


def compose_2d(ax, ay, atheta, bx, by, btheta):
    """Compose SE(2) transform A * B (apply B in A's frame)."""
    c, s = math.cos(atheta), math.sin(atheta)
    return (ax + c * bx - s * by,
            ay + s * bx + c * by,
            normalize_angle(atheta + btheta))


def invert_2d(x, y, theta):
    """Invert an SE(2) transform."""
    c, s = math.cos(theta), math.sin(theta)
    return (-(c * x + s * y), -(-s * x + c * y), -theta)


def relative_2d(from_x, from_y, from_theta, to_x, to_y, to_theta):
    """Transform of 'to' expressed in 'from' frame: inv(from) * to."""
    ix, iy, itheta = invert_2d(from_x, from_y, from_theta)
    return compose_2d(ix, iy, itheta, to_x, to_y, to_theta)


def point_to_robot_frame(px, py, rx, ry, rtheta):
    """Express map point (px, py) in the robot frame at pose (rx, ry, rtheta)."""
    dx, dy = px - rx, py - ry
    c, s = math.cos(-rtheta), math.sin(-rtheta)
    return (c * dx - s * dy, s * dx + c * dy)


def world_to_cell(x, y, origin_x, origin_y, res):
    return (int((x - origin_x) / res), int((y - origin_y) / res))


def cell_to_world(i, j, origin_x, origin_y, res):
    return (origin_x + (i + 0.5) * res, origin_y + (j + 0.5) * res)
