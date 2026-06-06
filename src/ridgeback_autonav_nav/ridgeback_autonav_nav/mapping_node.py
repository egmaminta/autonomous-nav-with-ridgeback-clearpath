#!/usr/bin/env python3
"""mapping_node — our own occupancy-grid SLAM front-end (no SLAM library).

Subscribes to LIDAR + EKF odometry, builds a log-odds occupancy grid via ray
casting, and refines pose with a correlative scan matcher to bound odometry
drift. It is the single source of truth for robot pose:

  * publishes ``/map``  (nav_msgs/OccupancyGrid, latched)
  * publishes ``/pose`` (geometry_msgs/PoseStamped, base_link in the map frame)
  * broadcasts the ``map -> odom`` TF (so perception can chain map<-camera)

The ``map`` frame is anchored at the robot's start pose, so HOME == (0, 0, 0)
and return-home is robust to global drift. The grid lives in RAM for the whole
process (destroyed only on exit) and is also serialized to disk periodically.
"""
import math
import os

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster

from ridgeback_autonav_nav.lib import scan_matcher as sm
from ridgeback_autonav_nav.lib.occupancy_grid import OccupancyGrid2D
from ridgeback_autonav_nav.lib.tf_utils import (compose_2d, invert_2d, normalize_angle,
                                   quaternion_from_yaw, yaw_from_quaternion)


class MappingNode(Node):
    def __init__(self):
        super().__init__('mapping_node')
        p = self.declare_parameter
        self.scan_topic = p('scan_topic', '/scan').value
        self.odom_topic = p('odom_topic', '/odom').value
        self.map_topic = p('map_topic', '/map').value
        self.pose_topic = p('pose_topic', '/pose').value
        self.map_frame = p('map_frame', 'map').value
        self.odom_frame = p('odom_frame', 'odom').value
        self.base_frame = p('base_frame', 'base_link').value
        self.laser_frame = p('laser_frame', 'lidar2d_0_laser').value
        self.res = p('resolution', 0.05).value
        self.size_m = p('initial_size_m', 30.0).value
        self.max_range = p('max_range', 10.0).value
        self.min_range = p('min_range', 0.10).value
        l_occ = p('log_odds_occ', 1.4).value
        l_free = p('log_odds_free', -0.35).value
        l_min = p('log_odds_min', -5.0).value
        l_max = p('log_odds_max', 5.0).value
        occ_th = p('occ_threshold', 0.65).value
        free_th = p('free_threshold', 0.25).value
        self.laser_offset_x = p('laser_offset_x', 0.42).value
        self.blind_enabled = p('blind_arc_enabled', True).value
        self.blind_center = math.radians(p('blind_arc_center_deg', 180.0).value)
        self.blind_half = math.radians(p('blind_arc_half_width_deg', 45.0).value)
        self.use_scan_matching = p('use_scan_matching', True).value
        self.min_trans = p('min_trans', 0.05).value
        self.min_rot = p('min_rot', 0.05).value
        self.win_xy = p('search_window_xy', 0.10).value
        self.step_xy = p('search_step_xy', 0.025).value
        self.win_th = p('search_window_theta', 0.08).value
        self.step_th = p('search_step_theta', 0.02).value
        self.match_min_score = p('match_min_score', 0.30).value
        self.max_match_beams = p('max_match_beams', 180).value
        self.map_pub_period = p('map_publish_period', 1.0).value
        self.tf_pub_period = p('tf_publish_period', 0.05).value
        self.save_dir = os.path.expanduser(p('save_dir', '~/ridgeback_autonav_maps').value)
        self.save_period = p('save_period', 5.0).value
        self.scan_max_age = p('scan_max_age', 0.5).value

        self.grid = OccupancyGrid2D(
            resolution=self.res, size_m=self.size_m,
            log_odds_occ=l_occ, log_odds_free=l_free,
            log_odds_min=l_min, log_odds_max=l_max,
            occ_threshold=occ_th, free_threshold=free_th)

        # map -> odom correction (starts identity; map anchored at start pose).
        self.corr = (0.0, 0.0, 0.0)
        self.odom_pose = None          # (x, y, yaw) base in odom
        self.odom_stamp = None
        self.last_kf_map = None        # (x, y, yaw) base in map at last keyframe
        self.latest_scan = None
        self.started = False
        self._save_stem = None

        sensor_qos = qos_profile_sensor_data
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Odometry, self.odom_topic, self._odom_cb, sensor_qos)
        self.create_subscription(LaserScan, self.scan_topic, self._scan_cb, sensor_qos)
        self.map_pub = self.create_publisher(OccupancyGrid, self.map_topic, latched)
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.tf_bc = TransformBroadcaster(self)

        self.create_timer(self.tf_pub_period, self._publish_tf)
        self.create_timer(self.map_pub_period, self._publish_map)
        if self.save_period > 0:
            self.create_timer(self.save_period, self._save_map)
        self.get_logger().info('mapping_node up; map anchored at start pose (HOME=0,0,0).')

    # ---- callbacks ---------------------------------------------------------
    def _odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        self.odom_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                          yaw_from_quaternion(q.x, q.y, q.z, q.w))
        self.odom_stamp = msg.header.stamp

    def _scan_cb(self, msg: LaserScan):
        self.latest_scan = msg
        if self.odom_pose is None:
            return
        self._update(msg)

    # ---- core update -------------------------------------------------------
    def _base_in_map(self, base_in_odom):
        return compose_2d(*self.corr, *base_in_odom)

    def _laser_in_map(self, base_in_map):
        return compose_2d(*base_in_map, self.laser_offset_x, 0.0, 0.0)

    def _update(self, scan: LaserScan):
        base_odom = self.odom_pose
        base_map_pred = self._base_in_map(base_odom)

        if not self.started:
            self.started = True
            self.last_kf_map = base_map_pred
            self._integrate(scan, self._laser_in_map(base_map_pred))
            self._publish_pose(base_map_pred)
            return

        # Keyframe gate: scan-match + integrate only after meaningful motion, but
        # ALWAYS publish the current pose so consumers (mission, nav, explorer)
        # have it even while the robot is stationary.
        dx = base_map_pred[0] - self.last_kf_map[0]
        dy = base_map_pred[1] - self.last_kf_map[1]
        dth = abs(normalize_angle(base_map_pred[2] - self.last_kf_map[2]))
        if math.hypot(dx, dy) >= self.min_trans or dth >= self.min_rot:
            laser_map = self._laser_in_map(base_map_pred)
            angles = self._angles(scan)
            ranges = np.asarray(scan.ranges, dtype=np.float64)

            if self.use_scan_matching:
                bx, by, bth, score = sm.match(
                    self.grid, laser_map[0], laser_map[1], laser_map[2], ranges, angles,
                    min_range=self.min_range, max_range=self.max_range,
                    max_beams=self.max_match_beams,
                    window_xy=self.win_xy, step_xy=self.step_xy,
                    window_theta=self.win_th, step_theta=self.step_th)
                if score >= self.match_min_score:
                    # Back out corrected base pose, refresh map->odom correction.
                    base_map_corr = compose_2d(bx, by, bth, -self.laser_offset_x, 0.0, 0.0)
                    self.corr = compose_2d(*base_map_corr, *invert_2d(*base_odom))
                    laser_map = (bx, by, bth)
                    base_map_pred = base_map_corr

            self._integrate(scan, laser_map)
            self.last_kf_map = base_map_pred

        self._publish_pose(base_map_pred)

    def _integrate(self, scan, laser_map):
        angles = self._angles(scan)
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        blind = None
        if self.blind_enabled:
            blind = np.abs(_wrap_arr(angles - self.blind_center)) <= self.blind_half
        self.grid.integrate_scan(
            laser_map[0], laser_map[1], laser_map[2], ranges, angles,
            min_range=self.min_range, max_range=self.max_range, blind_mask=blind)

    @staticmethod
    def _angles(scan):
        n = len(scan.ranges)
        return scan.angle_min + np.arange(n, dtype=np.float64) * scan.angle_increment

    # ---- publishers --------------------------------------------------------
    def _publish_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame
        t.transform.translation.x = float(self.corr[0])
        t.transform.translation.y = float(self.corr[1])
        qx, qy, qz, qw = quaternion_from_yaw(self.corr[2])
        t.transform.rotation.x, t.transform.rotation.y = qx, qy
        t.transform.rotation.z, t.transform.rotation.w = qz, qw
        self.tf_bc.sendTransform(t)

    def _publish_pose(self, base_map):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.position.x = float(base_map[0])
        msg.pose.position.y = float(base_map[1])
        qx, qy, qz, qw = quaternion_from_yaw(base_map[2])
        msg.pose.orientation.x, msg.pose.orientation.y = qx, qy
        msg.pose.orientation.z, msg.pose.orientation.w = qz, qw
        self.pose_pub.publish(msg)

    def _publish_map(self):
        if not self.started:
            return
        data = self.grid.to_int8()
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.info.resolution = self.grid.res
        msg.info.width = self.grid.width
        msg.info.height = self.grid.height
        msg.info.origin.position.x = self.grid.origin_x
        msg.info.origin.position.y = self.grid.origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = data.reshape(-1).astype(np.int8).tolist()
        self.map_pub.publish(msg)

    def _save_map(self):
        if not self.started:
            return
        if self._save_stem is None:
            stamp = self.get_clock().now().nanoseconds
            self._save_stem = os.path.join(self.save_dir, f'map_{stamp}')
        try:
            self.grid.save(self._save_stem)
        except Exception as exc:  # noqa: BLE001 - best-effort persistence
            self.get_logger().warn(f'map save failed: {exc}')

    def destroy_node(self):
        try:
            self._save_map()
        finally:
            super().destroy_node()


def _wrap_arr(a):
    return np.arctan2(np.sin(a), np.cos(a))


def main(args=None):
    rclpy.init(args=args)
    node = MappingNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
