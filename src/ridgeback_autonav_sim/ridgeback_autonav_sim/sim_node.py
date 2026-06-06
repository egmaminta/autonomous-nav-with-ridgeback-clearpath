#!/usr/bin/env python3
"""sim_node — a self-contained 2D Ridgeback simulator (no hardware, no Nav2).

Publishes exactly what the ridgeback_autonav stack consumes, on the real robot's namespaced
topics so the stack drops in unchanged:

  * ``/r100_0140/sensors/lidar2d_0/scan``  (raycast LIDAR, 270deg FOV)
  * ``/r100_0140/platform/odom`` (+ ``/filtered``)  (holonomic odometry)
  * TF ``odom -> base_link`` + static ``base_link -> {lidar2d_0_laser, camera_optical}``
  * synthetic RGB-D camera that draws room-number signs as readable text plaques
    with aligned depth, so the REAL perception_node (YOLO+PARSeq) can run in sim
  * ``/sim/ground_truth`` (true world pose) + ``/sim_map`` (true occupancy)

It subscribes ``/r100_0140/cmd_vel`` (holonomic: linear.x, linear.y, angular.z).
The ``map`` frame is owned by mapping_node; the sim only provides ``odom`` and
sensors, exactly like the real platform.
"""
import os

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (QoSProfile, DurabilityPolicy, ReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from ridgeback_autonav_nav.lib.tf_utils import (compose_2d, invert_2d, normalize_angle,
                                   quaternion_from_yaw, relative_2d)
from ridgeback_autonav_sim.lib import camera as cam
from ridgeback_autonav_sim.lib.raycast import raycast
from ridgeback_autonav_sim.lib.world import World

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


class SimNode(Node):
    def __init__(self):
        super().__init__('sim_node')
        p = self.declare_parameter
        world_file = p('world_file', '').value
        self.map_frame = p('map_frame', 'map').value
        self.odom_frame = p('odom_frame', 'odom').value
        self.base_frame = p('base_frame', 'base_link').value
        self.laser_frame = p('laser_frame', 'lidar2d_0_laser').value
        self.cam_frame = p('camera_optical_frame', 'camera_0_color_optical_frame').value
        # topics
        self.scan_topic = p('scan_topic', '/r100_0140/sensors/lidar2d_0/scan').value
        self.odom_topic = p('odom_topic', '/r100_0140/platform/odom').value
        self.odom_filtered_topic = p('odom_filtered_topic',
                                     '/r100_0140/platform/odom/filtered').value
        self.cmd_topic = p('cmd_vel_topic', '/r100_0140/cmd_vel').value
        # Match realsense2_camera topic names so sim and real are identical.
        self.color_topic = p('color_topic',
                             '/r100_0140/sensors/camera_0/color/image_raw').value
        self.color_info_topic = p('color_info_topic',
                                  '/r100_0140/sensors/camera_0/color/camera_info').value
        self.depth_topic = p('depth_topic',
                             '/r100_0140/sensors/camera_0/aligned_depth_to_color/image_raw').value
        self.depth_info_topic = p('depth_info_topic',
                                  '/r100_0140/sensors/camera_0/depth/camera_info').value
        # lidar
        self.laser_offset_x = p('laser_offset_x', 0.42).value
        self.num_beams = p('num_beams', 541).value
        self.fov_deg = p('fov_deg', 270.0).value
        self.range_max = p('range_max', 10.0).value
        self.range_min = p('range_min', 0.06).value
        self.scan_noise = p('scan_noise', 0.01).value
        # rates
        self.sim_rate = p('sim_rate', 50.0).value
        self.scan_rate = p('scan_rate', 15.0).value
        self.camera_rate = p('camera_rate', 10.0).value
        self.cmd_timeout = p('cmd_timeout', 0.5).value
        # kinematics / robot
        self.robot_radius = p('robot_radius', 0.43).value
        self.collision = p('collision', True).value
        self.odom_noise = p('odom_noise', 0.0).value
        # camera
        self.camera_enabled = p('camera_enabled', True).value
        self.cam_w = p('camera_width', 640).value
        self.cam_h = p('camera_height', 480).value
        self.fx = p('camera_fx', 525.0).value
        self.fy = p('camera_fy', 525.0).value
        self.cx = p('camera_cx', 320.0).value
        self.cy = p('camera_cy', 240.0).value
        self.cam_offset = (p('camera_x', 0.45).value, p('camera_y', 0.0).value,
                           p('camera_z', 0.6).value)

        self.world = self._load_world(world_file)
        self.pose = list(self.world.start)        # true world pose [x, y, theta]
        self.start = tuple(self.world.start)
        self.cmd = (0.0, 0.0, 0.0)
        self.cmd_t = 0.0

        sensor = qos_profile_sensor_data
        # Odometry is conventionally RELIABLE (matches the real platform and the
        # relay's Odometry subscription); LIDAR/camera stay best-effort sensor QoS.
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Twist, self.cmd_topic, self._cmd_cb, 10)
        self.scan_pub = self.create_publisher(LaserScan, self.scan_topic, sensor)
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, reliable)
        self.odom_f_pub = self.create_publisher(Odometry, self.odom_filtered_topic, reliable)
        self.gt_pub = self.create_publisher(PoseStamped, '/sim/ground_truth', 10)
        self.simmap_pub = self.create_publisher(OccupancyGrid, '/sim_map', LATCHED)
        self.tf_bc = TransformBroadcaster(self)
        self.static_bc = StaticTransformBroadcaster(self)

        if self.camera_enabled:
            try:
                import cv2  # noqa: F401
                from cv_bridge import CvBridge
                self.bridge = CvBridge()
                self.color_pub = self.create_publisher(Image, self.color_topic, sensor)
                self.depth_pub = self.create_publisher(Image, self.depth_topic, sensor)
                self.cinfo_pub = self.create_publisher(CameraInfo, self.color_info_topic, 10)
                self.dinfo_pub = self.create_publisher(CameraInfo, self.depth_info_topic, 10)
                self.create_timer(1.0 / self.camera_rate, self._camera_tick)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f'camera disabled ({exc})')
                self.camera_enabled = False

        self._publish_static_tf()
        self._publish_sim_map()
        self.create_timer(1.0 / self.sim_rate, self._physics_tick)
        self.create_timer(1.0 / self.scan_rate, self._scan_tick)
        self.get_logger().info(
            f'sim_node up: world={os.path.basename(world_file) or "default"} '
            f'start={self.start} signs={[s.text for s in self.world.signs]}')

    # ---- world -------------------------------------------------------------
    def _load_world(self, path):
        if path and os.path.exists(os.path.expanduser(path)):
            with open(os.path.expanduser(path)) as fh:
                spec = yaml.safe_load(fh)
            return World.from_spec(spec)
        self.get_logger().warn('no world_file; using a tiny default room.')
        return World.from_spec({
            'resolution': 0.05, 'bounds': [0, 0, 8, 6], 'border': True,
            'walls': [], 'signs': [{'text': '206', 'x': 7.0, 'y': 3.0, 'z': 1.4}],
            'start': {'x': 1.0, 'y': 1.0, 'theta': 0.0}})

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ---- cmd_vel -----------------------------------------------------------
    def _cmd_cb(self, msg: Twist):
        self.cmd = (msg.linear.x, msg.linear.y, msg.angular.z)
        self.cmd_t = self._now()

    # ---- physics -----------------------------------------------------------
    def _physics_tick(self):
        dt = 1.0 / self.sim_rate
        vx, vy, wz = self.cmd
        if self._now() - self.cmd_t > self.cmd_timeout:
            vx = vy = wz = 0.0
        if self.odom_noise > 0.0:
            n = self.odom_noise
            vx *= 1.0 + np.random.uniform(-n, n)
            vy *= 1.0 + np.random.uniform(-n, n)
            wz *= 1.0 + np.random.uniform(-n, n)
        x, y, th = self.pose
        nth = normalize_angle(th + wz * dt)
        c, s = np.cos(th), np.sin(th)
        nx = x + (vx * c - vy * s) * dt
        ny = y + (vx * s + vy * c) * dt
        if not (self.collision and self._blocked(nx, ny)):
            x, y = nx, ny
        self.pose = [x, y, nth]
        self._publish_odom_tf(vx, vy, wz)

    def _blocked(self, x, y):
        r = self.robot_radius
        for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            if self.world.is_occupied(x + r * np.cos(ang), y + r * np.sin(ang)):
                return True
        return self.world.is_occupied(x, y)

    def _odom_pose(self):
        """True pose expressed in the odom frame (odom origin == start)."""
        return relative_2d(*self.start, *self.pose)

    def _publish_odom_tf(self, vx, vy, wz):
        ox, oy, oth = self._odom_pose()
        now = self.get_clock().now().to_msg()
        qx, qy, qz, qw = quaternion_from_yaw(oth)

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = ox
        odom.pose.pose.position.y = oy
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = wz
        self.odom_pub.publish(odom)
        self.odom_f_pub.publish(odom)

        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = ox
        t.transform.translation.y = oy
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_bc.sendTransform(t)

        gt = PoseStamped()
        gt.header.stamp = now
        gt.header.frame_id = 'sim_world'
        gt.pose.position.x = self.pose[0]
        gt.pose.position.y = self.pose[1]
        gqx, gqy, gqz, gqw = quaternion_from_yaw(self.pose[2])
        gt.pose.orientation.x = gqx
        gt.pose.orientation.y = gqy
        gt.pose.orientation.z = gqz
        gt.pose.orientation.w = gqw
        self.gt_pub.publish(gt)

    # ---- scan --------------------------------------------------------------
    def _scan_tick(self):
        half = np.radians(self.fov_deg) / 2.0
        angles = np.linspace(-half, half, self.num_beams)
        x, y, th = self.pose
        lx = x + self.laser_offset_x * np.cos(th)
        ly = y + self.laser_offset_x * np.sin(th)
        ranges = raycast(self.world.occ, self.world.res,
                         self.world.origin_x, self.world.origin_y,
                         lx, ly, th, angles, self.range_max, self.range_min)
        if self.scan_noise > 0.0:
            ranges = ranges + np.random.normal(0.0, self.scan_noise, ranges.shape)
            ranges = np.clip(ranges, self.range_min, self.range_max)

        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.laser_frame
        msg.angle_min = float(-half)
        msg.angle_max = float(half)
        msg.angle_increment = float((2 * half) / (self.num_beams - 1))
        msg.range_min = float(self.range_min)
        msg.range_max = float(self.range_max)
        msg.ranges = [float(r) for r in ranges]
        self.scan_pub.publish(msg)

    # ---- camera ------------------------------------------------------------
    def _camera_tick(self):
        K = (self.fx, self.fy, self.cx, self.cy)
        # render_scene raycasts walls + floor/ceiling and perspective-warps the
        # wall-mounted signs itself (occlusion handled by the depth z-buffer).
        # Use a camera far-plane that spans the whole world (not the LIDAR range)
        # so a long corridor closes on a distant wall instead of leaving a gap.
        far = float(np.hypot(self.world.xmax - self.world.xmin,
                             self.world.ymax - self.world.ymin)) + 1.0
        color, depth_img = cam.render_scene(
            self.world, self.pose, self.cam_offset, K, self.cam_w, self.cam_h,
            max_depth=far)
        now = self.get_clock().now().to_msg()
        cmsg = self.bridge.cv2_to_imgmsg(color, encoding='bgr8')
        cmsg.header.stamp = now
        cmsg.header.frame_id = self.cam_frame
        self.color_pub.publish(cmsg)
        dmsg = self.bridge.cv2_to_imgmsg(depth_img, encoding='32FC1')
        dmsg.header.stamp = now
        dmsg.header.frame_id = self.cam_frame
        self.depth_pub.publish(dmsg)
        self._publish_caminfo(now)

    def _line_of_sight(self, sign):
        cx = self.pose[0] + self.cam_offset[0] * np.cos(self.pose[2])
        cy = self.pose[1] + self.cam_offset[0] * np.sin(self.pose[2])
        dx, dy = sign.x - cx, sign.y - cy
        dist = float(np.hypot(dx, dy))
        ang = float(np.arctan2(dy, dx))
        r = raycast(self.world.occ, self.world.res, self.world.origin_x,
                    self.world.origin_y, cx, cy, 0.0, np.array([ang]),
                    self.range_max, self.range_min)[0]
        return r >= dist - 0.25   # wall just behind the sign is fine

    def _publish_caminfo(self, stamp):
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self.cam_frame
        info.width = int(self.cam_w)
        info.height = int(self.cam_h)
        info.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        info.p = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        info.distortion_model = 'plumb_bob'
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.cinfo_pub.publish(info)
        self.dinfo_pub.publish(info)

    # ---- static TF + sim map ----------------------------------------------
    def _publish_static_tf(self):
        tfs = []
        # base -> laser
        laser = TransformStamped()
        laser.header.stamp = self.get_clock().now().to_msg()
        laser.header.frame_id = self.base_frame
        laser.child_frame_id = self.laser_frame
        laser.transform.translation.x = self.laser_offset_x
        laser.transform.rotation.w = 1.0
        tfs.append(laser)
        # base -> camera optical
        if self.camera_enabled:
            copt = TransformStamped()
            copt.header.stamp = self.get_clock().now().to_msg()
            copt.header.frame_id = self.base_frame
            copt.child_frame_id = self.cam_frame
            copt.transform.translation.x = self.cam_offset[0]
            copt.transform.translation.y = self.cam_offset[1]
            copt.transform.translation.z = self.cam_offset[2]
            qx, qy, qz, qw = cam.optical_static_quaternion()
            copt.transform.rotation.x = qx
            copt.transform.rotation.y = qy
            copt.transform.rotation.z = qz
            copt.transform.rotation.w = qw
            tfs.append(copt)
        # odom -> sim_world (so /sim_map aligns with everything in RViz)
        iw = invert_2d(*self.start)
        sw = TransformStamped()
        sw.header.stamp = self.get_clock().now().to_msg()
        sw.header.frame_id = self.odom_frame
        sw.child_frame_id = 'sim_world'
        sw.transform.translation.x = iw[0]
        sw.transform.translation.y = iw[1]
        wqx, wqy, wqz, wqw = quaternion_from_yaw(iw[2])
        sw.transform.rotation.x = wqx
        sw.transform.rotation.y = wqy
        sw.transform.rotation.z = wqz
        sw.transform.rotation.w = wqw
        tfs.append(sw)
        self.static_bc.sendTransform(tfs)

    def _publish_sim_map(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'sim_world'
        msg.info.resolution = self.world.res
        msg.info.width = self.world.width
        msg.info.height = self.world.height
        msg.info.origin.position.x = self.world.origin_x
        msg.info.origin.position.y = self.world.origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = self.world.to_int8().reshape(-1).astype(np.int8).tolist()
        self.simmap_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimNode()
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
