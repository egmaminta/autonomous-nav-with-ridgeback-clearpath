#!/usr/bin/env python3
"""explorer_node — frontier exploration with anti-loop guards.

A client of the ``navigate_to_pose`` action. Detects Wavefront frontiers on the
live ``/map``, scores them (size + travel cost + heading change), and drives to
the best one. The old system's "spin forever at the same goal" bug is closed by
three guards working together:

  * blacklist  — a frontier that aborts/times-out (or yields no map growth) is
                 banned for a TTL, so it is never re-sent.
  * hysteresis — candidates too close to the last goal are skipped while others
                 exist, preventing oscillation between two nearby frontiers.
  * watchdog   — if the map stops growing while a goal is active, the goal is
                 cancelled and blacklisted.

Completion is declared only after a debounce period with zero valid frontiers.
Controlled via the ``StartExploration`` service; state is published on
``/exploration_state`` (IDLE | EXPLORING | COMPLETE).
"""
import math
import threading

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (QoSProfile, DurabilityPolicy, ReliabilityPolicy)
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from ridgeback_autonav_msgs.action import NavigateToPose
from ridgeback_autonav_msgs.srv import StartExploration
from ridgeback_autonav_nav.lib import frontier
from ridgeback_autonav_nav.lib.frontier import detect_frontiers
from ridgeback_autonav_nav.lib.tf_utils import (cell_to_world, normalize_angle,
                                   quaternion_from_yaw, yaw_from_quaternion)


class ExplorerNode(Node):
    def __init__(self):
        super().__init__('explorer_node')
        p = self.declare_parameter
        self.map_frame = p('map_frame', 'map').value
        map_topic = p('map_topic', '/map').value
        pose_topic = p('pose_topic', '/pose').value
        self.period = p('process_period', 1.0).value
        self.min_frontier_size = p('min_frontier_size', 8).value
        self.max_range = p('max_frontier_range', 12.0).value
        self.min_range = p('min_frontier_range', 0.6).value
        self.w_size = p('w_size', 0.3).value     # frontier size (per 10 cells)
        self.w_cost = p('w_cost', 0.5).value     # travel-cost penalty (per metre)
        self.w_info = p('w_info', 1.0).value     # information gain (per 100 unknown cells)
        self.w_turn = p('w_turn', 0.3).value     # heading-change penalty (per rad)
        self.w_camera = p('w_camera', 0.6).value  # camera-guided: face walls (signs)
        self.info_gain_radius_m = p('info_gain_radius', 2.0).value
        self.camera_fov_deg = p('camera_fov_deg', 70.0).value
        self.camera_range_m = p('camera_range', 4.0).value
        self.camera_rays = int(p('camera_rays', 9).value)
        self.blacklist_radius = p('blacklist_radius', 0.7).value
        self.blacklist_ttl = p('blacklist_ttl', 60.0).value
        self.hysteresis_radius = p('hysteresis_radius', 0.5).value
        self.no_growth_timeout = p('no_growth_timeout', 12.0).value
        self.complete_debounce = p('complete_debounce', 5.0).value
        self.goal_xy_tol = p('goal_xy_tolerance', 0.30).value
        self.goal_yaw_tol = p('goal_yaw_tolerance', 1.57).value

        self._map = None
        self._pose = None
        self._lock = threading.Lock()
        self.exploring = False
        self.active_goal = None        # (x, y) world centroid
        self.goal_handle = None
        self.last_goal = None
        self.blacklist = []            # list of (x, y, expiry_s)
        self._empty_since = None
        self._known_cells = 0
        self._growth_t = 0.0
        self._goal_active = False

        cb = ReentrantCallbackGroup()
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, map_topic, self._map_cb, latched, callback_group=cb)
        self.create_subscription(PoseStamped, pose_topic, self._pose_cb, 10, callback_group=cb)
        self.state_pub = self.create_publisher(String, '/exploration_state', latched)
        self.marker_pub = self.create_publisher(MarkerArray, '/frontiers', 1)
        self.srv = self.create_service(
            StartExploration, 'start_exploration', self._srv_cb, callback_group=cb)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose',
                                       callback_group=cb)
        self.create_timer(self.period, self._tick, callback_group=cb)
        self._publish_state('IDLE')
        self.get_logger().info('explorer_node ready (service: start_exploration).')

    # ---- subscriptions / service -------------------------------------------
    def _map_cb(self, msg: OccupancyGrid):
        occ = np.asarray(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        self._map = (occ, msg.info.resolution,
                     msg.info.origin.position.x, msg.info.origin.position.y)

    def _pose_cb(self, msg: PoseStamped):
        q = msg.pose.orientation
        self._pose = (msg.pose.position.x, msg.pose.position.y,
                      yaw_from_quaternion(q.x, q.y, q.z, q.w))

    def _srv_cb(self, req, resp):
        with self._lock:
            if req.start:
                self.exploring = True
                self.blacklist.clear()
                self.last_goal = None
                self._empty_since = None
                self._known_cells = 0
                self._growth_t = self._now()
                self._publish_state('EXPLORING')
                resp.message = 'exploration started'
            else:
                self.exploring = False
                self._cancel_active()
                self._publish_state('IDLE')
                resp.message = 'exploration stopped'
            resp.ok = True
        return resp

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _publish_state(self, s):
        self.state_pub.publish(String(data=s))

    # ---- main loop ---------------------------------------------------------
    def _tick(self):
        with self._lock:
            if not self.exploring or self._map is None or self._pose is None:
                return
            now = self._now()
            self._expire_blacklist(now)

            # Track map growth for the no-growth watchdog.
            occ = self._map[0]
            known = int(np.count_nonzero(occ != -1))
            if known > self._known_cells + 20:
                self._known_cells = known
                self._growth_t = now

            if self._goal_active:
                if now - self._growth_t > self.no_growth_timeout:
                    self.get_logger().warn('no map growth -> blacklist + retarget')
                    self._blacklist(self.active_goal, now)
                    self._cancel_active()
                return

            frontiers = self._candidates(now)
            if not frontiers:
                if self._empty_since is None:
                    self._empty_since = now
                elif now - self._empty_since >= self.complete_debounce:
                    self.exploring = False
                    self._publish_state('COMPLETE')
                    self.get_logger().info('exploration complete (no frontiers).')
                self._publish_markers([])
                return

            self._empty_since = None
            best = max(frontiers, key=lambda f: f[2])  # (x, y, utility)
            self._publish_markers(frontiers)
            self._send_goal(best[0], best[1])

    def _candidates(self, now):
        occ, res, ox, oy = self._map
        px, py, pth = self._pose
        frs = detect_frontiers(occ, min_size=self.min_frontier_size)
        out = []
        others_exist = len(frs) > 1
        ig_radius_cells = max(1, int(self.info_gain_radius_m / res))
        fov_half = math.radians(self.camera_fov_deg) * 0.5
        for f in frs:
            wx, wy = cell_to_world(f.centroid_cell[0], f.centroid_cell[1], ox, oy, res)
            dist = math.hypot(wx - px, wy - py)
            if dist < self.min_range or dist > self.max_range:
                continue
            if self._is_blacklisted(wx, wy):
                continue
            if (others_exist and self.last_goal is not None and
                    math.hypot(wx - self.last_goal[0], wy - self.last_goal[1])
                    < self.hysteresis_radius):
                continue
            heading_to = math.atan2(wy - py, wx - px)
            turn = abs(normalize_angle(heading_to - pth))
            # Real information gain: unmapped area this frontier would reveal.
            ig = frontier.information_gain(occ, int(f.centroid_cell[0]),
                                           int(f.centroid_cell[1]),
                                           ig_radius_cells) / 100.0
            # Camera-guided bias: fraction of the FOV that faces walls when the
            # robot heads at this frontier (signs are on walls -> read more).
            cam = frontier.camera_visibility(
                occ, res, ox, oy, px, py, heading_to, fov_half,
                self.camera_range_m, self.camera_rays)
            utility = (self.w_info * ig + self.w_size * (f.size / 10.0)
                       + self.w_camera * cam
                       - self.w_cost * dist - self.w_turn * turn)
            out.append((wx, wy, utility))
        return out

    # ---- action plumbing ---------------------------------------------------
    def _send_goal(self, wx, wy):
        if not self.nav_client.server_is_ready():
            self.nav_client.wait_for_server(timeout_sec=0.1)
            if not self.nav_client.server_is_ready():
                self.get_logger().warn('nav server not ready; will retry')
                return
        px, py, _ = self._pose
        goal = NavigateToPose.Goal()
        goal.target_pose.header.frame_id = self.map_frame
        goal.target_pose.header.stamp = self.get_clock().now().to_msg()
        goal.target_pose.pose.position.x = float(wx)
        goal.target_pose.pose.position.y = float(wy)
        yaw = math.atan2(wy - py, wx - px)
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        goal.target_pose.pose.orientation.x = qx
        goal.target_pose.pose.orientation.y = qy
        goal.target_pose.pose.orientation.z = qz
        goal.target_pose.pose.orientation.w = qw
        goal.exploration_mode = True
        goal.xy_tolerance = self.goal_xy_tol
        goal.yaw_tolerance = self.goal_yaw_tol

        self.active_goal = (wx, wy)
        self.last_goal = (wx, wy)
        self._goal_active = True
        self._growth_t = self._now()
        self.get_logger().info(f'frontier goal -> ({wx:.2f}, {wy:.2f})')
        fut = self.nav_client.send_goal_async(goal)
        fut.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future):
        gh = future.result()
        with self._lock:
            if not gh.accepted:
                self.get_logger().warn('frontier goal rejected -> blacklist')
                self._blacklist(self.active_goal, self._now())
                self._goal_active = False
                self.active_goal = None
                return
            self.goal_handle = gh
        gh.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future):
        result = future.result().result
        with self._lock:
            if not result.success:
                self.get_logger().info(
                    f'frontier failed (code {result.error_code}) -> blacklist')
                self._blacklist(self.active_goal, self._now())
            self._goal_active = False
            self.goal_handle = None
            self.active_goal = None

    def _cancel_active(self):
        if self.goal_handle is not None:
            try:
                self.goal_handle.cancel_goal_async()
            except Exception:  # noqa: BLE001
                pass
        self._goal_active = False
        self.goal_handle = None
        self.active_goal = None

    # ---- blacklist ---------------------------------------------------------
    def _blacklist(self, xy, now):
        if xy is not None:
            self.blacklist.append((xy[0], xy[1], now + self.blacklist_ttl))

    def _is_blacklisted(self, wx, wy):
        for (bx, by, _) in self.blacklist:
            if math.hypot(wx - bx, wy - by) < self.blacklist_radius:
                return True
        return False

    def _expire_blacklist(self, now):
        self.blacklist = [b for b in self.blacklist if b[2] > now]

    # ---- viz ---------------------------------------------------------------
    def _publish_markers(self, frontiers):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for i, (wx, wy, _u) in enumerate(frontiers):
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'frontiers'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = wx
            m.pose.position.y = wy
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.25
            m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.8, 1.0, 0.9
            arr.markers.append(m)
        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = ExplorerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
