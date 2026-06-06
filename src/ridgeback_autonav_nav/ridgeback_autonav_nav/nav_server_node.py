#!/usr/bin/env python3
"""nav_server_node — go-to-pose Action server (no Nav2).

Exposes ``ridgeback_autonav_msgs/action/NavigateToPose``: a clean goal/feedback/cancel/result
interface that replaces the old, fragile ``SetParameters`` poking. Internally it
builds a costmap from the live ``/map``, plans with A*, follows the path with a
holonomic DWA controller, and runs bounded recovery + watchdogs.

Pose comes from ``/pose`` (published by mapping_node); obstacles come from the
live ``/scan`` transformed into the base frame via the laser offset. No TF
dependency here, which removes a whole class of timing races.
"""
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (QoSProfile, DurabilityPolicy, ReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import LaserScan

from ridgeback_autonav_msgs.action import NavigateToPose
from ridgeback_autonav_nav.lib import astar
from ridgeback_autonav_nav.lib import geometry
from ridgeback_autonav_nav.lib import path_utils
from ridgeback_autonav_nav.lib.dwa_controller import DWAConfig, compute_cmd
from ridgeback_autonav_nav.lib.occupancy_grid import costmap_from_occupancy
from ridgeback_autonav_nav.lib.tf_utils import (cell_to_world, normalize_angle,
                                   point_to_robot_frame, world_to_cell,
                                   yaw_from_quaternion)

ERR_OK, ERR_NO_PLAN, ERR_STUCK, ERR_TIMEOUT, ERR_CANCELED, ERR_COLLISION = range(6)


def _clamp(v, lim):
    return max(-lim, min(lim, v))


class NavServerNode(Node):
    def __init__(self):
        super().__init__('nav_server_node')
        p = self.declare_parameter
        self.map_frame = p('map_frame', 'map').value
        self.base_frame = p('base_frame', 'base_link').value
        self.laser_offset_x = p('laser_offset_x', 0.42).value
        scan_topic = p('scan_topic', '/scan').value
        map_topic = p('map_topic', '/map').value
        pose_topic = p('pose_topic', '/pose').value
        cmd_topic = p('cmd_vel_topic', '/r100_0140/cmd_vel').value
        plan_topic = p('plan_topic', '/plan').value
        self.robot_radius = p('robot_radius', 0.43).value
        self.infl = p('inflation_radius', 0.55).value
        self.infl_explore = p('inflation_radius_explore', 0.45).value
        self.unknown_cost = p('unknown_cost', 100.0).value
        self.unknown_cost_explore = p('unknown_cost_explore', 5.0).value
        self.lethal_cost = p('lethal_cost', 254.0).value
        self.allow_diag = p('allow_diagonal', True).value
        self.replan_period = p('replan_period', 1.0).value
        self.control_period = p('control_period', 0.1).value
        self.def_xy_tol = p('xy_tolerance', 0.20).value
        self.def_yaw_tol = p('yaw_tolerance', 0.10).value
        self.scan_max_age = p('scan_max_age', 0.25).value
        self.progress_timeout = p('progress_timeout', 6.0).value
        self.goal_timeout = p('goal_timeout', 120.0).value
        self.max_recoveries = p('max_recoveries', 4).value
        self.rec_rotate = p('recovery_rotate_speed', 0.4).value
        self.rec_strafe = p('recovery_strafe_speed', 0.2).value
        self.rec_backup = p('recovery_backup_speed', 0.15).value
        self.rec_step = p('recovery_step_time', 2.0).value
        self.lookahead = p('lookahead_distance', 0.8).value
        self.cmd_smoothing = p('cmd_smoothing', 0.35).value   # output velocity EMA (0=off)
        self.path_smooth_iters = p('path_smooth_iters', 2).value
        # Terminal precise approach: within approach_radius, drop the DWA/EMA and
        # servo straight to the goal (proportional) so it settles cleanly.
        self.approach_radius = p('approach_radius', 0.6).value
        self.approach_kp = p('approach_kp', 0.6).value
        self.hard_stop_distance = p('hard_stop_distance', 0.50).value
        self.hard_stop_speed = p('hard_stop_speed', 0.12).value
        self.reverse_threshold = p('reverse_threshold', 0.6).value  # allow DWA reverse only within this of an obstacle
        self.safety_margin = p('safety_margin', 0.15).value
        # No-go zones: list of "x1 y1 x2 y2 ..." map-frame polygon strings.
        self.no_go = geometry.parse_zone_strings(p('no_go_zones', []).value)
        self.no_go_edge_pts = geometry.polygon_edge_points(self.no_go, spacing=0.1)
        if self.no_go:
            self.get_logger().info(f'loaded {len(self.no_go)} no-go zone(s)')

        self.dwa = DWAConfig(
            max_vx=p('max_vx', 0.5).value, max_vy=p('max_vy', 0.30).value,
            max_wz=p('max_wz', 0.6).value, acc_vx=p('acc_vx', 0.6).value,
            acc_vy=p('acc_vy', 0.5).value, acc_wz=p('acc_wz', 1.2).value,
            vx_samples=p('vx_samples', 7).value, vy_samples=p('vy_samples', 5).value,
            wz_samples=p('wz_samples', 11).value, sim_time=p('sim_time', 1.5).value,
            sim_dt=p('sim_dt', 0.1).value, w_path=p('w_path', 2.0).value,
            w_goal=p('w_goal', 1.5).value, w_clearance=p('w_clearance', 0.8).value,
            w_smooth=p('w_smooth', 0.2).value, robot_radius=self.robot_radius,
            safety_margin=self.safety_margin,
            w_heading=p('w_heading', 0.6).value, w_strafe=p('w_strafe', 0.8).value)

        self._map = None        # (occ int8 HxW, res, ox, oy)
        self._pose = None       # (x, y, yaw)
        self._pose_t = 0.0
        self._scan = None       # LaserScan
        self._scan_t = 0.0
        self._cur_v = (0.0, 0.0, 0.0)

        cb = ReentrantCallbackGroup()
        sensor_qos = qos_profile_sensor_data
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, map_topic, self._map_cb, latched, callback_group=cb)
        self.create_subscription(PoseStamped, pose_topic, self._pose_cb, 10, callback_group=cb)
        self.create_subscription(LaserScan, scan_topic, self._scan_cb, sensor_qos, callback_group=cb)
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.plan_pub = self.create_publisher(Path, plan_topic, 1)

        self._action = ActionServer(
            self, NavigateToPose, 'navigate_to_pose',
            execute_callback=self._execute,
            goal_callback=lambda g: GoalResponse.ACCEPT,
            cancel_callback=lambda g: CancelResponse.ACCEPT,
            callback_group=cb)
        self.get_logger().info('nav_server_node ready (action: navigate_to_pose).')

    # ---- subscriptions -----------------------------------------------------
    def _map_cb(self, msg: OccupancyGrid):
        occ = np.asarray(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        self._map = (occ, msg.info.resolution,
                     msg.info.origin.position.x, msg.info.origin.position.y)

    def _pose_cb(self, msg: PoseStamped):
        q = msg.pose.orientation
        self._pose = (msg.pose.position.x, msg.pose.position.y,
                      yaw_from_quaternion(q.x, q.y, q.z, q.w))
        self._pose_t = self._now()

    def _scan_cb(self, msg: LaserScan):
        self._scan = msg
        self._scan_t = self._now()

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ---- helpers -----------------------------------------------------------
    def _stop(self):
        try:
            self.cmd_pub.publish(Twist())
        except Exception:  # noqa: BLE001 - context may be down during shutdown
            pass
        self._cur_v = (0.0, 0.0, 0.0)

    def _publish_cmd(self, vx, vy, wz, smooth=True):
        # Exponential smoothing on the output so commands don't hop between DWA
        # samples -> smooth, non-jerky motion. (Zero-stops bypass this via _stop;
        # terminal approach / recovery pass smooth=False for responsiveness.)
        a = self.cmd_smoothing if smooth else 0.0
        if a and a < 1.0:
            pvx, pvy, pwz = self._cur_v
            vx = pvx + a * (vx - pvx)
            vy = pvy + a * (vy - pvy)
            wz = pwz + a * (wz - pwz)
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = float(vx), float(vy), float(wz)
        self.cmd_pub.publish(t)
        self._cur_v = (vx, vy, wz)

    def _obstacles_base(self):
        """Live obstacle points in the base frame: LIDAR scan + no-go edges.

        No-go edges are included so the controller treats LIDAR-invisible
        boundaries (e.g. glass) as real obstacles, not just the global planner.
        """
        pts = []
        s = self._scan
        if s is not None:
            r = np.asarray(s.ranges, dtype=np.float64)
            a = s.angle_min + np.arange(r.shape[0]) * s.angle_increment
            ok = np.isfinite(r) & (r > 0.05) & (r <= s.range_max)
            r, a = r[ok], a[ok]
            pts.append(np.column_stack((self.laser_offset_x + r * np.cos(a), r * np.sin(a))))
        if self.no_go_edge_pts.size and self._pose is not None:
            px, py, pth = self._pose
            c, sn = math.cos(-pth), math.sin(-pth)
            dx = self.no_go_edge_pts[:, 0] - px
            dy = self.no_go_edge_pts[:, 1] - py
            bx = c * dx - sn * dy
            by = sn * dx + c * dy
            near = (bx * bx + by * by) <= (self.dwa.obstacle_check_range ** 2)
            pts.append(np.column_stack((bx[near], by[near])))
        if not pts:
            return np.zeros((0, 2))
        return np.vstack(pts)

    def _build_costmap(self, exploration):
        occ, res, ox, oy = self._map
        if self.no_go:
            # Stamp no-go zones as occupied BEFORE inflation so the planner keeps
            # the normal standoff around them (e.g. invisible glass walls).
            occ = occ.copy()
            geometry.stamp_polygons_occupied(occ, self.no_go, ox, oy, res)
        infl = self.infl_explore if exploration else self.infl
        ucost = self.unknown_cost_explore if exploration else self.unknown_cost
        cost = costmap_from_occupancy(occ, res, self.robot_radius, infl,
                                      unknown_cost=ucost, lethal_cost=self.lethal_cost)
        return cost, res, ox, oy

    def _plan(self, cost, res, ox, oy, goal_xy):
        px, py, _ = self._pose
        start = world_to_cell(px, py, ox, oy, res)
        goal = world_to_cell(goal_xy[0], goal_xy[1], ox, oy, res)
        start = astar.nearest_free_cell(cost, start, self.lethal_cost) or start
        goal = astar.nearest_free_cell(cost, goal, self.lethal_cost) or goal
        cells = astar.plan(cost, start, goal, self.lethal_cost, self.allow_diag)
        if not cells:
            return []
        world = [cell_to_world(c, r, ox, oy, res) for (c, r) in cells]
        # Smooth the grid staircase into a natural curve (collision-checked).
        return path_utils.smooth_path_safe(
            world, lambda x, y: self._is_free(x, y, cost, res, ox, oy),
            iterations=self.path_smooth_iters)

    def _is_free(self, x, y, cost, res, ox, oy):
        i, j = world_to_cell(x, y, ox, oy, res)
        if 0 <= j < cost.shape[0] and 0 <= i < cost.shape[1]:
            return cost[j, i] < self.lethal_cost
        return False

    def _carrot(self, path):
        """Pick a lookahead point ahead of the robot on the path."""
        px, py, _ = self._pose
        d = [math.hypot(wx - px, wy - py) for (wx, wy) in path]
        nearest = int(np.argmin(d))
        for k in range(nearest, len(path)):
            if math.hypot(path[k][0] - px, path[k][1] - py) >= self.lookahead:
                return path[k]
        return path[-1]

    def _publish_plan(self, path):
        msg = Path()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for (wx, wy) in path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = wx, wy
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.plan_pub.publish(msg)

    # ---- action execution --------------------------------------------------
    def _execute(self, goal_handle):
        req = goal_handle.request
        gx = req.target_pose.pose.position.x
        gy = req.target_pose.pose.position.y
        gq = req.target_pose.pose.orientation
        gyaw = yaw_from_quaternion(gq.x, gq.y, gq.z, gq.w)
        explore = req.exploration_mode
        xy_tol = req.xy_tolerance if req.xy_tolerance > 0 else self.def_xy_tol
        yaw_tol = req.yaw_tolerance if req.yaw_tolerance > 0 else self.def_yaw_tol

        self.get_logger().info(
            f'NavigateToPose -> ({gx:.2f}, {gy:.2f}, {math.degrees(gyaw):.0f}deg) '
            f'explore={explore}')

        rate = self.create_rate(1.0 / self.control_period)
        t_start = self._now()
        t_last_plan = -1e9
        path = []
        best_dist = math.inf
        t_progress = t_start
        recoveries = 0

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self._stop()
                goal_handle.canceled()
                return self._result(False, ERR_CANCELED, 'canceled')

            now = self._now()
            if now - t_start > self.goal_timeout:
                self._stop()
                goal_handle.abort()
                return self._result(False, ERR_TIMEOUT, 'goal timeout')

            # Watchdogs: require fresh sensors + pose, else hold.
            if (self._map is None or self._pose is None or self._scan is None or
                    now - self._scan_t > self.scan_max_age or
                    now - self._pose_t > 1.0):
                if self._scan is not None:
                    self.get_logger().warn(
                        f'hold: stale sensors (scan age '
                        f'{now - self._scan_t:.2f}s > {self.scan_max_age:.2f})',
                        throttle_duration_sec=1.0)
                else:
                    self.get_logger().warn('hold: waiting for map/pose/scan',
                                           throttle_duration_sec=1.0)
                self._stop()
                self._feedback(goal_handle, 'RECOVERING', math.inf)
                rate.sleep()
                continue

            px, py, pth = self._pose
            dist = math.hypot(gx - px, gy - py)
            yaw_err = normalize_angle(gyaw - pth)

            # Terminal precise approach: inside approach_radius, drop the DWA +
            # output EMA (which overshoot a tight tolerance) and servo straight to
            # the goal, then rotate to the final yaw. This settles cleanly.
            if dist < self.approach_radius:
                if dist <= xy_tol:
                    if abs(yaw_err) <= yaw_tol:
                        self._stop()
                        goal_handle.succeed()
                        return self._result(True, ERR_OK, 'reached')
                    wz = _clamp(1.5 * yaw_err, self.dwa.max_wz)
                    self._publish_cmd(0.0, 0.0, wz, smooth=False)
                else:
                    gxr, gyr = point_to_robot_frame(gx, gy, px, py, pth)
                    vx = _clamp(self.approach_kp * gxr, self.dwa.max_vx)
                    vy = _clamp(self.approach_kp * gyr, self.dwa.max_vy)
                    self._publish_cmd(vx, vy, 0.0, smooth=False)
                self._feedback(goal_handle, 'FOLLOWING', dist)
                rate.sleep()
                continue

            # Progress watchdog.
            if dist < best_dist - 0.05:
                best_dist = dist
                t_progress = now

            # (Re)plan periodically.
            if now - t_last_plan > self.replan_period or not path:
                cost, res, ox, oy = self._build_costmap(explore)
                path = self._plan(cost, res, ox, oy, (gx, gy))
                t_last_plan = now
                if path:
                    self._publish_plan(path)
                else:
                    recoveries += 1
                    self.get_logger().warn('no path to goal -> recovery')
                    if recoveries > self.max_recoveries:
                        self._stop()
                        goal_handle.abort()
                        return self._result(False, ERR_NO_PLAN, 'no path found')
                    self._do_recovery(recoveries)
                    rate.sleep()
                    continue

            # Follow with holonomic DWA.
            carrot = self._carrot(path)
            carrot_r = point_to_robot_frame(carrot[0], carrot[1], px, py, pth)
            goal_r = point_to_robot_frame(gx, gy, px, py, pth)
            obstacles = self._obstacles_base()
            nearest = math.inf
            if obstacles.shape[0]:
                d2 = obstacles[:, 0] ** 2 + obstacles[:, 1] ** 2
                k = int(np.argmin(d2))
                nearest = math.sqrt(float(d2[k]))

            # Hard safety stop: if anything is critically close, actively back
            # away from it — last resort, independent of the planner/DWA.
            if nearest < self.hard_stop_distance:
                norm = nearest or 1.0
                self._publish_cmd(-self.hard_stop_speed * obstacles[k, 0] / norm,
                                  -self.hard_stop_speed * obstacles[k, 1] / norm, 0.0,
                                  smooth=False)
                self.get_logger().warn(
                    f'SAFETY: obstacle at {nearest:.2f} m -> backing off',
                    throttle_duration_sec=1.0)
                self._feedback(goal_handle, 'RECOVERING', dist)
                t_last_plan = -1e9   # replan once clear
                rate.sleep()
                continue

            # Drive-like: in open space, NO reverse and NO sideways strafe — the
            # robot turns to face its target and drives forward like a car. Full
            # holonomic moves (reverse / strafe) are unlocked only when an
            # obstacle is close enough that they help escape a tight spot.
            holonomic = nearest < self.reverse_threshold
            cmd, _ = compute_cmd(self.dwa, self._cur_v, carrot_r, goal_r,
                                 normalize_angle(gyaw - pth), obstacles, dist,
                                 allow_reverse=holonomic, allow_strafe=holonomic)

            if cmd is None or (now - t_progress > self.progress_timeout):
                recoveries += 1
                reason = ('DWA found no collision-free trajectory' if cmd is None
                          else f'no progress for {self.progress_timeout:.0f}s')
                self.get_logger().warn(f'recovery #{recoveries}: {reason}')
                if recoveries > self.max_recoveries:
                    self._stop()
                    goal_handle.abort()
                    return self._result(False, ERR_STUCK, 'stuck / no progress')
                self._do_recovery(recoveries)
                t_progress = now
                best_dist = math.inf
                t_last_plan = -1e9  # force replan after recovery
                rate.sleep()
                continue

            self._publish_cmd(*cmd)
            self._feedback(goal_handle, 'FOLLOWING', dist)
            rate.sleep()

        self._stop()
        goal_handle.abort()
        return self._result(False, ERR_STUCK, 'shutdown')

    def _do_recovery(self, level):
        """Bounded, escalating recovery: rotate -> strafe -> backup -> clear."""
        end = self._now() + self.rec_step
        if level == 1:
            vx, vy, wz = 0.0, 0.0, self.rec_rotate
        elif level == 2:
            vx, vy, wz = 0.0, self.rec_strafe, 0.0       # holonomic lateral escape
        elif level == 3:
            vx, vy, wz = -self.rec_backup, 0.0, 0.0
        else:
            self._stop()
            return  # level 4: just clear/replan on next loop
        while self._now() < end and rclpy.ok():
            self._publish_cmd(vx, vy, wz)
            time.sleep(self.control_period)
        self._stop()

    def _feedback(self, goal_handle, state, dist):
        fb = NavigateToPose.Feedback()
        if self._pose is not None:
            fb.current_pose.header.frame_id = self.map_frame
            fb.current_pose.pose.position.x = self._pose[0]
            fb.current_pose.pose.position.y = self._pose[1]
        fb.distance_remaining = float(dist if math.isfinite(dist) else -1.0)
        fb.state = state
        goal_handle.publish_feedback(fb)

    @staticmethod
    def _result(success, code, message):
        r = NavigateToPose.Result()
        r.success = success
        r.error_code = code
        r.message = message
        return r


def main(args=None):
    rclpy.init(args=args)
    node = NavServerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node._stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
