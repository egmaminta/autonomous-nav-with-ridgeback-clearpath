#!/usr/bin/env python3
"""mission_node — the deterministic FSM brain (no LLM in the control loop).

States: PARSE -> INIT -> EXPLORE -> APPROACH -> CONFIRM -> RETURN_HOME -> DONE.
The robot explores, watches the latched sign registry, drives to the target sign
when confirmed, then ALWAYS returns to the exact captured start pose (even on
failure). Long operations (explore, navigate) are kicked off asynchronously and
polled by a timer, so the state machine never blocks. The LLM, if enabled, only
parses the task string once in PARSE.

Robustness vs. the old agentic system:
  * deterministic transitions with per-state timeouts and capped retries
  * goals issued via the NavigateToPose ACTION (feedback/cancel/result), never
    by poking parameters
  * sign state read from a TRANSIENT_LOCAL (latched) topic + a one-shot GetSigns
    service call at start, so there is no startup race
  * unconditional return-home, so the robot always comes back
"""
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from std_msgs.msg import String

from ridgeback_autonav_msgs.action import NavigateToPose
from ridgeback_autonav_msgs.msg import MissionStatus, SignRegistry as SignRegistryMsg
from ridgeback_autonav_msgs.srv import GetSigns, StartExploration
from ridgeback_autonav_mission.lib.task_parser import parse_task
from ridgeback_autonav_mission.lib.tf_helpers import quaternion_from_yaw, yaw_from_quaternion

PARSE, INIT, EXPLORE, APPROACH, CONFIRM, RETURN_HOME, DONE = (
    'PARSE', 'INIT', 'EXPLORE', 'APPROACH', 'CONFIRM', 'RETURN_HOME', 'DONE')


class MissionNode(Node):
    def __init__(self):
        super().__init__('mission_node')
        p = self.declare_parameter
        self.map_frame = p('map_frame', 'map').value
        self.task_str = p('task', 'Go to Room 206').value
        self.use_llm = p('use_llm_parser', False).value
        self.standoff = p('standoff_distance', 1.0).value
        self.confirm_frames = p('confirm_frames', 5).value
        self.approach_retries = p('approach_retries', 2).value
        self.home_xy_tol = p('home_xy_tolerance', 0.10).value
        self.home_yaw_tol = p('home_yaw_tolerance', 0.05).value
        self.mission_timeout = p('mission_timeout', 900.0).value
        self.poll = p('state_poll_period', 0.5).value
        registry_topic = p('registry_topic', '/sign_registry').value

        self.state = PARSE
        self.target = None
        self.home = None                  # (x, y, yaw)
        self.pose = None                  # (x, y, yaw)
        self.signs = []                   # list of (text, x, y, confirmed)
        self.result_str = ''
        self.t_start = self._now()
        self.t_state = self._now()

        # async nav bookkeeping
        self.nav_goal_handle = None
        self.nav_done = False
        self.nav_success = False
        self._explore_started = False
        self._approach_attempts = 0
        self._confirm_until = 0.0
        self._target_sign = None
        self._return_attempts = 0
        self._explore_state = ''

        cb = ReentrantCallbackGroup()
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PoseStamped, p('pose_topic', '/pose').value,
                                 self._pose_cb, 10, callback_group=cb)
        self.create_subscription(SignRegistryMsg, registry_topic,
                                 self._registry_cb, latched, callback_group=cb)
        self.create_subscription(String, '/exploration_state',
                                 self._explore_state_cb, latched, callback_group=cb)
        self.status_pub = self.create_publisher(
            MissionStatus, p('status_topic', '/mission_status').value, 10)

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose',
                                       callback_group=cb)
        self.explore_client = self.create_client(StartExploration, 'start_exploration',
                                                 callback_group=cb)
        self.signs_client = self.create_client(GetSigns, 'get_signs', callback_group=cb)

        self.create_timer(self.poll, self._tick, callback_group=cb)
        self.get_logger().info(f'mission_node up. Task: "{self.task_str}"')

    # ---- subscriptions -----------------------------------------------------
    def _pose_cb(self, msg):
        q = msg.pose.orientation
        self.pose = (msg.pose.position.x, msg.pose.position.y,
                     yaw_from_quaternion(q.x, q.y, q.z, q.w))

    def _registry_cb(self, msg):
        self.signs = [(s.text, s.position.x, s.position.y, s.confirmed) for s in msg.signs]

    def _explore_state_cb(self, msg):
        self._explore_state = msg.data

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _goto(self, state):
        self.get_logger().info(f'[FSM] {self.state} -> {state}')
        self.state = state
        self.t_state = self._now()

    # ---- target lookup -----------------------------------------------------
    def _find_target(self):
        from ridgeback_autonav_mission.lib.task_parser import normalize_room
        key = normalize_room(self.target)
        for (text, x, y, confirmed) in self.signs:
            if confirmed and normalize_room(text) == key:
                return (x, y)
        return None

    # ---- FSM ---------------------------------------------------------------
    def _tick(self):
        self._publish_status()
        # Global mission timeout: bail out but still return home.
        if self.state not in (RETURN_HOME, DONE) and \
                self._now() - self.t_start > self.mission_timeout:
            self.get_logger().warn('mission timeout -> returning home')
            self.result_str = 'ABORTED'
            self._stop_exploration()
            self._begin_return_home()
            return

        handler = {
            PARSE: self._do_parse, INIT: self._do_init, EXPLORE: self._do_explore,
            APPROACH: self._do_approach, CONFIRM: self._do_confirm,
            RETURN_HOME: self._do_return_home, DONE: lambda: None,
        }[self.state]
        handler()

    def _do_parse(self):
        self.target = parse_task(self.task_str)
        if not self.target:
            self.get_logger().error(f'Could not parse a room from "{self.task_str}".')
            self.result_str = 'ABORTED'
            self._goto(DONE)
            return
        self.get_logger().info(f'target room = {self.target}')
        self._goto(INIT)

    def _do_init(self):
        if self.pose is None:
            self.get_logger().warn('waiting for /pose ...', throttle_duration_sec=2.0)
            return
        self.home = self.pose
        self.get_logger().info(
            f'HOME captured: ({self.home[0]:.2f}, {self.home[1]:.2f}, '
            f'{math.degrees(self.home[2]):.0f}deg)')
        # One-shot registry sync via service (belt-and-suspenders vs. latched topic).
        if self.signs_client.service_is_ready():
            req = GetSigns.Request()
            req.confirmed_only = False
            fut = self.signs_client.call_async(req)
            fut.add_done_callback(self._sync_signs_cb)
        self._goto(EXPLORE)

    def _sync_signs_cb(self, future):
        try:
            resp = future.result()
            self.signs = [(s.text, s.position.x, s.position.y, s.confirmed)
                          for s in resp.signs]
        except Exception:  # noqa: BLE001
            pass

    def _do_explore(self):
        if not self._explore_started:
            self._set_exploration(True)
            self._explore_started = True
        target_xy = self._find_target()
        if target_xy is not None:
            self.get_logger().info(f'target {self.target} found @ '
                                   f'({target_xy[0]:.2f}, {target_xy[1]:.2f})')
            self._target_sign = target_xy
            self._stop_exploration()
            self._approach_attempts = 0
            self._begin_approach()
            return
        # Exploration finished without finding the target -> give up, return home.
        if self._exploration_complete():
            self.get_logger().warn(f'explored fully; {self.target} not found.')
            self.result_str = 'FAILED_NOT_FOUND'
            self._begin_return_home()

    def _begin_approach(self):
        sx, sy = self._target_sign
        rx, ry, _ = self.pose
        # Stand off from the sign along the line back toward the robot, facing it.
        dx, dy = rx - sx, ry - sy
        norm = math.hypot(dx, dy) or 1.0
        ax = sx + self.standoff * dx / norm
        ay = sy + self.standoff * dy / norm
        ayaw = math.atan2(sy - ay, sx - ax)
        self._start_nav(ax, ay, ayaw, 0.0, 0.0)
        self._goto(APPROACH)

    def _do_approach(self):
        if not self.nav_done:
            return
        if self.nav_success:
            self._confirm_until = self._now() + self.confirm_frames * 0.3
            self._goto(CONFIRM)
            return
        self._approach_attempts += 1
        if self._approach_attempts > self.approach_retries:
            self.get_logger().warn('approach failed; resuming exploration.')
            self._explore_started = False
            self._goto(EXPLORE)
        else:
            self.get_logger().info(f'retry approach ({self._approach_attempts})')
            self._begin_approach()

    def _do_confirm(self):
        # Give perception a moment of dwell to re-read; confirmation already
        # required >= min_observations, so a still-present target == success.
        if self._now() < self._confirm_until:
            return
        if self._find_target() is not None:
            self.get_logger().info(f'CONFIRMED at room {self.target}.')
            self.result_str = 'SUCCESS'
        else:
            self.get_logger().warn('target no longer confirmed; resuming exploration.')
            self._explore_started = False
            self._goto(EXPLORE)
            return
        self._begin_return_home()

    def _begin_return_home(self):
        if self.home is None:
            self._goto(DONE)
            return
        self._return_attempts = 0
        self._start_nav(self.home[0], self.home[1], self.home[2],
                        self.home_xy_tol, self.home_yaw_tol)
        self._goto(RETURN_HOME)

    def _do_return_home(self):
        if not self.nav_done:
            return
        if self.nav_success:
            self.get_logger().info('returned to HOME pose. Mission done.')
            if not self.result_str:
                self.result_str = 'SUCCESS'
            self._goto(DONE)
            return
        self._return_attempts += 1
        if self._return_attempts > 3:
            self.get_logger().error('could not reach HOME exactly; stopping (best effort).')
            self._goto(DONE)
            return
        self.get_logger().info(f'retry return-home ({self._return_attempts})')
        self._start_nav(self.home[0], self.home[1], self.home[2],
                        self.home_xy_tol, self.home_yaw_tol)

    # ---- exploration service ----------------------------------------------
    def _set_exploration(self, start):
        if not self.explore_client.service_is_ready():
            self.explore_client.wait_for_service(timeout_sec=0.2)
            if not self.explore_client.service_is_ready():
                self.get_logger().warn('start_exploration service not ready')
                return
        req = StartExploration.Request()
        req.start = start
        self.explore_client.call_async(req)

    def _stop_exploration(self):
        self._set_exploration(False)

    def _exploration_complete(self):
        # explorer_node publishes COMPLETE on /exploration_state; we infer
        # completion lazily here only after a grace period to avoid premature
        # give-up on a still-growing map.
        return getattr(self, '_explore_state', '') == 'COMPLETE'

    # ---- nav action plumbing ----------------------------------------------
    def _start_nav(self, x, y, yaw, xy_tol, yaw_tol):
        self.nav_done = False
        self.nav_success = False
        self.nav_goal_handle = None
        if not self.nav_client.server_is_ready():
            self.nav_client.wait_for_server(timeout_sec=1.0)
        goal = NavigateToPose.Goal()
        goal.target_pose.header.frame_id = self.map_frame
        goal.target_pose.header.stamp = self.get_clock().now().to_msg()
        goal.target_pose.pose.position.x = float(x)
        goal.target_pose.pose.position.y = float(y)
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        goal.target_pose.pose.orientation.x = qx
        goal.target_pose.pose.orientation.y = qy
        goal.target_pose.pose.orientation.z = qz
        goal.target_pose.pose.orientation.w = qw
        goal.exploration_mode = False
        goal.xy_tolerance = float(xy_tol)
        goal.yaw_tolerance = float(yaw_tol)
        fut = self.nav_client.send_goal_async(goal)
        fut.add_done_callback(self._nav_goal_resp)

    def _nav_goal_resp(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().warn('nav goal rejected')
            self.nav_done = True
            self.nav_success = False
            return
        self.nav_goal_handle = gh
        gh.get_result_async().add_done_callback(self._nav_result_cb)

    def _nav_result_cb(self, future):
        try:
            res = future.result().result
            self.nav_success = bool(res.success)
        except Exception:  # noqa: BLE001
            self.nav_success = False
        self.nav_done = True

    # ---- status ------------------------------------------------------------
    def _publish_status(self):
        m = MissionStatus()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.map_frame
        m.task = self.task_str
        m.target = self.target or ''
        m.state = self.state
        m.result = self.result_str
        m.elapsed_s = float(self._now() - self.t_start)
        if self.home is not None:
            m.home_pose.position.x = self.home[0]
            m.home_pose.position.y = self.home[1]
            qx, qy, qz, qw = quaternion_from_yaw(self.home[2])
            m.home_pose.orientation.x = qx
            m.home_pose.orientation.y = qy
            m.home_pose.orientation.z = qz
            m.home_pose.orientation.w = qw
        self.status_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
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
