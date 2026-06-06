#!/usr/bin/env python3
"""fake_perception_node — ground-truth sign perception for torch-free sim tests.

Drop-in stand-in for ``perception_node`` when you want to exercise the mission
FSM and navigation in the simulator without running YOLO/PARSeq. It reuses the
real ``SignRegistry`` (same confirmation logic) and publishes the same
``/sign_registry`` (latched) topic + ``GetSigns`` service, so ``mission_node`` is
byte-for-byte unaware which perception it is talking to.

Frame trick (so it stays frame-correct): a sign's position *relative to the
robot* is frame-independent. We compute that relative offset from the
ground-truth world pose (``/sim/ground_truth``) and the world sign coordinates,
then re-anchor it onto the robot's MAP pose (``/pose``) — yielding the sign in
the map frame exactly as the real depth+TF pipeline would.
"""
import math
import os

import rclpy
import yaml
from geometry_msgs.msg import Point, PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from ridgeback_autonav_msgs.msg import SignDetection, SignRegistry as SignRegistryMsg
from ridgeback_autonav_msgs.srv import GetSigns
from ridgeback_autonav_perception.lib.sign_registry import SignRegistry

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class FakePerceptionNode(Node):
    def __init__(self):
        super().__init__('fake_perception_node')
        p = self.declare_parameter
        world_file = p('world_file', '').value
        self.map_frame = p('map_frame', 'map').value
        registry_topic = p('registry_topic', '/sign_registry').value
        self.fov = math.radians(p('camera_fov_deg', 87.0).value)
        self.detect_range = p('detect_range', 4.5).value
        self.min_range = p('detect_min_range', 0.4).value

        self.signs_world = self._load_signs(world_file)
        self.registry = SignRegistry(
            cluster_radius=p('cluster_radius', 0.75).value,
            min_observations=p('min_observations', 3).value,
            max_spread=p('max_spread', 0.30).value,
            position_ema=p('position_ema', 0.4).value)

        self.gt = None        # (x, y, yaw) world pose
        self.map_pose = None  # (x, y, yaw) map pose

        self.create_subscription(PoseStamped, '/sim/ground_truth', self._gt_cb, 10)
        self.create_subscription(PoseStamped, p('pose_topic', '/pose').value,
                                 self._pose_cb, 10)
        self.registry_pub = self.create_publisher(SignRegistryMsg, registry_topic, LATCHED)
        self.marker_pub = self.create_publisher(MarkerArray, '/sign_markers', 1)
        self.create_service(GetSigns, 'get_signs', self._get_signs_cb)
        self.create_timer(0.2, self._tick)
        self.get_logger().info(
            f'fake_perception_node up: signs={[s[0] for s in self.signs_world]}')

    def _load_signs(self, path):
        if path and os.path.exists(os.path.expanduser(path)):
            with open(os.path.expanduser(path)) as fh:
                spec = yaml.safe_load(fh)
            return [(str(s['text']), float(s['x']), float(s['y']))
                    for s in spec.get('signs', [])]
        self.get_logger().warn('no world_file; fake perception has no signs.')
        return []

    def _gt_cb(self, msg):
        self.gt = (msg.pose.position.x, msg.pose.position.y, _yaw(msg.pose.orientation))

    def _pose_cb(self, msg):
        self.map_pose = (msg.pose.position.x, msg.pose.position.y, _yaw(msg.pose.orientation))

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _tick(self):
        if self.gt is None or self.map_pose is None:
            return
        gx, gy, gth = self.gt
        updated = False
        for (text, sx, sy) in self.signs_world:
            dx, dy = sx - gx, sy - gy
            dist = math.hypot(dx, dy)
            if not (self.min_range <= dist <= self.detect_range):
                continue
            bearing = math.atan2(dy, dx) - gth
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            if abs(bearing) > self.fov / 2.0:
                continue
            # Relative offset (robot frame) is frame-independent; re-anchor on map pose.
            rel_x = math.cos(-gth) * dx - math.sin(-gth) * dy
            rel_y = math.sin(-gth) * dx + math.cos(-gth) * dy
            mx, my, mth = self.map_pose
            wx = mx + math.cos(mth) * rel_x - math.sin(mth) * rel_y
            wy = my + math.sin(mth) * rel_x + math.cos(mth) * rel_y
            self.registry.observe(text, wx, wy, 0.95, self._now())
            updated = True
        if updated:
            self._publish()

    def _publish(self):
        msg = SignRegistryMsg()
        msg.header = self._header()
        for e in self.registry.all():
            msg.signs.append(self._entry(e))
        self.registry_pub.publish(msg)
        self._markers()

    def _entry(self, e):
        d = SignDetection()
        d.header = self._header()
        d.text = e.text
        d.position = Point(x=float(e.x), y=float(e.y), z=0.0)
        d.confidence = float(e.confidence)
        d.observations = int(e.observations)
        d.confirmed = bool(e.confirmed)
        return d

    def _header(self):
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        h.frame_id = self.map_frame
        return h

    def _get_signs_cb(self, req, resp):
        for e in self.registry.all(confirmed_only=req.confirmed_only):
            resp.signs.append(self._entry(e))
        return resp

    def _markers(self):
        arr = MarkerArray()
        clr = Marker()
        clr.action = Marker.DELETEALL
        arr.markers.append(clr)
        for i, e in enumerate(self.registry.all()):
            m = Marker()
            m.header = self._header()
            m.ns = 'signs'
            m.id = i
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = e.x, e.y, 0.3
            m.pose.orientation.w = 1.0
            m.scale.z = 0.4
            m.color.a = 1.0
            m.color.g = 1.0 if e.confirmed else 0.3
            m.color.r = 0.0 if e.confirmed else 1.0
            m.text = e.text
            arr.markers.append(m)
        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = FakePerceptionNode()
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
