#!/usr/bin/env python3
"""Cross-machine topic relay.

The Ridgeback onboard PC is near-saturated, so every DDS subscription a Jetson
node makes to a robot topic adds load on the robot side. This node collapses N
local subscribers per topic into a SINGLE cross-machine subscription, then
re-publishes on a local topic that all ridgeback_autonav nodes consume.

Configured via the ``relays`` parameter: a list of "remote|local|type" strings,
e.g. "/r100_0140/sensors/lidar2d_0/scan|/scan|sensor_msgs/msg/LaserScan".

Sensor data uses SensorDataQoS (best-effort) so it matches typical driver
publishers; everything else uses a small reliable depth-10 queue.
"""
from importlib import import_module

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy


SENSOR_TYPES = {
    'sensor_msgs/msg/LaserScan',
    'sensor_msgs/msg/Image',
    'sensor_msgs/msg/Imu',
    'sensor_msgs/msg/PointCloud2',
    'sensor_msgs/msg/CompressedImage',
}


def _load_msg_type(type_str):
    """Resolve 'pkg/msg/Type' to the Python message class."""
    pkg, kind, name = type_str.split('/')
    module = import_module(f'{pkg}.{kind}')
    return getattr(module, name)


class TopicRelay(Node):
    def __init__(self):
        super().__init__('topic_relay')
        default = [
            '/r100_0140/sensors/lidar2d_0/scan|/scan|sensor_msgs/msg/LaserScan',
            '/r100_0140/platform/odom/filtered|/odom|nav_msgs/msg/Odometry',
        ]
        relays = self.declare_parameter('relays', default).value
        self._pairs = []
        for spec in relays:
            try:
                remote, local, type_str = [s.strip() for s in spec.split('|')]
            except ValueError:
                self.get_logger().error(f'Bad relay spec (need remote|local|type): {spec!r}')
                continue
            msg_type = _load_msg_type(type_str)
            sensor = type_str in SENSOR_TYPES
            qos = qos_profile_sensor_data if sensor else QoSProfile(
                depth=10, reliability=ReliabilityPolicy.RELIABLE)
            pub = self.create_publisher(msg_type, local, qos)
            # Late-bound closure capture via default arg.
            self.create_subscription(
                msg_type, remote, lambda msg, p=pub: p.publish(msg), qos)
            self._pairs.append((remote, local))
            self.get_logger().info(f'Relaying {remote} -> {local} [{type_str}]')
        if not self._pairs:
            self.get_logger().warn('topic_relay started with no valid relays.')


def main(args=None):
    rclpy.init(args=args)
    node = TopicRelay()
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
