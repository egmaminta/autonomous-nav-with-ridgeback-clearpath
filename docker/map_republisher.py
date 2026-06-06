#!/usr/bin/env python3
"""Re-publish the latched occupancy grids as a continuous stream for web dashboards.

rosboard (and similar streaming viewers) forward ROS messages only as they
arrive — they don't re-deliver a stored/latched sample to a tile that subscribes
late. Our `/map` and `/sim_map` are latched (transient-local) and, once the
mission is DONE, stop being republished, so late-joining tiles wait forever.

This node subscribes to the latched grids (transient-local, so it receives the
retained sample immediately) and re-emits whatever it last saw on `*_live`
topics at a steady rate. Point the dashboard at `/map_live` and `/sim_map_live`
and the tiles always populate, regardless of join time or mission state.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid

# Match the publishers: latched = transient-local + reliable, keep-last depth 1.
LATCHED = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

PAIRS = [("/map", "/map_live"), ("/sim_map", "/sim_map_live")]
RATE_HZ = 2.0


class MapRepublisher(Node):
    def __init__(self):
        super().__init__("map_republisher")
        self._last = {}
        self._pubs = {}
        for src, dst in PAIRS:
            self._pubs[src] = self.create_publisher(OccupancyGrid, dst, 1)
            self.create_subscription(
                OccupancyGrid, src,
                lambda msg, s=src: self._last.__setitem__(s, msg),
                LATCHED,
            )
        self.create_timer(1.0 / RATE_HZ, self._tick)
        self.get_logger().info(
            "map_republisher up: " + ", ".join(f"{s}->{d}" for s, d in PAIRS))

    def _tick(self):
        for src, msg in self._last.items():
            self._pubs[src].publish(msg)


def main():
    rclpy.init()
    node = MapRepublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
