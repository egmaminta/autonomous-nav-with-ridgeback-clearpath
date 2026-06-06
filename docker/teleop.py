#!/usr/bin/env python3
"""Keyboard teleop for the holonomic Ridgeback sim.

Publishes geometry_msgs/Twist to /r100_0140/cmd_vel (override via argv[1]) so you
can drive the robot around the world by hand. Run it attached to a TTY:

    docker compose exec -it drive python3 /opt/teleop.py

Velocity is "sticky": a direction key sets that motion and it continues until you
press space (STOP). This matches teleop_twist_keyboard's model.
"""
import sys
import termios
import threading
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

HELP = """
ridgeback_autonav teleop (holonomic Ridgeback)
   w / s : forward / back        (linear x)
   a / d : strafe left / right   (linear y)
   q / e : rotate CCW / CW       (yaw)
   space : STOP
   + / - : faster / slower
   x or Ctrl-C : quit
"""

BASE = {"x": 0.4, "y": 0.3, "w": 0.6}   # m/s, m/s, rad/s


class Teleop(Node):
    def __init__(self, topic):
        super().__init__("teleop")
        self.pub = self.create_publisher(Twist, topic, 10)
        self.vx = self.vy = self.wz = 0.0
        self.spd = 1.0
        self.create_timer(0.1, self._tick)   # 10 Hz republish (keeps it moving)

    def _tick(self):
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = self.vx, self.vy, self.wz
        self.pub.publish(t)


def main():
    topic = sys.argv[1] if len(sys.argv) > 1 else "/r100_0140/cmd_vel"
    rclpy.init()
    node = Teleop(topic)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print(HELP)
    print(f"publishing -> {topic}\n", flush=True)

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)            # cbreak so Ctrl-C still raises SIGINT
        while True:
            c = sys.stdin.read(1)
            if c in ("x", "\x03"):
                break
            elif c == "w": node.vx, node.vy, node.wz = BASE["x"] * node.spd, 0.0, 0.0
            elif c == "s": node.vx, node.vy, node.wz = -BASE["x"] * node.spd, 0.0, 0.0
            elif c == "a": node.vx, node.vy, node.wz = 0.0, BASE["y"] * node.spd, 0.0
            elif c == "d": node.vx, node.vy, node.wz = 0.0, -BASE["y"] * node.spd, 0.0
            elif c == "q": node.vx, node.vy, node.wz = 0.0, 0.0, BASE["w"] * node.spd
            elif c == "e": node.vx, node.vy, node.wz = 0.0, 0.0, -BASE["w"] * node.spd
            elif c == " ": node.vx = node.vy = node.wz = 0.0
            elif c == "+": node.spd = min(3.0, node.spd * 1.25)
            elif c == "-": node.spd = max(0.1, node.spd / 1.25)
            print(f"\r vx={node.vx:+.2f} vy={node.vy:+.2f} wz={node.wz:+.2f} "
                  f"spd={node.spd:.2f}   ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        node.vx = node.vy = node.wz = 0.0
        node._tick()                 # final stop command
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        print("\nteleop stopped.")


if __name__ == "__main__":
    main()
