#!/usr/bin/env python3
"""camera_publisher — publish the D435 feed from the Jetson via pyrealsense2.

A lightweight alternative to the realsense2_camera ROS driver: it opens the
RealSense directly with the librealsense Python API, aligns depth to color, and
publishes exactly the topics perception_node consumes:

  * <ns>/color/image_raw                  sensor_msgs/Image (bgr8)
  * <ns>/aligned_depth_to_color/image_raw sensor_msgs/Image (16UC1, millimetres)
  * <ns>/color/camera_info                sensor_msgs/CameraInfo (live intrinsics)

Depth is aligned to color (our projection samples depth at a color-pixel bbox).
On USB-2 keep the profile small (default 640x480x15); move to a USB-3 port for
higher rates. If the device is on the robot's other PC, run this there instead.
"""
import threading

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster

try:                              # top-level import fails on some Jetson builds
    import pyrealsense2 as rs
    rs.pipeline
except Exception:                 # noqa: BLE001
    import pyrealsense2.pyrealsense2 as rs


class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        p = self.declare_parameter
        ns = p('camera_namespace', '/r100_0140/sensors/camera_0').value.rstrip('/')
        self.color_topic = p('color_topic', ns + '/color/image_raw').value
        self.depth_topic = p('depth_topic', ns + '/aligned_depth_to_color/image_raw').value
        self.info_topic = p('camera_info_topic', ns + '/color/camera_info').value
        self.frame_id = p('camera_optical_frame', 'camera_0_color_optical_frame').value
        # The RealSense ROS driver (which we replaced) normally publishes the
        # optical frame; since it's missing from the robot's TF tree, we add the
        # standard <camera_link> -> <optical> transform ourselves so perception
        # can project detections into the map. Assumes the D435 is in its URDF
        # mount (camera_0_link); if the camera was physically moved, set
        # publish_optical_tf:=false and provide the real transform elsewhere.
        self.parent_frame = p('parent_frame', 'camera_0_link').value
        self.publish_optical_tf = p('publish_optical_tf', True).value
        self.cw = p('color_width', 640).value
        self.ch = p('color_height', 480).value
        self.cfps = p('color_fps', 15).value
        self.dw = p('depth_width', 640).value
        self.dh = p('depth_height', 480).value
        self.dfps = p('depth_fps', 15).value
        self.serial = p('serial', '').value

        self.bridge = CvBridge()
        self.color_pub = self.create_publisher(Image, self.color_topic, qos_profile_sensor_data)
        self.depth_pub = self.create_publisher(Image, self.depth_topic, qos_profile_sensor_data)
        self.info_pub = self.create_publisher(CameraInfo, self.info_topic, qos_profile_sensor_data)

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if self.serial:
            cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.cw, self.ch, rs.format.bgr8, self.cfps)
        cfg.enable_stream(rs.stream.depth, self.dw, self.dh, rs.format.z16, self.dfps)
        try:
            self.profile = self.pipeline.start(cfg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f'RealSense start failed: {exc}. On USB-2 try a smaller profile '
                '(color_fps:=6) or move the camera to a USB-3 port.')
            raise
        self.align = rs.align(rs.stream.color)

        if self.publish_optical_tf:
            self._static_bc = StaticTransformBroadcaster(self)
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.parent_frame
            t.child_frame_id = self.frame_id
            # Standard ROS camera_link(FLU) -> optical(RDF) rotation.
            t.transform.rotation.x = -0.5
            t.transform.rotation.y = 0.5
            t.transform.rotation.z = -0.5
            t.transform.rotation.w = 0.5
            self._static_bc.sendTransform(t)
            self.get_logger().info(
                f'published static TF {self.parent_frame} -> {self.frame_id}')

        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.get_logger().info(
            f'camera_publisher streaming {self.cw}x{self.ch}@{self.cfps} '
            f'(depth aligned to color) -> {self.color_topic}')

    def _loop(self):
        while rclpy.ok() and not self._stop:
            try:
                frames = self.pipeline.wait_for_frames(5000)
            except Exception:  # noqa: BLE001
                self.get_logger().warn('no frames (5s) — is the camera on USB-3?',
                                       throttle_duration_sec=5.0)
                continue
            frames = self.align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue
            stamp = self.get_clock().now().to_msg()
            color_np = np.asanyarray(color.get_data())          # HxWx3 bgr8
            depth_np = np.asanyarray(depth.get_data())           # HxW uint16 (mm)

            cmsg = self.bridge.cv2_to_imgmsg(color_np, encoding='bgr8')
            cmsg.header.stamp = stamp
            cmsg.header.frame_id = self.frame_id
            self.color_pub.publish(cmsg)

            dmsg = self.bridge.cv2_to_imgmsg(depth_np, encoding='16UC1')
            dmsg.header.stamp = stamp
            dmsg.header.frame_id = self.frame_id
            self.depth_pub.publish(dmsg)

            self.info_pub.publish(self._camera_info(color, stamp))

    def _camera_info(self, color_frame, stamp):
        intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self.frame_id
        info.width = intr.width
        info.height = intr.height
        info.k = [intr.fx, 0.0, intr.ppx, 0.0, intr.fy, intr.ppy, 0.0, 0.0, 1.0]
        info.p = [intr.fx, 0.0, intr.ppx, 0.0, 0.0, intr.fy, intr.ppy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        info.distortion_model = 'plumb_bob'
        info.d = [float(c) for c in list(intr.coeffs)[:5]]
        return info

    def destroy_node(self):
        self._stop = True
        try:
            self.pipeline.stop()
        except Exception:  # noqa: BLE001
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
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
