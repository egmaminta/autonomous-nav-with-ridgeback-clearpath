#!/usr/bin/env python3
"""perception_node — read room-number signs and localize them in the map.

Pipeline (per the locked-in design): time-synced RGB + depth -> YOLO sign /
text-region detection -> PARSeq recognition on each crop -> room-number regex
filter -> robust depth at the bbox centre -> back-project through the camera
intrinsics -> TF into the map frame -> multi-observation sign registry.

State is exposed two ways so a late-joining consumer is never starved (the old
system's racy cache bug): a LATCHED ``/sign_registry`` topic AND a ``GetSigns``
service. Heavy models (torch) load lazily and failures degrade gracefully — the
node keeps running even if YOLO/PARSeq cannot be loaded.
"""
import re

import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401 (registers PointStamped transform)
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (QoSProfile, DurabilityPolicy, ReliabilityPolicy,
                       qos_profile_sensor_data)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from ridgeback_autonav_msgs.msg import SignDetection, SignRegistry as SignRegistryMsg
from ridgeback_autonav_msgs.srv import GetSigns
from ridgeback_autonav_perception.lib.projection import (intrinsics_from_k, pixel_to_camera,
                                            robust_depth)
from ridgeback_autonav_perception.lib.sign_registry import SignRegistry
from ridgeback_autonav_perception.lib.text_merge import merge_text_detections


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        p = self.declare_parameter
        self.color_topic = p('color_topic', '/r100_0140/sensors/camera_0/color/image').value
        self.depth_topic = p('depth_topic', '/r100_0140/sensors/camera_0/depth/image').value
        self.cinfo_topic = p('camera_info_topic',
                             '/r100_0140/sensors/camera_0/color/camera_info').value
        self.cam_frame = p('camera_optical_frame', 'camera_0_color_optical_frame').value
        self.map_frame = p('map_frame', 'map').value
        registry_topic = p('registry_topic', '/sign_registry').value
        self.sync_slop = p('sync_slop', 0.08).value
        self.detect_period = p('detect_period', 0.3).value
        self.crop_pad = p('crop_pad', 0.10).value
        self.room_re = re.compile(p('room_regex', r'\d{2,4}[A-Za-z]?').value)
        self.depth_is_mm = p('depth_is_mm', False).value
        self.depth_min = p('depth_min', 0.3).value
        self.depth_max = p('depth_max', 8.0).value
        self.depth_window = p('depth_window', 7).value
        self.depth_min_valid = p('depth_min_valid', 10).value
        self.parseq_min_conf = p('parseq_min_conf', 0.5).value
        # Combined detector*recognizer confidence gate (geometric mean), applied
        # per word before line-merging (slam_burger uses 0.3).
        self.min_text_conf = p('min_text_confidence', 0.3).value
        self.detector_backend = p('detector_backend', 'doctr').value  # 'doctr' | 'yolo'

        self.registry = SignRegistry(
            cluster_radius=p('cluster_radius', 0.75).value,
            min_observations=p('min_observations', 3).value,
            max_spread=p('max_spread', 0.30).value,
            position_ema=p('position_ema', 0.4).value)

        self.bridge = CvBridge()
        self.K = None
        self._busy = False
        self._last_t = 0.0

        self.device = p('yolo_device', 'cuda:0').value   # declared once, shared
        self._detector = self._load_detector(p)
        self._reader = self._load_reader(p)

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        cb = ReentrantCallbackGroup()
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        # best-effort so it accepts both our camera_publisher (best-effort) and a
        # stock RealSense driver (reliable); a reliable sub silently drops the former.
        self.create_subscription(CameraInfo, self.cinfo_topic, self._cinfo_cb,
                                 qos_profile_sensor_data, callback_group=cb)
        color_sub = Subscriber(self, Image, self.color_topic, qos_profile=qos_profile_sensor_data)
        depth_sub = Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.sync = ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=5,
                                                slop=self.sync_slop)
        self.sync.registerCallback(self._frame_cb)

        self.registry_pub = self.create_publisher(SignRegistryMsg, registry_topic, latched)
        self.marker_pub = self.create_publisher(MarkerArray, '/sign_markers', 1)
        self.create_service(GetSigns, 'get_signs', self._get_signs_cb, callback_group=cb)
        self.get_logger().info('perception_node ready (latched %s + get_signs service).'
                               % registry_topic)

    # ---- model loading (graceful) -----------------------------------------
    def _load_detector(self, p):
        try:
            if self.detector_backend == 'doctr':
                from ridgeback_autonav_perception.lib.doctr_detector import DocTRDetector
                return DocTRDetector(
                    arch=p('doctr_arch', 'db_mobilenet_v3_large').value,
                    device=self.device,
                    input_size=p('doctr_input_size', 512).value,
                    logger=self.get_logger())
            from ridgeback_autonav_perception.lib.yolo_detector import YoloDetector
            return YoloDetector(
                weights=p('yolo_weights', '').value,
                conf=p('yolo_conf', 0.35).value,
                iou=p('yolo_iou', 0.45).value,
                imgsz=p('yolo_imgsz', 640).value,
                device=self.device,
                hf_filename=p('yolo_hf_filename', 'best.pt').value,
                merge=False,   # raw word boxes; text-merge groups after recognition
                logger=self.get_logger())
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'detector load failed (detection disabled): {exc}')
            return None

    def _load_reader(self, p):
        try:
            from ridgeback_autonav_perception.lib.parseq_reader import ParseqReader
            return ParseqReader(
                hub=p('parseq_hub', 'baudm/parseq').value,
                model=p('parseq_model', 'parseq').value,
                device=self.device,
                min_conf=self.parseq_min_conf,
                logger=self.get_logger())
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'PARSeq load failed (recognition disabled): {exc}')
            return None

    # ---- callbacks ---------------------------------------------------------
    def _cinfo_cb(self, msg: CameraInfo):
        self.K = intrinsics_from_k(msg.k)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _frame_cb(self, color_msg: Image, depth_msg: Image):
        now = self._now()
        if self._busy or now - self._last_t < self.detect_period:
            return
        if self.K is None or self._detector is None or self._reader is None:
            return
        self._busy = True
        try:
            self._process(color_msg, depth_msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'frame processing error: {exc}')
        finally:
            self._last_t = now
            self._busy = False

    def _process(self, color_msg, depth_msg):
        bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        is_mm = self.depth_is_mm or (np.asarray(depth).dtype == np.uint16)
        fx, fy, cx, cy = self.K
        h, w = bgr.shape[:2]

        transform = self._lookup_tf(color_msg.header.stamp)
        if transform is None:
            return

        # 1. Detect word boxes -> 2. recognize EACH tight box with PARSeq
        # (PARSeq's strength) -> 3. combined det*rec confidence gate.
        items = []
        for (x1, y1, x2, y2, det_score, _cls) in self._detector.detect(bgr):
            crop = self._pad_crop(bgr, x1, y1, x2, y2, w, h)
            text, rec_conf = self._reader.read(crop)
            if not text.strip():
                continue
            conf = float((max(0.0, det_score) * max(0.0, rec_conf)) ** 0.5)
            if conf < self.min_text_conf:
                continue
            items.append({'text': text.strip(), 'confidence': conf,
                          'bbox_rect': (x1, y1, x2, y2)})

        # 4. Merge word texts into phrases by line ("RM" + "205" -> "RM 205").
        updated = False
        for ph in merge_text_detections(items):
            m = self.room_re.search(ph['text'])
            if not m:
                continue
            x1, y1, x2, y2 = ph['bbox_rect']
            z = robust_depth(depth, (x1, y1, x2, y2), window=self.depth_window,
                             is_mm=is_mm, dmin=self.depth_min, dmax=self.depth_max,
                             min_valid=self.depth_min_valid)
            if z is None:
                continue
            u, v = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            X, Y, Z = pixel_to_camera(u, v, z, fx, fy, cx, cy)
            pmap = self._to_map(X, Y, Z, transform)
            if pmap is None:
                continue
            entry = self.registry.observe(m.group(), pmap[0], pmap[1],
                                          ph['confidence'], self._now())
            if entry is not None:
                updated = True
                self.get_logger().info(
                    f"read '{ph['text']}' -> room '{entry.text}' @ "
                    f"({entry.x:.2f},{entry.y:.2f}) obs={entry.observations} "
                    f"confirmed={entry.confirmed}")
        if updated:
            self._publish_registry()

    def _pad_crop(self, bgr, x1, y1, x2, y2, w, h):
        pad_x = (x2 - x1) * self.crop_pad
        pad_y = (y2 - y1) * self.crop_pad
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(w, int(x2 + pad_x))
        cy2 = min(h, int(y2 + pad_y))
        return bgr[cy1:cy2, cx1:cx2]

    def _lookup_tf(self, stamp):
        for t in (Time.from_msg(stamp), Time()):
            try:
                return self.tf_buffer.lookup_transform(
                    self.map_frame, self.cam_frame, t,
                    timeout=Duration(seconds=0.1))
            except Exception:  # noqa: BLE001
                continue
        self.get_logger().warn('TF map<-camera unavailable; dropping detections',
                               throttle_duration_sec=5.0)
        return None

    def _to_map(self, X, Y, Z, transform):
        ps = PointStamped()
        ps.header.frame_id = self.cam_frame
        ps.point.x, ps.point.y, ps.point.z = float(X), float(Y), float(Z)
        try:
            out = tf2_geometry_msgs.do_transform_point(ps, transform)
            return (out.point.x, out.point.y)
        except Exception:  # noqa: BLE001
            return None

    # ---- outputs -----------------------------------------------------------
    def _publish_registry(self):
        msg = SignRegistryMsg()
        msg.header = self._header()
        for e in self.registry.all(confirmed_only=False):
            msg.signs.append(self._entry_to_msg(e))
        self.registry_pub.publish(msg)
        self._publish_markers()

    def _entry_to_msg(self, e):
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
            resp.signs.append(self._entry_to_msg(e))
        return resp

    def _publish_markers(self):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
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
    node = PerceptionNode()
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
