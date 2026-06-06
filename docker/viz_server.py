#!/usr/bin/env python3
"""Single-page browser viewer for the ridgeback_autonav sim — ego POV + live SLAM map, drive it.

A slam_burger-style interactive viewer ported to ridgeback_autonav: a stdlib ``http.server``
embedded in a rclpy node serves ONE self-contained HTML/Canvas page with

  * the robot's FIRST-PERSON view  — the onboard synthetic camera, color on top
    and a turbo-colormapped depth image below (server-side encoded JPEGs); and
  * the SLAM occupancy grid on the side — the ``/map`` that mapping_node builds
    up as the robot navigates (unknown -> free/occupied), with the robot pose,
    heading, footprint and live LIDAR overlaid in the map frame.

You drive with the keyboard directly in the page (no second terminal): the
browser POSTs velocities and the node republishes them as Twist on the robot's
cmd_vel with a short deadman.

Topics consumed (all already published by the sim + mapping stack):
    /map                                       OccupancyGrid (latched, frame map)
    /pose                                       PoseStamped (base_link in map frame)
    /r100_0140/sensors/lidar2d_0/scan           LaserScan
    /r100_0140/sensors/camera_0/color/image_raw                 Image (bgr8)
    /r100_0140/sensors/camera_0/aligned_depth_to_color/image_raw Image (32FC1, metres)
    -> publishes Twist on /r100_0140/cmd_vel

Run it next to the sim (see the ``view`` service in docker-compose.yml):
    python3 /opt/viz_server.py        # then open http://localhost:8088
"""
import base64
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (QoSProfile, DurabilityPolicy, ReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import Image, LaserScan

# Optional: camera feeds need OpenCV + cv_bridge + numpy. Guard so the viewer
# still runs (map + teleop only) if they are missing or the camera is disabled.
try:
    import cv2
    import numpy as np
    from cv_bridge import CvBridge
    _CAM_OK = True
except Exception:  # noqa: BLE001
    _CAM_OK = False

# Optional: click-to-navigate needs the nav action. Guard so the viewer still
# runs (map + teleop + camera) if ridgeback_autonav_msgs/nav_server are unavailable.
try:
    from rclpy.action import ActionClient
    from ridgeback_autonav_msgs.action import NavigateToPose
    _NAV_OK = True
except Exception:  # noqa: BLE001
    _NAV_OK = False

# Latched (transient-local) so we get the retained map sample at join.
LATCHED = QoSProfile(depth=1,
                     reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)

BASE = {"x": 0.4, "y": 0.3, "w": 0.6}  # matches docker/teleop.py
CMD_RATE_HZ = 20.0
CMD_DEADMAN_SEC = 0.4
DEPTH_MAX_M = 10.0  # sim depth range; turbo near->far = blue->red
JPEG_Q = 80


def yaw_from_quat(z, w):
    return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)


class VizServer(Node):

    def __init__(self):
        super().__init__("viz_server")
        self.port = int(self.declare_parameter("port", 8088).value)
        self.scan_topic = self.declare_parameter(
            "scan_topic", "/r100_0140/sensors/lidar2d_0/scan").value
        self.cmd_topic = self.declare_parameter("cmd_vel_topic",
                                                "/r100_0140/cmd_vel").value
        # Match the simulator's realsense2_camera-style topic names (see
        # ridgeback_autonav_sim/sim_node.py), so sim and real hardware are identical.
        self.color_topic = self.declare_parameter(
            "color_topic", "/r100_0140/sensors/camera_0/color/image_raw").value
        self.depth_topic = self.declare_parameter(
            "depth_topic",
            "/r100_0140/sensors/camera_0/aligned_depth_to_color/image_raw"
        ).value
        self.map_topic = self.declare_parameter("map_topic", "/map").value
        self.pose_topic = self.declare_parameter("pose_topic", "/pose").value
        self.laser_offset_x = float(
            self.declare_parameter("laser_offset_x", 0.42).value)
        self.robot_radius = float(
            self.declare_parameter("robot_radius", 0.43).value)

        self._lock = threading.Lock()
        self._map = None  # {w,h,res,ox,oy,seq,b64}
        self._map_seq = 0
        self._pose = None  # {x,y,theta}  (map frame)
        self._scan = None  # {amin,ainc,rmax,off,ranges}
        self._color_jpg = None  # bytes
        self._depth_jpg = None  # bytes
        self._cmd = (0.0, 0.0, 0.0)
        self._cmd_t = -1e9
        self._nav_active = False
        self._nav_state = "idle"  # idle|SENDING|PLANNING|FOLLOWING|RECOVERING|done|stopped|rejected
        self._nav_dist = float("inf")
        self._nav_msg = ""
        self._goal_xy = None  # (x,y) map frame, while a goal is live
        self._goal_handle = None
        self._goal_id = 0  # monotonic; stale callbacks (id mismatch) are ignored
        self._pending_goal = None  # (x,y) set by HTTP thread, consumed on executor
        self._cancel_req = False

        self.create_subscription(OccupancyGrid, self.map_topic, self._on_map,
                                 LATCHED)
        self.create_subscription(PoseStamped, self.pose_topic, self._on_pose,
                                 10)
        self.create_subscription(LaserScan, self.scan_topic, self._on_scan,
                                 qos_profile_sensor_data)
        if _CAM_OK:
            self._bridge = CvBridge()
            self.create_subscription(Image, self.color_topic, self._on_color,
                                     qos_profile_sensor_data)
            self.create_subscription(Image, self.depth_topic, self._on_depth,
                                     qos_profile_sensor_data)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.create_timer(1.0 / CMD_RATE_HZ, self._publish_cmd)

        # Click-to-navigate: action client + a slow tick that issues goals/cancels
        # from the executor thread (HTTP threads only ever set flags under _lock).
        self._nav_client = None
        if _NAV_OK:
            self._nav_client = ActionClient(self, NavigateToPose,
                                            "navigate_to_pose")
            self.create_timer(0.1, self._nav_tick)

        self._serve()
        cam = "color+depth" if _CAM_OK else "DISABLED (no cv2/cv_bridge)"
        nav = "navigate_to_pose" if _NAV_OK else "DISABLED (no ridgeback_autonav_msgs)"
        self.get_logger().info(
            f"viz_server up: http://localhost:{self.port}  "
            f"map={self.map_topic} camera={cam} click-to-nav={nav} "
            f"driving->{self.cmd_topic}")

    def _on_map(self, msg: OccupancyGrid):
        # int8 occupancy (-1 unknown / 0 free / 100 occ); -1 & 0xff == 255.
        data = bytes(v & 0xFF for v in msg.data)
        self._map_seq += 1
        with self._lock:
            self._map = {
                "w": msg.info.width,
                "h": msg.info.height,
                "res": msg.info.resolution,
                "ox": msg.info.origin.position.x,
                "oy": msg.info.origin.position.y,
                "seq": self._map_seq,
                "b64": base64.b64encode(data).decode("ascii"),
            }

    def _on_pose(self, msg: PoseStamped):
        th = yaw_from_quat(msg.pose.orientation.z, msg.pose.orientation.w)
        with self._lock:
            self._pose = {
                "x": msg.pose.position.x,
                "y": msg.pose.position.y,
                "theta": th
            }

    def _on_scan(self, msg: LaserScan):
        ranges = [
            round(float(r), 3) if math.isfinite(r) else 0.0 for r in msg.ranges
        ]
        with self._lock:
            self._scan = {
                "amin": float(msg.angle_min),
                "ainc": float(msg.angle_increment),
                "rmax": float(msg.range_max),
                "off": self.laser_offset_x,
                "ranges": ranges
            }

    def _on_color(self, msg: Image):
        try:
            img = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            ok, buf = cv2.imencode(".jpg", img,
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            if ok:
                with self._lock:
                    self._color_jpg = buf.tobytes()
        except Exception:  # noqa: BLE001
            pass

    def _on_depth(self, msg: Image):
        try:
            d = self._bridge.imgmsg_to_cv2(msg, "32FC1")
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
            norm = np.clip(d / DEPTH_MAX_M, 0.0, 1.0)
            u8 = (norm * 255.0).astype(np.uint8)
            turbo = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
            turbo[d <= 0.05] = (0, 0, 0)  # no return -> black
            ok, buf = cv2.imencode(".jpg", turbo,
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            if ok:
                with self._lock:
                    self._depth_jpg = buf.tobytes()
        except Exception:  # noqa: BLE001
            pass

    def _publish_cmd(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        with self._lock:
            if self._nav_active:
                return  # nav_server owns cmd_vel while a goal is live
            vx, vy, wz = self._cmd
            fresh = (now - self._cmd_t) < CMD_DEADMAN_SEC
        t = Twist()
        if fresh:
            t.linear.x, t.linear.y, t.angular.z = vx, vy, wz
        self.cmd_pub.publish(t)

    def set_cmd(self, vx, vy, wz):
        now = self.get_clock().now().nanoseconds * 1e-9
        with self._lock:
            self._cmd = (float(vx), float(vy), float(wz))
            self._cmd_t = now
            # Manual override: any real motion command cancels an active nav goal.
            if (vx or vy or wz) and self._nav_active:
                self._cancel_req = True

    def request_goal(self, x, y):
        if not _NAV_OK:
            return False
        with self._lock:
            self._pending_goal = (float(x), float(y))
        return True

    def request_cancel(self):
        with self._lock:
            if self._nav_active:
                self._cancel_req = True

    def _nav_tick(self):
        """Runs on the executor thread: issue pending goals / cancels safely."""
        with self._lock:
            pend = self._pending_goal
            self._pending_goal = None
            cancel = self._cancel_req
            self._cancel_req = False
            gh = self._goal_handle
        if cancel and gh is not None:
            try:
                gh.cancel_goal_async()
            except Exception:  # noqa: BLE001
                pass
        if pend is not None:
            self._send_goal(pend)

    def _send_goal(self, xy):
        if self._nav_client is None or not self._nav_client.server_is_ready():
            with self._lock:
                self._nav_state = "no nav server"
                self._nav_msg = "nav_server not available"
            self.get_logger().warn("click-to-nav: nav_server not ready")
            return
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.pose.position.x, ps.pose.position.y = xy[0], xy[1]
        ps.pose.orientation.w = 1.0
        goal = NavigateToPose.Goal()
        goal.target_pose = ps
        goal.exploration_mode = True  # traverse unknown cheaply on the building map
        goal.xy_tolerance = 0.0  # use node default
        goal.yaw_tolerance = 3.15  # final heading doesn't matter for click-to-go
        with self._lock:
            old = self._goal_handle  # supersede any in-flight goal
            self._goal_id += 1
            gid = self._goal_id
            self._goal_handle = None
            self._nav_active = True
            self._nav_state = "SENDING"
            self._nav_dist = float("inf")
            self._nav_msg = ""
            self._goal_xy = (xy[0], xy[1])
        if old is not None:
            try:
                old.cancel_goal_async()  # don't leave two goals driving cmd_vel
            except Exception:  # noqa: BLE001
                pass
        fut = self._nav_client.send_goal_async(
            goal, feedback_callback=lambda fb, g=gid: self._nav_fb(g, fb))
        fut.add_done_callback(lambda f, g=gid: self._nav_goal_response(g, f))
        self.get_logger().info(f"click-to-nav -> ({xy[0]:.2f}, {xy[1]:.2f})")

    def _nav_goal_response(self, gid, fut):
        try:
            gh = fut.result()
        except Exception:  # noqa: BLE001
            gh = None
        with self._lock:
            if gid != self._goal_id:
                return  # superseded by a newer goal
            if gh is None or not gh.accepted:
                self._nav_active = False
                self._nav_state = "rejected"
                self._goal_handle = None
                self._goal_xy = None
                return
            self._goal_handle = gh
        gh.get_result_async().add_done_callback(
            lambda f, g=gid: self._nav_result(g, f))

    def _nav_fb(self, gid, fb):
        f = fb.feedback
        with self._lock:
            if gid != self._goal_id:
                return
            self._nav_state = f.state or "FOLLOWING"
            self._nav_dist = float(f.distance_remaining)

    def _nav_result(self, gid, fut):
        try:
            res = fut.result().result
            ok, msg = bool(res.success), str(res.message)
        except Exception:  # noqa: BLE001
            ok, msg = False, "nav error"
        with self._lock:
            if gid != self._goal_id:
                return  # a newer goal is in charge now
            self._nav_active = False
            self._goal_handle = None
            self._goal_xy = None
            self._nav_state = "done" if ok else "stopped"
            self._nav_dist = float("inf")
            self._nav_msg = msg

    def map_json(self):
        with self._lock:
            return self._map

    def state_json(self):
        with self._lock:
            dist = self._nav_dist
            return {
                "pose": self._pose,
                "scan": self._scan,
                "map_seq": self._map["seq"] if self._map else 0,
                "robot_radius": self.robot_radius,
                "has_color": self._color_jpg is not None,
                "has_depth": self._depth_jpg is not None,
                "cam": _CAM_OK,
                "nav": {
                    "enabled": _NAV_OK,
                    "active": self._nav_active,
                    "state": self._nav_state,
                    "dist": None if dist == float("inf") else round(dist, 2),
                    "goal": list(self._goal_xy) if self._goal_xy else None,
                    "msg": self._nav_msg,
                },
            }

    def color_jpg(self):
        with self._lock:
            return self._color_jpg

    def depth_jpg(self):
        with self._lock:
            return self._depth_jpg

    def _serve(self):
        node = self

        class Handler(BaseHTTPRequestHandler):

            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype="application/json"):
                if isinstance(body, str):
                    body = body.encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                if body:
                    self.wfile.write(body)

            def _send_jpg(self, data):
                if not data:
                    self._send(503, b"", "image/jpeg")
                else:
                    self._send(200, data, "image/jpeg")

            def do_GET(self):
                p = self.path.split("?", 1)[0]
                if p == "/" or p.startswith("/index"):
                    self._send(200, INDEX_HTML, "text/html; charset=utf-8")
                elif p == "/api/map":
                    m = node.map_json()
                    self._send(200, json.dumps(m) if m else "null")
                elif p == "/api/state":
                    self._send(200, json.dumps(node.state_json()))
                elif p == "/api/camera/color":
                    self._send_jpg(node.color_jpg())
                elif p == "/api/camera/depth":
                    self._send_jpg(node.depth_jpg())
                else:
                    self._send(404, "{}")

            def do_POST(self):
                p = self.path.split("?", 1)[0]
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if n else b""
                try:
                    if p == "/api/teleop":
                        body = json.loads(raw or b"{}")
                        node.set_cmd(body.get("vx", 0.0), body.get("vy", 0.0),
                                     body.get("wz", 0.0))
                        self._send(200, '{"ok":true}')
                    elif p == "/api/goal":
                        body = json.loads(raw or b"{}")
                        ok = node.request_goal(body.get("x", 0.0),
                                               body.get("y", 0.0))
                        self._send(200 if ok else 503, json.dumps({"ok": ok}))
                    elif p == "/api/nav_cancel":
                        node.request_cancel()
                        self._send(200, '{"ok":true}')
                    else:
                        self._send(404, "{}")
                except Exception as exc:  # noqa: BLE001
                    self._send(400, json.dumps({"error": str(exc)}))

        httpd = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self._httpd = httpd


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ridgeback AutoNav simulator</title>
<style>
  :root{--ink:#1a1a1a;--mut:#666;--line:#bcbcbc;--hd:#f2f2f2;--bg:#e8e8e8}
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);overflow:hidden;
    font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  #bar{display:flex;justify-content:space-between;align-items:center;height:30px;padding:0 12px;
    border-bottom:1px solid var(--line);background:#fff;font-size:12px}
  #bar .t{font-weight:600}
  #bar .s{color:var(--mut);font-variant-numeric:tabular-nums}
  #body{display:grid;grid-template-columns:1fr 1fr;height:calc(100vh - 30px)}
  .col{display:flex;flex-direction:column;min-width:0;min-height:0}
  .panel{display:flex;flex-direction:column;min-height:0;border-right:1px solid var(--line);
    border-bottom:1px solid var(--line);background:#fff}
  .panel .hd{flex:0 0 auto;padding:3px 9px;border-bottom:1px solid var(--line);background:var(--hd);
    font-size:11px;color:#333;display:flex;justify-content:space-between;align-items:center}
  .panel .bd{flex:1 1 0;position:relative;min-height:0;display:flex;align-items:center;
    justify-content:center;overflow:hidden;background:#fafafa}
  .bd img{max-width:100%;max-height:100%;display:none}
  .wait{position:absolute;color:#999;font-size:12px}
  #map{position:absolute;inset:0;width:100%;height:100%;display:block;background:#fff;cursor:crosshair}
  .cbar{display:flex;align-items:center;gap:5px;color:#555}
  .cbar i{display:inline-block;width:84px;height:9px;border:1px solid #aaa;
    background:linear-gradient(90deg,#30123b,#3b9bf6,#1ae4b6,#b6f534,#fd9b2b,#d1342b)}
  #tel{position:absolute;top:8px;left:8px;background:#fff;border:1px solid var(--line);padding:5px 7px;
    font:11px/1.5 ui-monospace,Menlo,Consolas,monospace}
  #tel table{border-collapse:collapse}
  #tel td{padding:0 5px 0 0}
  #tel td.k{color:var(--mut)}
  #tel td.v{text-align:right;font-variant-numeric:tabular-nums}
  #help{position:absolute;left:8px;bottom:8px;right:8px;color:#555;font-size:11px;
    background:rgba(255,255,255,.9);border:1px solid var(--line);padding:3px 7px}
  #help kbd{font-family:ui-monospace,Menlo,monospace;background:#f0f0f0;border:1px solid #bbb;padding:0 4px}
</style>
</head>
<body>
<div id="bar">
  <span class="t">Ridgeback AutoNav simulator, onboard view and occupancy map</span>
  <span class="s" id="conn">connecting&hellip;</span>
</div>
<div id="body">
  <div class="col">
    <div class="panel" style="flex:1.3 1 0">
      <div class="hd"><span>Onboard camera (RGB)</span></div>
      <div class="bd"><span class="wait">waiting for camera&hellip;</span><img id="imgColor" alt=""></div>
    </div>
    <div class="panel" style="flex:1 1 0;border-bottom:none">
      <div class="hd"><span>Onboard depth</span>
        <span class="cbar">0<i></i>10&nbsp;m</span></div>
      <div class="bd"><span class="wait">waiting for depth&hellip;</span><img id="imgDepth" alt=""></div>
    </div>
  </div>
  <div class="col">
    <div class="panel" style="flex:1 1 0;border-right:none;border-bottom:none">
      <div class="hd"><span>Occupancy map (built online)</span>
        <span style="color:#888">click&nbsp;=&nbsp;navigation goal</span></div>
      <div class="bd" style="background:#fff">
        <canvas id="map"></canvas>
        <div id="tel"><table>
          <tr><td class="k">x</td><td class="v" id="px">&ndash;</td><td class="k">m</td></tr>
          <tr><td class="k">y</td><td class="v" id="py">&ndash;</td><td class="k">m</td></tr>
          <tr><td class="k">&theta;</td><td class="v" id="pt">&ndash;</td><td class="k">deg</td></tr>
          <tr><td class="k">v</td><td class="v" id="pv">0 0 0</td><td class="k"></td></tr>
          <tr><td class="k">spd</td><td class="v" id="ps">1.00</td><td class="k">&times;</td></tr>
          <tr id="navrow" style="display:none"><td class="k">nav</td>
            <td class="v" id="pn" colspan="2" style="text-align:left">&ndash;</td></tr>
        </table></div>
        <div id="help"><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> drive &middot;
          <kbd>Q</kbd><kbd>E</kbd> rotate &middot; <kbd>Space</kbd> stop &middot;
          <kbd>+</kbd>/<kbd>-</kbd> speed &middot; click map = goal &middot;
          <kbd>Esc</kbd> cancel &middot; wheel zoom &middot; drag pan</div>
      </div>
    </div>
  </div>
</div>
<script>
const BASE={x:0.4,y:0.3,w:0.6};
const cv=document.getElementById('map'), ctx=cv.getContext('2d');
let W=0,H=0;
function resize(){ const r=cv.getBoundingClientRect(); W=cv.width=Math.round(r.width); H=cv.height=Math.round(r.height); }
addEventListener('resize', resize);

// ---- view transform (world<->screen, y up) ----
let ppm=40, viewCx=0, viewCy=0, fitted=false;
function w2s(wx,wy){ return [W/2+(wx-viewCx)*ppm, H/2-(wy-viewCy)*ppm]; }
function s2w(sx,sy){ return [(sx-W/2)/ppm+viewCx, -(sy-H/2)/ppm+viewCy]; }
cv.addEventListener('wheel', e=>{ e.preventDefault();
  const r=cv.getBoundingClientRect(), ox=e.clientX-r.left, oy=e.clientY-r.top;
  const [bx,by]=s2w(ox,oy); ppm=Math.max(3,Math.min(400,ppm*Math.exp(-e.deltaY*0.0012)));
  const [ax,ay]=s2w(ox,oy); viewCx+=bx-ax; viewCy+=by-ay; }, {passive:false});
let drag=null, downPt=null, moved=false;
cv.addEventListener('mousedown', e=>{ if(e.button!==0) return;
  drag={cx:viewCx,cy:viewCy}; downPt={x:e.clientX,y:e.clientY}; moved=false; });
addEventListener('mousemove', e=>{ if(!drag) return;
  if(downPt && Math.hypot(e.clientX-downPt.x,e.clientY-downPt.y)>6) moved=true;
  viewCx=drag.cx-(e.clientX-downPt.x)/ppm; viewCy=drag.cy+(e.clientY-downPt.y)/ppm; });
addEventListener('mouseup', e=>{
  // A click that didn't pan = set a nav goal at that map point.
  if(drag && !moved && downPt){
    const r=cv.getBoundingClientRect();
    const [wx,wy]=s2w(downPt.x-r.left, downPt.y-r.top);
    sendGoal(wx,wy);
  }
  drag=null; downPt=null;
});
async function sendGoal(x,y){
  try{ await fetch('/api/goal',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({x,y})}); }catch(e){}
}
async function cancelNav(){
  try{ await fetch('/api/nav_cancel',{method:'POST'}); }catch(e){}
}

// ---- SLAM map (re-fetched on seq change; cached to offscreen) ----
let mapMeta=null, mapCanvas=null, mapSeq=-1, fetchingMap=false;
async function refreshMap(){
  if(fetchingMap) return; fetchingMap=true;
  try{
    const m=await (await fetch('/api/map')).json();
    if(!m){ fetchingMap=false; return; }
    const off=document.createElement('canvas'); off.width=m.w; off.height=m.h;
    const octx=off.getContext('2d'), img=octx.createImageData(m.w,m.h);
    const raw=atob(m.b64), n=raw.length;
    for(let i=0;i<n;i++){
      const v=raw.charCodeAt(i);                 // 0 free,100 occ,255(-1) unknown
      const col=i%m.w, row=(m.h-1)-((i/m.w)|0), p=(row*m.w+col)*4;
      let s;
      if(v===255){ img.data[p]=216;img.data[p+1]=216;img.data[p+2]=216; } // unknown: light grey
      else if(v>=65){ img.data[p]=34;img.data[p+1]=34;img.data[p+2]=34; }  // occupied: near-black
      else { s=255-Math.round(v*0.8); img.data[p]=s;img.data[p+1]=s;img.data[p+2]=s; } // free: white
      img.data[p+3]=255;
    }
    octx.putImageData(img,0,0);
    mapCanvas=off; mapMeta=m; mapSeq=m.seq;
    if(!fitted){ fitView(); fitted=true; }
  }catch(e){}
  fetchingMap=false;
}
function fitView(){ if(!mapMeta) return;
  const wm=mapMeta.w*mapMeta.res, hm=mapMeta.h*mapMeta.res;
  viewCx=mapMeta.ox+wm/2; viewCy=mapMeta.oy+hm/2;
  ppm=0.92*Math.min(W/wm,H/hm); }

// ---- live state ----
let state={}, connected=false, lastOk=0;
async function poll(){
  try{ const s=await (await fetch('/api/state')).json();
    state=s; connected=true; lastOk=performance.now();
    if(s.map_seq && s.map_seq!==mapSeq) refreshMap();
  }catch(e){ connected=false; }
}
setInterval(poll,100);

// ---- camera feeds (chained <img> reload; show on first load) ----
function feed(imgId, url, hzId){
  const img=document.getElementById(imgId);
  const wait=img.parentElement.querySelector('.wait');
  let t0=performance.now(), frames=0;
  const next=()=>{ img.src=url+'?t='+performance.now(); };
  img.onload=()=>{ img.style.display='block'; if(wait) wait.style.display='none';
    if(hzId){ frames++; const now=performance.now(); if(now-t0>1000){
      document.getElementById(hzId).textContent=Math.round(frames*1000/(now-t0))+' fps'; frames=0; t0=now; } }
    setTimeout(next,70); };
  img.onerror=()=>{ img.style.display='none'; if(wait) wait.style.display='block'; setTimeout(next,500); };
  next();
}
feed('imgColor','/api/camera/color',null);
feed('imgDepth','/api/camera/depth',null);

// ---- teleop (hold to move; release = stop = deadman) ----
const held=new Set(); let speed=1.0, lastSent='';
const KM={w:1,a:1,s:1,d:1,q:1,e:1};
addEventListener('keydown',e=>{ const k=e.key.toLowerCase();
  if(e.key==='Escape'){ cancelNav(); return; }
  if(k===' '){ held.clear(); e.preventDefault(); }
  else if(k==='+'||k==='='){ speed=Math.min(3,speed*1.25); }
  else if(k==='-'||k==='_'){ speed=Math.max(0.1,speed/1.25); }
  else if(KM[k]){ held.add(k); e.preventDefault(); } pushCmd(); });
addEventListener('keyup',e=>{ held.delete(e.key.toLowerCase()); pushCmd(); });
addEventListener('blur',()=>{ held.clear(); pushCmd(); });
function cmd(){ let vx=0,vy=0,wz=0;
  if(held.has('w'))vx+=BASE.x; if(held.has('s'))vx-=BASE.x;
  if(held.has('a'))vy+=BASE.y; if(held.has('d'))vy-=BASE.y;
  if(held.has('q'))wz+=BASE.w; if(held.has('e'))wz-=BASE.w;
  return {vx:vx*speed,vy:vy*speed,wz:wz*speed}; }
async function pushCmd(){ const c=cmd(), key=JSON.stringify(c);
  if(key===lastSent) return; lastSent=key;
  try{ await fetch('/api/teleop',{method:'POST',headers:{'Content-Type':'application/json'},body:key}); }catch(e){} }
setInterval(()=>{ if(held.size){ lastSent=''; pushCmd(); } },150);

// ---- render map ----
function draw(){
  if(!W||!H) resize();
  ctx.fillStyle='#ffffff'; ctx.fillRect(0,0,W,H);
  if(mapCanvas&&mapMeta){
    const wm=mapMeta.w*mapMeta.res, hm=mapMeta.h*mapMeta.res;
    const [sx,sy]=w2s(mapMeta.ox, mapMeta.oy+hm);
    ctx.imageSmoothingEnabled=false;
    ctx.drawImage(mapCanvas,sx,sy,wm*ppm,hm*ppm);
    ctx.strokeStyle='#bcbcbc'; ctx.lineWidth=1; ctx.strokeRect(sx,sy,wm*ppm,hm*ppm);
  } else {
    ctx.fillStyle='#999'; ctx.font='12px sans-serif'; ctx.textAlign='center';
    ctx.fillText('occupancy map — drive to build it', W/2, H/2); ctx.textAlign='left';
  }
  const p=state.pose, sc=state.scan;
  if(p&&sc&&sc.ranges){
    const lx=p.x+sc.off*Math.cos(p.theta), ly=p.y+sc.off*Math.sin(p.theta);
    const [lsx,lsy]=w2s(lx,ly);
    ctx.strokeStyle='rgba(40,40,40,0.05)'; ctx.lineWidth=1; ctx.beginPath(); ctx.fillStyle='#c0392b';
    for(let i=0;i<sc.ranges.length;i++){ const r=sc.ranges[i];
      if(!(r>0)||r>=sc.rmax-1e-3) continue;
      const a=p.theta+sc.amin+i*sc.ainc, ex=lx+r*Math.cos(a), ey=ly+r*Math.sin(a);
      const [esx,esy]=w2s(ex,ey); ctx.moveTo(lsx,lsy); ctx.lineTo(esx,esy); ctx.fillRect(esx-1,esy-1,2,2); }
    ctx.stroke();
  }
  // nav goal marker (target)
  const nav=state.nav;
  if(nav&&nav.goal){
    const [gx,gy]=w2s(nav.goal[0],nav.goal[1]);
    ctx.strokeStyle=nav.active?'#1565c0':'#999'; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.arc(gx,gy,8,0,Math.PI*2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(gx-12,gy); ctx.lineTo(gx+12,gy);
    ctx.moveTo(gx,gy-12); ctx.lineTo(gx,gy+12); ctx.stroke();
  }
  if(p){
    const [rx,ry]=w2s(p.x,p.y), rr=Math.max(4,(state.robot_radius||0.43)*ppm);
    ctx.beginPath(); ctx.arc(rx,ry,rr,0,Math.PI*2);
    ctx.fillStyle='rgba(21,101,192,0.14)'; ctx.fill();
    ctx.strokeStyle='#1565c0'; ctx.lineWidth=1.5; ctx.stroke();
    const hx=rx+Math.cos(p.theta)*rr*1.6, hy=ry-Math.sin(p.theta)*rr*1.6;
    ctx.beginPath(); ctx.moveTo(rx,ry); ctx.lineTo(hx,hy); ctx.strokeStyle='#0d3c78'; ctx.lineWidth=1.5; ctx.stroke();
    ctx.beginPath(); ctx.arc(rx,ry,2.5,0,Math.PI*2); ctx.fillStyle='#0d3c78'; ctx.fill();
  }
  const ok=connected&&(performance.now()-lastOk<800);
  document.getElementById('conn').textContent=ok?'connected':'offline';
  if(p){ document.getElementById('px').textContent=p.x.toFixed(2);
    document.getElementById('py').textContent=p.y.toFixed(2);
    document.getElementById('pt').textContent=(p.theta*180/Math.PI).toFixed(0); }
  const c=cmd();
  document.getElementById('pv').textContent=`${c.vx.toFixed(2)} ${c.vy.toFixed(2)} ${c.wz.toFixed(2)}`;
  document.getElementById('ps').textContent=speed.toFixed(2);
  const navrow=document.getElementById('navrow');
  if(nav&&(nav.active||nav.goal)){ navrow.style.display='table-row';
    let s=nav.state||''; if(nav.dist!=null) s+=' '+nav.dist.toFixed(1)+'m';
    document.getElementById('pn').textContent=s; }
  else navrow.style.display='none';
  requestAnimationFrame(draw);
}
resize(); refreshMap(); requestAnimationFrame(draw);
</script>
</body>
</html>
"""


def main():
    rclpy.init()
    node = VizServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._httpd.shutdown()
        except Exception:
            pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
