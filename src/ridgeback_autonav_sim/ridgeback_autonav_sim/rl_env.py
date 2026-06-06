#!/usr/bin/env python3
"""Gymnasium environment for "go read room N" on the ridgeback_autonav sim.

A fast, in-process RL / world-model environment (no ROS) that reuses the sim
core (occupancy world, LIDAR raycast, perspective camera render), so it steps
at roughly a thousand steps per second. Task: drive the holonomic Ridgeback to
a target room-number sign and face it, so the onboard camera reads it head-on.

Observation (Dict):
    scan    : (N,)      float32  LIDAR ranges, normalised to [0, 1]
    goal    : (3,)      float32  [distance, sin(bearing), cos(bearing)]
    image   : (84,84,4) uint8    onboard RGB-D, channels [R, G, B, depth],
                                 depth normalised over a fixed range
    proprio : (3,)      float32  body velocity [vx, vy, wz] / vmax (optional)

Action (Box, 3): holonomic body velocity command [vx, vy, wz], clipped to the
robot limits. The actual velocity slews toward the command at a per-axis
acceleration cap (first-order lag), so the robot has inertia.

Reward (potential-based progress plus a near-goal facing bonus):
    + k_prog * (phi_prev - phi)          progress on the potential phi
    + k_face * max(0, cos(bearing)) * g  facing bonus, ramped in near the goal
    - k_step                             per-step time cost
    - k_coll (terminates) or -k_coll_step (continues)  on collision
    + k_goal + k_goal_face * read_quality  on success
The potential phi is the straight-line distance, or the geodesic distance over
free space when geodesic_reward is set. Success is within the radius (lenient),
or within the radius and in front and facing above read_tau (strict). Episodes
end on success, collision (if collision_terminate), or max_steps (truncation).

Registered as RidgebackAutoNav-v1. The defaults give the realistic env (accel
limited motion, proprio obs, slide-along-wall collision, near-goal facing gate,
static obstacles). Richer behaviour (random obstacles, geodesic reward, strict
success, sensor noise) and a kinematic mode are opt-in via constructor kwargs.

Quick start (with `pip install gymnasium`):
    python3 -m ridgeback_autonav_sim.rl_env   # random + scripted smoke test
    RIDGEBACK_AUTONAV_WORLD=<world.yaml> python3 -m ridgeback_autonav_sim.rl_env
"""
import math
import os
from collections import deque

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # noqa: BLE001
    raise ImportError(
        "ridgeback_autonav_sim.rl_env needs gymnasium (pip install gymnasium)"
    ) from exc

from .lib import camera as cam
from .lib.raycast import raycast
from .lib.world import (World, rasterize_obstacle, inflate_occ, bfs_reachable,
                        bfs_distance_field, nearest_free_cell, OBSTACLE_PALETTE)


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


# Self-contained fallback world (small corridor + two signs) so the env runs
# without a spec. Pass your own via world_spec=<dict|path>, or set
# RIDGEBACK_AUTONAV_WORLD=<path>.
DEFAULT_WORLD = {
    "resolution": 0.05,
    "bounds": [0.0, 0.0, 8.0, 6.0],
    "border": True,
    "wall_thickness": 0.12,
    "walls": [[0, 4, 2.6, 4], [3.4, 4, 8, 4], [0, 2, 2.6, 2], [3.4, 2, 8, 2]],
    "signs": [
        {
            "text": "101",
            "x": 4.5,
            "y": 3.9,
            "z": 1.0,
            "yaw_deg": 270,
            "w": 0.6,
            "h": 0.3
        },
        {
            "text": "102",
            "x": 4.5,
            "y": 2.1,
            "z": 1.0,
            "yaw_deg": 90,
            "w": 0.6,
            "h": 0.3
        },
    ],
    "start": {
        "x": 0.8,
        "y": 3.0,
        "theta": 0.0
    },
}


class SimCore:
    """ROS-free physics and sensing over a World (footprint collision)."""

    def __init__(self,
                 world,
                 robot_radius=0.43,
                 laser_offset=0.42,
                 cam_offset=(0.45, 0.0, 0.6),
                 img_wh=(84, 84),
                 n_beams=32,
                 lidar_fov_deg=270.0,
                 max_range=10.0,
                 min_range=0.06,
                 depth_scale=10.0,
                 accel_max=None,
                 slide=False,
                 actuation_noise=0.0,
                 range_noise=0.0,
                 depth_noise=0.0,
                 pose_noise=0.0,
                 rng=None):
        self.world = world
        self.robot_radius = robot_radius
        self.laser_off = laser_offset
        self.cam_offset = cam_offset
        self.img_w, self.img_h = img_wh
        self.n_beams = n_beams
        self.max_range = max_range
        self.min_range = min_range
        self.depth_scale = depth_scale  # metres mapped to depth channel 0..255
        # Physics: accel_max=None applies the velocity instantly (kinematic). A
        # tuple (a_vx, a_vy, a_wz) caps how fast the actual body velocity slews
        # toward the command (a first-order velocity lag, i.e. inertia).
        self.accel_max = tuple(accel_max) if accel_max is not None else None
        self.slide = bool(slide)  # slide along walls vs full stop
        self.vel = [0.0, 0.0, 0.0]  # actual body velocity (vx, vy, wz)
        # Optional noise (all default 0 -> off, fully deterministic).
        self.actuation_noise = float(
            actuation_noise)  # std on commanded velocity
        self.range_noise = float(range_noise)  # std on LIDAR ranges (m)
        self.depth_noise = float(depth_noise)  # std on depth image (m)
        self.pose_noise = float(pose_noise)  # std on the observed pose (m)
        self._rng = rng if rng is not None else np.random.default_rng()
        half = math.radians(lidar_fov_deg) / 2.0
        self._angles = np.linspace(-half, half, n_beams)
        # Camera intrinsics preserving the 640-wide camera's horizontal FOV.
        f = 525.0 * (self.img_w / 640.0)
        self._K = (f, f, self.img_w / 2.0, self.img_h / 2.0)
        self.pose = [float(v) for v in world.start]

    def reset_pose(self, pose):
        self.pose = [float(pose[0]), float(pose[1]), float(pose[2])]
        self.vel = [0.0, 0.0, 0.0]  # a fresh episode starts at rest

    def observed_pose(self):
        """Pose as the agent sees it: the true pose plus optional Gaussian
        position noise (``pose_noise``). Used only to build the observation; the
        reward and collision always use the true ``self.pose``."""
        if self.pose_noise > 0.0:
            x, y, th = self.pose
            return [
                x + float(self._rng.normal(0.0, self.pose_noise)),
                y + float(self._rng.normal(0.0, self.pose_noise)), th
            ]
        return list(self.pose)

    def blocked(self, x, y):
        r = self.robot_radius
        for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            if self.world.is_occupied(x + r * math.cos(ang),
                                      y + r * math.sin(ang)):
                return True
        return self.world.is_occupied(x, y)

    def step(self, vx, vy, wz, dt, substeps=5):
        """Advances the base by one control step (``substeps`` Euler substeps).

        With ``accel_max`` the actual velocity slews toward the command at the
        per-axis acceleration cap (first-order lag), otherwise the command is
        applied instantly. Returns True if a footprint point hit an obstacle
        this step (the robot slides along it when ``slide`` is set).
        """
        sub = dt / substeps
        am = self.accel_max
        collided = False
        for _ in range(substeps):
            cvx, cvy, cwz = vx, vy, wz
            if self.actuation_noise > 0.0:
                n = self._rng.normal(0.0, self.actuation_noise, 3)
                cvx, cvy, cwz = cvx + n[0], cvy + n[1], cwz + n[2]
            if am is None:
                self.vel[0], self.vel[1], self.vel[
                    2] = cvx, cvy, cwz  # instantaneous
            else:
                for k, (cmd, amax) in enumerate(
                    ((cvx, am[0]), (cvy, am[1]), (cwz, am[2]))):
                    dv = cmd - self.vel[k]
                    lim = amax * sub
                    self.vel[k] += lim if dv > lim else (
                        -lim if dv < -lim else dv)
            x, y, th = self.pose
            avx, avy, awz = self.vel
            nth = _wrap(th + awz * sub)
            c, s = math.cos(th), math.sin(th)
            nx = x + (avx * c - avy * s) * sub
            ny = y + (avx * s + avy * c) * sub
            if not self.blocked(nx, ny):
                x, y = nx, ny
            else:
                collided = True
                if self.slide and not self.blocked(nx, y):
                    x = nx  # slide along x; kill the into-wall (y) component
                    self.vel[1] = 0.0
                elif self.slide and not self.blocked(x, ny):
                    y = ny  # slide along y; kill the into-wall (x) component
                    self.vel[0] = 0.0
                else:
                    self.vel[0] = self.vel[
                        1] = 0.0  # fully blocked / no-slide: stop
            self.pose = [x, y,
                         nth]  # rotation always applies (in-place spin is safe)
        return collided

    def lidar(self):
        x, y, th = self.pose
        lx = x + self.laser_off * math.cos(th)
        ly = y + self.laser_off * math.sin(th)
        r = raycast(self.world.occ, self.world.res, self.world.origin_x,
                    self.world.origin_y, lx, ly, th, self._angles,
                    self.max_range, self.min_range)
        r = np.asarray(r, dtype=np.float32)
        if self.range_noise > 0.0:
            r += self._rng.normal(0.0, self.range_noise,
                                  r.shape).astype(np.float32)
            np.clip(r, self.min_range, self.max_range, out=r)
        return r

    def image(self):
        """Onboard RGB-D as (H, W, 4) uint8, channels [R, G, B, depth]."""
        far = float(
            np.hypot(self.world.xmax - self.world.xmin,
                     self.world.ymax - self.world.ymin)) + 1.0
        color, depth = cam.render_scene(self.world,
                                        self.pose,
                                        self.cam_offset,
                                        self._K,
                                        self.img_w,
                                        self.img_h,
                                        max_depth=far)
        rgb = np.ascontiguousarray(color[:, :, ::-1])  # BGR -> RGB
        self._last_rgb = rgb
        if self.depth_noise > 0.0:
            depth = depth + self._rng.normal(0.0, self.depth_noise,
                                             depth.shape).astype(depth.dtype)
            np.clip(depth, 0.0, None, out=depth)
        du8 = (np.clip(depth / self.depth_scale, 0.0, 1.0) * 255.0).astype(
            np.uint8)
        return np.ascontiguousarray(np.dstack([rgb, du8]))  # (H, W, 4) RGB-D

    def rgb(self):
        """RGB part of the most recent frame, for gym render()."""
        return getattr(self, "_last_rgb", None)


class RidgebackAutoNavEnv(gym.Env):
    """Gymnasium env: navigate to and face a target room sign.

    Constructor knobs, grouped by concern (defaults give the realistic env; pass
    accel_max=None, slide=False, expose_velocity=False for a kinematic mode):
      core: world_spec, target_text, dt, max_steps, radius, reward,
            render_mode
      physics: accel_max=(ax, ay, aw) or None, slide, expose_velocity, vmax,
            action_delay (control latency in steps)
      noise: actuation_noise, range_noise, depth_noise, pose_noise (0=off)
      obstacles: randomize_obstacles, n_obstacles, obs_size, obstacle_types,
            obstacle_clearance, inflate_for_planning
      spawn: start_tries, min_start_goal_dist, require_reachable, spawn_region,
            start_dist_curriculum=(d0, d1, n_episodes)
      reward: geodesic_reward, require_facing_success, collision_terminate, and
            the weights in self.rw (k_prog, k_face, k_step, k_coll, k_goal,
            k_goal_face, face_gate_radius, k_coll_step, read_tau)
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(self,
                 world_spec=None,
                 target_text=None,
                 dt=0.1,
                 max_steps=400,
                 radius=0.8,
                 reward=None,
                 render_mode=None,
                 accel_max=(1.0, 1.0, 2.0),
                 slide=True,
                 expose_velocity=True,
                 vmax=(0.5, 0.3, 0.6),
                 action_delay=0,
                 actuation_noise=0.0,
                 range_noise=0.0,
                 depth_noise=0.0,
                 pose_noise=0.0,
                 randomize_obstacles=False,
                 n_obstacles=(2, 5),
                 obs_size=(0.25, 0.5),
                 obstacle_types=("box", "circle"),
                 obstacle_clearance=0.6,
                 inflate_for_planning=True,
                 start_tries=200,
                 min_start_goal_dist=2.0,
                 require_reachable=True,
                 spawn_region=None,
                 start_dist_curriculum=None,
                 geodesic_reward=False,
                 require_facing_success=False,
                 collision_terminate=True):
        super().__init__()
        if world_spec is None:
            env_path = os.environ.get("RIDGEBACK_AUTONAV_WORLD", "")
            world_spec = env_path if env_path and os.path.exists(
                env_path) else DEFAULT_WORLD
        if isinstance(world_spec, str):
            import yaml
            with open(world_spec) as fh:
                world_spec = yaml.safe_load(fh)
        self.world = World.from_spec(world_spec)
        if not self.world.signs:
            raise ValueError("world has no signs to navigate to")
        self.core = SimCore(self.world,
                            accel_max=accel_max,
                            slide=slide,
                            actuation_noise=actuation_noise,
                            range_noise=range_noise,
                            depth_noise=depth_noise,
                            pose_noise=pose_noise)
        self.dt = dt
        self.max_steps = max_steps
        self.radius = radius
        self.target_text = target_text
        self.render_mode = render_mode
        self.expose_velocity = bool(expose_velocity)
        self.randomize_obstacles = bool(randomize_obstacles)
        self.n_obstacles = n_obstacles
        self.obs_size = obs_size
        self.obstacle_types = tuple(obstacle_types)
        self.obstacle_clearance = float(obstacle_clearance)
        self.inflate_for_planning = bool(inflate_for_planning)
        self.start_tries = int(start_tries)
        self.min_start_goal_dist = float(min_start_goal_dist)
        self.require_reachable = bool(require_reachable)
        self.spawn_region = spawn_region  # None or (xlo, ylo, xhi, yhi) rect
        self.start_dist_curriculum = start_dist_curriculum  # (d0,d1,n) tuple
        self.geodesic_reward = bool(geodesic_reward)
        self.require_facing_success = bool(require_facing_success)
        self.collision_terminate = bool(collision_terminate)
        # control latency: the base reacts to a command this many steps late
        # (a deque of pending actions, filled on reset).
        self.action_delay = int(action_delay)
        self._act_queue = deque()
        self._n_obstacles = len(
            self.world.obstacles)  # current-episode obstacle count
        self._episode = 0
        self._dist_field = None  # geodesic field to the goal (when on)

        self.vmax = np.array(vmax, dtype=np.float32)
        self.action_space = spaces.Box(-self.vmax, self.vmax, dtype=np.float32)
        # goal = [distance, sin(bearing), cos(bearing)], bounded so check_env
        # is clean (distance <= world diagonal, trig in [-1, 1]).
        diag = float(
            np.hypot(self.world.xmax - self.world.xmin,
                     self.world.ymax - self.world.ymin))
        self._goal_hi = np.array([diag, 1.0, 1.0], dtype=np.float32)
        obs = {
            "scan":
                spaces.Box(0.0, 1.0, (self.core.n_beams,), np.float32),
            "goal":
                spaces.Box(np.array([0.0, -1.0, -1.0], np.float32),
                           self._goal_hi),
            "image":
                spaces.Box(0, 255, (self.core.img_h, self.core.img_w, 4),
                           np.uint8),
        }
        if self.expose_velocity:
            obs["proprio"] = spaces.Box(-1.0, 1.0, (3,),
                                        np.float32)  # vx, vy, wz / vmax
        self.observation_space = spaces.Dict(obs)
        self.rw = dict(k_prog=1.0,
                       k_face=0.05,
                       k_step=0.01,
                       k_coll=10.0,
                       k_goal=5.0,
                       k_goal_face=10.0,
                       face_gate_radius=1.5,
                       k_coll_step=0.5,
                       read_tau=0.6)
        if reward:
            unknown = set(reward) - set(self.rw)
            if unknown:
                raise ValueError(f"unknown reward keys {sorted(unknown)}; "
                                 f"known: {sorted(self.rw)}")
            self.rw.update(reward)
        self._steps = 0
        self._phi_prev = 0.0  # previous potential (geodesic or euclidean)
        self._target = None

    def _pick_target(self):
        signs = self.world.signs
        if self.target_text is not None:
            for s in signs:
                if s.text == str(self.target_text):
                    return s
        return signs[int(self.np_random.integers(len(signs)))]

    def _planning_grid(self):
        """Occupancy inflated by the footprint radius (config-space), so a point
        robot on this grid stands for the real robot. Identity grid if inflation
        is off."""
        rad = int(math.ceil(self.core.robot_radius /
                            self.world.res)) if self.inflate_for_planning else 0
        return inflate_occ(self.world.occ, rad) if rad else self.world.occ

    def _curriculum_max_dist(self):
        if not self.start_dist_curriculum:
            return None
        d0, d1, n = self.start_dist_curriculum
        return d0 + (d1 - d0) * min(1.0, self._episode / max(1, n))

    def _spawn_rects(self):
        """[xlo, ylo, xhi, yhi] rects to sample the start from: the ctor
        spawn_region if set, else the world's spawn_regions (the room interiors,
        so the agent has to drive out), else the world inset from walls."""
        if self.spawn_region is not None:
            return [self.spawn_region]
        if self.world.spawn_regions:
            return self.world.spawn_regions
        return [(self.world.xmin + 0.5, self.world.ymin + 0.5,
                 self.world.xmax - 0.5, self.world.ymax - 0.5)]

    def _sample_start(self):
        """Rejection-samples a free start pose.

        The pose lands in a spawn rect (the room interiors by default on the
        furnished world, so the agent must exit to reach a sign). It is not in
        collision, sits at least ``min_start_goal_dist`` from the goal (within a
        curriculum cap if set), and is reachable to the goal on the planning
        grid. One inflate plus one BFS per call, then O(1) per candidate. Falls
        back to the world start.
        """
        gx, gy = self._target.x, self._target.y
        plan = self._planning_grid()
        gi = nearest_free_cell(
            plan, self._goal_cell()) if self.require_reachable else None
        field = bfs_distance_field(plan, gi) if gi else None
        lo, hi = self.min_start_goal_dist, self._curriculum_max_dist()
        rects = self._spawn_rects()
        for _ in range(self.start_tries):
            xlo, ylo, xhi, yhi = rects[int(self.np_random.integers(len(rects)))]
            x = float(self.np_random.uniform(xlo, xhi))
            y = float(self.np_random.uniform(ylo, yhi))
            if self.core.blocked(x, y):
                continue
            dg = math.hypot(x - gx, y - gy)
            if dg < lo or (hi is not None and dg > hi):
                continue
            if field is not None:
                si = nearest_free_cell(plan, self.world.world_to_cell(x, y))
                if si is None or field[si[1], si[0]] < 0:
                    continue
            return [x, y, float(self.np_random.uniform(-math.pi, math.pi))]
        return [float(v) for v in self.world.start]

    def _sample_obstacle_candidates(self, start):
        """Samples obstacle dicts in open space, clear of the start and signs.

        Geometry-only (no grid ops); reachability is checked later in
        ``_scatter_obstacles``.
        """
        sx, sy = start[0], start[1]
        gx, gy = self._target.x, self._target.y
        k = int(
            self.np_random.integers(self.n_obstacles[0],
                                    self.n_obstacles[1] + 1))
        out = []
        tries = 0
        while len(out) < k and tries < k * 12:
            tries += 1
            cx = float(
                self.np_random.uniform(self.world.xmin + 0.6,
                                       self.world.xmax - 0.6))
            cy = float(
                self.np_random.uniform(self.world.ymin + 0.6,
                                       self.world.ymax - 0.6))
            sz = float(self.np_random.uniform(*self.obs_size))
            clr = self.obstacle_clearance + 0.5 * sz
            if math.hypot(cx - sx, cy - sy) < clr or math.hypot(
                    cx - gx, cy - gy) < clr:
                continue
            if any(
                    math.hypot(cx - s.x, cy - s.y) < clr
                    for s in self.world.signs):
                continue
            if self.world.is_occupied(cx, cy):  # not on a wall
                continue
            color = OBSTACLE_PALETTE[len(out) % len(OBSTACLE_PALETTE)]
            ht = float(self.np_random.uniform(0.5, 1.4))  # furniture-ish height
            typ = self.obstacle_types[int(
                self.np_random.integers(len(self.obstacle_types)))]
            if typ == "circle":
                out.append({
                    "type": "circle",
                    "x": cx,
                    "y": cy,
                    "r": 0.5 * sz,
                    "height": ht,
                    "color_bgr": color
                })
            else:
                out.append({
                    "type": "box",
                    "x": cx,
                    "y": cy,
                    "w": sz,
                    "h": float(self.np_random.uniform(*self.obs_size)),
                    "yaw_deg": float(self.np_random.uniform(0.0, 180.0)),
                    "height": ht,
                    "color_bgr": color
                })
        return out

    def _scatter_obstacles(self, start):
        """Rasterizes sampled obstacles into the live grid.

        Keeps the largest prefix that leaves the goal reachable from the start
        (a drop-last reachability guard).
        """
        res, ox, oy = self.world.res, self.world.origin_x, self.world.origin_y
        si = self.world.world_to_cell(start[0], start[1])
        gi = self._goal_cell()  # free cell near the sign
        rad = int(math.ceil(self.core.robot_radius /
                            res)) if self.inflate_for_planning else 0
        cands = self._sample_obstacle_candidates(start)
        while cands:
            trial = self.world.occ.copy()
            for o in cands:
                rasterize_obstacle(trial, None, None, res, ox, oy, o)
            plan = inflate_occ(trial, rad) if rad else trial
            sif = nearest_free_cell(plan,
                                    si) or si  # snap onto the inflated grid
            gif = nearest_free_cell(plan, gi) or gi
            if bfs_reachable(plan, sif, gif):
                break
            cands.pop()  # last one broke reachability
        for o in cands:
            rasterize_obstacle(self.world.occ, self.world.color_grid,
                               self.world.height_grid, res, ox, oy, o)
        self._n_obstacles = len(self.world.obstacles) + len(cands)

    def _dist(self):
        x, y, _ = self.core.pose
        return math.hypot(self._target.x - x, self._target.y - y)

    def _bearing_err(self):
        x, y, th = self.core.pose
        return _wrap(math.atan2(self._target.y - y, self._target.x - x) - th)

    def _in_front(self):
        x, y, _ = self.core.pose
        nx, ny = math.cos(self._target.yaw), math.sin(self._target.yaw)
        return (nx * (x - self._target.x) + ny * (y - self._target.y)) > 0.0

    def _goal_cell(self):
        """Returns a free cell near the sign that represents 'arrived'.

        The sign sits on a wall, so its own cell is occupied. We use a point
        just out from it along its outward normal (open space, within radius).
        """
        s = self._target
        nx, ny = math.cos(s.yaw), math.sin(s.yaw)
        for off in (0.7, 0.6, 0.8, 0.5, 0.9, 0.4):
            gx, gy = s.x + nx * off, s.y + ny * off
            if not self.world.is_occupied(gx, gy):
                return self.world.world_to_cell(gx, gy)
        return self.world.world_to_cell(s.x, s.y)

    def _phi(self):
        """Returns the shaping potential in metres.

        Geodesic distance over free space when ``geodesic_reward`` is on (so
        progress respects walls and obstacles), else the straight-line distance
        to the sign. Potential-based shaping is policy-invariant for either
        choice (Ng et al. 1999).
        """
        if self.geodesic_reward and self._dist_field is not None:
            ci, cj = self.world.world_to_cell(self.core.pose[0],
                                              self.core.pose[1])
            if 0 <= ci < self.world.width and 0 <= cj < self.world.height:
                d = self._dist_field[cj, ci]
                if d >= 0:
                    return float(d) * self.world.res
        return self._dist()  # euclidean fallback

    def _obs(self):
        scan = np.clip(self.core.lidar() / self.core.max_range, 0.0,
                       1.0).astype(np.float32)
        ox, oy, oth = self.core.observed_pose(
        )  # noisy pose for the obs (true pose drives reward)
        be = _wrap(math.atan2(self._target.y - oy, self._target.x - ox) - oth)
        d = min(math.hypot(self._target.x - ox, self._target.y - oy),
                float(self._goal_hi[0]))
        goal = np.array([d, math.sin(be), math.cos(be)], dtype=np.float32)
        obs = {"scan": scan, "goal": goal, "image": self.core.image()}
        if self.expose_velocity:
            obs["proprio"] = np.clip(
                np.asarray(self.core.vel, np.float32) / self.vmax, -1.0,
                1.0).astype(np.float32)
        return obs

    def _info(self, **extra):
        info = {
            "target": self._target.text,
            "dist": self._dist(),
            "bearing_err": self._bearing_err(),
            "pose": tuple(self.core.pose),
            "in_front": self._in_front(),
            "velocity": tuple(self.core.vel),
            "n_obstacles": self._n_obstacles,
            "start_goal_dist": getattr(self, "_start_goal_dist", self._dist())
        }
        if self.geodesic_reward:
            info["geodesic_dist"] = self._phi()
        info.update(extra)
        if "success" in info:
            info["is_success"] = bool(info["success"])  # SB3 Monitor convention
        return info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.core._rng = self.np_random  # seeded RNG -> determinism
        self._episode += 1
        if options and options.get("target_text") is not None:
            self.target_text = options["target_text"]
        self._target = self._pick_target()
        self._n_obstacles = len(self.world.obstacles)
        if self.randomize_obstacles:  # restore the static scene first
            self.world.occ = self.world.occ_base.copy()
            self.world.color_grid = self.world.color_base.copy()
            self.world.height_grid = self.world.height_base.copy()
        start = options.get("start") if options else None
        start = list(start) if start else self._sample_start()
        if self.randomize_obstacles:  # then scatter random obstacles
            self._scatter_obstacles(start)
        self.core.reset_pose(start)
        self._steps = 0
        self._act_queue = deque(
            [np.zeros(3, np.float32) for _ in range(self.action_delay)])
        self._start_goal_dist = self._dist()
        # geodesic field over free space to the goal cell (raw occ, point
        # robot), defined wherever the robot can stand; rebuilt per episode.
        self._dist_field = (bfs_distance_field(self.world.occ,
                                               self._goal_cell())
                            if self.geodesic_reward else None)
        self._phi_prev = self._phi()
        return self._obs(), self._info()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -self.vmax, self.vmax)
        if self.action_delay > 0:  # base reacts to commands a few steps late
            self._act_queue.append(a)
            a = self._act_queue.popleft()
        collided = self.core.step(float(a[0]), float(a[1]), float(a[2]),
                                  self.dt)
        self._steps += 1

        d = self._dist()  # euclidean to sign (success + info)
        facing = math.cos(self._bearing_err())
        rw = self.rw
        # progress on the (geodesic or euclidean) potential, policy-invariant
        phi = self._phi()
        # facing reward ramps in near the goal so it approaches, not stares
        gate = 1.0
        if rw["face_gate_radius"] > 0:
            gate = max(
                0.0,
                min(1.0, (rw["face_gate_radius"] - d) / rw["face_gate_radius"]))
        r = (rw["k_prog"] * (self._phi_prev - phi) +
             rw["k_face"] * max(0.0, facing) * gate - rw["k_step"])
        self._phi_prev = phi

        read_quality = max(0.0, facing) * (1.0 if self._in_front() else 0.3)
        terminated, success = False, False
        if collided:
            if self.collision_terminate:
                r -= rw["k_coll"]
                terminated = True
            else:
                r -= rw["k_coll_step"]  # penalise, keep going (slides via P1)
        if not terminated and d < self.radius:
            # lenient: inside radius. strict: in front and facing head-on.
            success = (not self.require_facing_success or
                       (self._in_front() and read_quality >= rw["read_tau"]))
            if success:
                terminated = True
                r += rw["k_goal"] + rw["k_goal_face"] * read_quality

        truncated = self._steps >= self.max_steps
        return (self._obs(), float(r), terminated, truncated,
                self._info(success=success,
                           collided=collided,
                           read_quality=read_quality))

    def render(self):
        if self.render_mode == "rgb_array":
            self.core.image()  # refresh the cached frame
            return self.core.rgb()  # RGB only (depth lives in the obs)
        if self.render_mode == "human":
            import cv2  # convenience window; never used in the hot path
            self.core.image()
            cv2.imshow("RidgebackAutoNav",
                       self.core.rgb()[:, :, ::-1])  # RGB -> BGR
            cv2.waitKey(1)

    def close(self):
        if self.render_mode == "human":
            try:
                import cv2
                cv2.destroyWindow("RidgebackAutoNav")
            except Exception:  # noqa: BLE001 (headless / never opened)
                pass


# Realism presets to pass to the env, e.g.
#   env = gym.make("RidgebackAutoNav-v1", **PRESETS["sim2real"])
# kinematic = easiest (instant velocity, no noise/obstacles); realistic = the v1
# defaults (inertia, slide, proprio, static furniture); sim2real = domain
# randomized for transfer (sensor/actuation noise, control latency, random
# obstacles). For sim2real, set vmax and accel_max to YOUR robot's configured
# limits and tune the noise and latency to the real platform.
PRESETS = {
    "kinematic":
        dict(accel_max=None,
             slide=False,
             expose_velocity=False,
             randomize_obstacles=False,
             geodesic_reward=False,
             require_facing_success=False,
             collision_terminate=True,
             actuation_noise=0.0,
             range_noise=0.0,
             depth_noise=0.0,
             pose_noise=0.0,
             action_delay=0,
             vmax=(0.5, 0.3, 0.6)),
    "realistic":
        dict(accel_max=(1.0, 1.0, 2.0),
             slide=True,
             expose_velocity=True,
             randomize_obstacles=False,
             geodesic_reward=False,
             require_facing_success=False,
             collision_terminate=True,
             actuation_noise=0.0,
             range_noise=0.0,
             depth_noise=0.0,
             pose_noise=0.0,
             action_delay=0,
             vmax=(0.5, 0.3, 0.6)),
    "sim2real":
        dict(accel_max=(1.0, 1.0, 2.0),
             slide=True,
             expose_velocity=True,
             randomize_obstacles=True,
             geodesic_reward=True,
             require_facing_success=False,
             collision_terminate=False,
             actuation_noise=0.05,
             range_noise=0.03,
             depth_noise=0.05,
             pose_noise=0.02,
             action_delay=1,
             vmax=(0.5, 0.3, 0.6)),
}


def register():
    """Registers the environment so ``gymnasium.make`` can build it.

    Registers the id ``RidgebackAutoNav-v1`` (idempotent). The environment's
    behaviour is controlled entirely through constructor keyword arguments, so a
    single version covers the kinematic-to-realistic range via configuration.
    """
    from gymnasium.envs.registration import register as _r, registry
    if "RidgebackAutoNav-v1" in registry:
        return
    _r(id="RidgebackAutoNav-v1",
       entry_point="ridgeback_autonav_sim.rl_env:RidgebackAutoNavEnv",
       max_episode_steps=400)


def _smoke():
    spec = os.environ.get("RIDGEBACK_AUTONAV_WORLD", "")
    env = RidgebackAutoNavEnv(world_spec=(spec or None))
    obs, info = env.reset(seed=0)
    print("obs:", {k: (v.shape, str(v.dtype)) for k, v in obs.items()})
    print("action_space:", env.action_space)
    print(f"target={info['target']}  start_dist={info['dist']:.2f}")

    # 1) random agent
    ret = 0.0
    for t in range(env.max_steps):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        ret += r
        if term or trunc:
            print(f"[random]   t={t} ok={info['success']} "
                  f"hit={info['collided']} d={info['dist']:.2f} R={ret:.2f}")
            break

    # 2) scripted "face + approach the sign" controller (naive: may hit a wall)
    obs, info = env.reset(seed=2)
    ret = 0.0
    for t in range(env.max_steps):
        be = info["bearing_err"]
        a = np.array([0.45, 0.0, float(np.clip(2.0 * be, -0.6, 0.6))],
                     dtype=np.float32)
        obs, r, term, trunc, info = env.step(a)
        ret += r
        if term or trunc:
            print(f"[scripted] t={t} ok={info['success']} "
                  f"front={info['in_front']} d={info['dist']:.2f} R={ret:.2f}")
            break

    # 3) deterministic SUCCESS demo: spawn 1 m in front of a sign, facing it,
    #    then drive straight in (proves the radius success + facing bonus).
    sign = env.world.signs[0]
    nx, ny = math.cos(sign.yaw), math.sin(sign.yaw)  # outward wall normal
    start = [sign.x + nx * 1.0, sign.y + ny * 1.0, _wrap(sign.yaw + math.pi)]
    obs, info = env.reset(options={"target_text": sign.text, "start": start})
    ret = 0.0
    for t in range(env.max_steps):
        obs, r, term, trunc, info = env.step(
            np.array([0.45, 0.0, 0.0], dtype=np.float32))
        ret += r
        if term or trunc:
            print(f"[success ] tgt={info['target']} t={t} ok={info['success']} "
                  f"front={info['in_front']} d={info['dist']:.2f} R={ret:.2f}")
            break

    # 4) v1 extras: per-episode random obstacles (reachability-guarded), the
    #    proprio (velocity) observation, and the geodesic potential.
    renv = RidgebackAutoNavEnv(world_spec=(spec or None),
                               randomize_obstacles=True,
                               geodesic_reward=True)
    obs, info = renv.reset(seed=1)
    counts = []
    for k in range(3):
        renv.reset(seed=k)
        counts.append(renv._n_obstacles)
        for _ in range(20):
            renv.step(renv.action_space.sample())
    print(f"[v1 extra] proprio_in_obs={'proprio' in obs} obstacles/ep={counts} "
          f"geodesic_dist={info.get('geodesic_dist', float('nan')):.2f}")
    print("smoke OK")


if __name__ == "__main__":
    _smoke()
