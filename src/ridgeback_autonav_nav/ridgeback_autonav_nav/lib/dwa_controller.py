"""Holonomic Dynamic Window Approach local controller for a mecanum base.

Everything is evaluated in the ROBOT body frame: trajectories start at the
origin and the carrot/goal/obstacles are passed already expressed relative to
the robot. Candidate body velocities (vx, vy, wz) — including lateral strafe
(vy) since the Ridgeback is omnidirectional — are rolled out, collision-checked
against the live scan, and scored on progress, heading, clearance and
smoothness. Returns the best (vx, vy, wz) or None if every trajectory collides.
"""
import math

import numpy as np


class DWAConfig:
    def __init__(self, **kw):
        self.max_vx = kw.get('max_vx', 0.5)
        self.max_vy = kw.get('max_vy', 0.30)
        self.max_wz = kw.get('max_wz', 0.6)
        self.acc_vx = kw.get('acc_vx', 0.6)
        self.acc_vy = kw.get('acc_vy', 0.5)
        self.acc_wz = kw.get('acc_wz', 1.2)
        self.vx_samples = kw.get('vx_samples', 7)
        self.vy_samples = kw.get('vy_samples', 5)
        self.wz_samples = kw.get('wz_samples', 11)
        self.sim_time = kw.get('sim_time', 1.5)
        self.sim_dt = kw.get('sim_dt', 0.1)
        self.w_path = kw.get('w_path', 2.0)
        self.w_goal = kw.get('w_goal', 1.5)
        self.w_clearance = kw.get('w_clearance', 0.8)
        self.w_smooth = kw.get('w_smooth', 0.2)
        self.w_heading = kw.get('w_heading', 0.6)   # prefer facing direction of travel
        self.w_strafe = kw.get('w_strafe', 0.8)     # discourage sideways crabbing
        self.robot_radius = kw.get('robot_radius', 0.43)
        # Extra standoff kept from obstacles on top of the footprint radius, so
        # the controller proactively avoids grazing walls.
        self.safety_margin = kw.get('safety_margin', 0.15)
        self.obstacle_check_range = kw.get('obstacle_check_range', 3.5)
        self.max_obstacles = kw.get('max_obstacles', 360)
        self.slow_radius = kw.get('slow_radius', 0.8)  # m; scale speed near goal


def _rollout(vx, vy, wz, steps, dt):
    """Roll out a constant body-velocity trajectory in the robot frame."""
    th = 0.0
    x = 0.0
    y = 0.0
    pts = np.empty((steps, 2), dtype=np.float64)
    for k in range(steps):
        th += wz * dt
        c, s = math.cos(th), math.sin(th)
        x += (vx * c - vy * s) * dt
        y += (vx * s + vy * c) * dt
        pts[k, 0] = x
        pts[k, 1] = y
    return pts, th


def compute_cmd(cfg, cur_v, carrot, goal, goal_yaw, obstacles, dist_to_goal,
                allow_reverse=True, allow_strafe=True):
    """Pick the best holonomic velocity command.

    Args:
        cfg: DWAConfig.
        cur_v: (vx, vy, wz) current body velocity.
        carrot: (x, y) lookahead point in the robot frame.
        goal: (x, y) final goal in the robot frame.
        goal_yaw: desired final heading relative to current heading (rad).
        obstacles: (N, 2) obstacle points in the robot frame (e.g. scan).
        dist_to_goal: scalar metres to the final goal (for speed scaling).
    Returns:
        ((vx, vy, wz), score) or (None, -inf) if every trajectory collides.
    """
    cvx, cvy, cwz = cur_v
    dt_w = cfg.sim_dt
    steps = max(1, int(cfg.sim_time / cfg.sim_dt))

    # Dynamic window: reachable velocities within one control period.
    win = cfg.sim_dt
    vx_lo = max(-cfg.max_vx, cvx - cfg.acc_vx * win)
    vx_hi = min(cfg.max_vx, cvx + cfg.acc_vx * win)
    if not allow_reverse:
        vx_lo = max(0.0, vx_lo)   # forward-only: turn-then-go like a driver
    vy_lo = max(-cfg.max_vy, cvy - cfg.acc_vy * win)
    vy_hi = min(cfg.max_vy, cvy + cfg.acc_vy * win)
    if not allow_strafe:
        vy_lo = vy_hi = 0.0   # no sideways crabbing: drive like a car (turn-then-go)
    wz_lo = max(-cfg.max_wz, cwz - cfg.acc_wz * win)
    wz_hi = min(cfg.max_wz, cwz + cfg.acc_wz * win)

    vxs = np.linspace(vx_lo, vx_hi, cfg.vx_samples)
    vys = np.linspace(vy_lo, vy_hi, cfg.vy_samples)
    wzs = np.linspace(wz_lo, wz_hi, cfg.wz_samples)

    # Speed scaling near the goal so the robot eases in.
    speed_scale = min(1.0, max(0.15, dist_to_goal / cfg.slow_radius))

    # Filter + cap obstacles for cheap clearance queries.
    obs = np.asarray(obstacles, dtype=np.float64).reshape(-1, 2)
    if obs.size:
        d = np.hypot(obs[:, 0], obs[:, 1])
        obs = obs[d <= cfg.obstacle_check_range]
        if obs.shape[0] > cfg.max_obstacles:
            sel = np.linspace(0, obs.shape[0] - 1, cfg.max_obstacles).astype(np.int64)
            obs = obs[sel]
    has_obs = obs.shape[0] > 0

    carrot = np.asarray(carrot, dtype=np.float64)
    best_cmd = None
    best_score = -math.inf

    for vx in vxs:
        for vy in vys:
            for wz in wzs:
                pts, th_final = _rollout(vx, vy, wz, steps, dt_w)
                # Collision + clearance.
                clearance = cfg.obstacle_check_range
                if has_obs:
                    # min distance from any trajectory point to any obstacle
                    dx = pts[:, 0][:, None] - obs[:, 0][None, :]
                    dy = pts[:, 1][:, None] - obs[:, 1][None, :]
                    dmin = math.sqrt(float(np.min(dx * dx + dy * dy)))
                    if dmin <= cfg.robot_radius + cfg.safety_margin:
                        continue  # too close (footprint + margin) -> discard
                    clearance = min(clearance, dmin)

                end = pts[-1]
                # Progress: closeness of trajectory end to the carrot.
                path_term = -float(np.hypot(end[0] - carrot[0], end[1] - carrot[1]))
                # Heading: align final body heading with desired terminal yaw,
                # weighted up only as we approach the goal.
                yaw_w = 1.0 if dist_to_goal < cfg.slow_radius else 0.2
                goal_term = -yaw_w * abs(_wrap(th_final - goal_yaw))
                clear_term = min(clearance, cfg.obstacle_check_range)
                smooth_term = -(abs(vx - cvx) + abs(vy - cvy) + abs(wz - cwz))

                # Drive-like motion: while travelling, prefer to FACE the carrot
                # (turn-then-go) and discourage sideways crabbing, so the robot
                # moves like a driver rather than strafing diagonally. Strafe is
                # still allowed (soft penalty) for tight maneuvers / obstacle dodging.
                if dist_to_goal > cfg.slow_radius:
                    bearing = math.atan2(carrot[1], carrot[0])
                    heading_term = -abs(_wrap(bearing - th_final))
                else:
                    heading_term = 0.0
                strafe_term = -abs(vy)

                score = (cfg.w_path * path_term +
                         cfg.w_goal * goal_term +
                         cfg.w_clearance * clear_term +
                         cfg.w_smooth * smooth_term +
                         cfg.w_heading * heading_term +
                         cfg.w_strafe * strafe_term)
                if score > best_score:
                    best_score = score
                    best_cmd = (vx * speed_scale, vy * speed_scale, wz)

    return best_cmd, best_score


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))
