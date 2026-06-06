"""Tests for the Gymnasium RL environment (RidgebackAutoNav).

Skipped where gymnasium is not installed (the lean sim image), so it never
breaks colcon test.
"""
import math

import numpy as np
import pytest

pytest.importorskip("gymnasium")
import gymnasium as gym  # noqa: E402
from gymnasium.utils.env_checker import check_env  # noqa: E402

import ridgeback_autonav_sim.rl_env as M  # noqa: E402
from ridgeback_autonav_sim.lib.world import (  # noqa: E402
    World, inflate_occ, bfs_reachable, nearest_free_cell)

YAML = "src/ridgeback_autonav_sim/config/sim_world.yaml"


def test_accel_limited_ramp_and_cap():
    w = World.from_spec(M.DEFAULT_WORLD)
    c = M.SimCore(w, accel_max=(1.0, 1.0, 2.0), rng=np.random.default_rng(0))
    c.reset_pose([4.0, 3.0, 0.0])
    c.step(0.5, 0.0, 0.0, 0.1)
    assert c.vel[0] <= 1.0 * 0.1 + 1e-9  # one step: dv <= accel*dt
    vs = [c.vel[0]]
    for _ in range(20):
        c.step(0.5, 0.0, 0.0, 0.1)
        vs.append(c.vel[0])
    assert all(
        b >= a - 1e-9 for a, b in zip(vs, vs[1:]))  # monotonic non-decreasing
    assert abs(vs[-1] - 0.5) < 1e-3  # reaches the commanded vmax


def test_instantaneous_when_no_accel():
    w = World.from_spec(M.DEFAULT_WORLD)
    c = M.SimCore(w, accel_max=None)
    c.reset_pose([4.0, 3.0, 0.0])
    c.step(0.5, 0.0, 0.0, 0.1)
    assert abs(c.vel[0] - 0.5) < 1e-9


def test_slide_vs_stop_at_wall():
    w = World.from_spec(M.DEFAULT_WORLD)
    c = M.SimCore(w, accel_max=None, slide=True)
    c.reset_pose([0.6, 3.0, 0.0])
    y0 = c.pose[1]
    for _ in range(20):
        c.step(-0.5, 0.5, 0.0, 0.1)  # diagonal into the left wall
    assert c.pose[1] - y0 > 0.3 and c.pose[0] < 0.6  # slid along y, pinned in x
    c2 = M.SimCore(w, accel_max=None, slide=False)
    c2.reset_pose([0.6, 3.0, 0.0])
    for _ in range(20):
        c2.step(-0.5, 0.0, 0.0, 0.1)
    assert abs(c2.pose[1] - 3.0) < 1e-6  # no slide -> held


def test_noise_is_seed_deterministic():
    w = World.from_spec(M.DEFAULT_WORLD)
    a = M.SimCore(w, range_noise=0.05, rng=np.random.default_rng(7))
    a.reset_pose([4, 3, 0])
    b = M.SimCore(w, range_noise=0.05, rng=np.random.default_rng(7))
    b.reset_pose([4, 3, 0])
    assert np.allclose(a.lidar(), b.lidar())
    assert not np.allclose(a.lidar(),
                           M.SimCore(w).lidar())  # noise actually changes it


def test_proprio_present_and_toggle():
    assert "proprio" in M.RidgebackAutoNavEnv(
        expose_velocity=True).observation_space.spaces
    assert "proprio" not in M.RidgebackAutoNavEnv(
        expose_velocity=False).observation_space.spaces


def test_goal_space_is_bounded():
    sp = M.RidgebackAutoNavEnv().observation_space.spaces["goal"]
    assert np.all(np.isfinite(sp.low)) and np.all(np.isfinite(sp.high))


def test_randomized_obstacles_reachable_and_deterministic():
    env = M.RidgebackAutoNavEnv(world_spec=YAML, randomize_obstacles=True)
    rad = int(math.ceil(env.core.robot_radius / env.world.res))
    for s in range(8):
        _, info = env.reset(seed=s)
        sx, sy, _ = info["pose"]
        assert not env.core.blocked(sx, sy)
        assert math.hypot(sx - env._target.x,
                          sy - env._target.y) >= env.min_start_goal_dist - 1e-6
        plan = inflate_occ(env.world.occ, rad)
        si = nearest_free_cell(plan, env.world.world_to_cell(sx, sy))
        gi = nearest_free_cell(plan, env._goal_cell())
        assert si and gi and bfs_reachable(plan, si, gi)
    env.reset(seed=3)
    a = env.world.occ.copy()
    env.reset(seed=3)
    b = env.world.occ.copy()
    assert np.array_equal(a, b)


def test_spawn_region_confines_start():
    env = M.RidgebackAutoNavEnv(world_spec=YAML,
                                spawn_region=(0.0, 3.0, 14.0, 5.0))
    for s in range(8):
        _, info = env.reset(seed=s)
        assert 3.0 <= info["pose"][1] <= 5.0


def test_spawns_inside_rooms_not_corridor():
    # the furnished world spawns the robot in a room (y<3 or y>5), never in the
    # corridor band (3<=y<=5), so the agent has to drive out to reach a sign.
    env = M.RidgebackAutoNavEnv(world_spec=YAML)
    for s in range(12):
        _, info = env.reset(seed=s)
        x, y, _ = info["pose"]
        assert y < 3.0 or y > 5.0, f"seed {s}: spawned in corridor at y={y:.2f}"
        assert not env.core.blocked(x, y)


def test_geodesic_greater_than_euclidean_around_wall():
    spec = {
        "resolution": 0.05,
        "bounds": [0, 0, 8, 6],
        "border": True,
        "walls": [[4.0, 0.0, 4.0, 4.0]],
        "signs": [{
            "text": "R",
            "x": 7.6,
            "y": 3.0,
            "z": 1.0,
            "yaw_deg": 180,
            "w": 0.5,
            "h": 0.3
        }],
        "start": {
            "x": 1,
            "y": 3,
            "theta": 0
        }
    }
    env = M.RidgebackAutoNavEnv(world_spec=spec,
                                geodesic_reward=True,
                                min_start_goal_dist=0.0)
    env.reset(options={"start": [1.0, 3.0, 0.0], "target_text": "R"})
    assert env._phi() > env._dist() + 0.8  # detour is longer than the line
    env.core.reset_pose([6.9, 3.0, 0.0])
    assert env._phi() < 0.5  # near the goal cell


def test_strict_success_requires_facing():
    env = M.RidgebackAutoNavEnv(require_facing_success=True)
    s = env.world.signs[0]
    nx, ny = math.cos(s.yaw), math.sin(s.yaw)
    env.reset(options={
        "target_text": s.text,
        "start": [s.x + nx * 0.5, s.y + ny * 0.5, s.yaw]
    })
    _, _, term, _, info = env.step([0, 0, 0])
    assert info["dist"] < env.radius and not info["is_success"] and not term
    for _ in range(60):
        _, _, term, _, info = env.step([0, 0, 0.6])
        if term:
            break
    assert info["is_success"]


def test_lenient_success_radius_only():
    env = M.RidgebackAutoNavEnv()  # default lenient
    s = env.world.signs[0]
    nx, ny = math.cos(s.yaw), math.sin(s.yaw)
    env.reset(options={
        "target_text": s.text,
        "start": [s.x + nx * 0.5, s.y + ny * 0.5, s.yaw]
    })
    _, _, term, _, info = env.step([0, 0, 0])
    assert info["is_success"] and term


def test_collision_continue_does_not_terminate():
    env = M.RidgebackAutoNavEnv(collision_terminate=False)
    s = env.world.signs[0]
    env.reset(options={"target_text": s.text, "start": [0.6, 4.0, math.pi]})
    hit = False
    for _ in range(40):
        _, _, term, _, info = env.step([0.5, 0, 0])
        hit |= info["collided"]
        assert not (info["collided"] and term)
    assert hit


def test_facing_gate_ramps_in_near_goal():
    # In an open room, stand in front of a sign and face it. The gated
    # facing bonus is larger nearer the goal, so the step reward is higher when
    # closer (both distances are outside the success radius).
    spec = {
        "resolution": 0.05,
        "bounds": [0, 0, 10, 10],
        "border": True,
        "signs": [{
            "text": "A",
            "x": 5.0,
            "y": 9.6,
            "z": 1.0,
            "yaw_deg": 270,
            "w": 0.6,
            "h": 0.3
        }],
        "start": {
            "x": 5,
            "y": 5,
            "theta": 0
        }
    }
    env = M.RidgebackAutoNavEnv(world_spec=spec)
    s = env.world.signs[0]
    nx, ny = math.cos(s.yaw), math.sin(s.yaw)
    face = s.yaw + math.pi  # heading toward the sign
    env.reset(options={
        "target_text": s.text,
        "start": [s.x + nx * 1.45, s.y + ny * 1.45, face]
    })
    _, r_far, _, _, info_far = env.step([0, 0, 0])
    env.reset(options={
        "target_text": s.text,
        "start": [s.x + nx * 1.0, s.y + ny * 1.0, face]
    })
    _, r_near, _, _, info_near = env.step([0, 0, 0])
    assert info_far["dist"] > env.radius and info_near["dist"] > env.radius
    assert r_near > r_far


def test_reward_key_guard():
    with pytest.raises(ValueError):
        M.RidgebackAutoNavEnv(reward={"k_bogus": 1.0})


def test_kinematic_mode_drops_inertia_and_proprio():
    # the simpler kinematic configuration: instantaneous velocity, no proprio.
    env = M.RidgebackAutoNavEnv(accel_max=None,
                                slide=False,
                                expose_velocity=False,
                                min_start_goal_dist=0.0,
                                require_reachable=False)
    assert set(env.observation_space.spaces) == {"scan", "goal", "image"}
    env.reset(seed=0)
    env.step([0.5, 0, 0])
    assert abs(env.core.vel[0] - 0.5) < 1e-9
    assert "proprio" in M.RidgebackAutoNavEnv().observation_space.spaces


def test_registration_and_check_env():
    M.register()
    assert "RidgebackAutoNav-v1" in gym.registry
    check_env(M.RidgebackAutoNavEnv(), skip_render_check=True)
    check_env(M.RidgebackAutoNavEnv(randomize_obstacles=True,
                                    geodesic_reward=True),
              skip_render_check=True)


def test_reset_seed_deterministic():
    env = M.RidgebackAutoNavEnv(world_spec=YAML)
    o1, i1 = env.reset(seed=11)
    o2, i2 = env.reset(seed=11)
    assert i1["target"] == i2["target"]
    assert np.allclose(i1["pose"], i2["pose"])
    assert np.allclose(o1["image"], o2["image"])


def test_vmax_configurable():
    env = M.RidgebackAutoNavEnv(world_spec=M.DEFAULT_WORLD,
                                vmax=(1.1, 0.5, 1.0),
                                accel_max=None)
    assert np.allclose(env.action_space.high, [1.1, 0.5, 1.0])
    env.reset(options={"target_text": "101", "start": [4.0, 3.0, 0.0]})
    env.step([5.0, 0.0, 0.0])  # over-limit command, clipped to vmax
    assert abs(env.core.vel[0] - 1.1) < 1e-6


def test_action_delay_lags_commands():
    # 2-step latency + instant velocity: the base ignores the command for two
    # steps (buffered zeros), then applies it.
    env = M.RidgebackAutoNavEnv(world_spec=M.DEFAULT_WORLD,
                                accel_max=None,
                                action_delay=2)
    env.reset(options={"target_text": "101", "start": [4.0, 3.0, 0.0]})
    for _ in range(2):
        env.step([0.5, 0.0, 0.0])
        assert abs(env.core.vel[0]) < 1e-9
    env.step([0.5, 0.0, 0.0])
    assert abs(env.core.vel[0] - 0.5) < 1e-6


def test_presets_construct_and_step():
    assert set(M.PRESETS) == {"kinematic", "realistic", "sim2real"}
    for cfg in M.PRESETS.values():
        env = M.RidgebackAutoNavEnv(world_spec=YAML, **cfg)
        env.reset(seed=0)
        env.step(env.action_space.sample())
