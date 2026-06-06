#!/usr/bin/env python3
"""Minimal PPO training on RidgebackAutoNav-v1, the ROS-free RL env.

The env (ridgeback_autonav_sim/rl_env.py) gives a Dict observation {scan, goal,
image, proprio}, where image is an 84x84 RGB-D frame (4 channels, [R, G, B,
depth]) and proprio is the body velocity. We use SB3's MultiInputPolicy (a small
CNN over the RGB-D image plus an MLP over scan/goal/proprio).

This needs stable-baselines3 + torch, which are NOT in the image by default
(they are large). Install them inside the `rl` container first:

    pip3 install "stable-baselines3[extra]"

Then train (the `rl` service sets RIDGEBACK_AUTONAV_WORLD for you):

    docker compose run --rm rl python3 /opt/rl_train.py --steps 200000
    docker compose run --rm rl python3 /opt/rl_train.py --target 206

For world models instead of model-free RL, use the same env but record rollouts
of obs['image'] (84x84 RGB-D) + actions, and train your dynamics model offline.
"""
import argparse
import sys

import ridgeback_autonav_sim.rl_env as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--target",
                    default=None,
                    help="fix the target room, e.g. 206")
    ap.add_argument("--envs", type=int, default=4, help="parallel envs")
    ap.add_argument("--env-id",
                    default="RidgebackAutoNav-v1",
                    help="Gymnasium env id")
    ap.add_argument("--out", default="/tmp/ridgeback_autonav_ppo")
    args = ap.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except Exception:
        sys.exit("stable-baselines3 not installed. Run: "
                 "pip3 install 'stable-baselines3[extra]'")

    M.register()
    import gymnasium as gym

    def mk():
        return gym.make(args.env_id, target_text=args.target)

    env = make_vec_env(mk, n_envs=args.envs)
    model = PPO("MultiInputPolicy",
                env,
                verbose=1,
                n_steps=512,
                batch_size=256,
                gamma=0.99,
                gae_lambda=0.95)
    model.learn(total_timesteps=args.steps)
    model.save(args.out)
    print("saved policy ->", args.out)


if __name__ == "__main__":
    main()
