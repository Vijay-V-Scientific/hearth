#!/usr/bin/env python
"""Bring up the Isaac Lab G1 velocity-locomotion task (Objective 1).

Run with Isaac Lab's environment::

    cd ~/projects/isaac-sim/IsaacLab
    uv run isaaclab -p /home/vijay/projects/hearth/scripts/hearth_locomotion.py --livestream 0

Instantiates the task ``Isaac-Velocity-Flat-G1`` (the G1 velocity-tracking
locomotion environment shipped with Isaac Lab), resets it, and steps it under
physics with sampled actions. Objective 1 of the brief is the locomotion stack:
0.0-1.5 m/s forward, 150 N push recovery, 500 Hz low-level control. This script
proves the environment instantiates and steps under physics - the velocity
ranges it commands (lin_vel_x up to 1.0 m/s in the flat cfg) are the target the
policy will be trained to track, not a policy result yet.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Velocity-Flat-G1", help="Isaac Lab task id")
parser.add_argument("--envs", type=int, default=1, help="Number of parallel envs")
parser.add_argument("--steps", type=int, default=50, help="Physics steps to take after reset")

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

add_launcher_args(parser)
args_cli = parser.parse_args()


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.005)
    with launch_simulation(sim_cfg, args_cli):
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.envs)
        print(
            f"[INFO] task={args_cli.task} num_envs={env_cfg.scene.num_envs} "
            f"dt={env_cfg.sim.dt} decimation={env_cfg.decimation}",
            flush=True,
        )

        import gymnasium as gym

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None).unwrapped
        try:
            import torch

            obs, _ = env.reset()
            device = env.device
            print(
                f"[INFO] reset: obs shape {obs['policy'].shape if isinstance(obs, dict) else obs.shape}",
                flush=True,
            )
            robot = env.scene["robot"]
            action_dim = env.action_space.shape[-1]
            start_z = float(robot.data.root_pos_w.torch[0, 2])
            for _ in range(args_cli.steps):
                actions = 2 * torch.rand(env.action_space.shape, device=device) - 1
                obs, reward, terminated, truncated, _ = env.step(actions)
            end_z = float(robot.data.root_pos_w.torch[0, 2])
            reward_mean = float(reward.mean())
            finite = bool(torch.isfinite(reward).all().item())
            # Random ±1 joint targets for ~1 s must not explode the G1: base stays on the
            # ground plane and finite. This is a numerical-stability gate, not a balance test.
            stable = math.isfinite(end_z) and 0.1 < end_z < 1.6
            print(
                f"[INFO] stepped {args_cli.steps} env steps | reward={reward_mean:.4f} finite={finite} "
                f"terminated={int(terminated.sum())} base_z {start_z:.3f}->{end_z:.3f}",
                flush=True,
            )
            ok = finite and stable
            print(
                f"[RESULT] task={args_cli.task} envs={args_cli.envs} steps={args_cli.steps} "
                f"action_dim={action_dim} base_z_end={end_z:.3f} ok={'true' if ok else 'false'}",
                flush=True,
            )
            if not ok:
                raise SystemExit(2)
        finally:
            env.close()


if __name__ == "__main__":
    main()
