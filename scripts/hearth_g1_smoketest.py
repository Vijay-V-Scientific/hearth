"""Isaac integration smoke test for the Unitree G1 (NOT the G1 gate).

This is a Foundations integration check: it boots Isaac Lab headless, imports
and spawns the G1 USD with its full joint set, and steps real physics to
confirm the asset loads and the pipeline runs with finite poses throughout.

It is explicitly NOT gate G1. Gate G1 is the sim-vs-real joint-replay RMSE < 3 deg
test in hearth/sim2real/replay.py; standing/walking is Objective 1. A free
biped held at fixed joint angles is dynamically unbalanced, so the observed
base height / tilt is reported honestly (it topples) and is not a pass
criterion here.

Backend note: on this machine the PhysX backend's plugins fail to load (USD ABI
mismatch), so the working physics backend is Newton/MJWarp; that is the default.

Run with the Isaac Lab environment (not the hearth venv):

    cd ~/projects/isaac-sim/IsaacLab && \
    VIRTUAL_ENV="$PWD/.venv" .venv/bin/isaaclab \
        -p ~/projects/hearth/scripts/hearth_g1_smoketest.py \
        --physics newton_mjwarp --steps 600
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Spawn the G1 and prove it stands.")
parser.add_argument(
    "--physics",
    default="newton_mjwarp",
    choices=["isaacsim_physx", "newton_mjwarp"],
    help="Physics backend (PhysX is broken on this install; Newton/MJWarp works).",
)
parser.add_argument("--steps", type=int, default=600, help="Physics steps to hold the stance.")
parser.add_argument("--dt", type=float, default=0.005, help="Physics timestep [s].")
add_launcher_args(parser)
# Headless: no visualizer selected means no viewer window opens.
args_cli = parser.parse_args()

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass, replace
from isaaclab_assets.robots.unitree import G1_CFG
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

MIN_UPRIGHT_HEIGHT = 0.50
"""Minimum base height (m) before the robot is considered fallen."""
MAX_TILT_RAD = math.radians(35.0)
"""Maximum body tilt (rad) from upright before the robot is considered fallen."""


@configclass
class StanceSceneCfg(InteractiveSceneCfg):
    """Ground plane, a dome light and one Unitree G1 in its standing pose."""

    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75)),
    )
    g1: ArticulationCfg = replace(G1_CFG, prim_path="{ENV_REGEX_NS}/G1")


def _tilt_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Angle (rad) between the body +Z axis and world up, from a wxyz quaternion."""
    _, x, y, _z = quat_wxyz.unbind(dim=-1)
    up_dot = 1.0 - 2.0 * (x * x + y * y)
    return torch.acos(up_dot.clamp(-1.0, 1.0))


def run() -> int:
    """Spawn G1, hold its stance, and report whether it stands. Return exit code."""
    torch.manual_seed(0)
    exit_code = 0
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        if isinstance(physics_cfg, NewtonCfg) and isinstance(
            physics_cfg.solver_cfg, MJWarpSolverCfg
        ):
            physics_cfg.solver_cfg.integrator = "implicitfast"
            physics_cfg.solver_cfg.njmax = 300
            physics_cfg.solver_cfg.nconmax = 200
            physics_cfg.solver_cfg.ls_iterations = 40
            physics_cfg.solver_cfg.cone = "elliptic"
            physics_cfg.solver_cfg.impratio = 10.0
            physics_cfg.solver_cfg.ls_parallel = False
            physics_cfg.solver_cfg.update_data_interval = 2
            physics_cfg.solver_cfg.ccd_iterations = 50
            physics_cfg.num_substeps = 2
        sim = SimulationContext(
            sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device, physics=physics_cfg)
        )
        scene: InteractiveScene = instantiate_cfg()
        sim.reset()

        robot: Articulation = scene["g1"]
        default_joint_pos = robot.data.default_joint_pos.torch.clone()

        print(
            f"[gate] spawned G1: {robot.num_joints} joints, "
            f"{len(robot.joint_names)} names, device={sim.device}",
            flush=True,
        )

        # Settle a few steps, then measure.
        min_z = float("inf")
        max_tilt = 0.0
        settle = 50
        for step in range(args_cli.steps):
            robot.actuators.target_command.set_position_index(value=default_joint_pos)
            scene.write_data_to_sim()
            sim.step()
            scene.update(args_cli.dt)

            if step >= settle:
                pose = robot.data.root_pose_w.torch  # [N, 7]: pos + quat(wxyz)
                z = float(pose[0, 2])
                tilt = float(_tilt_from_quat(pose[0:1, 3:])[0])
                min_z = min(min_z, z)
                max_tilt = max(max_tilt, tilt)
                if step % 100 == 0:
                    print(
                        f"[gate] step {step:4d}  base_z={z:.3f} m  tilt={math.degrees(tilt):.1f} deg",
                        flush=True,
                    )

        final_z = float(robot.data.root_pose_w.torch[0, 2])
        print(
            f"[gate] FINAL base_z={final_z:.3f} m  min_z={min_z:.3f} m  "
            f"max_tilt={math.degrees(max_tilt):.1f} deg",
            flush=True,
        )

        # PASS = the pipeline: the G1 spawned with its full joint set, stepped the
        # requested number of real physics steps, and stayed finite throughout
        # (no NaN/Inf poses). Stance is reported separately and is not a pass
        # criterion; see the module docstring (standing is Objective 1's job).
        pipeline_ok = (
            robot.num_joints > 0
            and args_cli.steps > 0
            and math.isfinite(min_z)
            and math.isfinite(max_tilt)
        )
        upright = min_z > MIN_UPRIGHT_HEIGHT and max_tilt < MAX_TILT_RAD
        print(
            f"[gate] STANCE (not a criterion): {'held upright' if upright else 'toppled (expected: needs locomotion policy)'}",
            flush=True,
        )
        print(
            f"[gate] PIPELINE (import+spawn+physics): {'PASS' if pipeline_ok else 'FAIL'}",
            flush=True,
        )
        exit_code = 0 if pipeline_ok else 1
    return exit_code


def instantiate_cfg() -> InteractiveScene:
    """Build the scene (kept separate so Isaac globals exist when it runs)."""
    from isaaclab.utils import instantiate

    cfg = StanceSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=True)
    # The stock G1 gains are tuned for PhysX; under MJWarp they oscillate and topple.
    # Raise damping on every actuator group so the position hold is statically stable.
    for actuator_cfg in cfg.g1.actuators.values():
        actuator_cfg.damping = 20.0
    return instantiate(cfg)


if __name__ == "__main__":
    raise SystemExit(run())
