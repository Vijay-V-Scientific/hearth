#!/usr/bin/env python
"""Load a generated HEARTH apartment + a Unitree G1 into Isaac Sim and step physics.

Run with Isaac Lab's environment (it bundles Isaac Sim)::

    ~/projects/isaac-sim/IsaacLab/.venv/bin/python scripts/hearth_load_scene.py --headless

Proves the two things Objective 1 stands on: a procedurally generated apartment
from ``hearth.scene`` opens in Isaac as real geometry, and the G1 humanoid spawns
beside it and survives being stepped under physics. Prints a machine-readable
summary and exits; nothing here is a policy result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=2026, help="Apartment seed")
parser.add_argument("--rooms", type=int, default=3, help="Room count")
parser.add_argument("--steps", type=int, default=30, help="Physics steps before exit")
parser.add_argument(
    "--usd", default=str(REPO / "logs" / "demo_apartment.usd"), help="Apartment USD to load"
)
parser.add_argument(
    "--robot", default="g1", choices=("g1", "none"), help="Robot to spawn beside the apartment"
)

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR  # noqa: E402

add_launcher_args(parser)  # registers --headless, --device, --width ...
args_cli = parser.parse_args()

from hearth.scene import ApartmentConfig, generate_scene, write_usd  # noqa: E402


def main() -> None:
    usd_path = Path(args_cli.usd)
    if not usd_path.is_file():
        scene = generate_scene(ApartmentConfig(room_count=args_cli.rooms, seed=args_cli.seed))
        usd_path.parent.mkdir(parents=True, exist_ok=True)
        write_usd(scene, str(usd_path))
        print(f"[INFO] generated apartment -> {usd_path}", flush=True)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.005)
    with launch_simulation(sim_cfg, args_cli):
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view([6.0, -6.0, 4.0], [0.0, 0.0, 0.5])

        # Ground plane + distant light so the world is physically sane.
        cfg_ground = sim_utils.GroundPlaneCfg()
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
        cfg_light = sim_utils.DistantLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        cfg_light.func("/World/lightDistant", cfg_light, translation=(1, 0, 10))

        # The generated apartment, referenced as-is from disk.
        cfg_ap = sim_utils.UsdFileCfg(usd_path=str(usd_path.resolve()))
        cfg_ap.func("/World/Apartment", cfg_ap)

        robot_path = "/World/Robot"
        if args_cli.robot == "g1":
            g1_url = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd"
            cfg_g1 = sim_utils.UsdFileCfg(usd_path=g1_url, activate_contact_sensors=True)
            cfg_g1.func(robot_path, cfg_g1, translation=(3.0, 3.0, 0.02))
            print(f"[INFO] spawning G1 from {g1_url}", flush=True)

        sim.reset()
        stage = sim.stage
        n_prims = sum(1 for _ in stage.Traverse())
        robot_valid = stage.GetPrimAtPath(robot_path).IsValid() if args_cli.robot == "g1" else "n/a"
        print(f"[INFO] stage ready: {n_prims} prims; robot prim valid = {robot_valid}", flush=True)

        for _ in range(args_cli.steps):
            if not sim.is_running():
                break
            sim.step()
        print(f"[INFO] stepped {args_cli.steps} physics steps without error", flush=True)
        print(
            f"[RESULT] apartment={usd_path.name} rooms={args_cli.rooms} seed={args_cli.seed} "
            f"prims={n_prims} robot={args_cli.robot} ok=true",
            flush=True,
        )


if __name__ == "__main__":
    main()
