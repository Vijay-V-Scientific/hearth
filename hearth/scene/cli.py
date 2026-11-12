"""CLI for the procedural scene generator (DONE).

Usage:
    python -m hearth.scene.cli generate --seed 0 --out /tmp/hearth/scene0.usd
    python -m hearth.scene.cli list-objects
    python -m hearth.scene.cli stats --config configs/scene/example.yaml
"""

from __future__ import annotations

import argparse
import sys
from .types import ApartmentConfig
from .generator import generate_scene
from .writer import write_usd


def _cmd_generate(args: argparse.Namespace) -> int:
    cfg = ApartmentConfig(
        width=args.width,
        depth=args.depth,
        room_count=args.rooms,
        min_object_spacing=args.min_spacing,
        task_objects_per_scene=args.task_objects,
        seed=args.seed,
    )
    out = args.out or f"/tmp/hearth/scene_{args.seed}.usd"
    scene = generate_scene(cfg)
    path = write_usd(scene, out)
    md = scene.metadata
    print(f"Wrote {path}")
    print(f"  scene_id={md.scene_id} seed={md.seed}")
    print(f"  rooms={md.num_rooms} objects={md.num_objects} task_objects={md.num_task_objects}")
    print(f"  total_area={md.total_area:.1f} m^2")
    return 0


def _cmd_list_objects(args: argparse.Namespace) -> int:
    from collections import Counter

    cfg = ApartmentConfig(width=args.width, depth=args.depth, room_count=1, seed=args.seed)
    scene = generate_scene(cfg)
    counts = Counter(cat for cat, _ in scene.objects)
    print("Objects in a 1-room scene:")
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n}")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    import yaml

    with open(args.config) as f:
        data = yaml.safe_load(f)
    cfg = ApartmentConfig(**data)
    scene = generate_scene(cfg)
    md = scene.metadata
    print(f"Config {args.config}: rooms={md.num_rooms} objects={md.num_objects} "
          f"task={md.num_task_objects} area={md.total_area:.1f} m^2")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hearth-scene", description="Procedural apartment scene generator")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Generate and write a USD scene")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--width", type=float, default=12.0)
    g.add_argument("--depth", type=float, default=10.0)
    g.add_argument("--rooms", type=int, default=4)
    g.add_argument("--task-objects", type=int, default=3)
    g.add_argument("--min-spacing", type=float, default=0.5)
    g.add_argument("--out", type=str, default=None)
    g.set_defaults(func=_cmd_generate)

    l = sub.add_parser("list-objects", help="List object categories in a 1-room scene")
    l.add_argument("--seed", type=int, default=0)
    l.add_argument("--width", type=float, default=6.0)
    l.add_argument("--depth", type=float, default=6.0)
    l.set_defaults(func=_cmd_list_objects)

    s = sub.add_parser("stats", help="Print stats for a scene config YAML")
    s.add_argument("--config", type=str, required=True)
    s.set_defaults(func=_cmd_stats)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
