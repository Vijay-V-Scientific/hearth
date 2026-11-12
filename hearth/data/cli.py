"""Command-line interface for the HEARTH data pipeline.

    hearth-data save episodes.h5 rec1.jsonl --robot g1 --task clear_table
    hearth-data load episodes.h5 --stats
    hearth-data format episodes.h5

Importable without Isaac Sim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .storage import load_episodes, save_episodes, storage_format
from .types import StorageFormat


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hearth-data",
        description="Project HEARTH episode data pipeline.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    save = sub.add_parser("save", help="merge JSONL records into a storage file")
    save.add_argument("source", help="input JSONL file (records)")
    save.add_argument("target", help="output file (format from extension)")
    save.add_argument("--fmt", default=None, help="storage format: hdf5 (default), parquet, jsonl")

    load = sub.add_parser("load", help="load records and print stats")
    load.add_argument("source", help="storage file to read")
    load.add_argument("--stats", action="store_true", help="print per-episode stats (default)")

    fmt = sub.add_parser("format", help="detect the storage format of a file")
    fmt.add_argument("source", help="storage file to inspect")

    return p


def cmd_save(args: argparse.Namespace) -> int:
    records = load_episodes(args.source)
    target: StorageFormat | None = None
    if args.fmt:
        target = StorageFormat(args.fmt)
    out = save_episodes(args.target, records, fmt=target or StorageFormat.HDF5)
    print(f"Wrote {len(records)} episode(s) to {out} ({storage_format(out).value}).")
    return 0


def cmd_load(args: argparse.Namespace) -> int:
    records = load_episodes(args.source)
    fmt = storage_format(args.source)
    print(f"{len(records)} episode(s) from {args.source} ({fmt.value}).")
    for rec in records:
        print(
            f"  {rec.episode_id}: task={rec.task} robot={rec.robot} "
            f"seed={rec.seed} steps={rec.length} dt={rec.dt}"
        )
    return 0


def cmd_format(args: argparse.Namespace) -> int:
    fmt = storage_format(args.source)
    print(f"{args.source}: {fmt.value}")
    return 0


_COMMANDS = {
    "save": cmd_save,
    "load": cmd_load,
    "format": cmd_format,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _COMMANDS[args.command]
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
