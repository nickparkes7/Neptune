#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
from pathlib import Path

import yaml

STATUS_PATH = Path("status/phase1.yml")


def load_status():
    if not STATUS_PATH.exists():
        raise SystemExit(f"Missing {STATUS_PATH}")
    with STATUS_PATH.open() as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("steps", [])
    data.setdefault("goals", [])
    return data


def save_status(data):
    with STATUS_PATH.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def cmd_list(args):
    data = load_status()
    steps = data["steps"]
    counts = {s: 0 for s in ["pending", "in_progress", "blocked", "done"]}
    for step in steps:
        counts[step.get("status", "pending")] = counts.get(step.get("status", "pending"), 0) + 1
    print("Goals:")
    for g in data.get("goals", []):
        print(f"- {g}")
    print()
    print("Progress summary:")
    total = len(steps)
    print(f"  done {counts.get('done',0)}/{total} · in_progress {counts.get('in_progress',0)} · blocked {counts.get('blocked',0)}")
    print()
    print("Steps:")
    for step in steps:
        sid = step["id"]
        print(f"- {sid:<12} {step.get('status','pending'):<12} owner={step.get('owner','')}")


def cmd_set(args):
    data = load_status()
    steps = {s["id"]: s for s in data["steps"]}
    if args.step_id not in steps:
        raise SystemExit(f"Unknown step_id: {args.step_id}")
    step = steps[args.step_id]
    now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    step["status"] = args.status
    if args.owner is not None:
        step["owner"] = args.owner
    if args.notes:
        step["notes"] = args.notes
    if args.artifact:
        arts = step.setdefault("artifacts", [])
        arts.append(args.artifact)
    if args.status == "in_progress" and not step.get("started"):
        step["started"] = now
    if args.status == "done":
        step.setdefault("started", now)
        step["finished"] = now
    save_status(data)
    print(f"Updated {args.step_id} → {args.status}")


def cmd_goals(args):
    data = load_status()
    for g in data.get("goals", []):
        print(f"- {g}")


def main():
    ap = argparse.ArgumentParser(description="Phase 1 status CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_list = sub.add_parser("list", help="List progress and steps")
    ap_list.set_defaults(func=cmd_list)

    ap_set = sub.add_parser("set", help="Update a step's status")
    ap_set.add_argument("step_id")
    ap_set.add_argument("status", choices=["pending", "in_progress", "blocked", "done"])
    ap_set.add_argument("--owner")
    ap_set.add_argument("--notes")
    ap_set.add_argument("--artifact")
    ap_set.set_defaults(func=cmd_set)

    ap_goals = sub.add_parser("goals", help="Print guiding goals")
    ap_goals.set_defaults(func=cmd_goals)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

