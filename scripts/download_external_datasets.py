#!/usr/bin/env python3
"""Download external trajectory datasets into the local gitignored data tree.

This script recreates the directory layout used by the experiments:

    data/external/{aftraj,agenterrorbench,agentrx,atbench,mast_mad,tau-bench,...}

Hugging Face revisions and git commits are pinned to the snapshots used in the
initial local experiments so another laptop can reproduce the same inputs.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


HF_SNAPSHOTS = {
    "aftraj": {
        "repo_id": "ZBox008003/AFTraj",
        "local_name": "aftraj",
        "revision": "11191169d3d99298df098eee8d83db425117faba",
        "gated": False,
    },
    "agenterrorbench": {
        "repo_id": "davide221/agenterrorbench",
        "local_name": "agenterrorbench",
        "revision": "ac71bfa93c9743319060debb0d2278e02b00640f",
        "gated": False,
    },
    "agentrx": {
        "repo_id": "microsoft/AgentRx",
        "local_name": "agentrx",
        "revision": "88e871fecb58b2d090449f37ec80b8865594e0b5",
        "gated": True,
    },
    "atbench": {
        "repo_id": "AI45Research/ATBench",
        "local_name": "atbench",
        "revision": "4476ef92ed8f85c8d58d8a5b9dfdf55aa7893138",
        "gated": False,
    },
}

HF_FILES = {
    "mast_mad": {
        "repo_id": "mcemri/MAD",
        "local_name": "mast_mad",
        "revision": "5a82e32347f70a701a3c68637de12f8a0be3de3c",
        "files": [
            "MAD_full_dataset.json",
            "MAD_human_labelled_dataset.json",
        ],
    },
}

GIT_REPOS = {
    "tau_bench": {
        "url": "https://github.com/sierra-research/tau-bench.git",
        "local_name": "tau-bench",
        "revision": "59a200c6d575d595120f1cb70fea53cef0632f6b",
        "large": False,
    },
    "mast_repo": {
        "url": "https://github.com/multi-agent-systems-failure-taxonomy/MAST.git",
        "local_name": "mast_repo",
        "revision": "a70542e541b2104ef8fcd785778179e173fb8d70",
        "large": True,
    },
}

CORE_DATASETS = [
    "aftraj",
    "agenterrorbench",
    "agentrx",
    "atbench",
    "mast_mad",
    "tau_bench",
]
ALL_DATASETS = CORE_DATASETS + ["mast_repo"]


def _require_hf() -> tuple[object, object]:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install with:\n"
            "  uv pip install -e '.[data]'\n"
            "or:\n"
            "  python -m pip install huggingface_hub"
        ) from exc
    return hf_hub_download, snapshot_download


def _run(cmd: list[str], dry_run: bool) -> None:
    print("+", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def _prepare_target(path: Path, force: bool, dry_run: bool) -> bool:
    if not path.exists():
        return True
    if not force:
        print(f"skip existing: {path}")
        return False
    print(f"remove existing: {path}")
    if not dry_run:
        shutil.rmtree(path)
    return True


def _download_hf_snapshot(name: str, root: Path, force: bool, dry_run: bool) -> None:
    spec = HF_SNAPSHOTS[name]
    target = root / str(spec["local_name"])
    if not _prepare_target(target, force, dry_run):
        return
    print(
        f"download HF dataset {spec['repo_id']}@{spec['revision']} -> {target}"
        + (" (gated; requires prior HF access/login)" if spec["gated"] else "")
    )
    if dry_run:
        return
    _hf_hub_download, snapshot_download = _require_hf()
    snapshot_download(
        repo_id=str(spec["repo_id"]),
        repo_type="dataset",
        revision=str(spec["revision"]),
        local_dir=str(target),
    )


def _download_hf_files(name: str, root: Path, force: bool, dry_run: bool) -> None:
    spec = HF_FILES[name]
    target = root / str(spec["local_name"])
    if target.exists() and force:
        print(f"remove existing: {target}")
        if not dry_run:
            shutil.rmtree(target)
    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)
    for filename in spec["files"]:
        print(
            f"download HF file {spec['repo_id']}:{filename}@{spec['revision']} -> {target}"
        )
        if dry_run:
            continue
        hf_hub_download, _snapshot_download = _require_hf()
        hf_hub_download(
            repo_id=str(spec["repo_id"]),
            repo_type="dataset",
            revision=str(spec["revision"]),
            filename=filename,
            local_dir=str(target),
        )


def _clone_git_repo(name: str, root: Path, force: bool, dry_run: bool) -> None:
    spec = GIT_REPOS[name]
    target = root / str(spec["local_name"])
    if not _prepare_target(target, force, dry_run):
        return
    _run(["git", "clone", "--depth", "1", str(spec["url"]), str(target)], dry_run)
    if dry_run:
        return
    revision = str(spec["revision"])
    try:
        subprocess.run(["git", "-C", str(target), "checkout", revision], check=True)
    except subprocess.CalledProcessError:
        _run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", revision], False)
        _run(["git", "-C", str(target), "checkout", revision], False)


def _resolve_requested(values: Iterable[str]) -> list[str]:
    requested: list[str] = []
    for value in values:
        if value == "core":
            requested.extend(CORE_DATASETS)
        elif value == "all":
            requested.extend(ALL_DATASETS)
        else:
            requested.append(value)

    valid = set(ALL_DATASETS)
    unknown = sorted(set(requested) - valid)
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown)}")

    deduped: list[str] = []
    for item in requested:
        if item not in deduped:
            deduped.append(item)
    return deduped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/external"),
        help="Output directory that will contain dataset subdirectories.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["core"],
        help=(
            "Dataset names to download. Use 'core' for paper-relevant data "
            "or 'all' to also clone the large MAST trace repo. "
            f"Valid names: {', '.join(ALL_DATASETS)}"
        ),
    )
    parser.add_argument(
        "--skip-gated",
        action="store_true",
        help="Skip gated datasets such as AgentRx if this machine has no HF access.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove and re-download existing dataset directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without downloading.",
    )
    args = parser.parse_args(argv)

    root = args.root
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)

    selected = _resolve_requested(args.datasets)
    if args.skip_gated:
        selected = [
            name
            for name in selected
            if not (name in HF_SNAPSHOTS and HF_SNAPSHOTS[name].get("gated"))
        ]

    for name in selected:
        if name in HF_SNAPSHOTS:
            _download_hf_snapshot(name, root, args.force, args.dry_run)
        elif name in HF_FILES:
            _download_hf_files(name, root, args.force, args.dry_run)
        elif name in GIT_REPOS:
            _clone_git_repo(name, root, args.force, args.dry_run)
        else:
            raise AssertionError(name)

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
