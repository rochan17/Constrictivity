"""
inventory.py
============

Task 1: inventory the data directory. Read-only -- does not touch the
existing pipeline/cache code, only consumes `cache_utils` to resolve
pointers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cache_utils as cu

METHODS = ("taufactor", "berg_voxel", "berg_cc")


def find_best_full_result(ms_dir: Path, latest: int, method: str) -> Path | None:
    """The latest version's pointer chain gives ONE real-run directory, but
    that need not be the only real run sharing its config hash -- and it need
    not be the one with full_result.pkl (see microstructure1/berg_cc: v4's
    pointer resolves to v2, which lacks full_result.pkl, while v3 is a
    same-config real run that has it). Search all real runs of `method`
    across every version whose effective config hash matches the latest
    version's, and prefer one with full_result.pkl if any exists.
    """
    target_hash = cu.effective_config_hash(ms_dir, latest, method)
    if target_hash is None:
        return None

    candidates = []
    for v in cu.list_versions(ms_dir):
        method_dir = cu.version_dir(ms_dir, v) / method
        if not cu.is_real_run(method_dir):
            continue
        if cu.effective_config_hash(ms_dir, v, method) != target_hash:
            continue
        candidates.append((v, method_dir))

    if not candidates:
        return None

    with_full = [(v, d) for v, d in candidates if (d / cu.FULL_RESULT_FILENAME).exists()]
    pool = with_full if with_full else candidates
    return max(pool, key=lambda vd: vd[0])[1]


def inventory_microstructure(ms_dir: Path) -> Dict[str, Any]:
    latest = cu.read_latest_version(ms_dir)
    row: Dict[str, Any] = {"microstructure": ms_dir.name, "latest_version": latest}

    for method in METHODS:
        if latest is None:
            row[method] = {"status": "missing", "has_full": False}
            continue

        method_dir = cu.version_dir(ms_dir, latest) / method
        is_ptr = cu.is_pointer(method_dir)
        is_real = cu.is_real_run(method_dir)

        if is_ptr:
            ptr = cu.read_pointer(method_dir)
            status = f"pointer->v{ptr['points_to_version']}"
        elif is_real:
            status = "real"
        else:
            status = "missing"

        best_dir = find_best_full_result(ms_dir, latest, method)
        naive_dir = cu.resolve_real_dir(ms_dir, latest, method)
        has_full = best_dir is not None and (best_dir / cu.FULL_RESULT_FILENAME).exists()

        row[method] = {
            "status": status,
            "has_full": has_full,
            "resolved_dir": str(best_dir) if best_dir else None,
            "naive_dir": str(naive_dir),
            "differs_from_naive": best_dir is not None and str(best_dir) != str(naive_dir),
        }

    return row


def build_inventory(data_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for ms_dir in sorted(data_dir.iterdir()):
        if not ms_dir.is_dir():
            continue
        if not (ms_dir / cu.LATEST_VERSION_FILENAME).exists():
            continue
        rows.append(inventory_microstructure(ms_dir))
    return rows


def print_inventory(rows: List[Dict[str, Any]]) -> None:
    header = f"{'microstructure':<28}{'latest_v':<10}" + "".join(
        f"{m:<28}" for m in METHODS
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['microstructure']:<28}{str(row['latest_version']):<10}"
        for m in METHODS:
            info = row[m]
            tag = info["status"] + (" [full]" if info["has_full"] else " [NO-FULL]")
            if info.get("differs_from_naive"):
                tag += "*"
            line += f"{tag:<28}"
        print(line)
    print("(* = best-full-result dir differs from the naive latest-version pointer resolution)")

    print()
    counts = {m: {"real": 0, "pointer": 0, "missing": 0,
                  "full": 0, "no_full": 0} for m in METHODS}
    for row in rows:
        for m in METHODS:
            info = row[m]
            if info["status"] == "real":
                counts[m]["real"] += 1
            elif info["status"].startswith("pointer"):
                counts[m]["pointer"] += 1
            else:
                counts[m]["missing"] += 1
            if info["has_full"]:
                counts[m]["full"] += 1
            else:
                counts[m]["no_full"] += 1

    print(f"n_microstructures = {len(rows)}")
    print(f"{'method':<12}{'real':<8}{'pointer':<10}{'missing':<10}{'has_full':<10}{'no_full':<10}")
    for m in METHODS:
        c = counts[m]
        print(f"{m:<12}{c['real']:<8}{c['pointer']:<10}{c['missing']:<10}{c['full']:<10}{c['no_full']:<10}")

    return counts


if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data"
    rows = build_inventory(data_dir)
    counts = print_inventory(rows)

    berg_cc_full = counts["berg_cc"]["full"]
    n = len(rows)
    if n > 0 and berg_cc_full < n / 2:
        print()
        print("HARD STOP: full_result.pkl is missing for berg_cc on most microstructures.")
        print("The batch appears to have been run without --save-full. Re-run with")
        print("--save-full before continuing. Refusing to reconstruct missing arrays.")
        sys.exit(1)
