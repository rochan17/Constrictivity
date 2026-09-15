"""
collect_results.py
===================

Aggregate cached per-microstructure results (written by pipeline.py /
run_batch.py under <microstructure>/version<N>/<method>/ -- see
cache_utils.py's module docstring for the full layout) across an entire
data directory into pandas DataFrames. Nothing here recomputes anything --
it only reads latest_version.json/summary.json files already on disk, so
it's cheap to re-run any time after a batch finishes (or while it's still
running, to check on partial progress).

Versioning is shared across all three methods per microstructure (one
version counter, not independent per method) -- see cache_utils.py. A
method's entry at a given version is either a real run or a pointer to an
earlier version where it actually ran; both views below surface which.

Two views:

    latest_summary_table(data_dir)
        Wide table, one row per microstructure, using that microstructure's
        latest version (its latest_version.json). Columns are prefixed by
        method name, e.g. "berg_cc.tau_sq_c", plus "<method>.version" (the
        version a method's data actually lives in -- may be earlier than
        the microstructure's latest version if that method was a pointer)
        and "<method>.is_pointer".

    all_versions_table(data_dir)
        Long table, one row per (microstructure, version) -- NOT per
        (microstructure, method, hash), since versioning is microstructure-
        wide. Built from each version's summary.json, which already has
        every method's resolved metrics. Per method: "<method>.status"
        ("real"/"pointer"), "<method>.config_hash", "<method>.resolved_version",
        plus "<method>.<metric_key>" for every metric, e.g.:

            df = all_versions_table("data")
            df[["microstructure", "version", "berg_cc.status",
                "berg_cc.tau_sq_c"]]

CLI usage:
    python collect_results.py --data-dir data
    python collect_results.py --data-dir data --out summary.csv
    python collect_results.py --data-dir data --all-versions --out-long all_runs.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

import cache_utils

METHODS = ("taufactor", "berg_voxel", "berg_cc")


def latest_summary_table(data_dir) -> pd.DataFrame:
    data_dir = Path(data_dir)
    rows = []
    for ms_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        versions = cache_utils.list_versions(ms_dir)
        if not versions:
            continue
        latest_v = cache_utils.read_latest_version(ms_dir)
        if latest_v is None or latest_v not in versions:
            latest_v = max(versions)

        summary = cache_utils.load_version_summary(ms_dir, latest_v)
        if summary is None:
            continue

        row: Dict[str, Any] = {"microstructure": ms_dir.name, "latest_version": latest_v}
        for method in METHODS:
            metrics = summary.get(method)
            if metrics is None:
                continue
            row[f"{method}.version"] = metrics.get("_resolved_version", latest_v)
            row[f"{method}.is_pointer"] = bool(metrics.get("_cache_hit", False))
            for k, v in metrics.items():
                if k.startswith("_"):
                    continue
                row[f"{method}.{k}"] = v
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("microstructure")


def all_versions_table(data_dir) -> pd.DataFrame:
    data_dir = Path(data_dir)
    rows = []
    for ms_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for version in cache_utils.list_versions(ms_dir):
            summary = cache_utils.load_version_summary(ms_dir, version)
            if summary is None:
                continue
            row: Dict[str, Any] = {"microstructure": ms_dir.name, "version": version}
            for method in METHODS:
                metrics = summary.get(method)
                if metrics is None:
                    continue
                is_pointer = bool(metrics.get("_cache_hit", False))
                row[f"{method}.status"] = "pointer" if is_pointer else "real"
                row[f"{method}.config_hash"] = metrics.get("_version_hash")
                row[f"{method}.resolved_version"] = metrics.get("_resolved_version", version)
                for k, v in metrics.items():
                    if k.startswith("_"):
                        continue
                    row[f"{method}.{k}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=None, help="CSV path for the wide (latest-only) table.")
    ap.add_argument("--out-long", default=None, help="CSV path for the long (all-versions) table.")
    ap.add_argument("--all-versions", action="store_true",
                     help="Print the long all-versions table instead of the latest-only wide table.")
    args = ap.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    if args.all_versions:
        df = all_versions_table(args.data_dir)
    else:
        df = latest_summary_table(args.data_dir)
    print(df)

    if args.out:
        latest_summary_table(args.data_dir).to_csv(args.out)
        print(f"wrote {args.out}")
    if args.out_long:
        all_versions_table(args.data_dir).to_csv(args.out_long, index=False)
        print(f"wrote {args.out_long}")


if __name__ == "__main__":
    main()
