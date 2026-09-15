"""
cache_utils.py
==============

Version-based result caching for per-microstructure runs. ONE version
counter is shared across all three methods (taufactor, berg_voxel, berg_cc)
per microstructure -- "version 3" means "the 3rd distinct state of this
microstructure's full pipeline", not an independent per-method history.

Layout:

    <microstructure_dir>/
        latest_version.json           {"version": N, "updated": ...}
        version<N>/
            summary.json               resolved metrics for all 3 methods
            <method>/
                # real run:
                config.json, metrics.json, full_result.pkl?, _SUCCESS, log.txt?
                # OR pointer (config unchanged vs an earlier real run):
                pointer.json

A method directory's kind is determined purely by which marker it has:
pointer.json present -> pointer; else _SUCCESS present -> real run. The two
are mutually exclusive. See pipeline.py's run_microstructure for how these
are decided and written.

Because every later version with the same config hash always resolves to a
pointer (never a second real run -- see find_prior_real_run), there is at
most one real run per unique config hash per microstructure at any time.
This makes the design self-healing: if a real run is deleted by hand and its
hash is needed again later, find_prior_real_run simply finds nothing and the
caller recomputes it fresh. The one unrepaired case is *other* pointers that
still name a since-deleted version (dangling) -- not auto-fixed, flagged as
a known limitation.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_VERSION_RE = re.compile(r"^version(\d+)$")

LATEST_VERSION_FILENAME = "latest_version.json"
SUMMARY_FILENAME = "summary.json"
CONFIG_FILENAME = "config.json"
METRICS_FILENAME = "metrics.json"
FULL_RESULT_FILENAME = "full_result.pkl"
SUCCESS_FILENAME = "_SUCCESS"
POINTER_FILENAME = "pointer.json"


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def config_hash(config: Dict[str, Any]) -> str:
    """Stable short hash of a hyperparameter dict."""
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# =============================================================================
# Version discovery / allocation
# =============================================================================

def list_versions(microstructure_dir) -> list:
    """Sorted list of existing version numbers, from version<N> subdirs.
    Scans the directory itself (not a persisted counter) so a deleted
    version's number is never reused."""
    microstructure_dir = Path(microstructure_dir)
    if not microstructure_dir.exists():
        return []
    versions = []
    for p in microstructure_dir.iterdir():
        if not p.is_dir():
            continue
        m = _VERSION_RE.match(p.name)
        if m:
            versions.append(int(m.group(1)))
    return sorted(versions)


def next_version_number(microstructure_dir) -> int:
    versions = list_versions(microstructure_dir)
    return (max(versions) + 1) if versions else 1


def version_dir(microstructure_dir, version: int) -> Path:
    return Path(microstructure_dir) / f"version{version}"


# =============================================================================
# latest_version.json
# =============================================================================

def read_latest_version(microstructure_dir) -> Optional[int]:
    obj = _read_json(Path(microstructure_dir) / LATEST_VERSION_FILENAME)
    return obj["version"] if obj else None


def write_latest_version(microstructure_dir, version: int) -> None:
    _write_json(
        Path(microstructure_dir) / LATEST_VERSION_FILENAME,
        {"version": version, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")},
    )


# =============================================================================
# Real run / pointer marker helpers
# =============================================================================

def is_real_run(method_dir) -> bool:
    return (Path(method_dir) / SUCCESS_FILENAME).exists()


def is_pointer(method_dir) -> bool:
    return (Path(method_dir) / POINTER_FILENAME).exists()


def load_metrics(method_dir) -> Dict[str, Any]:
    with open(Path(method_dir) / METRICS_FILENAME) as f:
        return json.load(f)


def save_real_run(
    microstructure_dir, version: int, method: str,
    config: Dict[str, Any], metrics: Dict[str, Any],
    full_result: Optional[Dict[str, Any]] = None,
) -> Path:
    """Writes config.json + metrics.json (+ optional full_result.pkl), then
    _SUCCESS last -- a crash mid-write can never look like a valid entry."""
    method_dir = version_dir(microstructure_dir, version) / method
    method_dir.mkdir(parents=True, exist_ok=True)

    _write_json(method_dir / CONFIG_FILENAME, config)
    _write_json(method_dir / METRICS_FILENAME, metrics)
    if full_result is not None:
        import pickle
        with open(method_dir / FULL_RESULT_FILENAME, "wb") as f:
            pickle.dump(full_result, f)

    (method_dir / SUCCESS_FILENAME).touch()
    return method_dir


def save_pointer(
    microstructure_dir, version: int, method: str,
    points_to_version: int, config_hash_value: str,
) -> Path:
    method_dir = version_dir(microstructure_dir, version) / method
    message = (
        f"{method} already run in version {points_to_version} "
        f"(config hash {config_hash_value}); see "
        f"version{points_to_version}/{method}/ for metrics.json, config.json, "
        f"and full_result.pkl if present."
    )
    _write_json(method_dir / POINTER_FILENAME, {
        "type": "pointer",
        "method": method,
        "config_hash": config_hash_value,
        "points_to_version": points_to_version,
        "message": message,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    return method_dir


def read_pointer(method_dir) -> Optional[Dict[str, Any]]:
    return _read_json(Path(method_dir) / POINTER_FILENAME)


def resolve_real_dir(microstructure_dir, version: int, method: str) -> Path:
    """If version<N>/<method> is a pointer, follow its single hop to the
    real run it names (pointers never chain, so one hop always suffices);
    else return the dir itself."""
    method_dir = version_dir(microstructure_dir, version) / method
    pointer = read_pointer(method_dir)
    if pointer is not None:
        return version_dir(microstructure_dir, pointer["points_to_version"]) / method
    return method_dir


# =============================================================================
# Prior real-run resolution (across earlier versions)
# =============================================================================

def find_prior_real_run(
    microstructure_dir, method: str, config_hash_value: str,
    before_version: int, require_full: bool = False,
) -> Optional[Tuple[int, Path]]:
    """Scans version1 .. before_version-1 (strictly earlier than the version
    being decided) for a REAL run of `method` whose config.json hashes to
    config_hash_value. Skips any version whose `method` dir is a pointer
    (only real runs are ever matched/returned -- callers never chain
    pointer-to-pointer). If require_full=True, a candidate only counts if it
    also has full_result.pkl. Returns (version_number, real_run_dir) for the
    first match found, or None.
    """
    microstructure_dir = Path(microstructure_dir)
    for v in list_versions(microstructure_dir):
        if v >= before_version:
            break
        method_dir = version_dir(microstructure_dir, v) / method
        if not is_real_run(method_dir):
            continue
        config = _read_json(method_dir / CONFIG_FILENAME)
        if config is None or config_hash(config) != config_hash_value:
            continue
        if require_full and not (method_dir / FULL_RESULT_FILENAME).exists():
            continue
        return v, method_dir
    return None


# =============================================================================
# Per-version summary.json
# =============================================================================

def save_version_summary(microstructure_dir, version: int, summary: Dict[str, Any]) -> None:
    _write_json(version_dir(microstructure_dir, version) / SUMMARY_FILENAME, summary)


def load_version_summary(microstructure_dir, version: int) -> Optional[Dict[str, Any]]:
    return _read_json(version_dir(microstructure_dir, version) / SUMMARY_FILENAME)


# =============================================================================
# Effective config hash "in effect" at a given version
# =============================================================================

def effective_config_hash(microstructure_dir, version: int, method: str) -> Optional[str]:
    """The config hash in effect for `method` at `version`: for a real run,
    recomputed from its config.json (safe/deterministic -- sha256 of
    sort_keys=True JSON round-trips exactly); for a pointer, read directly
    from pointer.json's stored config_hash field."""
    method_dir = version_dir(microstructure_dir, version) / method
    pointer = read_pointer(method_dir)
    if pointer is not None:
        return pointer["config_hash"]
    config = _read_json(method_dir / CONFIG_FILENAME)
    return config_hash(config) if config is not None else None
