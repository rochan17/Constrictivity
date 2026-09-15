"""
throat_analysis.py
===================

Read-only analysis layer over an existing microstructure transport pipeline
(pipeline.py / berg_cc.py / berg_voxel_fast.py / cache_utils.py). This module
never writes into the pipeline's own `<microstructure>/version<N>/` trees --
all cache/output goes to a separate tree passed in by the caller.

Pure functions, no side effects at import. See REPORT.md for the numbers
this produces and the constraints (0b in the task spec) that shaped it.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import cache_utils as cu
import berg_cc

METHODS = ("taufactor", "berg_voxel", "berg_cc")

# Classification cuts (Task 4) -- module-level constants, not literals.
C_NARROW_CUT = 0.85          # C < this = geometrically narrow
F_BUSY_FRAC_OF_MAX = 0.25    # f >= this * max(f) = transport-busy
F_IDLE_CUT = 0.005           # f < this = idle


# =============================================================================
# 1. Inventory
# =============================================================================

def find_best_full_result_dir(ms_dir: Path, method: str) -> Optional[Path]:
    """The latest version's pointer chain resolves to ONE real-run directory,
    but that need not be the only real run sharing its effective config hash,
    and it need not be the one with full_result.pkl persisted (observed on
    microstructure1/berg_cc: the latest pointer resolves to a real run
    without full_result.pkl, while a later same-config real run has it).
    Search all real runs of `method` across every version whose effective
    config hash matches the version currently in effect, and prefer one that
    has full_result.pkl.
    """
    latest = cu.read_latest_version(ms_dir)
    if latest is None:
        return None
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


# Mirrors pipeline.py's own config.get(key, default) calls for berg_cc, so a
# config.json that omits a key can still be matched against that key's real
# pipeline default rather than against whatever value the caller is querying
# for (see find_matching_version_dir).
BERG_CC_CONFIG_DEFAULTS: Dict[str, Any] = {
    "direction": "x",
    "min_throat_size": 5,
    "dilate_both": True,
    "surface_axis": None,
    "skip_surface_surface": True,
    "area_method": "vox_projection",
    "alpha": 0.5,
    "split_volume_equal": True,
    "streamtube_method": "greedy_fast",
}


def find_matching_version_dir(ms_dir: Path, method: str, criteria: Dict[str, Any]) -> Optional[Path]:
    """Like find_best_full_result_dir, but instead of following
    latest_version.json, scans all versions for a real run of `method` whose
    config.json matches `criteria`. A key missing from config.json falls back
    to that key's real pipeline default (BERG_CC_CONFIG_DEFAULTS), not to the
    queried value -- so a version that omits e.g. min_throat_size only
    matches a criteria value of 5 (the pipeline's actual default), never an
    arbitrary queried value. Prefers the run with full_result.pkl; ties
    broken by highest version."""
    candidates = []
    for v in cu.list_versions(ms_dir):
        method_dir = cu.version_dir(ms_dir, v) / method
        if not cu.is_real_run(method_dir):
            continue
        config = json.loads((method_dir / cu.CONFIG_FILENAME).read_text())
        if all(config.get(k, BERG_CC_CONFIG_DEFAULTS.get(k)) == val for k, val in criteria.items()):
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
            row[method] = {"status": "missing", "has_full": False, "resolved_dir": None}
            continue
        method_dir = cu.version_dir(ms_dir, latest) / method
        if cu.is_pointer(method_dir):
            ptr = cu.read_pointer(method_dir)
            status = f"pointer->v{ptr['points_to_version']}"
        elif cu.is_real_run(method_dir):
            status = "real"
        else:
            status = "missing"
        best_dir = find_best_full_result_dir(ms_dir, method)
        has_full = best_dir is not None and (best_dir / cu.FULL_RESULT_FILENAME).exists()
        row[method] = {"status": status, "has_full": has_full,
                        "resolved_dir": str(best_dir) if best_dir else None}
    return row


def build_inventory(data_dir: Path) -> pd.DataFrame:
    rows = []
    for ms_dir in sorted(Path(data_dir).iterdir()):
        if not ms_dir.is_dir() or not (ms_dir / cu.LATEST_VERSION_FILENAME).exists():
            continue
        rows.append(inventory_microstructure(ms_dir))
    flat = []
    for r in rows:
        flat_row = {"microstructure": r["microstructure"], "latest_version": r["latest_version"]}
        for m in METHODS:
            flat_row[f"{m}_status"] = r[m]["status"]
            flat_row[f"{m}_has_full"] = r[m]["has_full"]
            flat_row[f"{m}_dir"] = r[m]["resolved_dir"]
        flat.append(flat_row)
    return pd.DataFrame(flat)


# =============================================================================
# 2. Loading raw results
# =============================================================================

@dataclass
class MicrostructureData:
    name: str
    ms_dir: Path
    phase_id: int
    direction: str
    berg: Dict[str, Any]          # unpickled berg_cc full_result
    berg_metrics: Dict[str, Any]
    berg_config: Dict[str, Any]
    tau: Dict[str, Any]           # unpickled taufactor full_result
    tau_metrics: Dict[str, Any]
    segmented: np.ndarray
    body_array: np.ndarray


def _build_microstructure_data(ms_dir: Path, berg_dir: Optional[Path], tau_dir: Optional[Path]) -> MicrostructureData:
    manifest_path = ms_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    if berg_dir is None or not (berg_dir / cu.FULL_RESULT_FILENAME).exists():
        raise FileNotFoundError(f"No berg_cc full_result.pkl found for {ms_dir}")
    if tau_dir is None or not (tau_dir / cu.FULL_RESULT_FILENAME).exists():
        raise FileNotFoundError(f"No taufactor full_result.pkl found for {ms_dir}")

    with open(berg_dir / cu.FULL_RESULT_FILENAME, "rb") as f:
        berg = pickle.load(f)
    berg_metrics = cu.load_metrics(berg_dir)
    berg_config = json.loads((berg_dir / cu.CONFIG_FILENAME).read_text())

    with open(tau_dir / cu.FULL_RESULT_FILENAME, "rb") as f:
        tau = pickle.load(f)
    tau_metrics = cu.load_metrics(tau_dir)
    tau_config = json.loads((tau_dir / cu.CONFIG_FILENAME).read_text())

    # phase_id is set by pipeline.py per-microstructure (default 1) and
    # persisted in taufactor's config.json; manifest.json's phase_id (if
    # present) is an explicit override, matching pipeline._load_manifest.
    phase_id = manifest.get("phase_id", tau_config.get("phase_id", 1))

    segmented = np.load(ms_dir / "segmented.npy")
    body_array = np.load(ms_dir / "body_array.npy")

    direction = berg_config.get("direction", "x")

    return MicrostructureData(
        name=ms_dir.name, ms_dir=ms_dir, phase_id=phase_id, direction=direction,
        berg=berg, berg_metrics=berg_metrics, berg_config=berg_config,
        tau=tau, tau_metrics=tau_metrics,
        segmented=segmented, body_array=body_array,
    )


def load_microstructure(ms_dir: Path) -> MicrostructureData:
    ms_dir = Path(ms_dir)
    berg_dir = find_best_full_result_dir(ms_dir, "berg_cc")
    tau_dir = find_best_full_result_dir(ms_dir, "taufactor")
    return _build_microstructure_data(ms_dir, berg_dir, tau_dir)


def load_microstructure_matching(ms_dir: Path, berg_cc_criteria: Dict[str, Any]) -> MicrostructureData:
    """Like load_microstructure, but instead of using whichever berg_cc
    version latest_version.json currently points to, picks the berg_cc
    version whose config.json matches `berg_cc_criteria` (see
    find_matching_version_dir). taufactor is still resolved via the latest
    version, since its config isn't part of the filter."""
    ms_dir = Path(ms_dir)
    berg_dir = find_matching_version_dir(ms_dir, "berg_cc", berg_cc_criteria)
    if berg_dir is None:
        raise FileNotFoundError(
            f"No berg_cc version matching {berg_cc_criteria} found for {ms_dir}")
    tau_dir = find_best_full_result_dir(ms_dir, "taufactor")
    return _build_microstructure_data(ms_dir, berg_dir, tau_dir)


# =============================================================================
# 2b. Acceptance-test-2 style checks (importable for tests)
# =============================================================================

def check_shapes(msd: MicrostructureData) -> Dict[str, Tuple[int, ...]]:
    return {
        "segmented": msd.segmented.shape,
        "body_array": msd.body_array.shape,
        "pore_mask": msd.tau["pore_mask"].shape,
        "J_x": msd.tau["J_x"].shape,
    }


def check_pore_mask_agreement(msd: MicrostructureData) -> Dict[str, Any]:
    pore_mask = msd.tau["pore_mask"]
    body_pos = msd.body_array > 0
    agree = pore_mask == body_pos
    n_total = agree.size
    n_disagree = int(n_total - agree.sum())
    return {
        "n_total": n_total,
        "n_disagree": n_disagree,
        "frac_agree": float(agree.sum()) / n_total,
    }


def get_inlet_outlet(msd: MicrostructureData) -> Tuple[np.ndarray, np.ndarray]:
    """Recomputes inlet/outlet body indices from persisted arrays. Not stored
    in full_result.pkl (pipeline.py only promotes total_current /
    effective_conductance from solution_data into metrics.json) -- this is a
    pure function of data we already have, so recomputing it does not require
    re-running the solve.
    """
    return berg_cc.get_inlet_outlet_indices(msd.berg["arrays"], msd.body_array, direction=msd.direction)


def check_kirchhoff(msd: MicrostructureData) -> Dict[str, Any]:
    arrays = msd.berg["arrays"]
    current = msd.berg["current"]
    conns = arrays["conns"]
    N_bodies = len(arrays["body_ids"])
    total_current = msd.berg_metrics["total_current"]

    net_current = np.zeros(N_bodies)
    valid = (conns[:, 0] >= 0) & (conns[:, 1] >= 0)
    i_idx = conns[valid, 0]
    j_idx = conns[valid, 1]
    I = current[valid]
    np.add.at(net_current, i_idx, -I)
    np.add.at(net_current, j_idx, I)

    inlet_idx, outlet_idx = get_inlet_outlet(msd)
    boundary_set = set(inlet_idx.tolist()) | set(outlet_idx.tolist())
    active_bodies = set(i_idx.tolist()) | set(j_idx.tolist())
    interior_active = np.array([
        (b in active_bodies) and (b not in boundary_set) for b in range(N_bodies)
    ])
    residuals = net_current[interior_active]
    max_abs_residual = float(np.max(np.abs(residuals))) if len(residuals) else 0.0
    return {
        "n_interior_checked": int(interior_active.sum()),
        "max_abs_residual": max_abs_residual,
        "total_current": total_current,
        "max_residual_frac_of_total": max_abs_residual / total_current,
    }


def check_inlet_current_consistency(msd: MicrostructureData) -> Dict[str, Any]:
    arrays = msd.berg["arrays"]
    current = msd.berg["current"]
    conns = arrays["conns"]
    total_current = msd.berg_metrics["total_current"]
    inlet_idx, _ = get_inlet_outlet(msd)
    inlet_set = set(inlet_idx.tolist())

    valid = (conns[:, 0] >= 0) & (conns[:, 1] >= 0)
    i_idx, j_idx = conns[valid, 0], conns[valid, 1]
    I = current[valid]
    inlet_adjacent = np.array([(i in inlet_set) or (j in inlet_set) for i, j in zip(i_idx, j_idx)])
    sum_abs = float(np.sum(np.abs(I[inlet_adjacent])))

    sign = np.where(np.isin(i_idx, list(inlet_set)) & ~np.isin(j_idx, list(inlet_set)), 1.0,
             np.where(np.isin(j_idx, list(inlet_set)) & ~np.isin(i_idx, list(inlet_set)), -1.0, 0.0))
    net_from_inlet = float(np.sum(sign * I))

    return {
        "n_inlet_adjacent_edges": int(inlet_adjacent.sum()),
        "sum_abs_inlet_current": sum_abs,
        "total_current": total_current,
        "ratio_sum_abs_to_total": sum_abs / total_current,
        "net_signed_current_out_of_inlet": net_from_inlet,
        "ratio_net_to_total": abs(net_from_inlet) / total_current,
    }


# =============================================================================
# 3. EDT geometric reference
# =============================================================================

@dataclass
class GeometryData:
    edt: np.ndarray
    r_body_by_idx: np.ndarray          # (N_bodies,) aligned to arrays['body_ids'] order
    interface_lo: np.ndarray           # (M,) body LABEL (not idx), lower
    interface_hi: np.ndarray           # (M,) body LABEL, higher
    interface_r_throat: np.ndarray     # (M,) max EDT on the interface
    interface_voxel_count: np.ndarray  # (M,)
    interface_centroid: np.ndarray     # (M, 3) z,y,x


def compute_edt(segmented: np.ndarray, phase_id: int) -> np.ndarray:
    return ndi.distance_transform_edt(segmented == phase_id)


def compute_body_radii(edt: np.ndarray, body_array: np.ndarray, body_ids: np.ndarray) -> np.ndarray:
    """r_body per label in `body_ids`, vectorised via ndi.maximum(index=...)."""
    return np.asarray(ndi.maximum(edt, labels=body_array, index=body_ids))


def compute_interfaces(body_array: np.ndarray, edt: np.ndarray) -> GeometryData:
    """Vectorised interface extraction: for each axis, compare body_array to
    its neighbour shifted by one voxel; keep positions where both labels are
    nonzero and different; group by (min,max) label pair via an int64-packed
    key and np.unique + reduceat (no Python loop over voxels or pairs).

    Note: this is a strict 6-connectivity face-adjacency interface, distinct
    from berg_cc's own throat-voxel definition (particulate_claude.py
    _extract_one_pair), which uses `dilate(body==b1) & dilate(body==b2)` when
    `dilate_both=True` (the default) -- so a handful of berg_cc throats can
    have no matching 6-connectivity interface here (their bodies are close
    but not literally touching), and conversely a handful of 6-connectivity
    interfaces are absent from berg_cc's throat list, mostly because
    `skip_surface_surface` and `min_throat_size` filter them out upstream.
    Both directions are quantified in Task 3's acceptance test 2, not treated
    as errors.
    """
    all_lo, all_hi, all_eface, all_z, all_y, all_x = [], [], [], [], [], []
    for axis in range(3):
        a = body_array
        b = np.roll(a, -1, axis=axis)
        e_a = edt
        e_b = np.roll(edt, -1, axis=axis)
        valid = np.ones_like(a, dtype=bool)
        idx = [slice(None)] * 3
        idx[axis] = -1
        valid[tuple(idx)] = False

        both_nonzero = (a > 0) & (b > 0) & (a != b) & valid
        if not np.any(both_nonzero):
            continue
        la, lb = a[both_nonzero], b[both_nonzero]
        ea, eb = e_a[both_nonzero], e_b[both_nonzero]
        zs, ys, xs = np.where(both_nonzero)

        all_lo.append(np.minimum(la, lb))
        all_hi.append(np.maximum(la, lb))
        all_eface.append(np.minimum(ea, eb))
        all_z.append(zs); all_y.append(ys); all_x.append(xs)

    lo = np.concatenate(all_lo)
    hi = np.concatenate(all_hi)
    e_face = np.concatenate(all_eface)
    zc = np.concatenate(all_z); yc = np.concatenate(all_y); xc = np.concatenate(all_x)

    pair_key = (lo.astype(np.int64) << 32) | hi.astype(np.int64)
    order = np.argsort(pair_key, kind="stable")
    pk_sorted = pair_key[order]
    e_sorted = e_face[order]
    z_sorted, y_sorted, x_sorted = zc[order], yc[order], xc[order]

    unique_keys, start_idx, counts = np.unique(pk_sorted, return_index=True, return_counts=True)
    r_throat_vals = np.maximum.reduceat(e_sorted, start_idx)
    cz = np.add.reduceat(z_sorted.astype(np.float64), start_idx) / counts
    cy = np.add.reduceat(y_sorted.astype(np.float64), start_idx) / counts
    cx = np.add.reduceat(x_sorted.astype(np.float64), start_idx) / counts

    lo_u = (unique_keys >> 32).astype(np.int64)
    hi_u = (unique_keys & 0xFFFFFFFF).astype(np.int64)

    return GeometryData(
        edt=edt,
        r_body_by_idx=np.array([]),  # filled in by caller (needs body_ids ordering)
        interface_lo=lo_u,
        interface_hi=hi_u,
        interface_r_throat=r_throat_vals,
        interface_voxel_count=counts.astype(np.int64),
        interface_centroid=np.stack([cz, cy, cx], axis=1),
    )


def build_geometry(msd: MicrostructureData) -> GeometryData:
    edt = compute_edt(msd.segmented, msd.phase_id)
    body_ids = msd.berg["arrays"]["body_ids"]
    r_body = compute_body_radii(edt, msd.body_array, body_ids)
    geom = compute_interfaces(msd.body_array, edt)
    geom.r_body_by_idx = r_body
    return geom


def interface_lookup(geom: GeometryData) -> Dict[Tuple[int, int], int]:
    """(lo_label, hi_label) -> row index into geom.interface_* arrays."""
    return {
        (int(lo), int(hi)): i
        for i, (lo, hi) in enumerate(zip(geom.interface_lo, geom.interface_hi))
    }


# =============================================================================
# 3b. Cache (EDT + interface maps are expensive and deterministic)
# =============================================================================

def geometry_cache_path(cache_dir: Path, ms_name: str) -> Path:
    return Path(cache_dir) / f"{ms_name}_geometry.npz"


def save_geometry_cache(cache_dir: Path, ms_name: str, geom: GeometryData) -> Path:
    path = geometry_cache_path(cache_dir, ms_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        edt=geom.edt.astype(np.float32),
        r_body_by_idx=geom.r_body_by_idx,
        interface_lo=geom.interface_lo,
        interface_hi=geom.interface_hi,
        interface_r_throat=geom.interface_r_throat,
        interface_voxel_count=geom.interface_voxel_count,
        interface_centroid=geom.interface_centroid,
    )
    return path


def load_geometry_cache(cache_dir: Path, ms_name: str) -> Optional[GeometryData]:
    path = geometry_cache_path(cache_dir, ms_name)
    if not path.exists():
        return None
    d = np.load(path)
    return GeometryData(
        edt=d["edt"],
        r_body_by_idx=d["r_body_by_idx"],
        interface_lo=d["interface_lo"],
        interface_hi=d["interface_hi"],
        interface_r_throat=d["interface_r_throat"],
        interface_voxel_count=d["interface_voxel_count"],
        interface_centroid=d["interface_centroid"],
    )


def get_geometry(msd: MicrostructureData, cache_dir: Path, use_cache: bool = True) -> GeometryData:
    if use_cache:
        cached = load_geometry_cache(cache_dir, msd.name)
        if cached is not None:
            return cached
    geom = build_geometry(msd)
    if use_cache:
        save_geometry_cache(cache_dir, msd.name, geom)
    return geom


# =============================================================================
# 4. Throat table assembly
# =============================================================================

def classify_throats(C: np.ndarray, f: np.ndarray) -> np.ndarray:
    f_max = f.max() if len(f) and f.max() > 0 else 1.0
    busy = f >= F_BUSY_FRAC_OF_MAX * f_max
    idle = f < F_IDLE_CUT
    narrow = C < C_NARROW_CUT

    cls = np.full(len(C), "secondary_or_filler", dtype=object)
    cls[narrow & busy] = "bottleneck"
    cls[narrow & idle] = "geometric_only"
    cls[~narrow & busy] = "highway"
    # everything else stays secondary_or_filler (includes narrow&!busy&!idle, etc.)
    return cls


def build_throat_table(msd: MicrostructureData, geom: GeometryData) -> pd.DataFrame:
    arrays = msd.berg["arrays"]
    body_ids = arrays["body_ids"]
    conns = arrays["conns"]
    current = msd.berg["current"]
    total_current = msd.berg_metrics["total_current"]

    valid = (conns[:, 0] >= 0) & (conns[:, 1] >= 0)
    idx_i = conns[valid, 0]
    idx_j = conns[valid, 1]
    throat_row_idx = np.where(valid)[0]

    label_i = body_ids[idx_i]
    label_j = body_ids[idx_j]
    lo_label = np.minimum(label_i, label_j)
    hi_label = np.maximum(label_i, label_j)

    lookup = interface_lookup(geom)
    n = len(throat_row_idx)
    r_throat = np.full(n, np.nan)
    a_interface_voxels = np.full(n, np.nan)
    centroid = np.full((n, 3), np.nan)
    has_interface = np.zeros(n, dtype=bool)

    for k in range(n):
        key = (int(lo_label[k]), int(hi_label[k]))
        row = lookup.get(key)
        if row is not None:
            r_throat[k] = geom.interface_r_throat[row]
            a_interface_voxels[k] = geom.interface_voxel_count[row]
            centroid[k] = geom.interface_centroid[row]
            has_interface[k] = True

    r_body1 = geom.r_body_by_idx[idx_i]
    r_body2 = geom.r_body_by_idx[idx_j]
    C = r_throat / np.minimum(r_body1, r_body2)
    C_geomean = r_throat / np.sqrt(r_body1 * r_body2)

    G, cdata = berg_cc.calculate_conductances(arrays, sigma=msd.berg_config.get("sigma", 1.0), verbose=False)
    G = G[throat_row_idx]
    R1 = cdata["R1"][throat_row_idx]
    R_throat = cdata["R_throat"][throat_row_idx]
    R2 = cdata["R2"][throat_row_idx]
    R_edge = cdata["R_total"][throat_row_idx]

    I_throat = current[throat_row_idx]
    f = np.abs(I_throat) / total_current

    # Dissipation, two levels per constraint (c): edge-level (the whole
    # body1-throat-body2 series segment, physically what matters -- see
    # R_throat/R_edge often << 1 in this network) and throat-only (the
    # channel segment alone, for comparison). Same I_throat flows through
    # all three sub-resistors in series, so P_edge = I^2 R_edge decomposes
    # exactly as P1 + P_throat_only + P2.
    P_edge = I_throat**2 * R_edge
    P_throat_only = I_throat**2 * R_throat

    # Normalised to a fraction of total dissipated power (own level's total,
    # not each other's -- P_edge_frac sums to 1 over all throats, and so does
    # P_throat_only_frac, separately). Raw P is not comparable across
    # microstructures (scales with I_total^2, which varies by orders of
    # magnitude with domain size/porosity) and even within one
    # microstructure conflates "carries a lot of current" with "is a
    # bottleneck" -- P_frac makes it a unitless share, directly comparable to
    # f the same way f already is.
    P_edge_frac = P_edge / P_edge.sum() if P_edge.sum() > 0 else np.zeros_like(P_edge)
    P_throat_only_frac = (P_throat_only / P_throat_only.sum()
                          if P_throat_only.sum() > 0 else np.zeros_like(P_throat_only))

    # coordination number per body (for diagnostics, Task 4.3)
    N_bodies = len(body_ids)
    z = np.zeros(N_bodies, dtype=int)
    np.add.at(z, conns[valid, 0], 1)
    np.add.at(z, conns[valid, 1], 1)

    df = pd.DataFrame({
        "throat_row": throat_row_idx,
        "body1": label_i,
        "body2": label_j,
        "body1_idx": idx_i,
        "body2_idx": idx_j,
        "r_throat": r_throat,
        "r_body1": r_body1,
        "r_body2": r_body2,
        "C": C,
        "C_geomean": C_geomean,
        "has_interface": has_interface,
        "A_interface_voxels": a_interface_voxels,
        "geometric_throat_area": arrays["geometric_throat_area"][throat_row_idx],
        "throat_area": arrays["throat_area"][throat_row_idx],
        "is_fallback": arrays["is_fallback"][throat_row_idx],
        "A_body1": arrays["A_body1"][throat_row_idx],
        "A_body2": arrays["A_body2"][throat_row_idx],
        "L_body1": arrays["L_body1"][throat_row_idx],
        "L_throat": arrays["L_throat"][throat_row_idx],
        "L_body2": arrays["L_body2"][throat_row_idx],
        "G": G,
        "R1": R1,
        "R_throat": R_throat,
        "R2": R2,
        "R_edge": R_edge,
        "I_throat": I_throat,
        "f": f,
        "P_edge": P_edge,
        "P_throat_only": P_throat_only,
        "P_edge_frac": P_edge_frac,
        "P_throat_only_frac": P_throat_only_frac,
        "centroid_z": centroid[:, 0],
        "centroid_y": centroid[:, 1],
        "centroid_x": centroid[:, 2],
        "z_body1": z[idx_i],
        "z_body2": z[idx_j],
    })
    df["class"] = classify_throats(df["C"].values, df["f"].values)
    return df


# =============================================================================
# 5. Voxel-vs-network cross-check
# =============================================================================

def compute_inlet_plane_total_current(msd: MicrostructureData) -> float:
    """Total current through the domain, from the SAME J field, computed as
    I = D_eff * A_cross * delta_V / L (taufactor's own converged D_eff
    scaled to this domain's geometry and applied potential drop) rather than
    summed off any single voxel plane.

    A single-plane sum was tried first and rejected: on microstructure1
    (404 bodies, dense/well-connected network) every interior plane agrees
    to within 0.05% (std/mean = 5e-4) and matches D_eff*A*dV/L almost
    exactly, so a plane sum looks fine there. But on Coarse_microstructure_0
    (37 bodies, 8.5% porosity, a much sparser/more tortuous network) the
    SIGNED per-plane sum of J along the flow axis is not even
    consistently signed across the domain (ranges from -2.14 to +0.33
    depending on which plane), i.e. current recirculates through the
    off-axis directions between sparse pathways rather than flowing
    monotonically down the flow axis -- so no single plane is a reliable
    "total current" reference there. D_eff is a whole-domain, boundary-
    condition-derived scalar (see pipeline.py's run_taufactor) and is not
    subject to this per-plane noise.
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis = axis_map[msd.direction]
    shape = msd.segmented.shape
    cross_area = float(np.prod([shape[i] for i in range(3) if i != axis]))
    L = float(shape[axis])
    pot = msd.tau["potential_field"]
    delta_V = float(pot.max() - pot.min())
    D_eff = msd.tau_metrics["D_eff"]
    return D_eff * cross_area * delta_V / L


def compute_voxel_throat_currents(msd: MicrostructureData, geom: GeometryData, df: pd.DataFrame) -> np.ndarray:
    """For every interface voxel pair (from geom, keyed by axis-adjacency),
    take the J component normal to that face and sum, signed body1->body2
    matching df's (body1,body2) ordering. Returns f_voxel aligned to df rows.
    """
    body_array = msd.body_array
    Js = {0: msd.tau["J_x"], 1: msd.tau["J_y"], 2: msd.tau["J_z"]}

    pair_flux = {}  # (lo_label, hi_label) -> signed flux sum (lo->hi convention)
    for axis in range(3):
        a = body_array
        b = np.roll(a, -1, axis=axis)
        valid = np.ones_like(a, dtype=bool)
        idx = [slice(None)] * 3
        idx[axis] = -1
        valid[tuple(idx)] = False
        both_nonzero = (a > 0) & (b > 0) & (a != b) & valid
        if not np.any(both_nonzero):
            continue

        la, lb = a[both_nonzero], b[both_nonzero]
        lo = np.minimum(la, lb).astype(np.int64)
        hi = np.maximum(la, lb).astype(np.int64)
        # sign: +1 if a==lo (flux measured at face from a's voxel toward b),
        # i.e. positive J_axis at voxel a means flow toward larger index (a->b)
        sign = np.where(la == lo, 1.0, -1.0)

        J_here = Js[axis][both_nonzero]
        signed_flux = sign * J_here

        pair_key = (lo << 32) | hi
        order = np.argsort(pair_key, kind="stable")
        pk_sorted = pair_key[order]
        flux_sorted = signed_flux[order]
        uk, start_idx = np.unique(pk_sorted, return_index=True)
        counts = np.diff(np.append(start_idx, len(pk_sorted)))
        sums = np.add.reduceat(flux_sorted, start_idx)
        for k, s in zip(uk, sums):
            key = (int(k >> 32), int(k & 0xFFFFFFFF))
            pair_flux[key] = pair_flux.get(key, 0.0) + float(s)

    I_total_plane = compute_inlet_plane_total_current(msd)

    lo_label = np.minimum(df["body1"].values, df["body2"].values)
    hi_label = np.maximum(df["body1"].values, df["body2"].values)
    body1 = df["body1"].values

    I_voxel = np.full(len(df), np.nan)
    for k in range(len(df)):
        key = (int(lo_label[k]), int(hi_label[k]))
        if key not in pair_flux:
            continue
        raw = pair_flux[key]  # signed lo->hi
        # convert to body1->body2 convention
        I_voxel[k] = raw if body1[k] == lo_label[k] else -raw

    f_voxel = np.abs(I_voxel) / I_total_plane
    return f_voxel, I_voxel, I_total_plane


def check_divergence(msd: MicrostructureData) -> Dict[str, Any]:
    """Net flux out of each interior pore voxel, as a fraction of mean |J|.
    taufactor's J is a finite-difference gradient, not a staggered
    finite-volume flux, so this will not be machine zero."""
    pore = msd.tau["pore_mask"]
    Jx, Jy, Jz = msd.tau["J_x"], msd.tau["J_y"], msd.tau["J_z"]

    div = (
        (np.roll(Jx, -1, axis=0) - np.roll(Jx, 1, axis=0)) / 2.0 +
        (np.roll(Jy, -1, axis=1) - np.roll(Jy, 1, axis=1)) / 2.0 +
        (np.roll(Jz, -1, axis=2) - np.roll(Jz, 1, axis=2)) / 2.0
    )
    interior = pore.copy()
    interior[0, :, :] = interior[-1, :, :] = False
    interior[:, 0, :] = interior[:, -1, :] = False
    interior[:, :, 0] = interior[:, :, -1] = False

    mean_absJ = float(np.mean(np.sqrt(Jx[pore]**2 + Jy[pore]**2 + Jz[pore]**2)))
    div_interior = div[interior]
    return {
        "mean_abs_J": mean_absJ,
        "mean_abs_div": float(np.mean(np.abs(div_interior))),
        "median_abs_div": float(np.median(np.abs(div_interior))),
        "p90_abs_div": float(np.percentile(np.abs(div_interior), 90)),
        "mean_abs_div_over_mean_absJ": float(np.mean(np.abs(div_interior))) / mean_absJ if mean_absJ > 0 else np.nan,
    }


def cross_check_stats(df: pd.DataFrame) -> Dict[str, Any]:
    from scipy import stats as sstats
    sub = df.dropna(subset=["f_voxel"]).copy()
    spearman = sstats.spearmanr(sub["f_voxel"], sub["f"])
    pearson = sstats.pearsonr(sub["f_voxel"], sub["f"])

    busy = sub[sub["f"] > 0.01].copy()
    busy["rel_err"] = np.abs(busy["f_voxel"] - busy["f"]) / busy["f"]
    idle = sub[sub["f"] <= 0.01].copy()
    idle["abs_err"] = np.abs(idle["f_voxel"] - idle["f"])

    return {
        "n_matched": len(sub),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "n_busy_subset": len(busy),
        "median_rel_err_busy": float(busy["rel_err"].median()) if len(busy) else np.nan,
        "p90_rel_err_busy": float(busy["rel_err"].quantile(0.9)) if len(busy) else np.nan,
        "n_idle_subset": len(idle),
        "median_abs_err_idle": float(idle["abs_err"].median()) if len(idle) else np.nan,
        "p90_abs_err_idle": float(idle["abs_err"].quantile(0.9)) if len(idle) else np.nan,
        "pass": bool(spearman.statistic > 0.95 and (len(busy) == 0 or busy["rel_err"].median() < 0.10)),
    }
