"""
pipeline.py
===========

Per-microstructure pipeline: run taufactor, berg_voxel_fast, and berg_cc on
one microstructure folder, with version-based caching (cache_utils.py). ONE
version counter is shared across all three methods per microstructure: if
none of the three methods' configs changed since the latest version, running
again is a true no-op (nothing written, nothing recomputed). If at least one
changed, a new version is allocated, and WITHIN that version each method is
decided independently: a method whose config exactly matches an earlier
version's real run gets a small pointer file instead of being recomputed
("<method> already run in version X"); only methods with a genuinely new
config are actually run. See cache_utils.py's module docstring for the full
on-disk layout.

Expected folder layout (one per microstructure):

    <microstructure_dir>/
        segmented.npy    - 3D array, phase-labelled (any dtype); phase_id
                            selects the pore/conducting phase
        body_array.npy   - 3D int array, watershed-labelled pore bodies
                            (0 = solid), already computed upstream
        manifest.json    - OPTIONAL per-microstructure overrides (filenames,
                            phase_id, hyperparameters); see _load_manifest.
        latest_version.json, version<N>/...  - written by this module (see
                            cache_utils.py)

Thread/process budget: `cores_per_task` is threaded through explicitly to
every library that offers a runtime (not just import-time) thread-count
control -- torch.set_num_threads, numba.set_num_threads, and
particulate_claude's own `n_jobs` multiprocessing pool -- since relying on
OMP_NUM_THREADS/MKL_NUM_THREADS alone is only reliable if those env vars are
set before the relevant native libraries are first loaded in this process
(see run_batch.py, which sets them in a lightweight ProcessPoolExecutor
initializer before pipeline.py -- and therefore numpy/torch/numba -- is ever
imported in the worker).
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

import berg_cc
import berg_voxel_fast as bvf
import particulate_claude as pc

import cache_utils
from cache_utils import config_hash


def _set_runtime_thread_limits(n_cores: int) -> None:
    """Best-effort thread-count pinning via runtime APIs (import-order safe)."""
    n_cores = max(1, int(n_cores))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(var, str(n_cores))
    try:
        import numba
        numba.set_num_threads(n_cores)
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(n_cores)
    except Exception:
        pass


def _load_taufactor_warm_start(folder: Path, taufactor_version: Optional[int],
                                target_shape) -> Optional[np.ndarray]:
    """Load a prior taufactor run's saved potential_field, to seed
    berg_voxel's KCL solve initial guess (see run_berg_voxel's docstring).
    Requires that taufactor run have been saved with save_full=True at some
    point (its own or an earlier version's, via find_prior_real_run/
    resolve_real_dir) -- returns None (no warm start, not an error) if
    unavailable for any reason: no resolved taufactor version yet, no
    full_result.pkl there, or a shape mismatch against berg_voxel's own
    segmented array (possible if the two methods were ever pointed at
    different segmented_file/phase_id overrides).
    """
    if taufactor_version is None:
        return None
    real_dir = cache_utils.resolve_real_dir(folder, taufactor_version, "taufactor")
    full_path = real_dir / "full_result.pkl"
    if not full_path.exists():
        return None
    import pickle
    with open(full_path, "rb") as f:
        full = pickle.load(f)
    field = full.get("potential_field")
    if field is None or tuple(field.shape) != tuple(target_shape):
        return None
    return field


# =============================================================================
# Stage 1 -- taufactor + flux-reconciled constrictivity
# =============================================================================

def run_taufactor(segmented_binary: np.ndarray, config: Dict[str, Any],
                   verbose: bool = False, save_full: bool = False):
    """
    Run taufactor's Solver on a binary pore mask (1 = pore, 0 = solid) and
    compute the flux-reconciled tortuosity/constrictivity in addition to
    taufactor's own D_eff/tau.

    config keys: iter_limit, conv_crit, plot_interval (taufactor.Solver.solve
    kwargs; defaults match taufactor's own defaults unless overridden), axis.

    axis : taufactor.Solver ALWAYS solves along literal array axis 0 (see
    its own source: Nx = img.shape[0], the applied potential gradient and
    every flux/tau calculation run along that axis, with no parameter to
    change it -- confirmed by reading taufactor/taufactor.py directly).
    Unlike berg_voxel_fast (axis=) and berg_cc (direction=), which both
    accept any of the 3 array axes directly, this function has to fake that
    by moving `axis` to position 0 before solving (np.moveaxis), then
    moving every save_full array back to the original orientation before
    returning it, so its arrays still line up spatially with
    segmented.npy/body_array.npy as stored on disk. The returned scalar
    metrics need no such correction -- they're axis-order invariant once
    the correct flow axis has been solved along.

    Returns (metrics, full) where full is None unless save_full is True, in
    which case it holds the full potential field, pore mask, and per-voxel
    flux components (J_x/J_y/J_z, cell-centred) needed to re-derive or
    visualise the flux map -- not just its scalar averages.
    """
    import torch
    import taufactor as tau

    axis = config.get("axis", 0)
    work = np.moveaxis(segmented_binary, axis, 0) if axis != 0 else segmented_binary

    t0 = time.perf_counter()
    solver = tau.Solver(work, device="cpu")
    solver.solve(
        iter_limit=config.get("iter_limit", 10000),
        verbose=verbose,
        conv_crit=config.get("conv_crit", 0.01),
    )

    inner_field = solver.field[:, 1:-1, 1:-1, 1:-1].cpu().numpy()[0]
    torch_img = torch.tensor(solver.cpu_img, device=solver.device)
    pore_mask = (solver.return_mask(torch_img).cpu().numpy()[0] == 1)

    F_x = np.diff(inner_field, axis=0)
    F_y = np.diff(inner_field, axis=1)
    F_z = np.diff(inner_field, axis=2)

    mask_x = pore_mask[:-1, :, :] & pore_mask[1:, :, :]
    mask_y = pore_mask[:, :-1, :] & pore_mask[:, 1:, :]
    mask_z = pore_mask[:, :, :-1] & pore_mask[:, :, 1:]

    F_x[~mask_x] = 0.0
    F_y[~mask_y] = 0.0
    F_z[~mask_z] = 0.0

    J_x = (np.pad(F_x, ((1, 0), (0, 0), (0, 0))) + np.pad(F_x, ((0, 1), (0, 0), (0, 0)))) / 2.0
    J_y = (np.pad(F_y, ((0, 0), (1, 0), (0, 0))) + np.pad(F_y, ((0, 0), (0, 1), (0, 0)))) / 2.0
    J_z = (np.pad(F_z, ((0, 0), (0, 0), (1, 0))) + np.pad(F_z, ((0, 0), (0, 0), (0, 1)))) / 2.0

    J_x_pore = J_x[pore_mask]
    J_y_pore = J_y[pore_mask]
    J_z_pore = J_z[pore_mask]

    J_mag_pore = np.sqrt(J_x_pore ** 2 + J_y_pore ** 2 + J_z_pore ** 2)
    flux_average_pore = float(np.mean(J_mag_pore))
    flux_x_average_pore = float(np.abs(np.mean(J_x_pore)))
    tau_reconciled = flux_average_pore / flux_x_average_pore

    metrics = {
        "porosity": float(segmented_binary.mean()),
        "D_eff": float(solver.D_eff[0]),
        "tau_solver": float(solver.tau[0]),
        "tau_reconciled": float(tau_reconciled),
        "constrictivity": float(tau_reconciled ** 2 / solver.tau[0]),
        "elapsed_s": time.perf_counter() - t0,
    }

    if verbose:
        print(f"[taufactor] D_eff={metrics['D_eff']:.4f} tau={metrics['tau_solver']:.4f} "
              f"tau_reconciled={metrics['tau_reconciled']:.4f} "
              f"constrictivity={metrics['constrictivity']:.4f}")

    full = None
    if save_full:
        def _restore(arr):
            return np.moveaxis(arr, 0, axis) if axis != 0 else arr
        full = {
            "potential_field": _restore(inner_field),
            "pore_mask": _restore(pore_mask),
            "J_x": _restore(J_x), "J_y": _restore(J_y), "J_z": _restore(J_z),
        }

    return metrics, full


# =============================================================================
# Stage 2 -- berg_voxel_fast (voxel-network Berg pipeline)
# =============================================================================

def run_berg_voxel(segmented: np.ndarray, config: Dict[str, Any],
                    verbose: bool = False, save_full: bool = False,
                    warm_start_field: Optional[np.ndarray] = None):
    """
    config keys: any BergPNConfig field (axis, phase_id, sigma,
    conductance_mode, length_mode, decomposition_method, ...).

    warm_start_field : optional 3D array, same shape as `segmented`, used
    purely to seed berg_voxel_fast's iterative KCL solve's initial guess
    (never replaces the solve -- see solve_kcl's own docstring). Not a
    config key: it's an actual array, not a scalar hyperparameter, so it's
    passed as a real argument by run_microstructure (which loads it from a
    prior taufactor run's saved potential_field, if available -- see
    run_microstructure for exactly when that applies) rather than exposed
    through the JSON config dict.

    Returns (metrics, full) where full is None unless save_full is True. The
    full result intentionally does NOT include the per-streamtube list: at
    full voxel resolution there can be hundreds of thousands of streamtubes
    (measured ~245k on a 96^3 sample), and even a compressed columnar/CSR
    dump of that came out to ~930MB per microstructure per config version --
    too large to keep by default across a batch. `full` here is just the
    solved potential/current fields, which are cheap (a few MB) and enough
    to recompute or re-decompose streamtubes later if ever needed. If you
    need the actual per-streamtube paths at voxel resolution, call
    decompose_streamtubes() directly rather than going through this cache.
    """
    t0 = time.perf_counter()
    # segmented_file is folded into this config dict only so it participates
    # in the cache-key hash (see run_microstructure) -- BergPNConfig has no
    # such field, so it must be excluded here before unpacking as **kwargs.
    cfg_kwargs = {k: v for k, v in config.items() if k not in ("verbose", "segmented_file")}
    cfg = bvf.BergPNConfig(verbose=verbose, **cfg_kwargs)

    net = bvf.build_network(segmented, cfg)
    sol = bvf.solve_kcl(net, cfg, warm_start_field=warm_start_field)
    transport = bvf.effective_transport(net, sol, cfg)
    streamtubes, decomp_info = bvf.decompose_streamtubes(net, sol, transport, cfg)
    bvf.assign_node_volume(net, streamtubes, cfg)
    berg = bvf.compute_berg_quantities(net, sol, transport, streamtubes, cfg)

    metrics = {k: v for k, v in berg.items() if k != "streamtubes"}
    metrics.update({
        "decomposition_method": decomp_info["method"],
        "decomposition_engine": decomp_info["engine"],
        "decomposition_elapsed_s": decomp_info["elapsed_s"],
        "one_over_F": transport["one_over_F"],
        "F": transport["F"],
        "G_eff": transport["G_eff"],
        "L_sample": transport["L_sample"],
        "elapsed_s": time.perf_counter() - t0,
    })

    full = None
    if save_full:
        full = {
            "phi": sol["phi"],
            "edge_current": transport["edge_current"],
            "node_coords": net["coords"],
        }

    return metrics, full


# =============================================================================
# Stage 3 -- berg_cc (body/throat network Berg pipeline)
# =============================================================================

def run_berg_cc(body_array: np.ndarray, config: Dict[str, Any],
                 verbose: bool = False, n_jobs: int = 1, save_full: bool = False):
    """
    config keys:
        min_throat_size, dilate_both, skip_surface_surface
            -> particulate_claude.extract_throats_from_bodies_voxel_parallel
        area_method, alpha, split_volume_equal
            -> particulate_claude.calculate_network_effective_properties
        sigma, direction, delta_V, voxel_size, streamtube_method
            -> berg_cc.compute_all_berg

    `direction` ('x'/'y'/'z', default 'x') is the single source of truth for
    flow direction here: it's read once and passed to both
    extract_throats_from_bodies_voxel_parallel (which derives surface_axis
    from it, so is_surface detection follows the same axis as everything
    else) and berg_cc.compute_all_berg. An explicit `surface_axis` in config
    is still honoured as a legacy override, but only if it agrees with
    `direction` -- see particulate_claude._resolve_axis.

    Returns (metrics, full) where full is None unless save_full is True, in
    which case it holds the full per-streamtube list (node_path, throat_path,
    I_gamma, volume_shares, V_gamma, L_gamma for every streamtube -- on the
    body/throat network this is at most one per throat, so it's small: a few
    hundred to a few thousand entries, not the hundreds of thousands seen at
    voxel resolution) plus the network arrays, solved potential/current, and
    per-throat/per-segment ι² needed to reproduce every plot in Berg (2012).
    """
    t0 = time.perf_counter()
    direction = config.get("direction", "x")

    axis = pc.DIRECTION_TO_AXIS[direction]
    body_array, _, _ = pc.merge_boundary_only_bodies_iterative(body_array, axis=axis, verbose=verbose)
    body_array, _, _ = pc.merge_short_circuit_bodies(body_array, axis=axis, verbose=verbose)

    network = pc.extract_throats_from_bodies_voxel_parallel(
        body_array,
        min_throat_size=config.get("min_throat_size", 5),
        dilate_both=config.get("dilate_both", True),
        n_jobs=n_jobs,
        direction=direction,
        surface_axis=config.get("surface_axis"),
        skip_surface_surface=config.get("skip_surface_surface", True),
        verbose=verbose,
    )
    network, warnings = pc.calculate_network_effective_properties(
        network,
        area_method=config.get("area_method", "vox_projection"),
        alpha=config.get("alpha", 0.5),
        split_volume_equal=config.get("split_volume_equal", True),
        verbose=verbose,
    )

    result = berg_cc.compute_all_berg(
        network, body_array,
        area_method=config.get("area_method", "vox_projection"),
        sigma=config.get("sigma", 1.0),
        direction=direction,
        delta_V=config.get("delta_V", 1.0),
        voxel_size=config.get("voxel_size", 1.0),
        streamtube_method=config.get("streamtube_method", "greedy_fast"),
        verbose=verbose,
    )

    metrics = {
        "n_bodies": network.num_bodies,
        "n_throats": network.num_throats,
        "n_streamtubes": len(result["streamtubes"]),
        "n_fallback_throats": int(sum(1 for t in network.throats.values() if t.is_fallback)),
        "Omega": result["Omega"],
        "Omega_c": result["Omega_c"],
        "V_stuck_total": result["V_stuck_total"],
        "iota_sq_g": result["iota_sq_g"],
        "iota_sq_c": result["global_berg"]["iota_sq_c"],
        "tau_sq_c": result["global_berg"]["tau_sq_c"],
        "C_c": result["global_berg"]["C_c"],
        "consistency": result["global_berg"]["consistency"],
        "phi_total": result["phi_total"],
        "phi_c": result["phi_c"],
        "F_total": result["F_total"],
        "F_c": result["F_c"],
        "total_current": result["solution_data"]["total_current"],
        "effective_conductance": result["solution_data"]["effective_conductance"],
        "elapsed_s": time.perf_counter() - t0,
    }

    full = None
    if save_full:
        full = {
            "streamtubes": result["streamtubes"],
            "per_streamtube": result["per_streamtube"],
            "arrays": result["arrays"],
            "potential": result["potential"],
            "current": result["current"],
            "iota_sq": result["iota_sq"],
            "Phi_1t": result["Phi_1t"],
            "Phi_2t": result["Phi_2t"],
        }

    return metrics, full


# =============================================================================
# Orchestration -- one microstructure, all three methods, cached/versioned
# =============================================================================

MANIFEST_FILENAME = "manifest.json"


def _load_manifest(folder: Path) -> Dict[str, Any]:
    """
    Optional <folder>/manifest.json for microstructures that need to differ
    from the batch-wide defaults -- e.g. a different phase_id, a different
    flow axis, different array filenames, or a hyperparameter override for
    just this sample. Absent entirely is fine; every key is optional.
    Example -- a sample whose premesh/body_array was generated with the flow
    axis along y instead of the batch default x:

        {
          "segmented_file": "segmented_premesh.npy",
          "axis": 1,
          "berg_cc_config": {"min_throat_size": 8}
        }

    Any *_config block here is merged one level deep on top of the run's
    global config (this microstructure's keys win, everything else from the
    global config is kept) -- so you only need to specify what's different
    for this sample, not repeat the whole config.
    """
    path = folder / MANIFEST_FILENAME
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def run_microstructure(
    folder,
    cores_per_task: int,
    phase_id: int = 1,
    axis: int = 0,
    segmented_file: str = "segmented.npy",
    body_array_file: str = "body_array.npy",
    taufactor_config: Optional[Dict[str, Any]] = None,
    berg_voxel_config: Optional[Dict[str, Any]] = None,
    berg_cc_config: Optional[Dict[str, Any]] = None,
    save_full: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run (or resolve from a cached version) taufactor + berg_voxel_fast +
    berg_cc for one microstructure folder, sharing a single version counter
    across all three methods (see cache_utils.py's module docstring). If
    every method's config matches the latest existing version exactly, this
    is a true no-op: nothing recomputed, nothing written, the existing
    version's summary is returned as-is. Otherwise a new version is
    allocated, and each method is independently either pointed at an
    earlier version with a matching config (no recompute) or actually run
    and saved fresh into the new version.

    `phase_id`/`axis`/`segmented_file`/`body_array_file`/`taufactor_config`/
    `berg_voxel_config`/`berg_cc_config` are the batch-wide defaults (e.g.
    from run_batch.py's CLI flags); a per-microstructure manifest.json (see
    _load_manifest) can override any of them for just this folder.

    `axis` is the single flow-direction knob shared by all three methods by
    default (0/1/2 for x/y/z, matching berg_voxel_fast's own `axis` and
    berg_cc's `direction` conventions) -- it's folded into each method's own
    config below (taufactor_config["axis"], berg_voxel_config["axis"],
    berg_cc_config["direction"], berg_cc_config["surface_axis"]) via
    setdefault, so setting just this one top-level value keeps all three
    methods solving along the same physical axis without having to remember
    to set it four separate ways. Any method's own config can still
    override it individually if a genuine per-method difference is ever
    needed.
    """
    _set_runtime_thread_limits(cores_per_task)

    folder = Path(folder)
    manifest = _load_manifest(folder)

    phase_id = manifest.get("phase_id", phase_id)
    axis = manifest.get("axis", axis)
    segmented_file = manifest.get("segmented_file", segmented_file)
    body_array_file = manifest.get("body_array_file", body_array_file)

    taufactor_config = {**(taufactor_config or {}), **manifest.get("taufactor_config", {})}
    berg_voxel_config = {**(berg_voxel_config or {}), **manifest.get("berg_voxel_config", {})}
    berg_cc_config = {**(berg_cc_config or {}), **manifest.get("berg_cc_config", {})}

    # phase_id/axis/input filenames determine the actual data and direction
    # each stage sees, so they must be part of the cache key even though
    # taufactor/berg_cc don't take them as literal solve kwargs the same way
    # berg_voxel does -- otherwise two runs that differ only in phase_id,
    # axis, or a swapped-in array would collide on one cache entry. berg_cc
    # never touches segmented.npy/phase_id (it works directly off the
    # already-labelled body_array), so it only keys on its own input file
    # plus its direction (derived from axis).
    #
    # We only ever set berg_cc_config["direction"] here, not "surface_axis":
    # run_berg_cc/extract_throats_from_bodies_voxel_parallel now derive
    # surface_axis from direction themselves (particulate_claude._resolve_axis),
    # using the same {0:'x',1:'y',2:'z'} convention as taufactor/berg_voxel's
    # `axis`, so this holds whether run_berg_cc is called from here or
    # directly -- no separate surface_axis to keep in sync. (A real crash on
    # Coarse_microstructure_0 once came from surface_axis being set
    # independently and going stale relative to direction; that class of bug
    # is now structurally impossible since surface_axis is derived, not
    # passed around.)
    taufactor_config.setdefault("phase_id", phase_id)
    taufactor_config.setdefault("axis", axis)
    taufactor_config.setdefault("segmented_file", segmented_file)
    berg_voxel_config.setdefault("phase_id", phase_id)
    berg_voxel_config.setdefault("axis", axis)
    berg_voxel_config.setdefault("segmented_file", segmented_file)
    berg_cc_config.setdefault("direction", pc.AXIS_TO_DIRECTION[axis])
    berg_cc_config.setdefault("body_array_file", body_array_file)

    segmented = np.load(folder / segmented_file)
    body_array = np.load(folder / body_array_file)
    segmented_binary = (segmented == phase_id).astype(np.uint8)

    configs = {
        "taufactor": taufactor_config,
        "berg_voxel": berg_voxel_config,
        "berg_cc": berg_cc_config,
    }
    current_hashes = {method: config_hash(cfg) for method, cfg in configs.items()}

    # ── Whole-version no-op check ──────────────────────────────────────────
    # If every method's config already matches what's in effect at the
    # latest existing version, this run changes nothing: return that
    # version's summary verbatim, write nothing new to disk.
    latest_v = cache_utils.read_latest_version(folder)
    if latest_v is not None and latest_v in cache_utils.list_versions(folder):
        all_match = all(
            current_hashes[m] == cache_utils.effective_config_hash(folder, latest_v, m)
            for m in configs
        )
        if all_match:
            full_ok = (not save_full) or all(
                (cache_utils.resolve_real_dir(folder, latest_v, m) / "full_result.pkl").exists()
                for m in configs
            )
            if full_ok:
                return cache_utils.load_version_summary(folder, latest_v)

    # ── Allocate a new version; decide each method independently ──────────
    new_v = cache_utils.next_version_number(folder)
    summary: Dict[str, Any] = {"microstructure": folder.name, "version": new_v}
    resolved_versions: Dict[str, int] = {}

    runners = {
        "taufactor": lambda: run_taufactor(segmented_binary, taufactor_config,
                                            verbose=verbose, save_full=save_full),
        "berg_cc": lambda: run_berg_cc(body_array, berg_cc_config, verbose=verbose,
                                        n_jobs=cores_per_task, save_full=save_full),
    }

    for method, cfg in configs.items():
        h = current_hashes[method]
        prior = cache_utils.find_prior_real_run(
            folder, method, h, before_version=new_v, require_full=save_full)

        if prior is not None:
            prior_version, prior_dir = prior
            cache_utils.save_pointer(folder, new_v, method, prior_version, h)
            metrics = cache_utils.load_metrics(prior_dir)
            metrics["_cache_hit"] = True
            metrics["_resolved_version"] = prior_version
        else:
            method_dir = cache_utils.version_dir(folder, new_v) / method
            method_dir.mkdir(parents=True, exist_ok=True)
            log_path = method_dir / "log.txt"
            with open(log_path, "w") as log_f, \
                 contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
                try:
                    if method == "berg_voxel":
                        # Warm-start berg_voxel's KCL solve from taufactor's
                        # saved potential_field, if taufactor has already
                        # been resolved this round (dict iteration order
                        # guarantees "taufactor" runs before "berg_voxel")
                        # and was saved with save_full=True at some point.
                        # Requires save_full=True on THIS run too (no field
                        # would be loadable to pass through otherwise).
                        warm_start_field = None
                        if save_full:
                            warm_start_field = _load_taufactor_warm_start(
                                folder, resolved_versions.get("taufactor"),
                                target_shape=segmented.shape)
                        metrics, full = run_berg_voxel(
                            segmented, berg_voxel_config, verbose=verbose,
                            save_full=save_full, warm_start_field=warm_start_field)
                    else:
                        metrics, full = runners[method]()
                except Exception:
                    traceback.print_exc(file=log_f)
                    raise
            cache_utils.save_real_run(folder, new_v, method, cfg, metrics, full_result=full)
            metrics["_cache_hit"] = False
            metrics["_resolved_version"] = new_v

        resolved_versions[method] = metrics["_resolved_version"]
        metrics["_version_hash"] = h
        summary[method] = metrics

    cache_utils.save_version_summary(folder, new_v, summary)
    cache_utils.write_latest_version(folder, new_v)

    return summary
