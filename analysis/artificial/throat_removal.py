"""
throat_removal.py
==================

What-if experiment: remove "problematic" (bottleneck) throats from a
body/throat pore network, recompute Berg effective geometry and
coordination numbers for the surviving connections, re-run the berg_cc
solve, and report how the effective transport properties (tau_sq_c, C_c,
effective_conductance, phi_total, ...) change relative to the unmodified
network.

Read-only with respect to the existing pipeline: never writes into
`data/<microstructure>/version<N>/`, and never mutates a caller-supplied
PoreNetwork in place (remove_throats works on a deep copy). Reuses
analysis/throat_analysis.py's geometry machinery and
analysis/bottleneck_overlap.py's overlap definition, plus the root-level
particulate_claude.py / berg_cc.py solver stack directly.

Default criterion ("bottleneck"): the geometric-bottleneck /
transport-bottleneck OVERLAP from bottleneck_overlap.py -- bottom 10% of
C_geomean (narrow relative to both neighbour bodies) AND top 10% of
P_throat_only_frac (throat-channel dissipated-power share), both by rank.
This intentionally does NOT use throat_analysis.classify_throats' "f"
(current fraction) based label, since analysis/REPORT.md documents f as
having non-trivial cross-check error against voxel-resolution ground
truth (median relative error 33-81%, Spearman 0.77-0.79 vs a 0.95
target); P_throat_only_frac comes from the same solved current but
combined with resistance into a dissipation ranking, which is what
bottleneck_overlap.py already uses for its "transport bottleneck" set.
`criterion` is a plain string/callable switch so any other rule (a
different N, C instead of C_geomean, P_edge_frac instead of
P_throat_only_frac, ...) can be swapped in without touching the rest of
this module.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd

import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import berg_cc
import particulate_claude as pc

_ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(_ANALYSIS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_ANALYSIS_DIR))

import throat_analysis as ta


# =============================================================================
# Network mutation
# =============================================================================

def remove_throats(network: "pc.PoreNetwork", throat_ids: Iterable[int]) -> "pc.PoreNetwork":
    """Return a deep copy of `network` with the given throats removed and
    each endpoint body's connected_bodies/connected_throats updated to
    match. Bodies that end up with zero remaining throats are kept in
    network.bodies (coordination_number becomes 0) -- they are excluded
    from the solve automatically by berg_cc's boundary-connectivity
    filter, not by deleting them here.

    Does not touch PoreBody.volume_voxels: raw voxel volume comes from the
    segmented body_array and is independent of throat connectivity.
    """
    pruned = copy.deepcopy(network)
    throat_ids = list(throat_ids)

    for tid in throat_ids:
        throat = pruned.throats.get(tid)
        if throat is None:
            continue
        b1_id, b2_id = throat.body1_id, throat.body2_id

        for bid, other_id in ((b1_id, b2_id), (b2_id, b1_id)):
            body = pruned.bodies.get(bid)
            if body is None:
                continue
            if tid in body.connected_throats:
                body.connected_throats.remove(tid)
            if other_id in body.connected_bodies:
                body.connected_bodies.remove(other_id)
            body.connections_to_throats.pop(tid, None)

        del pruned.throats[tid]

    return pruned


# =============================================================================
# Bottleneck identification (reuses analysis/throat_analysis.py)
# =============================================================================

def _make_msd_stub(network: "pc.PoreNetwork", body_array: np.ndarray,
                    segmented: np.ndarray, phase_id: int, direction: str,
                    berg_config: Dict[str, Any], berg_result: Dict[str, Any]) -> "ta.MicrostructureData":
    """Build a throat_analysis.MicrostructureData in memory, wrapping an
    already-computed berg_cc result instead of loading full_result.pkl
    from disk -- build_throat_table only reads the fields set here.
    """
    berg = {
        "arrays": berg_result["arrays"],
        "current": berg_result["current"],
    }
    berg_metrics = {
        "total_current": berg_result["solution_data"]["total_current"],
    }
    return ta.MicrostructureData(
        name="in_memory", ms_dir=Path("."), phase_id=phase_id, direction=direction,
        berg=berg, berg_metrics=berg_metrics, berg_config=berg_config,
        tau={}, tau_metrics={},
        segmented=segmented, body_array=body_array,
    )


def rank_criterion(
    column: str,
    percentile: float,
    tail: str = "bottom",
    column2: Optional[str] = None,
    percentile2: Optional[float] = None,
    tail2: str = "top",
    combine: str = "and",
) -> Callable[[pd.DataFrame], np.ndarray]:
    """Build a criterion callable for identify_bottleneck_throats /
    prune_and_recompute out of simple percentile rules, instead of writing
    a lambda by hand each time.

    Single-column use (e.g. "just bottom 5% of C", or "just top 10% of
    P_throat_only_frac" -- edge Joule heating I^2*R_edge):
        rank_criterion("C", percentile=5, tail="bottom")
        rank_criterion("P_edge_frac", percentile=10, tail="top")

    Two-column overlap use (e.g. reproduce the default "bottleneck" set --
    bottom 10% C_geomean AND top 10% P_throat_only_frac):
        rank_criterion("C_geomean", 10, "bottom",
                        "P_throat_only_frac", 10, "top", combine="and")

    Parameters
    ----------
    column : str
        A column in the throat table (e.g. "C", "C_geomean", "f", "G",
        "P_edge_frac", "P_throat_only_frac", "R_edge", ...).
    percentile : float
        0-100. With tail="bottom", selects throats with `column` at or
        below this percentile (the narrowest/smallest N). With
        tail="top", selects at or above the (100 - percentile)th
        percentile (the largest N).
    tail : str
        "bottom" or "top".
    column2, percentile2, tail2 : optional
        A second rule to combine with the first, e.g. bottom-C AND
        top-P for a geometric+transport bottleneck overlap. percentile2
        defaults to `percentile` and tail2 defaults to "top" (the common
        "narrow AND high-dissipation" pattern) if column2 is given.
    combine : str
        "and" (intersection, default) or "or" (union) when column2 is
        given; ignored otherwise.

    Returns
    -------
    A callable(df) -> bool array suitable for the `criterion` argument of
    identify_bottleneck_throats / prune_and_recompute.
    """
    def _mask(df: pd.DataFrame, col: str, pct: float, tl: str) -> np.ndarray:
        valid = df[col].notna()
        vals = df[col]
        if tl == "bottom":
            cutoff = vals[valid].quantile(pct / 100.0)
            m = valid & (vals <= cutoff)
        elif tl == "top":
            cutoff = vals[valid].quantile(1.0 - pct / 100.0)
            m = valid & (vals >= cutoff)
        else:
            raise ValueError(f"tail must be 'bottom' or 'top', got {tl!r}")
        return m.to_numpy()

    def criterion(df: pd.DataFrame) -> np.ndarray:
        mask = _mask(df, column, percentile, tail)
        if column2 is not None:
            pct2 = percentile if percentile2 is None else percentile2
            mask2 = _mask(df, column2, pct2, tail2)
            if combine == "and":
                mask = mask & mask2
            elif combine == "or":
                mask = mask | mask2
            else:
                raise ValueError(f"combine must be 'and' or 'or', got {combine!r}")
        return mask

    return criterion


def random_criterion(
    percentile: Optional[float] = None,
    seed: Optional[int] = None,
    *,
    n: Optional[int] = None,
    match: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
) -> Callable[[pd.DataFrame], np.ndarray]:
    """Build a criterion that removes a uniformly random set of throats,
    regardless of any geometric/transport property -- a null/control
    comparison against rank_criterion's targeted selections. Exactly one
    of `percentile`, `n`, or `match` must be given, to say how many
    throats to remove:

    - `percentile`: remove round(percentile/100 * n_throats) throats
      (same rounding convention as rank_criterion).
    - `n`: remove exactly this many throats.
    - `match`: another criterion callable(df) -> bool array (e.g. from
      rank_criterion) -- remove the SAME NUMBER of throats it would have
      selected, for an apples-to-apples "my targeted rule removed K
      throats, here's a random K instead" comparison. This runs `match`
      against the same df to count its selection, then draws a random K
      independently (a different, non-overlapping-by-construction random
      set -- not "K throats from match's own selection").

    Example (the common case -- match a targeted rule's count exactly):
        targeted = rank_criterion("C_geomean", 10, "bottom",
                                   "P_throat_only_frac", 10, "top")
        random_matched = random_criterion(match=targeted, seed=0)
        # both criteria remove the identical NUMBER of throats when run
        # on the same throat table, so results are directly comparable.

    Parameters
    ----------
    seed : int, optional
        Seed for reproducibility. None (default) draws a different random
        set each call -- pass a fixed seed to compare the SAME random
        removal across multiple criteria/metrics, or different seeds to
        get a distribution of random-removal outcomes (see
        analysis/notebooks/throat_pruning_experiment.ipynb for a
        multi-seed sweep pattern).

    Returns
    -------
    A callable(df) -> bool array suitable for the `criterion` argument of
    identify_bottleneck_throats / prune_and_recompute / run_on_microstructures.
    """
    n_given = sum(x is not None for x in (percentile, n, match))
    if n_given != 1:
        raise ValueError("Pass exactly one of percentile, n, or match")

    def criterion(df: pd.DataFrame) -> np.ndarray:
        total = len(df)
        if match is not None:
            n_remove = int(np.asarray(match(df), dtype=bool).sum())
        elif n is not None:
            n_remove = n
        else:
            n_remove = max(1, round(percentile / 100.0 * total))
        n_remove = min(n_remove, total)

        rng = np.random.default_rng(seed)
        chosen = rng.choice(total, size=n_remove, replace=False)
        mask = np.zeros(total, dtype=bool)
        mask[chosen] = True
        return mask

    return criterion


def identify_bottleneck_throats(
    network: "pc.PoreNetwork",
    body_array: np.ndarray,
    segmented: np.ndarray,
    phase_id: int,
    direction: str,
    berg_config: Dict[str, Any],
    berg_result: Dict[str, Any],
    criterion: Union[str, Callable[[pd.DataFrame], np.ndarray]] = "bottleneck",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Build the per-throat table (constriction ratio C/C_geomean, current
    fraction f, dissipated-power fractions, conductance) via
    throat_analysis.build_throat_table, then select throat IDs to remove
    according to `criterion`:

      - "bottleneck" (default): geometric bottleneck ∩ transport bottleneck,
        matching analysis/bottleneck_overlap.py's overlap definition --
        bottom 10% of C_geomean (narrowest relative to both neighbour
        bodies) AND top 10% of P_throat_only_frac (throat-channel-only
        dissipated power share), both by rank (nsmallest/nlargest N,
        N = round(0.10 * n_throats_with_valid_C_geomean)). This is the
        criterion that gave 7 throats on microstructure1.
      - callable: called as criterion(df) -> boolean array over df's rows,
        for a caller-supplied rule (e.g. a different N, C instead of
        C_geomean, or P_edge_frac instead of P_throat_only_frac).

    Returns (throat_ids_to_remove, full_table). `full_table` has one row
    per throat with a "throat_id" column (arrays["throat_ids"][throat_row])
    mapping each row back to network.throats keys, in addition to the
    columns build_throat_table already provides (C, C_geomean, f,
    P_edge_frac, P_throat_only_frac, G, ...).
    """
    msd = _make_msd_stub(network, body_array, segmented, phase_id, direction,
                          berg_config, berg_result)
    geom = ta.build_geometry(msd)
    df = ta.build_throat_table(msd, geom)
    df = df.copy()
    df["throat_id"] = berg_result["arrays"]["throat_ids"][df["throat_row"].to_numpy()]

    if criterion == "bottleneck":
        default_criterion = rank_criterion(
            "C_geomean", 10, "bottom",
            "P_throat_only_frac", 10, "top", combine="and",
        )
        mask = default_criterion(df)
    elif callable(criterion):
        mask = np.asarray(criterion(df), dtype=bool)
    else:
        raise ValueError(f"Unknown criterion {criterion!r}; use 'bottleneck' or a callable(df) -> bool array")

    return df.loc[mask, "throat_id"].to_numpy(), df


# =============================================================================
# Main entry point
# =============================================================================

_METRIC_KEYS = (
    "n_bodies", "n_throats", "n_streamtubes", "n_fallback_throats",
    "Omega", "Omega_c", "V_stuck_total",
    "iota_sq_g", "iota_sq_c", "tau_sq_c", "C_c", "consistency",
    "phi_total", "phi_c", "F_total", "F_c",
    "total_current", "effective_conductance",
    "inv_tau_sq_c", "inv_C_c", "inv_F_c",
)


def _safe_inv(x: float) -> float:
    return 1.0 / x if x not in (0.0, None) and not np.isnan(x) else np.nan


def _metrics_from_result(network: "pc.PoreNetwork", result: Dict[str, Any]) -> Dict[str, Any]:
    tau_sq_c = result["global_berg"]["tau_sq_c"]
    C_c = result["global_berg"]["C_c"]
    F_c = result["F_c"]
    return {
        "n_bodies": network.num_bodies,
        "n_throats": network.num_throats,
        "n_streamtubes": len(result["streamtubes"]),
        "n_fallback_throats": int(sum(1 for t in network.throats.values() if t.is_fallback)),
        "Omega": result["Omega"],
        "Omega_c": result["Omega_c"],
        "V_stuck_total": result["V_stuck_total"],
        "iota_sq_g": result["iota_sq_g"],
        "iota_sq_c": result["global_berg"]["iota_sq_c"],
        "tau_sq_c": tau_sq_c,
        "C_c": C_c,
        "consistency": result["global_berg"]["consistency"],
        "phi_total": result["phi_total"],
        "phi_c": result["phi_c"],
        "F_total": result["F_total"],
        "F_c": F_c,
        "total_current": result["solution_data"]["total_current"],
        "effective_conductance": result["solution_data"]["effective_conductance"],
        # NOTE on sign conventions:
        #   tau_sq_c : Berg's tau_sq_c is ~(L_sample/L_path)^2, inverted from
        #              the traditional (L_path/L_sample)^2 >= 1 convention --
        #              it sits <=1, ideal (straight) at 1, and LOWER means
        #              MORE tortuous. So inv_tau_sq_c >= 1, ideal at 1 (its
        #              min), and HIGHER inv_tau_sq_c = worse (more tortuous).
        #   C_c      : C_c >= 1, ideal (unconstricted) at 1, and HIGHER C_c
        #              means MORE constricted -- already on the traditional
        #              "higher = worse" scale, no inversion needed to read
        #              it that way. inv_C_c <= 1, ideal at 1 (its max), and
        #              LOWER inv_C_c = worse (more constricted).
        #   F_c      : higher = better sigma_eff, so inv_F_c: higher = worse.
        # Net: inv_tau_sq_c and inv_F_c are "higher = worse"; inv_C_c is
        # "lower = worse" (ideal/max = 1). Read C_c itself directly as
        # "higher = worse" without inverting.
        "inv_tau_sq_c": _safe_inv(tau_sq_c),
        "inv_C_c": _safe_inv(C_c),
        "inv_F_c": _safe_inv(F_c),
    }


def prune_and_recompute(
    body_array: np.ndarray,
    segmented: np.ndarray,
    config: Dict[str, Any],
    criterion: Union[str, Callable[[pd.DataFrame], np.ndarray]] = "bottleneck",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Build a network from `body_array`, identify bottleneck throats, remove
    them, recompute effective geometry + coordination numbers, and re-solve
    -- returning baseline vs. pruned metrics side by side.

    `config` has the same shape pipeline.run_berg_cc expects:
        min_throat_size, dilate_both, skip_surface_surface
            -> particulate_claude.extract_throats_from_bodies_voxel_parallel
        area_method, alpha, split_volume_equal
            -> particulate_claude.calculate_network_effective_properties
        sigma, direction, delta_V, voxel_size, streamtube_method
            -> berg_cc.compute_all_berg
    `segmented` is the phase-labeled voxel array (needed only for the EDT
    geometry underlying bottleneck classification); `config["phase_id"]`
    selects which label is the pore phase (default 3, matching
    particulate_claude's convention).

    Returns a dict with keys "baseline", "pruned" (each a metrics dict, see
    _METRIC_KEYS; "pruned" is None if removal disconnected the network --
    see "solve_failed" below), "delta" (pruned - baseline per numeric key,
    empty if the pruned solve failed), "n_throats_removed",
    "n_bodies_newly_isolated", "removed_throat_ids", "throat_table" (the
    classification table used for selection), and "solve_failed" (the
    exception message if berg_cc's solve raised on the pruned network --
    this happens when removing the selected throats disconnects inlet from
    outlet, e.g. the default "bottleneck" criterion can select a large
    enough fraction of throats to sever percolation; a stricter or
    conductance-aware criterion may be needed in that case).
    """
    direction = config.get("direction", "x")
    phase_id = config.get("phase_id", 3)

    network = pc.extract_throats_from_bodies_voxel_parallel(
        body_array,
        min_throat_size=config.get("min_throat_size", 5),
        dilate_both=config.get("dilate_both", True),
        n_jobs=config.get("n_jobs", 16),
        direction=direction,
        surface_axis=config.get("surface_axis"),
        skip_surface_surface=config.get("skip_surface_surface", True),
        verbose=verbose,
    )
    network, _warnings = pc.calculate_network_effective_properties(
        network,
        area_method=config.get("area_method", "vox_projection"),
        alpha=config.get("alpha", 0.5),
        split_volume_equal=config.get("split_volume_equal", True),
        verbose=verbose,
    )

    baseline_result = berg_cc.compute_all_berg(
        network, body_array,
        area_method=config.get("area_method", "vox_projection"),
        sigma=config.get("sigma", 1.0),
        direction=direction,
        delta_V=config.get("delta_V", 1.0),
        voxel_size=config.get("voxel_size", 1.0),
        streamtube_method=config.get("streamtube_method", "greedy_fast"),
        verbose=verbose,
    )
    baseline_metrics = _metrics_from_result(network, baseline_result)
    baseline_coord = network.body_coordination_numbers

    removed_throat_ids, throat_table = identify_bottleneck_throats(
        network, body_array, segmented, phase_id, direction,
        config, baseline_result, criterion=criterion,
    )

    pruned_network = remove_throats(network, removed_throat_ids)
    pruned_network, _warnings = pc.calculate_network_effective_properties(
        pruned_network,
        area_method=config.get("area_method", "vox_projection"),
        alpha=config.get("alpha", 0.5),
        split_volume_equal=config.get("split_volume_equal", True),
        verbose=verbose,
    )

    pruned_coord = pruned_network.body_coordination_numbers
    n_newly_isolated = int(np.sum((baseline_coord > 0) & (pruned_coord == 0)))

    try:
        pruned_result = berg_cc.compute_all_berg(
            pruned_network, body_array,
            area_method=config.get("area_method", "vox_projection"),
            sigma=config.get("sigma", 1.0),
            direction=direction,
            delta_V=config.get("delta_V", 1.0),
            voxel_size=config.get("voxel_size", 1.0),
            streamtube_method=config.get("streamtube_method", "greedy_fast"),
            verbose=verbose,
        )
    except RuntimeError as exc:
        return {
            "baseline": baseline_metrics,
            "pruned": None,
            "delta": {},
            "n_throats_removed": len(removed_throat_ids),
            "n_bodies_newly_isolated": n_newly_isolated,
            "removed_throat_ids": removed_throat_ids,
            "throat_table": throat_table,
            "solve_failed": str(exc),
        }

    pruned_metrics = _metrics_from_result(pruned_network, pruned_result)

    delta = {
        k: (pruned_metrics[k] - baseline_metrics[k])
        for k in _METRIC_KEYS
        if isinstance(baseline_metrics[k], (int, float, np.integer, np.floating))
    }

    return {
        "baseline": baseline_metrics,
        "pruned": pruned_metrics,
        "delta": delta,
        "n_throats_removed": len(removed_throat_ids),
        "n_bodies_newly_isolated": n_newly_isolated,
        "removed_throat_ids": removed_throat_ids,
        "throat_table": throat_table,
        "solve_failed": None,
    }


# =============================================================================
# Multi-microstructure sweep
# =============================================================================

def run_on_microstructures(
    microstructures: Dict[str, Dict[str, Any]],
    criterion: Union[str, Callable[[pd.DataFrame], np.ndarray]] = "bottleneck",
    verbose: bool = False,
    skip_errors: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Run prune_and_recompute on several microstructures with the same
    criterion and collect the results into one tidy comparison table.

    Parameters
    ----------
    microstructures : dict
        {ms_name: {"body_array": ..., "segmented": ..., "config": {...}}}
        -- one entry per microstructure, each with its own arrays and its
        own berg_cc-shaped config (direction/surface_axis/phase_id/
        min_throat_size typically differ per microstructure -- match each
        one's cached data/<ms_name>/version*/berg_cc/config.json).
    criterion : same as prune_and_recompute's `criterion`, applied
        identically to every microstructure in the sweep.
    skip_errors : bool
        If True (default), a microstructure whose prune_and_recompute call
        raises (bad/mismatched arrays, a config error, an unexpected solver
        exception -- distinct from a disconnected-network solve, which
        prune_and_recompute already reports via "solve_failed" without
        raising) is skipped: its row gets "errored"=True and an "error"
        message, all baseline_/pruned_/delta_ columns are NaN, and it is
        NOT added to raw_results. The rest of the sweep still runs. If
        False, the exception propagates immediately (stops the sweep).

    Returns
    -------
    (summary_df, raw_results) where summary_df has one row per
    microstructure with columns "ms_name", "errored", "error",
    "solve_failed", "n_throats_removed", "n_bodies_newly_isolated", and
    baseline_<metric>/pruned_<metric>/delta_<metric> for every key in
    _METRIC_KEYS (NaN for pruned_*/delta_* if that microstructure's solve
    failed or it errored), and raw_results maps ms_name -> the full
    prune_and_recompute return dict for every microstructure that did NOT
    error (for per-microstructure follow-up, e.g. inspecting throat_table
    or removed_throat_ids).
    """
    raw_results: Dict[str, Dict[str, Any]] = {}
    rows = []

    for ms_name, ms in microstructures.items():
        try:
            result = prune_and_recompute(
                ms["body_array"], ms["segmented"], ms["config"],
                criterion=criterion, verbose=verbose,
            )
        except Exception as exc:
            if not skip_errors:
                raise
            rows.append({
                "ms_name": ms_name,
                "errored": True,
                "error": f"{type(exc).__name__}: {exc}",
                "solve_failed": False,
                "n_throats_removed": np.nan,
                "n_bodies_newly_isolated": np.nan,
                **{f"baseline_{key}": np.nan for key in _METRIC_KEYS},
                **{f"pruned_{key}": np.nan for key in _METRIC_KEYS},
                **{f"delta_{key}": np.nan for key in _METRIC_KEYS},
            })
            continue

        raw_results[ms_name] = result

        row: Dict[str, Any] = {
            "ms_name": ms_name,
            "errored": False,
            "error": None,
            "n_throats_removed": result["n_throats_removed"],
            "n_bodies_newly_isolated": result["n_bodies_newly_isolated"],
            "solve_failed": result["solve_failed"] is not None,
        }
        for key in _METRIC_KEYS:
            row[f"baseline_{key}"] = result["baseline"][key]
            row[f"pruned_{key}"] = result["pruned"][key] if result["pruned"] is not None else np.nan
            row[f"delta_{key}"] = result["delta"].get(key, np.nan)
        rows.append(row)

    return pd.DataFrame(rows), raw_results


def with_before_after_columns(
    df: pd.DataFrame,
    metrics: Iterable[str],
    fmt: str = "{:.4g}",
) -> pd.DataFrame:
    """Add "<metric>_before/after" string columns (e.g. "0.3795/0.3457") to a
    copy of a summary_df / sweep_df produced by run_on_microstructures (or
    any DataFrame with matching baseline_<metric>/pruned_<metric> columns),
    for a compact side-by-side read without doubling the column count with
    separate baseline_/pruned_ columns.

    `metrics` are the bare names (e.g. "inv_tau_sq_c", "inv_C_c", "inv_F_c",
    "phi_c") -- looks up "baseline_<metric>" and "pruned_<metric>" for each.
    Rows where the pruned solve failed (pruned_<metric> is NaN) render as
    "<baseline>/failed".
    """
    out = df.copy()
    for metric in metrics:
        b_col, p_col = f"baseline_{metric}", f"pruned_{metric}"
        if b_col not in out.columns or p_col not in out.columns:
            raise KeyError(f"Missing {b_col!r}/{p_col!r} -- pass a df from run_on_microstructures")

        def _render(b, p):
            b_str = fmt.format(b) if pd.notna(b) else "nan"
            p_str = fmt.format(p) if pd.notna(p) else "failed"
            return f"{b_str}/{p_str}"

        out[f"{metric}_before/after"] = [
            _render(b, p) for b, p in zip(out[b_col], out[p_col])
        ]
    return out
