"""
body_analysis.py
=================

What distinguishes the bodies flanking a bottleneck throat from an ordinary
body? And do bottleneck throats/bodies cluster spatially, or scatter
throughout the domain?

Builds one row per body (volume, r_body, coordination number z,
is_surface, position along the flow axis) and marks which bodies touch a
top-N geometric-bottleneck / dissipation-highway / overlap throat (from
bottleneck_overlap.py), so the two populations can be compared.

Pure functions, no side effects at import. Depends on throat_analysis.py's
MicrostructureData/GeometryData and a throat_df already annotated by
bottleneck_overlap.annotate_bottleneck_sets().
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats as sstats

import throat_analysis as ta


def conducting_body_mask(msd: ta.MicrostructureData) -> np.ndarray:
    """Bodies carrying non-negligible current, i.e. the nodes of the directed
    conducting graph `build_streamtubes` builds internally (edge kept iff
    |current| >= I_threshold). berg_cc.compute_all_berg never persists that
    graph, and never calls its own boundary-connectivity / directed-reachability
    filters (they're dead code called with conducting_mask=None) -- so
    `arrays` in full_result.pkl still includes every body in the
    microstructure, percolating or not. This reconstructs the same mask from
    the persisted `current` array so downstream analysis only sees bodies
    that are actually part of the solved resistor network.
    """
    arrays = msd.berg["arrays"]
    conns = arrays["conns"]
    current = msd.berg["current"]
    # full_result.pkl doesn't persist I_threshold (see ta.get_inlet_outlet's
    # docstring re: pipeline.py trimming solution_data) -- recompute it with
    # the exact recipe berg_cc.compute_all_berg uses before building the
    # conducting graph in build_streamtubes.
    I_total = abs(msd.berg_metrics["total_current"])
    I_max = np.max(np.abs(current))
    I_threshold = min(1e-10 * I_total, 1e-10 * I_max, 1e-10)
    N = len(arrays["body_ids"])

    valid = (conns[:, 0] >= 0) & (conns[:, 1] >= 0)
    conducting_throat = valid & (np.abs(current) >= I_threshold)

    mask = np.zeros(N, dtype=bool)
    mask[conns[conducting_throat, 0]] = True
    mask[conns[conducting_throat, 1]] = True
    return mask


def build_body_table(msd: ta.MicrostructureData, geom: ta.GeometryData, df: pd.DataFrame) -> pd.DataFrame:
    """One row per CONDUCTING body only (see `conducting_body_mask`) --
    bodies with no path carrying non-negligible current are excluded, since
    they aren't part of the resistor network the Berg quantities are computed
    over. `df` must already carry the bottleneck-overlap boolean columns
    (is_geo_bottleneck, is_edge_highway, is_throat_highway, overlap_edge,
    overlap_throat_only) from bottleneck_overlap.annotate_bottleneck_sets().
    """
    arrays = msd.berg["arrays"]
    body_ids = arrays["body_ids"]
    coords = arrays["coords"]
    volumes = arrays["volumes"]
    is_surface = arrays["is_surface"]
    conns = arrays["conns"]
    N = len(body_ids)
    is_conducting = conducting_body_mask(msd)

    # Coordination number restricted to conducting throats between conducting
    # bodies -- a throat carrying negligible current, or touching a
    # non-conducting body, shouldn't count toward z for the resistor network.
    current = msd.berg["current"]
    I_total = abs(msd.berg_metrics["total_current"])
    I_max = np.max(np.abs(current))
    I_threshold = min(1e-10 * I_total, 1e-10 * I_max, 1e-10)
    valid = (conns[:, 0] >= 0) & (conns[:, 1] >= 0)
    conducting_throat = valid & (np.abs(current) >= I_threshold)
    z = np.zeros(N, dtype=int)
    np.add.at(z, conns[conducting_throat, 0], 1)
    np.add.at(z, conns[conducting_throat, 1], 1)

    axis_map = {"x": 0, "y": 1, "z": 2}
    axis = axis_map[msd.direction]
    domain_len = msd.segmented.shape[axis]
    # position along flow axis, normalised to [0, 1] (0 = inlet face, 1 = outlet face)
    pos_frac = coords[:, axis] / max(domain_len - 1, 1)

    body_df = pd.DataFrame({
        "body_idx": np.arange(N),
        "body_id": body_ids,
        "volume": volumes,
        "r_body": geom.r_body_by_idx,
        "z_coordination": z,
        "is_surface": is_surface,
        "pos_frac_along_flow": pos_frac,
        "centroid_z": coords[:, 0],
        "centroid_y": coords[:, 1],
        "centroid_x": coords[:, 2],
    })

    def _touches(mask_col: str) -> np.ndarray:
        ids = set(df.loc[df[mask_col], "body1"]) | set(df.loc[df[mask_col], "body2"])
        return body_df["body_id"].isin(ids).values

    body_df["touches_geo_bottleneck"] = _touches("is_geo_bottleneck")
    body_df["touches_edge_highway"] = _touches("is_edge_highway")
    body_df["touches_throat_highway"] = _touches("is_throat_highway")
    body_df["touches_overlap_edge"] = _touches("overlap_edge")
    body_df["touches_overlap_throat_only"] = _touches("overlap_throat_only")
    body_df["touches_any_topn"] = (body_df["touches_geo_bottleneck"]
                                   | body_df["touches_edge_highway"]
                                   | body_df["touches_throat_highway"])
    # geo AND (either dissipation level) -- throats that are both
    # geometrically narrow and transport-relevant, not merely one or the other.
    body_df["touches_overlap_any"] = (body_df["touches_overlap_edge"]
                                      | body_df["touches_overlap_throat_only"])

    n_dropped = int((~is_conducting).sum())
    if n_dropped:
        print(f"[build_body_table] dropping {n_dropped}/{N} non-conducting "
              f"bodies (no path above I_threshold)")
    body_df = body_df.loc[is_conducting].reset_index(drop=True)

    return body_df


def compare_groups(body_df: pd.DataFrame, group_col: str,
                   props=("volume", "r_body", "z_coordination", "pos_frac_along_flow")) -> Dict[str, Dict]:
    """Mann-Whitney U (distribution-free, robust to the skewed volume/r_body
    distributions) comparing bodies where group_col is True vs. False, for
    each property in `props`. Reports medians of both groups and the test.
    """
    out = {}
    in_group = body_df[body_df[group_col]]
    out_group = body_df[~body_df[group_col]]
    out["n_in_group"] = len(in_group)
    out["n_out_group"] = len(out_group)

    for prop in props:
        a = in_group[prop].dropna().astype(float)
        b = out_group[prop].dropna().astype(float)
        if len(a) < 3 or len(b) < 3:
            out[prop] = {"median_in": float(a.median()) if len(a) else np.nan,
                         "median_out": float(b.median()) if len(b) else np.nan,
                         "mannwhitney_p": np.nan, "note": "too few samples for a test"}
            continue
        stat = sstats.mannwhitneyu(a, b, alternative="two-sided")
        out[prop] = {
            "median_in": float(a.median()),
            "median_out": float(b.median()),
            "mean_in": float(a.mean()),
            "mean_out": float(b.mean()),
            "mannwhitney_u": float(stat.statistic),
            "mannwhitney_p": float(stat.pvalue),
        }

    frac_surface_in = float(in_group["is_surface"].mean()) if len(in_group) else np.nan
    frac_surface_out = float(out_group["is_surface"].mean()) if len(out_group) else np.nan
    out["is_surface"] = {"frac_in": frac_surface_in, "frac_out": frac_surface_out}

    return out


def nearest_neighbor_distances(coords: np.ndarray) -> np.ndarray:
    """Euclidean distance from each point to its nearest OTHER point in the
    same set, vectorised via a full pairwise distance matrix (fine at these
    throat counts -- hundreds, not tens of thousands)."""
    if len(coords) < 2:
        return np.array([])
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1)


def spatial_clustering_report(df: pd.DataFrame, mask_col: str, n_random: int = 200,
                              random_state: int = 0) -> Dict[str, object]:
    """Are throats in df[mask_col] spatially clustered relative to a random
    same-size subset of all throats? Compares the mean nearest-neighbour
    distance within the marked set against a null distribution built by
    repeatedly drawing random subsets of the same size from all throats
    with valid centroids. A mean NN distance well BELOW the null range means
    the marked throats cluster together spatially; well ABOVE means they are
    more spread out than random; within range means no detectable
    clustering at this sample size.
    """
    valid = df["C_geomean"].notna()
    df_valid = df[valid]
    coords_all = df_valid[["centroid_z", "centroid_y", "centroid_x"]].values
    marked = df_valid[mask_col].values
    n_marked = int(marked.sum())

    if n_marked < 3:
        return {"n_marked": n_marked, "note": "too few marked throats for a clustering test"}

    observed_nn = nearest_neighbor_distances(coords_all[marked])
    observed_mean_nn = float(observed_nn.mean())

    rng = np.random.default_rng(random_state)
    null_means = []
    n_total = len(df_valid)
    for _ in range(n_random):
        idx = rng.choice(n_total, size=n_marked, replace=False)
        null_means.append(nearest_neighbor_distances(coords_all[idx]).mean())
    null_means = np.array(null_means)

    percentile = float((null_means < observed_mean_nn).mean() * 100)

    return {
        "n_marked": n_marked,
        "n_total_valid": n_total,
        "observed_mean_nn_distance": observed_mean_nn,
        "null_mean_nn_distance": float(null_means.mean()),
        "null_std_nn_distance": float(null_means.std()),
        "percentile_of_observed_in_null": percentile,
        "clustered": bool(percentile < 5),
        "dispersed": bool(percentile > 95),
    }
