"""
bottleneck_overlap.py
======================

Two independent throat rankings, and where they agree:

- Geometric bottleneck: throats with the smallest C_geomean = r_throat /
  sqrt(r_body1 * r_body2). Geometric mean instead of min(r_body1, r_body2)
  (throat_analysis.py's default C) so a throat between one large and one
  small body isn't automatically called "narrow" just because the small
  body is small -- geomean rewards a throat that's narrow relative to BOTH
  neighbours, not just the smaller one.

- Transport ("highway") bottleneck: throats with the largest dissipated
  power, as a FRACTION of total dissipated power (P_edge_frac = P_edge /
  sum(P_edge); P_throat_only_frac = P_throat_only / sum(P_throat_only) --
  each level normalised to its own total, so both are unitless shares that
  sum to 1 over all throats, the same way f = |I|/I_total already is).
  Raw P scales with I_total^2, which varies by orders of magnitude with
  domain size/porosity, so it is not comparable across microstructures and
  even within one microstructure a raw-P ranking conflates "carries a lot
  of current" with "is a bottleneck" more than the fraction does. Ranking
  and rank-correlation results are identical to using raw P (nlargest and
  Spearman/Pearson are invariant to a positive rescaling by a constant) --
  the fraction changes interpretability and cross-microstructure
  comparability, not the ranking itself.

  Two separate levels, kept separate rather than combined, per constraint
  (c) (R_throat alone under-ranks throats where the physics puts most of
  the resistance in the body spreading segments):
    - P_edge_frac         (whole body1-throat-body2 segment's share)
    - P_throat_only_frac  (throat channel alone's share)

"Overlap" is reported two ways for each of the two dissipation rankings
against the geometric ranking: a concrete top-N set intersection, and the
overall Spearman/Pearson rank correlation.

Pure functions, no side effects at import. Depends only on the columns
build_throat_table() in throat_analysis.py already produces
(C_geomean, P_edge_frac, P_throat_only_frac).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats as sstats


def top_n_narrow(df: pd.DataFrame, n: int, col: str = "C_geomean") -> pd.Index:
    """Smallest C_geomean = most geometrically constricted."""
    return df.nsmallest(n, col).index


def top_n_dissipating(df: pd.DataFrame, n: int, col: str) -> pd.Index:
    """Largest dissipation = most transport-relevant."""
    return df.nlargest(n, col).index


def overlap_report(df: pd.DataFrame, n: int) -> Dict[str, object]:
    """n = size of each top-N set (e.g. round(0.1 * len(df)) for top 10%,
    or a fixed count like 20). Same n used for both the geometric set and
    each dissipation set, so precision/recall of the intersection is
    directly interpretable as "what fraction of the top-N narrow throats are
    also top-N dissipators" and vice versa.

    Throats with no matching geometric interface (Task 3 -- a handful,
    typically <1% -- see REPORT.md) have NaN C_geomean and are excluded from
    both the top-N narrow set and the correlation, rather than letting them
    silently propagate NaN into scipy.stats.
    """
    valid = df["C_geomean"].notna()
    df_valid = df[valid]
    geo_set = set(top_n_narrow(df_valid, n))

    out: Dict[str, object] = {"n": n, "n_throats": len(df), "n_valid_C_geomean": int(valid.sum())}

    for label, col in (("edge", "P_edge_frac"), ("throat_only", "P_throat_only_frac")):
        diss_set = set(top_n_dissipating(df_valid, n, col))
        inter = geo_set & diss_set
        rho = sstats.spearmanr(df_valid["C_geomean"], df_valid[col])
        pear = sstats.pearsonr(df_valid["C_geomean"], df_valid[col])

        out[label] = {
            "dissipation_col": col,
            "intersection_throat_rows": sorted(df.loc[list(inter), "throat_row"].tolist()),
            "n_intersection": len(inter),
            "jaccard": len(inter) / len(geo_set | diss_set) if (geo_set | diss_set) else 0.0,
            "precision_of_top_n": len(inter) / n if n else 0.0,
            "sum_dissipation_frac_in_top_n": float(df_valid.nlargest(n, col)[col].sum()),
            "spearman_C_geomean_vs_dissipation": float(rho.statistic),
            "spearman_p": float(rho.pvalue),
            "pearson_C_geomean_vs_dissipation": float(pear.statistic),
            "pearson_p": float(pear.pvalue),
        }

    return out


def annotate_bottleneck_sets(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Adds boolean columns marking membership in each top-N set, and the
    dual-overlap columns (in both geometric AND a dissipation set), for
    downstream use (tables, napari points layers). Does not mutate df.

    Throats with NaN C_geomean (no matching geometric interface, see
    overlap_report) can never be a geometric bottleneck by construction, but
    remain eligible for the dissipation sets.
    """
    df = df.copy()
    df_valid = df[df["C_geomean"].notna()]
    geo_idx = top_n_narrow(df_valid, n)
    edge_idx = top_n_dissipating(df, n, "P_edge_frac")
    throat_idx = top_n_dissipating(df, n, "P_throat_only_frac")

    df["is_geo_bottleneck"] = df.index.isin(geo_idx)
    df["is_edge_highway"] = df.index.isin(edge_idx)
    df["is_throat_highway"] = df.index.isin(throat_idx)
    df["overlap_edge"] = df["is_geo_bottleneck"] & df["is_edge_highway"]
    df["overlap_throat_only"] = df["is_geo_bottleneck"] & df["is_throat_highway"]
    return df
