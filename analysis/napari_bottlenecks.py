#!/usr/bin/env python3
"""
napari_bottlenecks.py
======================

Interactive 3D viewer for the geometric-vs-dissipation bottleneck analysis
(bottleneck_overlap.py). Run this directly with an interactive display --
it blocks on napari.run() and is not meant to be invoked from a headless
tool call.

Usage:
    python napari_bottlenecks.py --microstructure microstructure1
    python napari_bottlenecks.py --microstructure Coarse_microstructure_0 --top-frac 0.15

Layers added:
    - "pore space"        : the segmented pore phase, as a translucent volume
    - "|J| (taufactor)"    : the voxel current-magnitude field, as a volume
                             (toggle visibility off by default -- dense/slow)
    - "body labels (all)"  : the watershed body_array, as a napari Labels
                             layer, every body (click a body to see its id)
    - "body labels (top-N sets)" : same Labels layer, but every body that is
                             NOT an endpoint of a throat in any top-N set
                             (geometric OR edge-dissipation OR throat-only-
                             dissipation) is zeroed out -- toggle this on and
                             the "all" layer off to declutter down to just
                             the bodies actually relevant to some bottleneck
                             ranking. Off by default.
    - "body labels (overlap only)" : same, but zeroed down to only bodies
                             touching an overlap throat (both geometric AND
                             a dissipation top-N) -- the tightest, most
                             minimal view. Off by default.
    - "all throats"        : every throat centroid, small grey points
    - "geometric bottleneck (top-N)"  : blue points, smallest C_geomean
    - "edge dissipation (top-N)"      : orange points, largest I^2*R_edge
    - "throat-only dissipation (top-N)": points, largest I^2*R_throat
    - "overlap: geo & edge"           : black-ringed, larger points
    - "overlap: geo & throat_only"    : black-ringed, larger points

Each points layer carries `properties` (throat_row, C_geomean, P_edge_frac,
P_throat_only_frac, f, is_fallback, ...) so napari's built-in point
inspector (hover / click, or the layer's "properties" table) shows the
numbers behind each point -- no need to cross-reference the CSV by hand.
Dissipation is shown as a fraction of total dissipated power (own level's
total), not raw P, so it's comparable across microstructures the same way
f already is.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import throat_analysis as ta
import bottleneck_overlap as bo


def build_viewer(ms_name: str, data_dir: Path, cache_dir: Path, top_frac: float):
    import napari

    msd = ta.load_microstructure(data_dir / ms_name)
    geom = ta.get_geometry(msd, cache_dir, use_cache=True)
    df = ta.build_throat_table(msd, geom)
    n = max(3, round(top_frac * len(df)))
    df = bo.annotate_bottleneck_sets(df, n)
    overlap = bo.overlap_report(df, n)

    print(f"[{ms_name}] n_throats={len(df)}  top-N={n}")
    print(f"  overlap geo & edge         : {overlap['edge']['n_intersection']}")
    print(f"  overlap geo & throat_only  : {overlap['throat_only']['n_intersection']}")

    pore = (msd.segmented == msd.phase_id)
    Jmag = np.sqrt(msd.tau["J_x"] ** 2 + msd.tau["J_y"] ** 2 + msd.tau["J_z"] ** 2)

    viewer = napari.Viewer(title=f"Bottleneck overlap -- {ms_name}", ndisplay=3)

    viewer.add_image(pore.astype(np.uint8), name="pore space", opacity=0.15,
                     colormap="gray", rendering="translucent")
    viewer.add_image(Jmag, name="|J| (taufactor)", opacity=0.35, colormap="magma",
                     rendering="mip", visible=False)
    viewer.add_labels(msd.body_array, name="body labels (all)", opacity=0.25, visible=False)

    # Body id -> whether it's an endpoint of a top-N-set / overlap throat.
    # Zeroing a label out of a COPY of body_array (not the original -- keep
    # "body labels (all)" untouched) turns it into background in a napari
    # Labels layer, so only the relevant bodies render.
    def _filtered_labels(body_mask_ids: set) -> np.ndarray:
        keep = np.isin(msd.body_array, list(body_mask_ids)) if body_mask_ids else np.zeros_like(msd.body_array, dtype=bool)
        out = np.where(keep, msd.body_array, 0)
        return out.astype(msd.body_array.dtype)

    topn_body_ids = set(df.loc[df["is_geo_bottleneck"] | df["is_edge_highway"] | df["is_throat_highway"], "body1"]) \
        | set(df.loc[df["is_geo_bottleneck"] | df["is_edge_highway"] | df["is_throat_highway"], "body2"])
    overlap_body_ids = set(df.loc[df["overlap_edge"] | df["overlap_throat_only"], "body1"]) \
        | set(df.loc[df["overlap_edge"] | df["overlap_throat_only"], "body2"])

    print(f"  bodies touching any top-N throat : {len(topn_body_ids)} / {len(msd.berg['arrays']['body_ids'])}")
    print(f"  bodies touching an overlap throat: {len(overlap_body_ids)} / {len(msd.berg['arrays']['body_ids'])}")

    viewer.add_labels(_filtered_labels(topn_body_ids), name="body labels (top-N sets)",
                      opacity=0.5, visible=False)
    viewer.add_labels(_filtered_labels(overlap_body_ids), name="body labels (overlap only)",
                      opacity=0.6, visible=False)

    coords_all = df[["centroid_z", "centroid_y", "centroid_x"]].values
    valid_coords = df["C_geomean"].notna().values  # throats w/o geometric match have NaN centroid too

    props_cols = ["throat_row", "body1", "body2", "C", "C_geomean", "f",
                  "P_edge_frac", "P_throat_only_frac", "R_throat", "R_edge", "is_fallback"]

    def _props(mask):
        return {c: df.loc[mask, c].values for c in props_cols}

    all_mask = valid_coords
    viewer.add_points(coords_all[all_mask], name="all throats", size=2.5,
                      face_color="#bbbbbb", opacity=0.35, properties=_props(all_mask),
                      border_width=0)

    geo_mask = (df["is_geo_bottleneck"] & valid_coords).values
    viewer.add_points(coords_all[geo_mask], name=f"geometric bottleneck (top-{n})",
                      size=5, face_color="#0072B2", properties=_props(geo_mask),
                      border_width=0)

    edge_mask = (df["is_edge_highway"] & valid_coords).values
    viewer.add_points(coords_all[edge_mask], name=f"edge dissipation (top-{n})",
                      size=5, face_color="#D55E00", properties=_props(edge_mask),
                      border_width=0)

    throat_mask = (df["is_throat_highway"] & valid_coords).values
    viewer.add_points(coords_all[throat_mask], name=f"throat-only dissipation (top-{n})",
                      size=5, face_color="#CC79A7", properties=_props(throat_mask),
                      border_width=0, visible=False)

    ov_edge_mask = (df["overlap_edge"] & valid_coords).values
    if ov_edge_mask.sum():
        viewer.add_points(coords_all[ov_edge_mask], name="overlap: geo & edge",
                          size=9, face_color="transparent", border_color="black",
                          border_width=0.25, symbol="ring", properties=_props(ov_edge_mask))

    ov_throat_mask = (df["overlap_throat_only"] & valid_coords).values
    if ov_throat_mask.sum():
        viewer.add_points(coords_all[ov_throat_mask], name="overlap: geo & throat_only",
                          size=9, face_color="transparent", border_color="black",
                          border_width=0.25, symbol="ring", properties=_props(ov_throat_mask),
                          visible=False)

    viewer.dims.ndisplay = 3
    return viewer, df, overlap


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--microstructure", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--cache-dir", type=Path, default=Path(__file__).resolve().parent / "output" / "cache")
    parser.add_argument("--top-frac", type=float, default=0.10,
                        help="fraction of throats in each top-N set (default 0.10 = top 10%%)")
    args = parser.parse_args()

    import napari
    viewer, df, overlap = build_viewer(args.microstructure, args.data_dir, args.cache_dir, args.top_frac)
    napari.run()


if __name__ == "__main__":
    main()
