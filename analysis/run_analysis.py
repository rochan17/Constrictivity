#!/usr/bin/env python3
"""
run_analysis.py
================

CLI driver for the throat-level constriction analysis (Tasks 1-6 of the
task spec). Writes per-microstructure throat tables (CSV + parquet) and
figures to a separate output tree -- never touches the existing pipeline's
<microstructure>/version<N>/ directories.

Usage:
    python run_analysis.py --data-dir ../data --out analysis/output \
        --microstructures microstructure1 Coarse_microstructure_0

Note: Task 5 (voxel-vs-network cross-check) failed on both available
microstructures as of this writing (see REPORT.md). This CLI still
produces the figures/tables Task 6 asks for, since they are part of the
documented failure analysis, but per the task spec no aggregation across
microstructures or correlation against tau/constrictivity/F is performed
here (Task 8) until Task 5 passes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import throat_analysis as ta
import bottleneck_overlap as bo
import body_analysis as ba

CLASS_COLORS = {
    "bottleneck": "#D55E00",         # vermillion
    "geometric_only": "#0072B2",     # blue
    "highway": "#009E73",            # bluish green
    "secondary_or_filler": "#999999",  # neutral grey
}
CLASS_ORDER = ["bottleneck", "geometric_only", "highway", "secondary_or_filler"]


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_main_scatter(df: pd.DataFrame, ms_name: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)
    sizes = 15 + 60 * (
        (df["geometric_throat_area"] - df["geometric_throat_area"].min())
        / (df["geometric_throat_area"].max() - df["geometric_throat_area"].min() + 1e-12)
    )

    # f can be exactly 0 (or ~1e-17 numerical noise), which breaks a log
    # axis -- plot a real, positive "idle floor" just below the smallest
    # f value that's actually meaningful (> F_IDLE_CUT threshold), and put
    # every f below that floor there instead of letting it set the axis
    # range down to 1e-17.
    f_meaningful = df.loc[df["f"] > ta.F_IDLE_CUT, "f"]
    floor = (f_meaningful.min() * 0.3) if len(f_meaningful) else 1e-4
    f_plot = df["f"].clip(lower=floor)

    for cls in CLASS_ORDER:
        sub = df[df["class"] == cls]
        if len(sub) == 0:
            continue
        ax.scatter(sub["C"], f_plot.loc[sub.index], s=sizes.loc[sub.index],
                   c=CLASS_COLORS[cls], label=cls.replace("_", " "),
                   alpha=0.75, edgecolors="white", linewidths=0.4, zorder=3)

    ax.axvline(ta.C_NARROW_CUT, color="black", linestyle="--", linewidth=0.8, alpha=0.6, zorder=2)
    f_max = df["f"].max()
    ax.axhline(ta.F_BUSY_FRAC_OF_MAX * f_max, color="black", linestyle="--", linewidth=0.8, alpha=0.6, zorder=2)
    ax.axhline(floor * 3, color="#bbbbbb", linestyle=":", linewidth=0.8, zorder=1)
    ax.text(ax.get_xlim()[0], floor * 3.3, " f ≈ 0 (clipped to floor)", fontsize=7, color="#888888", va="bottom")

    top5 = df.nlargest(5, "f")
    for _, row in top5.iterrows():
        ax.annotate(str(int(row["throat_row"])), (row["C"], f_plot.loc[row.name]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8, color="black")

    ax.set_yscale("log")
    ax.set_ylim(floor * 0.5, f_max * 3)
    ax.set_xlabel("Constriction ratio  C = r_throat / min(r_body1, r_body2)")
    ax.set_ylabel("Current fraction  f = |I_throat| / I_total")
    ax.set_title(f"Constriction vs. transport relevance -- {ms_name}")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(out_dir / f"{ms_name}_main_scatter.{ext}", **kwargs)
    plt.close(fig)


def fig_cross_check(df: pd.DataFrame, ms_name: str, out_dir: Path):
    sub = df.dropna(subset=["f_voxel"]).copy()
    # Restrict the PLOT to throats with a meaningful current on at least one
    # side (> F_IDLE_CUT) -- both f and f_voxel can be ~1e-17 numerical
    # noise for genuinely dead throats, and including those stretches a log
    # axis across 16 decades, crushing the meaningful cluster into a corner.
    # The idle/near-zero population is summarised separately as an
    # absolute-error histogram per the task spec (see cross_check_stats'
    # idle-subset numbers in REPORT.md), not plotted here.
    sub = sub[(sub["f"] > ta.F_IDLE_CUT) & (sub["f_voxel"] > ta.F_IDLE_CUT)]
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=150)

    for is_fb, color, label in [(False, "#0072B2", "real throat"), (True, "#D55E00", "fallback throat")]:
        m = sub["is_fallback"] == is_fb
        if m.sum() == 0:
            continue
        ax.scatter(sub.loc[m, "f"], sub.loc[m, "f_voxel"], s=22, c=color, alpha=0.7,
                   edgecolors="white", linewidths=0.3, label=label, zorder=3)

    lims = [min(sub["f"].min(), sub["f_voxel"].min()) * 0.5,
            max(sub["f"].max(), sub["f_voxel"].max()) * 2]
    ax.plot(lims, lims, color="black", linewidth=1.0, linestyle="--", alpha=0.6, zorder=2, label="y = x")

    stats = ta.cross_check_stats(df)
    txt = (f"Spearman r = {stats['spearman_r']:.3f}\n"
           f"Pearson r = {stats['pearson_r']:.3f}\n"
           f"median rel. err = {stats['median_rel_err_busy']*100:.1f}%\n"
           f"PASS: {stats['pass']}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#cccccc"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("f_network = |I_throat| / I_total  (berg_cc)")
    ax.set_ylabel("f_voxel = |flux through interface| / I_total  (taufactor)")
    ax.set_title(f"Voxel-vs-network cross-check -- {ms_name}")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(out_dir / f"{ms_name}_cross_check.{ext}", **kwargs)
    plt.close(fig)


def fig_resistance_split(df: pd.DataFrame, ms_name: str, out_dir: Path):
    ratio = (df["R_throat"] / df["R_edge"]).dropna()
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150)
    sorted_r = np.sort(ratio.values)
    ecdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
    ax.plot(sorted_r, ecdf, color="#0072B2", linewidth=1.8)
    med = np.median(ratio)
    ax.axvline(med, color="#D55E00", linestyle="--", linewidth=1.0)
    ax.text(med, 0.05, f"  median = {med:.3f}", color="#D55E00", fontsize=9)
    ax.set_xlabel("R_throat / R_edge")
    ax.set_ylabel("ECDF")
    ax.set_title(f"Where the resistance sits -- {ms_name}")
    _style_axes(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(out_dir / f"{ms_name}_resistance_split.{ext}", **kwargs)
    plt.close(fig)


def fig_bottleneck_overlap(df: pd.DataFrame, ms_name: str, n: int, out_dir: Path):
    """Two panels: C_geomean vs P_edge_frac, and C_geomean vs
    P_throat_only_frac, each log-log, marking the top-N-narrow /
    top-N-dissipating / overlap sets. Dissipation is plotted as a fraction
    of total dissipated power (own level's total), not raw P -- rankings
    are identical either way, but the fraction is a unitless share
    comparable across microstructures, the same way f already is."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), dpi=150)
    df_valid = df[df["C_geomean"].notna()]

    for ax, col, label in zip(axes, ("P_edge_frac", "P_throat_only_frac"),
                               ("edge dissipation fraction  P_edge / ΣP_edge",
                                "throat-only dissipation fraction  P_throat / ΣP_throat")):
        geo_idx = bo.top_n_narrow(df_valid, n)
        diss_idx = bo.top_n_dissipating(df_valid, n, col)
        overlap_idx = geo_idx.intersection(diss_idx)

        # P can be exactly (or numerically ~) 0 for a dead throat, which
        # breaks a log axis across ~30 decades and crushes the meaningful
        # cluster into a sliver -- clip to a visible floor just below the
        # smallest value in the top-n dissipating set, same approach as
        # fig_main_scatter's f-floor.
        p_diss_min = df_valid.loc[diss_idx, col].min() if len(diss_idx) else df_valid[col].max()
        floor = max(p_diss_min * 1e-4, df_valid.loc[df_valid[col] > 0, col].min() * 0.3
                    if (df_valid[col] > 0).any() else 1e-12)
        y = df_valid[col].clip(lower=floor)

        ax.scatter(df_valid["C_geomean"], y, s=16, c="#cccccc", alpha=0.6,
                   edgecolors="none", zorder=2, label="other")
        ax.scatter(df_valid.loc[geo_idx, "C_geomean"], y.loc[geo_idx], s=28, c="#0072B2",
                   alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3,
                   label=f"top-{n} narrow (geometric)")
        ax.scatter(df_valid.loc[diss_idx, "C_geomean"], y.loc[diss_idx], s=28, c="#D55E00",
                   alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3,
                   label=f"top-{n} dissipating")
        if len(overlap_idx):
            ax.scatter(df_valid.loc[overlap_idx, "C_geomean"], y.loc[overlap_idx], s=90,
                       facecolors="none", edgecolors="black", linewidths=1.3, zorder=4,
                       label=f"overlap (n={len(overlap_idx)})")

        ax.set_yscale("log")
        ax.set_ylim(floor * 0.5, df_valid[col].max() * 3)
        ax.set_xlabel("C_geomean = r_throat / sqrt(r_body1 · r_body2)")
        ax.set_ylabel(label)
        ax.legend(loc="upper right", frameon=False, fontsize=7.5)
        _style_axes(ax)

    fig.suptitle(f"Geometric bottleneck vs. dissipation bottleneck -- {ms_name}")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(out_dir / f"{ms_name}_bottleneck_overlap.{ext}", **kwargs)
    plt.close(fig)


def fig_body_property_comparison(body_df: pd.DataFrame, ms_name: str, out_dir: Path):
    """Bottleneck-touching bodies vs. all others: volume, r_body,
    coordination number, and surface-fraction. Jittered strip + box, since
    n can be small (tens of bodies on Coarse_microstructure_0)."""
    props = [("volume", "Body volume [vox³]", True),
             ("r_body", "r_body (EDT max) [vox]", False),
             ("z_coordination", "Coordination number z", False)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), dpi=150)
    rng = np.random.default_rng(0)

    for ax, (col, label, logy) in zip(axes[:3], props):
        groups = [body_df.loc[~body_df["touches_overlap_any"], col],
                  body_df.loc[body_df["touches_overlap_any"], col]]
        bp = ax.boxplot(groups, positions=[0, 1], widths=0.5, showfliers=False,
                        patch_artist=True, medianprops=dict(color="black"))
        for patch, color in zip(bp["boxes"], ["#999999", "#D55E00"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        for i, g in enumerate(groups):
            x = rng.normal(i, 0.06, size=len(g))
            ax.scatter(x, g, s=10, c=["#999999", "#D55E00"][i], alpha=0.6, edgecolors="none", zorder=3)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["other bodies", "touches\nbottleneck"])
        ax.set_ylabel(label)
        _style_axes(ax)

    ax = axes[3]
    frac_in = body_df.loc[body_df["touches_overlap_any"], "is_surface"].mean()
    frac_out = body_df.loc[~body_df["touches_overlap_any"], "is_surface"].mean()
    ax.bar([0, 1], [frac_out, frac_in], color=["#999999", "#D55E00"], alpha=0.7, width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["other bodies", "touches\nbottleneck"])
    ax.set_ylabel("Fraction is_surface")
    ax.set_ylim(0, 1)
    _style_axes(ax)

    fig.suptitle(f"Bodies touching a geo+dissipation overlap throat vs. other bodies -- {ms_name}")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(out_dir / f"{ms_name}_body_property_comparison.{ext}", **kwargs)
    plt.close(fig)


def fig_bottleneck_spatial_map(msd: ta.MicrostructureData, df: pd.DataFrame, body_df: pd.DataFrame,
                               ms_name: str, out_dir: Path):
    """2D projection (max-intensity along the shortest axis) of the domain,
    with all bottleneck-touching body centroids and bottleneck throat
    centroids overlaid, to visually check spatial clustering."""
    shape = msd.segmented.shape
    proj_axis = int(np.argmin(shape))
    pore = (msd.segmented == msd.phase_id)
    proj = pore.mean(axis=proj_axis)
    other_axes = [i for i in range(3) if i != proj_axis]
    axis_names = ["z", "y", "x"]

    fig, ax = plt.subplots(figsize=(7, 6.5), dpi=150)
    ax.imshow(proj.T, origin="lower", cmap="Greys", alpha=0.5, aspect="auto")

    body_coords = body_df[["centroid_z", "centroid_y", "centroid_x"]].values
    other_mask = ~body_df["touches_overlap_any"].values
    bneck_mask = body_df["touches_overlap_any"].values

    ax.scatter(body_coords[other_mask, other_axes[0]], body_coords[other_mask, other_axes[1]],
              s=10, c="#bbbbbb", alpha=0.4, edgecolors="none", label="other bodies", zorder=2)
    ax.scatter(body_coords[bneck_mask, other_axes[0]], body_coords[bneck_mask, other_axes[1]],
              s=30, c="#D55E00", alpha=0.75, edgecolors="white", linewidths=0.4,
              label="touches geo+dissipation overlap throat", zorder=3)

    ax.set_xlabel(f"{axis_names[other_axes[0]]} [vox]")
    ax.set_ylabel(f"{axis_names[other_axes[1]]} [vox]")
    ax.set_title(f"Spatial distribution of bottleneck-adjacent bodies -- {ms_name}\n"
                f"(projected along {axis_names[proj_axis]}, pore fraction background)")
    ax.legend(loc="upper right", frameon=True, fontsize=8, framealpha=0.85)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(out_dir / f"{ms_name}_bottleneck_spatial_map.{ext}", **kwargs)
    plt.close(fig)


def fig_current_concentration(dfs: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(dfs), 2)))
    for (ms_name, df), color in zip(dfs.items(), colors):
        f_sorted = np.sort(df["f"].values)[::-1]
        cum = np.cumsum(f_sorted) / f_sorted.sum()
        rank = np.arange(1, len(f_sorted) + 1)
        ax.plot(rank, cum, label=ms_name, color=color, linewidth=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("Throat rank (sorted by f, descending)")
    ax.set_ylabel("Cumulative fraction of total current")
    ax.set_title("Current concentration across throats")
    ax.legend(frameon=False, fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(out_dir / f"current_concentration.{ext}", **kwargs)
    plt.close(fig)


def fig_spatial_view(msd: ta.MicrostructureData, df: pd.DataFrame, ms_name: str, out_dir: Path):
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis = axis_map[msd.direction]
    J = {0: msd.tau["J_x"], 1: msd.tau["J_y"], 2: msd.tau["J_z"]}[axis]
    Jmag = np.sqrt(msd.tau["J_x"] ** 2 + msd.tau["J_y"] ** 2 + msd.tau["J_z"] ** 2)

    top = df.loc[df["f"].idxmax()]
    slice_axis = 2 if axis != 2 else 0
    slice_idx = int(round(top[["centroid_z", "centroid_y", "centroid_x"]].values[slice_axis]))
    slice_idx = int(np.clip(slice_idx, 0, msd.segmented.shape[slice_axis] - 1))

    def take_slice(arr):
        idx = [slice(None)] * 3
        idx[slice_axis] = slice_idx
        return arr[tuple(idx)]

    pore_slice = take_slice(msd.segmented == msd.phase_id)
    Jmag_slice = take_slice(Jmag)
    Jmag_masked = np.where(pore_slice, Jmag_slice, np.nan)

    other_axes = [i for i in range(3) if i != slice_axis]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    im = ax.imshow(Jmag_masked, cmap="magma", origin="lower")
    ax.imshow(np.where(pore_slice, 0, 1), cmap="Greys", alpha=0.15, origin="lower")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|J|")

    for cls in CLASS_ORDER:
        sub = df[df["class"] == cls]
        if len(sub) == 0:
            continue
        pos_a = sub[["centroid_z", "centroid_y", "centroid_x"]].values[:, other_axes[0]]
        pos_b = sub[["centroid_z", "centroid_y", "centroid_x"]].values[:, other_axes[1]]
        on_slice = np.abs(sub[["centroid_z", "centroid_y", "centroid_x"]].values[:, slice_axis] - slice_idx) < 3
        if on_slice.sum() == 0:
            continue
        ax.scatter(pos_b[on_slice], pos_a[on_slice], s=28, c=CLASS_COLORS[cls],
                   edgecolors="white", linewidths=0.5, label=cls.replace("_", " "), zorder=3)

    ax.set_title(f"Spatial view through top-f bottleneck -- {ms_name} (slice axis {slice_axis}={slice_idx})")
    ax.legend(loc="upper right", frameon=True, fontsize=8, framealpha=0.85)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(out_dir / f"{ms_name}_spatial_view.{ext}", **kwargs)
    plt.close(fig)


def run_one(ms_name: str, data_dir: Path, out_dir: Path, cache_dir: Path, fig_dir: Path,
           overlap_top_frac: float = 0.10, berg_cc_criteria: dict | None = None):
    ms_dir = data_dir / ms_name
    if berg_cc_criteria:
        msd = ta.load_microstructure_matching(ms_dir, berg_cc_criteria)
    else:
        msd = ta.load_microstructure(ms_dir)
    geom = ta.get_geometry(msd, cache_dir, use_cache=True)
    df = ta.build_throat_table(msd, geom)
    f_voxel, I_voxel, I_total_plane = ta.compute_voxel_throat_currents(msd, geom, df)
    df["f_voxel"] = f_voxel
    df["I_voxel"] = I_voxel

    n = max(3, round(overlap_top_frac * len(df)))
    df = bo.annotate_bottleneck_sets(df, n)
    overlap = bo.overlap_report(df, n)

    body_df = ba.build_body_table(msd, geom, df)
    body_comparison = ba.compare_groups(body_df, "touches_overlap_any")
    clustering = {
        "geo_bottleneck": ba.spatial_clustering_report(df, "is_geo_bottleneck"),
        "edge_highway": ba.spatial_clustering_report(df, "is_edge_highway"),
        "throat_highway": ba.spatial_clustering_report(df, "is_throat_highway"),
    }

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / f"{ms_name}_throats.csv", index=False)
    df.to_parquet(tables_dir / f"{ms_name}_throats.parquet", index=False)
    body_df.to_csv(tables_dir / f"{ms_name}_bodies.csv", index=False)
    with open(tables_dir / f"{ms_name}_bottleneck_overlap.json", "w") as f:
        json.dump(overlap, f, indent=2)
    with open(tables_dir / f"{ms_name}_body_comparison.json", "w") as f:
        json.dump({"group_comparison": body_comparison, "spatial_clustering": clustering}, f, indent=2)

    fig_main_scatter(df, ms_name, fig_dir)
    fig_cross_check(df, ms_name, fig_dir)
    fig_resistance_split(df, ms_name, fig_dir)
    fig_bottleneck_overlap(df, ms_name, n, fig_dir)
    fig_body_property_comparison(body_df, ms_name, fig_dir)
    fig_bottleneck_spatial_map(msd, df, body_df, ms_name, fig_dir)
    fig_spatial_view(msd, df, ms_name, fig_dir)

    return msd, df, overlap, body_df, body_comparison, clustering


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--microstructures", nargs="*", default=None,
                        help="Subset of microstructure names; default = all with a manifest.json")
    parser.add_argument("--min-throat-size", type=int, default=None,
                        help="Only use the berg_cc version whose config.json has this min_throat_size")
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--split-volume-equal", dest="split_volume_equal",
                             action="store_true", default=None,
                             help="Only use the berg_cc version with split_volume_equal true (or unset, which defaults to true)")
    split_group.add_argument("--no-split-volume-equal", dest="split_volume_equal",
                             action="store_false",
                             help="Only use the berg_cc version with split_volume_equal false")
    args = parser.parse_args()

    berg_cc_criteria = {}
    if args.min_throat_size is not None:
        berg_cc_criteria["min_throat_size"] = args.min_throat_size
    if args.split_volume_equal is not None:
        berg_cc_criteria["split_volume_equal"] = args.split_volume_equal
    berg_cc_criteria = berg_cc_criteria or None

    out_dir = args.out
    cache_dir = out_dir / "cache"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    names = args.microstructures or sorted(
        p.name for p in args.data_dir.iterdir() if p.is_dir() and (p / "segmented.npy").exists()
    )

    all_dfs = {}
    for name in names:
        print(f"[run_analysis] {name} ...")
        try:
            msd, df, overlap, body_df, body_comparison, clustering = run_one(
                name, args.data_dir, out_dir, cache_dir, fig_dir, berg_cc_criteria=berg_cc_criteria)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue
        all_dfs[name] = df
        stats = ta.cross_check_stats(df)
        print(f"  n_throats={len(df)}  spearman={stats['spearman_r']:.3f}  "
              f"median_rel_err={stats['median_rel_err_busy']:.3f}  pass={stats['pass']}")
        print(f"  bottleneck overlap (top-{overlap['n']}): "
              f"edge={overlap['edge']['n_intersection']}  "
              f"throat_only={overlap['throat_only']['n_intersection']}")
        vol_p = body_comparison["volume"]["mannwhitney_p"]
        z_p = body_comparison["z_coordination"]["mannwhitney_p"]
        print(f"  bottleneck-adjacent bodies: median volume "
              f"{body_comparison['volume']['median_in']:.0f} vs {body_comparison['volume']['median_out']:.0f} "
              f"(p={vol_p:.1e}), median z {body_comparison['z_coordination']['median_in']:.1f} "
              f"vs {body_comparison['z_coordination']['median_out']:.1f} (p={z_p:.1e})")

    if len(all_dfs) >= 1:
        fig_current_concentration(all_dfs, fig_dir)

    print(f"[run_analysis] done. tables -> {out_dir/'tables'}  figures -> {fig_dir}")


if __name__ == "__main__":
    main()
