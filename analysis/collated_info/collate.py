#!/usr/bin/env python3
"""
collate.py
==========

Cross-microstructure collation over the per-microstructure output that
run_analysis.py already wrote to <out>/tables/*.csv|*.json (throat_analysis.py
/ bottleneck_overlap.py / body_analysis.py results) plus each microstructure's
own data/<name>/version<N>/summary.json (taufactor tau/constrictivity, berg_cc
F, berg_voxel F).

This is the Task 8 aggregate/tau-correlation study, run here across all 30
available microstructures rather than the 2 that were available when
REPORT.md's Task 8 section said it was withheld pending Task 5 (voxel-vs-
network cross-check) passing. Task 5 still does not pass in general (see
`cross_check_pass` per microstructure in the output table) -- that is
reported per-microstructure below rather than silently ignored, so any trend
here should be read as exploratory, not validated the way Tasks 1-7 were.

Usage:
    python collate.py --data-dir ../../data --tables-dir ../output/tables \
        --out .
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats

SIZE_ORDER = ["Coarse", "Medium", "Fine"]
SIZE_COLORS = {"Coarse": "#0072B2", "Medium": "#009E73", "Fine": "#D55E00"}


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def _size_class(ms_name: str) -> str:
    m = re.match(r"([A-Za-z]+)_microstructure_\d+", ms_name)
    return m.group(1) if m else "unknown"


def load_summary(data_dir: Path, ms_name: str) -> dict | None:
    latest_path = data_dir / ms_name / "latest_version.json"
    if not latest_path.exists():
        return None
    latest = json.loads(latest_path.read_text())["version"]
    summary_path = data_dir / ms_name / f"version{latest}" / "summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text())


def build_collated_table(data_dir: Path, tables_dir: Path) -> pd.DataFrame:
    rows = []
    for throats_csv in sorted(tables_dir.glob("*_throats.csv")):
        ms_name = throats_csv.name[: -len("_throats.csv")]
        summary = load_summary(data_dir, ms_name)
        if summary is None:
            continue

        df = pd.read_csv(throats_csv)
        overlap_path = tables_dir / f"{ms_name}_bottleneck_overlap.json"
        overlap = json.loads(overlap_path.read_text()) if overlap_path.exists() else {}

        sub = df.dropna(subset=["f_voxel"])
        spearman = sstats.spearmanr(sub["f_voxel"], sub["f"]) if len(sub) > 1 else None
        busy = sub[sub["f"] > 0.01]
        median_rel_err = float(np.abs(busy["f_voxel"] - busy["f"]).div(busy["f"]).median()) if len(busy) else np.nan
        cross_check_pass = bool(
            spearman is not None and spearman.statistic > 0.95
            and (len(busy) == 0 or median_rel_err < 0.10)
        )

        tau = summary.get("taufactor", {})
        bv = summary.get("berg_voxel", {})
        bcc = summary.get("berg_cc", {})

        rows.append({
            "microstructure": ms_name,
            "size_class": _size_class(ms_name),
            "n_throats": len(df),
            "n_bodies": bcc.get("n_bodies", np.nan),
            "porosity": tau.get("porosity", np.nan),
            "D_eff": tau.get("D_eff", np.nan),
            "tau_reconciled": tau.get("tau_reconciled", np.nan),
            "tau_solver": tau.get("tau_solver", np.nan),
            "constrictivity": tau.get("constrictivity", np.nan),
            "F_voxel": bv.get("F", np.nan),
            "F_cc": bcc.get("F_total", np.nan),
            "median_C": float(df["C"].median()),
            "median_C_geomean": float(df["C_geomean"].median()) if df["C_geomean"].notna().any() else np.nan,
            "frac_bottleneck_class": float((df["class"] == "bottleneck").mean()),
            "frac_highway_class": float((df["class"] == "highway").mean()),
            "current_gini_top10pct_frac": float(
                df.nlargest(max(1, round(0.10 * len(df))), "f")["f"].sum() / df["f"].sum()
            ) if df["f"].sum() > 0 else np.nan,
            "cross_check_spearman": float(spearman.statistic) if spearman is not None else np.nan,
            "cross_check_median_rel_err": median_rel_err,
            "cross_check_pass": cross_check_pass,
            "overlap_edge_jaccard": overlap.get("edge", {}).get("jaccard", np.nan),
            "overlap_throat_only_jaccard": overlap.get("throat_only", {}).get("jaccard", np.nan),
            "overlap_edge_spearman": overlap.get("edge", {}).get("spearman_C_geomean_vs_dissipation", np.nan),
            "overlap_throat_only_spearman": overlap.get("throat_only", {}).get("spearman_C_geomean_vs_dissipation", np.nan),
        })

    out = pd.DataFrame(rows)
    if len(out):
        out["size_class"] = pd.Categorical(out["size_class"], categories=SIZE_ORDER, ordered=True)
        out = out.sort_values(["size_class", "microstructure"]).reset_index(drop=True)
    return out


def _scatter_by_size(ax, table, x, y):
    for cls in SIZE_ORDER:
        sub = table[table["size_class"] == cls]
        if len(sub) == 0:
            continue
        ax.scatter(sub[x], sub[y], s=45, c=SIZE_COLORS[cls], label=cls,
                   alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3)


def _annotate_corr(ax, table, x, y):
    sub = table.dropna(subset=[x, y])
    if len(sub) < 3:
        return
    rho = sstats.spearmanr(sub[x], sub[y])
    ax.text(0.97, 0.03, f"Spearman r = {rho.statistic:.2f} (p={rho.pvalue:.2g}, n={len(sub)})",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#cccccc"))


def fig_tau_vs_porosity(table: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    ax = axes[0]
    _scatter_by_size(ax, table, "porosity", "tau_reconciled")
    _annotate_corr(ax, table, "porosity", "tau_reconciled")
    ax.set_xlabel("Porosity")
    ax.set_ylabel("tau_reconciled")
    ax.set_title("Tortuosity vs. porosity")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _style_axes(ax)

    ax = axes[1]
    _scatter_by_size(ax, table, "porosity", "constrictivity")
    _annotate_corr(ax, table, "porosity", "constrictivity")
    ax.set_xlabel("Porosity")
    ax.set_ylabel("Constrictivity")
    ax.set_title("Constrictivity vs. porosity")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _style_axes(ax)

    fig.suptitle("Bulk transport metrics vs. porosity, across all microstructures")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(out_dir / f"tau_constrictivity_vs_porosity.{ext}", **kwargs)
    plt.close(fig)


def fig_constrictivity_vs_bottleneck_geometry(table: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    ax = axes[0]
    _scatter_by_size(ax, table, "constrictivity", "median_C_geomean")
    _annotate_corr(ax, table, "constrictivity", "median_C_geomean")
    ax.set_xlabel("Constrictivity (taufactor, bulk)")
    ax.set_ylabel("Median throat C_geomean (network, per-throat)")
    ax.set_title("Bulk constrictivity vs. median throat narrowness")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _style_axes(ax)

    ax = axes[1]
    _scatter_by_size(ax, table, "constrictivity", "frac_bottleneck_class")
    _annotate_corr(ax, table, "constrictivity", "frac_bottleneck_class")
    ax.set_xlabel("Constrictivity (taufactor, bulk)")
    ax.set_ylabel("Fraction of throats classified 'bottleneck'")
    ax.set_title("Bulk constrictivity vs. bottleneck throat prevalence")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _style_axes(ax)

    fig.suptitle("Does the bulk constrictivity metric track network-level throat geometry?")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(out_dir / f"constrictivity_vs_bottleneck_geometry.{ext}", **kwargs)
    plt.close(fig)


def fig_F_comparison(table: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=150)
    _scatter_by_size(ax, table, "F_voxel", "F_cc")
    _annotate_corr(ax, table, "F_voxel", "F_cc")
    sub = table.dropna(subset=["F_voxel", "F_cc"])
    if len(sub):
        lims = [min(sub["F_voxel"].min(), sub["F_cc"].min()) * 0.7,
                max(sub["F_voxel"].max(), sub["F_cc"].max()) * 1.3]
        ax.plot(lims, lims, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="y = x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("F (berg_voxel, voxel-level formation factor)")
    ax.set_ylabel("F (berg_cc, network-level formation factor)")
    ax.set_title("Voxel-level vs. network-level formation factor F")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(out_dir / f"F_voxel_vs_cc.{ext}", **kwargs)
    plt.close(fig)


def fig_overlap_by_size_class(table: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)
    x = np.arange(len(SIZE_ORDER))
    width = 0.35
    for i, (col, label, color) in enumerate([
        ("overlap_edge_jaccard", "geometric vs. edge-dissipation", "#0072B2"),
        ("overlap_throat_only_jaccard", "geometric vs. throat-only-dissipation", "#D55E00"),
    ]):
        means = [table.loc[table["size_class"] == cls, col].mean() for cls in SIZE_ORDER]
        sems = [table.loc[table["size_class"] == cls, col].sem() for cls in SIZE_ORDER]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=sems, color=color, alpha=0.8,
               label=label, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(SIZE_ORDER)
    ax.set_ylabel("Top-N Jaccard overlap (mean ± SEM across microstructures in class)")
    ax.set_title("Geometric- vs. dissipation-bottleneck overlap, by microstructure size class")
    ax.legend(frameon=False, fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(out_dir / f"bottleneck_overlap_by_size_class.{ext}", **kwargs)
    plt.close(fig)


def fig_cross_check_pass_rate(table: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    ax = axes[0]
    rates = [table.loc[table["size_class"] == cls, "cross_check_pass"].mean() for cls in SIZE_ORDER]
    ns = [int((table["size_class"] == cls).sum()) for cls in SIZE_ORDER]
    bars = ax.bar(SIZE_ORDER, rates, color=[SIZE_COLORS[c] for c in SIZE_ORDER], alpha=0.8)
    for bar, n, r in zip(bars, ns, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.02, f"n={n}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fraction passing Task-5 cross-check\n(Spearman > 0.95 and median busy rel. err < 10%)")
    ax.set_title("Voxel-vs-network cross-check pass rate")
    _style_axes(ax)

    ax = axes[1]
    _scatter_by_size(ax, table, "n_throats", "cross_check_median_rel_err")
    ax.axhline(0.10, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(table["n_throats"].min(), 0.11, "pass threshold", fontsize=7, color="#666666")
    ax.set_xlabel("n_throats (proxy for microstructure size/complexity)")
    ax.set_ylabel("Median relative error, busy throats")
    ax.set_title("Does cross-check accuracy trend with microstructure size?")
    ax.legend(frameon=False, fontsize=9)
    _style_axes(ax)

    fig.suptitle("Task 5 (voxel-vs-network cross-check) across all microstructures -- still exploratory")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(out_dir / f"cross_check_pass_rate.{ext}", **kwargs)
    plt.close(fig)


def fig_current_concentration_by_size(table: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    rng = np.random.default_rng(0)
    for cls in SIZE_ORDER:
        sub = table[table["size_class"] == cls]
        x = rng.normal(SIZE_ORDER.index(cls), 0.06, size=len(sub))
        ax.scatter(x, sub["current_gini_top10pct_frac"], s=35, c=SIZE_COLORS[cls],
                   alpha=0.8, edgecolors="white", linewidths=0.4, zorder=3)
    ax.set_xticks(range(len(SIZE_ORDER)))
    ax.set_xticklabels(SIZE_ORDER)
    ax.set_ylabel("Fraction of total current carried by top 10% of throats")
    ax.set_title("How concentrated is current flow, by microstructure size class")
    _style_axes(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(out_dir / f"current_concentration_by_size.{ext}", **kwargs)
    plt.close(fig)


def write_report(table: pd.DataFrame, out_dir: Path):
    lines = []
    lines.append("# Cross-microstructure collation (Task 8, exploratory)\n")
    lines.append(
        "REPORT.md's Task 8 section withheld the aggregate/tau-correlation study "
        "because Task 5 (voxel-vs-network cross-check) did not pass on either of "
        "the 2 microstructures available at the time. This collation runs the "
        "same idea across all "
        f"{len(table)} microstructures now present in `data/`, but Task 5 still "
        f"does not pass in general -- only "
        f"{int(table['cross_check_pass'].sum())}/{len(table)} pass here (see "
        "`cross_check_pass_rate.png`). Treat every trend below as exploratory, "
        "not as a validated result the way Tasks 1-7 are.\n"
    )

    lines.append("## Coverage\n")
    for cls in SIZE_ORDER:
        n = int((table["size_class"] == cls).sum())
        lines.append(f"- {cls}: {n} microstructures")
    lines.append("")

    def corr_line(x, y, label):
        sub = table.dropna(subset=[x, y])
        if len(sub) < 3:
            return f"- {label}: too few points (n={len(sub)})"
        rho = sstats.spearmanr(sub[x], sub[y])
        return f"- {label}: Spearman r = {rho.statistic:.2f} (p={rho.pvalue:.2g}, n={len(sub)})"

    lines.append("## Headline correlations\n")
    lines.append(corr_line("porosity", "tau_reconciled", "porosity vs. tau_reconciled"))
    lines.append(corr_line("porosity", "constrictivity", "porosity vs. constrictivity"))
    lines.append(corr_line("constrictivity", "median_C_geomean",
                           "bulk constrictivity vs. median network throat C_geomean"))
    lines.append(corr_line("constrictivity", "frac_bottleneck_class",
                           "bulk constrictivity vs. fraction of throats classed 'bottleneck'"))
    lines.append(corr_line("F_voxel", "F_cc", "voxel-level F vs. network-level F"))
    lines.append("")

    lines.append("## Bottleneck/dissipation overlap by size class\n")
    for cls in SIZE_ORDER:
        sub = table[table["size_class"] == cls]
        if len(sub) == 0:
            continue
        lines.append(
            f"- {cls} (n={len(sub)}): mean edge-overlap Jaccard = "
            f"{sub['overlap_edge_jaccard'].mean():.3f}, mean throat-only-overlap Jaccard = "
            f"{sub['overlap_throat_only_jaccard'].mean():.3f}"
        )
    lines.append("")

    lines.append("## Files\n")
    lines.append("- `collated_table.csv` -- one row per microstructure, all fields aggregated below")
    lines.append("- `tau_constrictivity_vs_porosity.png/pdf`")
    lines.append("- `constrictivity_vs_bottleneck_geometry.png/pdf`")
    lines.append("- `F_voxel_vs_cc.png/pdf`")
    lines.append("- `bottleneck_overlap_by_size_class.png/pdf`")
    lines.append("- `cross_check_pass_rate.png/pdf`")
    lines.append("- `current_concentration_by_size.png/pdf`")

    (out_dir / "COLLATED_REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=here.parent.parent / "data")
    parser.add_argument("--tables-dir", type=Path, default=here.parent / "output" / "tables")
    parser.add_argument("--out", type=Path, default=here)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    table = build_collated_table(args.data_dir, args.tables_dir)
    if len(table) == 0:
        raise SystemExit(
            f"No microstructure tables found in {args.tables_dir} -- run run_analysis.py first."
        )
    table.to_csv(args.out / "collated_table.csv", index=False)

    fig_tau_vs_porosity(table, args.out)
    fig_constrictivity_vs_bottleneck_geometry(table, args.out)
    fig_F_comparison(table, args.out)
    fig_overlap_by_size_class(table, args.out)
    fig_cross_check_pass_rate(table, args.out)
    fig_current_concentration_by_size(table, args.out)
    write_report(table, args.out)

    print(f"[collate] {len(table)} microstructures collated -> {args.out}")
    print(f"[collate] cross-check pass rate: {table['cross_check_pass'].mean():.2f} "
          f"({int(table['cross_check_pass'].sum())}/{len(table)})")


if __name__ == "__main__":
    main()
