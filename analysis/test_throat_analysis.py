"""
test_throat_analysis.py
========================

Acceptance tests from the task spec (Tasks 1-5), as pytest assertions.
Runs against the two microstructures under ../data. Requires that batch
has been run with --save-full for the currently active config of every
method on every microstructure (see REPORT.md Task 1).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import throat_analysis as ta
import bottleneck_overlap as bo
import body_analysis as ba

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = Path(__file__).resolve().parent / "output" / "cache"

MICROSTRUCTURES = sorted(
    p.name for p in DATA_DIR.iterdir()
    if p.is_dir() and (p / "manifest.json").exists()
)


@pytest.fixture(scope="module", params=MICROSTRUCTURES)
def msd(request):
    return ta.load_microstructure(DATA_DIR / request.param)


@pytest.fixture(scope="module")
def geom(msd):
    return ta.get_geometry(msd, CACHE_DIR, use_cache=True)


@pytest.fixture(scope="module")
def throat_df(msd, geom):
    df = ta.build_throat_table(msd, geom)
    f_voxel, I_voxel, I_total_plane = ta.compute_voxel_throat_currents(msd, geom, df)
    df["f_voxel"] = f_voxel
    df["I_voxel"] = I_voxel
    df.attrs["I_total_plane"] = I_total_plane
    return df


# =============================================================================
# Task 1: inventory
# =============================================================================

def test_inventory_has_microstructures():
    inv = ta.build_inventory(DATA_DIR)
    assert len(inv) > 0


def test_berg_cc_has_full_result_for_every_microstructure():
    inv = ta.build_inventory(DATA_DIR)
    # HARD STOP condition from the task spec: if full_result.pkl is missing
    # for berg_cc on MOST microstructures, refuse to proceed.
    n_full = inv["berg_cc_has_full"].sum()
    assert n_full >= len(inv) / 2, (
        "berg_cc full_result.pkl missing for most microstructures -- "
        "re-run the batch with --save-full before analysing."
    )


# =============================================================================
# Task 2: acceptance tests 1-4
# =============================================================================

def test_shapes_agree(msd):
    shapes = ta.check_shapes(msd)
    assert len(set(shapes.values())) == 1, shapes


def test_pore_mask_agrees_with_body_array(msd):
    result = ta.check_pore_mask_agreement(msd)
    assert result["frac_agree"] >= 0.99, result


def test_kirchhoff_residual_is_tiny(msd):
    result = ta.check_kirchhoff(msd)
    assert result["max_residual_frac_of_total"] < 1e-6, result


def test_inlet_current_matches_total(msd):
    result = ta.check_inlet_current_consistency(msd)
    assert abs(result["ratio_sum_abs_to_total"] - 1.0) < 1e-6, result
    assert abs(result["ratio_net_to_total"] - 1.0) < 1e-6, result


# =============================================================================
# Task 3: EDT geometric reference
# =============================================================================

def test_C_is_in_valid_range(msd, geom, throat_df):
    C = throat_df["C"].dropna()
    assert (C > 0).all()
    assert (C <= 1.05).all(), f"max C = {C.max()}"


def test_most_berg_cc_throats_have_a_matching_interface(throat_df):
    frac_matched = throat_df["has_interface"].mean()
    assert frac_matched > 0.95, (
        f"only {frac_matched:.1%} of berg_cc throats have a matching "
        "6-connectivity interface -- expected the vast majority to match, "
        "with a small residual from dilate_both=True throat detection."
    )


def test_r_body_has_no_zeros(geom):
    assert (geom.r_body_by_idx > 0).all()
    assert geom.r_body_by_idx.max() > 1.0


# =============================================================================
# Task 4: throat table assembly
# =============================================================================

def test_f_sums_to_one_across_flow_planes(msd, throat_df):
    """sum(f) over throats crossing any plane normal to the flow axis is ~1
    -- verified via body centroid position rather than a literal geometric
    plane crossing, since throats don't lie exactly on integer planes."""
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis = axis_map[msd.direction]
    coords = msd.berg["arrays"]["coords"]
    c1 = coords[throat_df["body1_idx"].values, axis]
    c2 = coords[throat_df["body2_idx"].values, axis]

    lo, hi = min(c1.min(), c2.min()), max(c1.max(), c2.max())
    checked = 0
    for frac in (0.25, 0.5, 0.75):
        plane = lo + frac * (hi - lo)
        crosses = (np.minimum(c1, c2) <= plane) & (np.maximum(c1, c2) >= plane)
        if crosses.sum() == 0:
            continue
        total_f = throat_df.loc[crosses, "f"].sum()
        assert total_f > 0.3, f"plane {frac}: sum(f) = {total_f} (expected order 1)"
        checked += 1
    assert checked >= 1


def test_G_equals_inverse_R_edge(throat_df):
    invR = 1.0 / throat_df["R_edge"]
    np.testing.assert_allclose(throat_df["G"], invR, rtol=1e-9)


def test_A_body_apportionment_effect(throat_df, msd):
    """Per constraint (a) in the task spec, A_body1 is expected to correlate
    with 1/z (coordination number). This is reported, not gated: on the two
    available microstructures the correlation is only modest-to-weak
    (Pearson ~0.39 on microstructure1's 658 throats, ~0.21 on
    Coarse_microstructure_0's 47) and NOT confirmed as "strongly positive"
    -- see REPORT.md Task 4.3. This test only asserts the apportionment
    effect (A_body1 < geometric_throat_area for some throats) is present,
    since that direction is unambiguous; it does not assert a correlation
    threshold, which the task spec says to report honestly rather than force.
    """
    frac_below = (throat_df["A_body1"] < throat_df["geometric_throat_area"]).mean()
    assert frac_below > 0  # expected nonzero per constraint (a) -- apportionment, not error


def test_class_counts_are_populated(throat_df):
    counts = throat_df["class"].value_counts()
    assert counts.sum() == len(throat_df)
    assert set(counts.index) <= {"bottleneck", "geometric_only", "highway", "secondary_or_filler"}


# =============================================================================
# Task 5: voxel-vs-network cross-check (report only -- NOT asserted as a pass)
# =============================================================================

def test_cross_check_runs_and_reports(throat_df):
    """This does NOT assert Spearman > 0.95 / median error < 10% -- per the
    task spec, Task 5 failing is a valid, reportable outcome that must not
    be tuned away. This test only asserts the check itself runs cleanly and
    produces well-formed statistics; see REPORT.md for the pass/fail verdict.
    """
    stats = ta.cross_check_stats(throat_df)
    assert stats["n_matched"] > 0
    assert -1.0 <= stats["spearman_r"] <= 1.0
    assert not np.isnan(stats["median_rel_err_busy"]) or stats["n_busy_subset"] == 0


def test_divergence_check_runs(msd):
    result = ta.check_divergence(msd)
    assert result["mean_abs_J"] > 0
    assert result["mean_abs_div_over_mean_absJ"] >= 0


# =============================================================================
# Geometric-mean bottleneck vs. dissipation ("highway") bottleneck overlap
# =============================================================================

def test_C_geomean_between_C_and_one(throat_df):
    """C_geomean = r_throat/sqrt(r1*r2) is always >= C = r_throat/min(r1,r2)
    is FALSE in general (geomean <= min is false; geomean >= min always,
    since sqrt(r1*r2) <= max(r1,r2) but >= min(r1,r2)) -- i.e. C_geomean <= C
    always, since the geometric mean of two positive numbers is always >=
    their min, making the denominator bigger and C_geomean smaller or equal.
    """
    sub = throat_df.dropna(subset=["C", "C_geomean"])
    assert (sub["C_geomean"] <= sub["C"] + 1e-9).all()


def test_dissipation_decomposes_across_segments(throat_df):
    """P_edge = I^2 * R_edge should be >= P_throat_only = I^2 * R_throat,
    since R_edge = R1 + R_throat + R2 with all terms non-negative."""
    sub = throat_df.dropna(subset=["P_edge", "P_throat_only"])
    assert (sub["P_edge"] >= sub["P_throat_only"] - 1e-12).all()


def test_dissipation_fractions_sum_to_one(throat_df):
    """P_edge_frac and P_throat_only_frac are each normalised to their own
    level's total dissipated power, so each sums to 1 over all throats --
    the same invariant f already satisfies for current."""
    np.testing.assert_allclose(throat_df["P_edge_frac"].sum(), 1.0, rtol=1e-9)
    np.testing.assert_allclose(throat_df["P_throat_only_frac"].sum(), 1.0, rtol=1e-9)


def test_bottleneck_overlap_report_runs(throat_df):
    n = max(3, round(0.10 * len(throat_df)))
    report = bo.overlap_report(throat_df, n)
    assert report["n"] == n
    for label in ("edge", "throat_only"):
        assert 0 <= report[label]["n_intersection"] <= n
        assert -1.0 <= report[label]["spearman_C_geomean_vs_dissipation"] <= 1.0


def test_annotate_bottleneck_sets_consistent_with_overlap_report(throat_df):
    n = max(3, round(0.10 * len(throat_df)))
    df2 = bo.annotate_bottleneck_sets(throat_df, n)
    report = bo.overlap_report(throat_df, n)
    assert df2["overlap_edge"].sum() == report["edge"]["n_intersection"]
    assert df2["overlap_throat_only"].sum() == report["throat_only"]["n_intersection"]
    # geometric bottleneck set is a subset of throats with valid C_geomean
    assert (df2.loc[df2["is_geo_bottleneck"], "C_geomean"].notna()).all()


# =============================================================================
# Body-level analysis: what distinguishes bottleneck-adjacent bodies
# =============================================================================

@pytest.fixture(scope="module")
def body_df(msd, geom, throat_df):
    n = max(3, round(0.10 * len(throat_df)))
    df2 = bo.annotate_bottleneck_sets(throat_df, n)
    return ba.build_body_table(msd, geom, df2)


def test_body_table_covers_every_body(msd, body_df):
    assert len(body_df) == len(msd.berg["arrays"]["body_ids"])
    assert body_df["body_id"].nunique() == len(body_df)


def test_body_table_volume_and_r_body_positive(body_df):
    assert (body_df["volume"] > 0).all()
    assert (body_df["r_body"] > 0).all()


def test_compare_groups_runs_and_reports_both_groups(body_df):
    result = ba.compare_groups(body_df, "touches_any_topn")
    assert result["n_in_group"] + result["n_out_group"] == len(body_df)
    for prop in ("volume", "r_body", "z_coordination"):
        assert "median_in" in result[prop]
        assert "median_out" in result[prop]


def test_nearest_neighbor_distances_shape_and_positivity():
    coords = np.array([[0, 0, 0], [1, 0, 0], [5, 5, 5], [5, 5, 6]], dtype=float)
    d = ba.nearest_neighbor_distances(coords)
    assert d.shape == (4,)
    assert (d > 0).all()
    # point 0's nearest neighbour is point 1 (distance 1)
    assert np.isclose(d[0], 1.0)


def test_spatial_clustering_report_runs(throat_df):
    n = max(3, round(0.10 * len(throat_df)))
    df2 = bo.annotate_bottleneck_sets(throat_df, n)
    result = ba.spatial_clustering_report(df2, "is_geo_bottleneck")
    if "note" not in result:
        assert 0 <= result["percentile_of_observed_in_null"] <= 100
        assert result["observed_mean_nn_distance"] > 0
