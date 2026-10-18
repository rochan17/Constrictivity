# Berg RN: speedups, a `vanilla` / `short_circuit` toggle, and validation against TauFactor

Follow-up to `berg_rn_improvements_report.md` in this folder. That report proposed the speedups and the "1-Pass Geometric Short-Circuit" from scratch prototypes. This one implements them in `particulate_claude.py`, `berg_cc.py` and `pipeline.py`. I tested them on `TRACE_fine_test` (Fine_0–4), then ran all 25 microstructures in `data_new`, and then explained the remaining discrepancies against TauFactor with diagnostics that use the TauFactor potential field itself.

Environment: conda env `rbenv` (numpy 1.26.4, scipy 1.14.1, shapely 2.1.2, scikit-image 0.20.0, numba 0.59.1). All scripts and raw results are in `Test Run/rn_testbench/` (see §8).

---

## 0. Summary

1. **Speed: ~9.5× end to end (69 s → 7.3 s per 150³ sample, single core), with unchanged results.** In `vanilla` mode the merged body array, the body properties, the throat voxels and the throat areas are bit-for-bit identical to the old code on all 5 test samples. The network arrays (lengths, areas, volumes) are identical too. The only difference comes from the sparse solve: potentials agree to ≤1.2×10⁻¹⁵, and every reported metric to ≤2.5×10⁻¹⁴ relative.
2. **Toggle.** `rn_method='vanilla'` is the original centroid-star RN, exactly. `rn_method='short_circuit'` shortens the half-link lengths before the solve and keeps every volume exactly. The default is the single constant `particulate_claude.DEFAULT_RN_METHOD` (currently `'short_circuit'`). It can be overridden per run through the berg_cc config, and the pipeline records it in the cache hash, so the two variants never share a cached version.
3. **Merging moved into the berg_cc path.** `extract_throats_from_bodies_voxel_parallel(merge_bodies=True)` now runs the boundary-only and short-circuit body merges itself. `pipeline.run_berg_cc` no longer calls them directly, and `merge_bodies` is a hashed config key.
4. **Physics, fine media (10 samples).** Short-circuit removes the path-length error of the vanilla RN. The current-weighted network tortuosity matches TauFactor's flux-based τ to −1.2% on average (vanilla: +24%). 1/F goes from −38% to −6.6% on average.
5. **Physics, medium and coarse media (15 samples).** Here short-circuit makes 1/F *worse* (Medium +40%, Coarse +130%), and that is the expected and honest outcome. On these samples the vanilla RN only looked acceptable through a cancellation of errors: paths too long, constriction far too weak. The link-level comparison with the TauFactor field (§6.5) shows the RN overestimates the conductance of narrow necks (≤16 shared voxel faces) by 2–3.6×. Those necks carry 22–69% of the dissipation in Medium/Coarse samples, but only ~5% in Fine. Short-circuit fixes geometry, not constriction, so it cannot fix this. It also slightly over-shortens paths in these more irregular bodies (T_I −11% to −14%).

**Recommendation.** Use `short_circuit` as the default RN: it is the physically correct path-length correction, and it is the variant whose agreement with TauFactor does not depend on cancelling errors. Report Medium/Coarse 1/F values with the caveat in §6. The next modelling target is the conductance of narrow throats, not the path length.

---

## 1. What changed in the code

### 1.1 Files touched

| File | Change |
|---|---|
| `particulate_claude.py` | `_LabelGraph` + graph-based `merge_boundary_only_bodies_iterative` / `merge_short_circuit_bodies`; vectorised `compute_initial_body_props`; intersection crop in throat extraction; vectorised per-voxel hulls in `get_planar_area`; `merge_bodies=` in extraction; `compute_short_circuit_distances`; `rn_method=` / `direction=` in `calculate_network_effective_properties`; `d_override=` in `_compute_body_side`; new `PoreNetwork` fields `flow_axis`, `body_array` (merged labels), `rn_method` |
| `berg_cc.py` | `solve_potential_field` assembles the Dirichlet/pinned system from filtered COO triplets (no CSR mutation, no `SparseEfficiencyWarning`); `recover_segment_potentials` and `compute_local_iota_squared` vectorised; removed the leftover debug `print` of the whole augmented DAG in `_prepare_cc_augmented_dag` |
| `pipeline.py` | merges removed from `run_berg_cc` (now `merge_bodies=True` in the extraction call); `rn_method` and `direction` passed to the effective-property step; `network.body_array` handed to `compute_all_berg`; `rn_method` added to the berg_cc metrics; `run_microstructure` sets defaults for `merge_bodies` and `rn_method` in `berg_cc_config` so both are part of the cache key |

Your earlier uncommitted edits (`phase_id` 1 → 3 in `pipeline.py` / `berg_voxel_fast.py`, and the `analysis/` files) are untouched.

### 1.2 How to switch between the two RNs

```python
# one-off, any caller
pc.calculate_network_effective_properties(net, rn_method='vanilla')        # original
pc.calculate_network_effective_properties(net, rn_method='short_circuit')  # new

# batch runs: per-run override (cached as its own version)
python run_batch.py --data-dir data_new --cores-per-microstructure 4 \
    --berg-cc-config '{"rn_method": "vanilla"}'

# global revert: one line in particulate_claude.py
DEFAULT_RN_METHOD = 'vanilla'
```

Every caller that doesn't pass `rn_method` (for example `analysis/artificial/throat_removal.py`) follows `DEFAULT_RN_METHOD`. `merge_bodies` defaults to `False` in the library function, because `throat_removal.py` reuses the *unmerged* `body_array` labels for its bottleneck classification. It defaults to `True` in the pipeline. The merged labels are always available as `network.body_array`.

### 1.3 Cache consequences

`berg_cc_config` now always contains `merge_bodies` and `rn_method`, so the next `run_batch` run hashes differently for berg_cc and recomputes it. TauFactor and berg_voxel still resolve to their existing cached versions. This is deliberate and fixes a latent problem: the merge step added to `run_berg_cc` earlier was not part of the hash. A berg_cc version computed *before* the merges existed could therefore have been served as a cache hit for a config that now merges.

---

## 2. Speedups

### 2.1 Measured timings (Fine_0–4, 150³, `n_jobs=1`, mean of 5 samples)

Baseline = frozen copy of the modules as they were at the start of this work. Both columns were run with 5 samples concurrently on a 10-core machine, so the absolute numbers carry the same contention.

| Stage | Baseline (s) | New (s) | Speedup | Output |
|---|---:|---:|---:|---|
| `merge_boundary_only_bodies_iterative` | 8.25 | 0.055 | 150× | identical label array (same dtype) |
| `merge_short_circuit_bodies` | 28.6 | 0.056 | 510× | identical label array |
| `compute_initial_body_props` | 9.43 | 0.52 | 18× | identical (all fields, exact float equality) |
| extraction total (EDT, boundaries, body props, throats, assembly) | 22.2 | 6.0 | 3.7× | identical throat voxels and areas |
| `calculate_network_effective_properties` | 0.01 | 0.01 | — | identical |
| `compute_all_berg` | 0.08 | 0.06 | 1.3× | ≤2.5×10⁻¹⁴ relative on every metric |
| **Total** | **69.1** | **7.3** | **9.5×** | |

Through `pipeline.run_berg_cc`, vanilla Fine_0 gives 1/F = 0.1047, τ_c = 1.6578, 1/C_c = 0.6302. These equal the cached `version9` numbers.

### 2.2 Why each change is exact

- **Merges.** The old code tested adjacency with `binary_dilation(mask_a) & mask_b`, using the default 6-connected cross. Two labels pass that test exactly when they share a voxel face. `_LabelGraph` reads every such face pair in one pass, by comparing each voxel with its +1 neighbour along every axis. Merges are applied to the in-memory graph, and one lookup-table relabel is done at the end. The decision logic copies the original exactly:
  - The same sorted iteration order.
  - The same stale-within-a-pass face/neighbour snapshots.
  - Sets built with the same expressions, so `max()` ties resolve the same way. CPython set order depends only on hashes and insertion sequence, and `hash(np.int32(k)) == hash(k)`.
- **Body properties.** `ndi.center_of_mass` divides an integer-valued index sum by the voxel count. Doing it for all bodies at once with `labels=` performs the same exact integer sums, so the quotient is bit-identical. Max radii come from `ndi.maximum` (exact), and volumes and surface counts from `bincount`.
- **Intersection crop.** A throat voxel must lie in `dilate(b1) & dilate(b2)`, so it lies inside *both* padded boxes. A voxel on the edge of the intersection box is, by construction, outside the dilation of the body whose box defines that edge. Every voxel one step further in has all of its 6 neighbours inside the crop. So the dilation is never cut short. This holds for both `pad=2` (`dilate_both`) and `pad=1`.
- **Planar area.** `shapely.convex_hull(shapely.multipoints(proj2d))` builds the same GEOS geometries as the old per-voxel `MultiPoint(...).convex_hull` loop, from the same projected coordinates. The union is the same GEOS `UnaryUnion`. I did *not* use the previous report's translate-one-unit-hull trick: it changes the area at the 10⁻¹⁴ level, and I wanted vanilla to stay bit-exact.
- **Solver.** Dropping COO entries before assembly gives the same matrix values as zeroing rows and columns in a CSR matrix. The only difference is the stored explicit zeros, which can change SuperLU's pivot order. That is the source of the 10⁻¹⁵ differences.

### 2.3 What's left

About 3.4 s of the remaining ~6 s extraction is GEOS `union_all` inside the vox-projection area, over 691 throats. I kept it because the area definition is the one you want. In production the pipeline passes `n_jobs=cores_per_task`, and extraction parallelises over body pairs, so this term divides by the core count. The previous report's "watershed 2.13 s" stage doesn't exist in this code path: `body_array.npy` is precomputed.

---

## 3. The short-circuit RN

### 3.1 Why the vanilla RN is too resistive

Each throat k of body i is a half-link from the body centroid c_i to the throat centroid p_k. Its length is d_ik = |p_k − c_i| and its volume is V_ik = V_i / z_i (equal split). Its conductance is

  G_ik = σ A_ik / d_ik = σ V_ik / d_ik².

Current crossing body i from throat p to throat r is charged the star length d_ip + d_ir instead of something close to the chord |p_p − p_r|. Inlet and outlet bodies are worse: their node is the *centroid of the body's face patch*, so each throat is also charged a transverse detour along the equipotential plane. If every path is λ times too long *at fixed volume*, the area is λ times too small and the conductance λ² times too small. The measured shortening below is ≈ 0.77 (λ ≈ 1.30), which predicts a 1/F deficit of 1 − 1/λ² ≈ 41%. Vanilla on Fine media is at −38% on average.

### 3.2 The rules (`compute_short_circuit_distances`)

Only d_ik changes. V_ik is untouched, so A_ik = V_ik / d̃_ik and total pore volume Ω is conserved exactly (verified: Ω is identical between the two variants on every sample).

- **Dirichlet (inlet/outlet) bodies**, classified with the same test as `berg_cc.get_inlet_outlet_indices`: d̃_ik = min(d_ik, max(|p_k,x − x_plane|, 0.5)). The whole contact patch is one equipotential, so current leaves it perpendicular to the plane.
- **Interior bodies with ≥ 2 throats:**

  w_pr = A_r (|Δx_pr| + 0.5),  ratio_p = Σ_r w_pr min(chord_pr, star_pr) / Σ_r w_pr star_pr,  d̃_ip = d_ip · clip(ratio_p, 0.25, 1).

  The weight is a geometry-only proxy for how much current throat p exchanges with throat r: larger, and more axially separated, partner throats carry more of it.
- **Dead-end bodies (one throat):** unchanged.

The Berg effective geometry (α-formula, `RATIO_THRESHOLD`, uniform-pipe fallback) is then built exactly as before on the shortened lengths.

### 3.3 Ablation and sensitivity (Fine_0–3, equal volume split)

Errors are relative to TauFactor. Mean (MAE in brackets).

| Variant | 1/F | τ_c (Berg) | T_I (current-weighted) | 1/C_c |
|---|---:|---:|---:|---:|
| vanilla | −38.6 (38.6) | +30.2 | +23.9 | +7.7 (7.8) |
| **short_circuit (shipped)** | **−8.4 (8.4)** | **+3.2** | **−1.0** | **+1.0 (4.4)** |
| plane rule only | −35.6 | +25.5 | +19.2 | +5.0 |
| chord rule only | −13.9 | +8.3 | +3.6 | +4.4 |
| weights = 1 | +4.1 (5.5) | −3.3 | −7.8 | +0.7 |
| weights = A_r | −0.5 (4.0) | −1.5 | −5.8 | −0.1 |
| weights = \|Δx\| + 0.5 | −6.8 | +2.5 | −1.8 | +1.3 |
| clip at 0 / 0.5 | unchanged | unchanged | unchanged | ±0.4 |
| pure axial projection (lower bound) | +34.3 | −19.7 | −20.7 | −10.4 |

Reading this:
- The interior chord rule does most of the work. The plane rule adds the last ~5 points.
- The clip never binds: ≤ 2 of ~1,400 half-links hit 0.25. Mean shortening factor: 0.77 interior, 0.75 boundary.
- **I didn't pick the variant with the best 1/F.** Area-only weights give −0.5%, but only because the paths become too short (T_I −5.8%), compensating a constriction error with a tortuosity error. The shipped A·|Δx| weighting is the one whose *paths* match TauFactor's flux-weighted tortuosity (T_I −1%). That is the quantity the short-circuit is meant to correct.
- The pure axial projection overshoots massively. Real streamlines do move transversely inside bodies.

---

## 4. Results on TRACE_fine_test (testbench)

Config: `min_throat_size=1`, `dilate_both`, `skip_surface_surface`, `vox_projection`, α = 0.5, `split_volume_equal=True`, `flow_decomposition` (= your cached `version9`). TauFactor values come from the cached runs. Fine_4 has no TauFactor in `TRACE_fine_test`, but it does in `data_new` (§5).

| Sample | Variant | 1/F (err) | τ_c Berg (err) | T_I (err) | 1/C_c (err) | fallback throats |
|---|---|---|---|---|---|---:|
| Fine_0 (TF 0.1817 / 1.272 / 0.631) | vanilla | 0.1047 (−42.4%) | 1.658 (+30.3%) | 1.591 (+25.1%) | 0.630 (−0.1%) | 483 |
| | short_circuit | 0.1552 (−14.6%) | 1.314 (+3.3%) | 1.270 (−0.2%) | 0.587 (−6.9%) | 411 |
| Fine_1 (0.1544 / 1.296 / 0.610) | vanilla | 0.0922 (−40.3%) | 1.701 (+31.3%) | 1.625 (+25.4%) | 0.656 (+7.5%) | 528 |
| | short_circuit | 0.1406 (−8.9%) | 1.341 (+3.5%) | 1.290 (−0.5%) | 0.622 (+1.9%) | 471 |
| Fine_2 (0.1126 / 1.353 / 0.511) | vanilla | 0.0745 (−33.8%) | 1.753 (+29.6%) | 1.661 (+22.8%) | 0.589 (+15.1%) | 552 |
| | short_circuit | 0.1109 (−1.5%) | 1.388 (+2.6%) | 1.328 (−1.9%) | 0.549 (+7.5%) | 464 |
| Fine_3 (0.1187 / 1.345 / 0.556) | vanilla | 0.0735 (−38.1%) | 1.746 (+29.8%) | 1.647 (+22.4%) | 0.602 (+8.3%) | 561 |
| | short_circuit | 0.1086 (−8.5%) | 1.391 (+3.4%) | 1.324 (−1.6%) | 0.564 (+1.5%) | 475 |

These match the previous report's "1-Pass Geometric SC" rows to the third decimal. The small differences come from using the exact `get_inlet_outlet_indices` thresholds for the plane rule.

---

## 5. Results on data_new (25 samples)

TauFactor references come from `data_new/all_runs_versions.csv` (latest version per sample). Same berg_cc config as §4. All 25 samples ran with no errors in all four variants, in about 40 s wall time per variant for the whole set (6 in parallel).

### 5.1 Per class (geometric-mean ratio of 1/F to D_eff; arithmetic mean of T_I error)

| Class | Variant | 1/F ÷ D_TF | T_I error | fallback throats |
|---|---|---:|---:|---:|
| Fine (10) | vanilla | 0.618 | +24.3% | 68% |
| | **short_circuit** | **0.934** | **−1.2%** | 58% |
| Medium (10) | vanilla | 0.968 | +12.1% | 56% |
| | short_circuit | 1.397 | −10.7% | 46% |
| Coarse (5) | vanilla | 1.620 | +8.5% | 59% |
| | short_circuit | 2.282 | −14.4% | 48% |

### 5.2 Per sample

| Sample | TF D_eff | vanilla 1/F | short_circuit 1/F | T_I err (van) | T_I err (SC) | TF C | 1/C_c (van) | 1/C_c (SC) | φ_c / ε |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| Coarse 0 | 0.0038 | 0.0061 (+60%) | 0.0087 (+127%) | +12.6% | −11.9% | 0.039 | 0.192 | 0.166 | 0.52 |
| Coarse 1 | 0.0024 | 0.0042 (+73%) | 0.0055 (+128%) | +4.5% | −12.6% | 0.018 | 0.129 | 0.112 | 0.52 |
| Coarse 2 | 0.0026 | 0.0035 (+32%) | 0.0053 (+101%) | +17.5% | −12.3% | 0.054 | 0.289 | 0.245 | 0.37 |
| Coarse 7 | 0.0040 | 0.0059 (+48%) | 0.0081 (+105%) | +11.8% | −14.2% | 0.038 | 0.185 | 0.148 | 0.43 |
| Coarse 9 | 0.0006 | 0.0013 (+106%) | 0.0018 (+190%) | −3.8% | −21.1% | 0.018 | 0.175 | 0.165 | 0.20 |
| Fine 0 | 0.1817 | 0.1047 (−42%) | 0.1552 (−15%) | +25.1% | −0.2% | 0.631 | 0.630 | 0.587 | 0.98 |
| Fine 1 | 0.1544 | 0.0922 (−40%) | 0.1406 (−9%) | +25.4% | −0.5% | 0.610 | 0.656 | 0.622 | 0.96 |
| Fine 2 | 0.1126 | 0.0745 (−34%) | 0.1109 (−1%) | +22.8% | −1.9% | 0.511 | 0.589 | 0.549 | 0.96 |
| Fine 3 | 0.1187 | 0.0735 (−38%) | 0.1086 (−8%) | +22.4% | −1.6% | 0.556 | 0.602 | 0.564 | 0.96 |
| Fine 4 | 0.1498 | 0.0899 (−40%) | 0.1363 (−9%) | +26.6% | +0.7% | 0.597 | 0.646 | 0.608 | 0.99 |
| Fine 5 | 0.1258 | 0.0788 (−37%) | 0.1213 (−4%) | +25.5% | −1.5% | 0.564 | 0.639 | 0.601 | 0.95 |
| Fine 6 | 0.1407 | 0.0870 (−38%) | 0.1333 (−5%) | +24.8% | −0.7% | 0.575 | 0.634 | 0.602 | 0.98 |
| Fine 7 | 0.1191 | 0.0767 (−36%) | 0.1154 (−3%) | +22.0% | −3.3% | 0.555 | 0.597 | 0.560 | 0.97 |
| Fine 8 | 0.1198 | 0.0756 (−37%) | 0.1181 (−1%) | +25.5% | −1.5% | 0.536 | 0.621 | 0.589 | 0.96 |
| Fine 9 | 0.1254 | 0.0768 (−39%) | 0.1140 (−9%) | +22.7% | −1.2% | 0.566 | 0.610 | 0.580 | 0.95 |
| Medium 0 | 0.0749 | 0.0554 (−26%) | 0.0807 (+8%) | +21.0% | −3.8% | 0.351 | 0.479 | 0.438 | 0.91 |
| Medium 1 | 0.0395 | 0.0375 (−5%) | 0.0534 (+35%) | +12.0% | −10.1% | 0.224 | 0.381 | 0.340 | 0.82 |
| Medium 2 | 0.0257 | 0.0247 (−4%) | 0.0364 (+41%) | +10.3% | −13.1% | 0.208 | 0.375 | 0.339 | 0.75 |
| Medium 3 | 0.1181 | 0.0808 (−32%) | 0.1167 (−1%) | +19.9% | −3.0% | 0.445 | 0.526 | 0.488 | 0.91 |
| Medium 4 | 0.0255 | 0.0240 (−6%) | 0.0345 (+36%) | +8.6% | −13.3% | 0.191 | 0.311 | 0.283 | 0.81 |
| Medium 5 | 0.0251 | 0.0279 (+11%) | 0.0406 (+62%) | +8.5% | −14.1% | 0.163 | 0.305 | 0.285 | 0.81 |
| Medium 6 | 0.0208 | 0.0236 (+14%) | 0.0332 (+60%) | +8.0% | −13.6% | 0.178 | 0.337 | 0.298 | 0.79 |
| Medium 7 | 0.0359 | 0.0326 (−9%) | 0.0467 (+30%) | +13.2% | −9.2% | 0.241 | 0.363 | 0.326 | 0.87 |
| Medium 8 | 0.0165 | 0.0211 (+27%) | 0.0311 (+88%) | +10.1% | −13.6% | 0.128 | 0.284 | 0.258 | 0.76 |
| Medium 9 | 0.0271 | 0.0309 (+14%) | 0.0439 (+62%) | +9.0% | −13.0% | 0.179 | 0.353 | 0.308 | 0.81 |

Medium_0 and Medium_3 have the highest TauFactor constrictivity (C ≈ 0.35 and 0.45) of their class. They behave like Fine samples (SC 1/F within 8%). So the split is about how constricted the pore space is, not about the class label.

---

## 6. Why the discrepancies exist

### 6.1 The quantities are not defined the same way

- **Tortuosity.** TauFactor's `tau_reconciled` = ⟨|J|⟩ / |⟨J_x⟩| over pore voxels. For a field split into streamtubes Γ carrying current I_Γ along length L_Γ, ∫|J| dV = Σ I_Γ L_Γ and ∫J_x dV = Σ I_Γ Δs. So τ_rec is the **current-weighted mean of L_Γ/Δs**. The like-for-like network quantity is T_I = Σ I_Γ L_Γ / (I Δs), which the testbench reports. Berg's τ²_c is instead a **volume-weighted mean of (Δs/L_Γ)²**, and 1/√τ²_c is systematically larger than T_I, since volume weighting emphasises slow, long tubes. The gap is +3–5% on Fine samples and +6–13% on Medium/Coarse (e.g. Coarse_0 SC: 1.631 vs 1.444), where path lengths are more widely spread. Comparing τ_c directly with TauFactor's τ therefore mixes a definitional offset into the error. This is why τ_c looks +3% off on Fine while T_I is −1%.
- **Constrictivity.** TauFactor's constrictivity is defined to close D = ε C / T² with the *total* porosity ε. It therefore absorbs everything not explained by T, including dead-end and non-conducting pore volume. Berg's 1/C_c closes 1/F = φ_c τ²_c / C_c with the *conducting* porosity φ_c. The two agree only when φ_c ≈ ε, which holds for Fine (0.97) but not Medium (0.82) or Coarse (0.39).

### 6.2 An exact error budget

On the network, 1/F = φ_c · τ²_c / C_c holds exactly (Berg 2012, Eq. 17). TauFactor defines D = ε C_TF / T_TF². So

  (1/F_RN) / D_TF = (φ_c / ε) × ((1/C_c) / C_TF) × (T_TF / τ_c,B)².

The identity holds on every sample to ≤0.5%. The residual comes from Ω_c being counted from streamtubes in one place and from currents in the other.

| Class | Variant | 1/F ratio | = φ_c/ε | × constriction | × tortuosity |
|---|---|---:|---:|---:|---:|
| Fine | vanilla | 0.618 | 0.967 | 1.093 | **0.585** |
| | short_circuit | 0.934 | 0.967 | 1.029 | 0.939 |
| Medium | vanilla | 0.968 | 0.824 | **1.688** | **0.696** |
| | short_circuit | 1.397 | 0.824 | 1.528 | 1.109 |
| Coarse | vanilla | 1.620 | 0.386 | **6.172** | **0.679** |
| | short_circuit | 2.282 | 0.386 | 5.334 | 1.108 |

### 6.3 Fine media: only the geometry was wrong

The vanilla error is almost entirely the tortuosity factor (0.585 ≈ 1/1.31²), as predicted in §3.1. Short-circuit moves it to 0.94. The constriction and porosity factors stay within a few percent of 1. The −6.6% average 1/F deficit that remains is split between:
- (a) the Berg τ_c definitional offset (§6.1), which by itself costs ≈ (1/1.035)² ≈ −7%. This is not a flaw in the paths: T_I is within 1.2%;
- (b) a φ_c/ε of 0.97: the RN counts ~3% of the pore volume as non-conducting, while TauFactor's constrictivity absorbs that;
- (c) a sample-to-sample constriction scatter of ±7% (Fine_0 −7%, Fine_8 +10%).

A ~0.7% systematic also comes from `delta_s = shape[axis] = 150`, while the RN's Dirichlet nodes sit at voxel centres 0 and 149 (149 voxels apart). I left this unchanged so vanilla stays exact.

### 6.4 Medium/Coarse: vanilla's agreement was a cancellation of errors

For Medium, vanilla's 1/F ratio of 0.968 looks good. But it is 0.824 × 1.688 × 0.696: paths 20% too long cancel a constriction that is 69% too weak. For Coarse the constriction factor is 6.2× and swamps everything. Short-circuit removes most of the tortuosity error, so the constriction error becomes visible. That is why SC 1/F is worse there. No path-length model can fix a constriction error. Pretending otherwise, for example by keeping vanilla because its Medium mean looks better, would be tuning to the wrong physics.

Short-circuit also over-shortens paths in these samples: T_I is −11% to −14% (tortuosity factor 1.11). The chord between two throats is a lower bound on the streamline length only inside a convex body. In the larger, more irregular bodies of Medium/Coarse samples (Coarse: 144 bodies per 150³ vs ~357 for Fine), the chord can cut through solid. The truth lies between the star (vanilla, +8–12%) and the chord (SC, −11–14%). This is a limit of the geometric heuristic: it uses no information about body shape.

### 6.5 Where the constriction is lost: link-level test against the TauFactor field

To find *which* links are wrong, I compared every interior link's RN conductance G^RN with the conductance the TauFactor field gives it:
- I_ij^TF = sum of finite-volume face fluxes over all voxel faces shared by bodies i and j (unit conductance per face, the same normalisation as the RN with σ = 1);
- ΔΦ^TF = difference of body-mean potentials;
- G^TF = I^TF / ΔΦ^TF.

This decomposition is power-exact: by a discrete Green identity, Σ_links I·ΔΦ plus the Dirichlet-boundary terms equals the total dissipation, so no dissipation is hidden inside bodies. The ratios are aggregated as dissipation-weighted geometric means. TauFactor was solved on `body_array > 0`; Fine_0 uses the field cached in `TRACE_fine_test/.../version1`.

| Sample (TF C) | Variant | G^RN/G^TF all links | fallback links | Berg links | narrow (≤16 faces) | wide (>16 faces) | share of dissipation in narrow links |
|---|---|---:|---:|---:|---:|---:|---:|
| Fine_0 (0.63) | vanilla | 0.47 | 0.39 | 0.82 | 1.38 | 0.44 | 5% |
| | short_circuit | 0.70 | 0.58 | 1.16 | 2.01 | 0.66 | |
| Medium_0 (0.35) | vanilla | 0.66 | 0.53 | 1.02 | 1.16 | 0.56 | 23% |
| | short_circuit | 0.96 | 0.73 | 1.30 | 1.57 | 0.82 | |
| Medium_8 (0.13) | vanilla | 1.15 | 0.52 | 1.51 | 2.17 | 0.70 | 44% |
| | short_circuit | 1.73 | 0.81 | 2.19 | 3.07 | 1.10 | |
| Coarse_0 (0.04) | vanilla | 1.21 | 1.16 | 1.23 | 2.28 | 1.01 | 22% |
| | short_circuit | 1.94 | 1.78 | 2.01 | 3.59 | 1.64 | |
| Coarse_9 (0.02) | vanilla | 2.63 | 7.83 | 2.24 | 2.23 | 3.79 | 69% |
| | short_circuit | 3.57 | 12.07 | 3.01 | 3.06 | 5.05 | |

Body-mean potentials are not exactly the RN's node potentials, so read these as relative comparisons. The absolute level of the "all links" column shouldn't be compared directly with the global 1/F. The trends are robust, though:

1. **Narrow necks are always over-conductive in the RN (×1.4–3.6), and their share of the dissipation grows from 5% (Fine) to 22–69% (Medium_8, Coarse).** That is what turns a small, harmless bias in Fine media into a large 1/F overestimate in constricted media.
2. In Fine media the opposite bias dominates. The uniform-pipe fallback links (72–77% of the dissipation) are *under*-conductive, and short-circuit partly corrects them. That is consistent with §6.3.
3. Two mechanisms make narrow links too conductive, and both can be identified:
   - *Area.* For throats with ≤4 shared faces, the vox-projection area is 2.0–3.0× the shared-face count (the cross-section the finite-volume solve actually conducts through). For 5–16 faces it is 1.4–1.5×, and for >16 faces it is ≈1. Weighted by current, though, the area excess over a whole sample is only 0–9% (Fine 1.00, Medium 1.09, Coarse 1.07). So area alone cannot explain a 2–5× constriction miss. Its correlation with the constriction factor across the 25 samples is r = 0.46.
   - *Constriction scaling.* In the Berg segment model, the throat resistance per side is α d / (σ √(A_itk A_tk)), so the conductance scales like √A_tk. That is the right scaling for a short orifice in a wide body (Holm/Maxwell spreading resistance R ≈ 1/(4σa) per side, and it matches the Berg term to within ~10% for compact bodies). It is too weak for a *channel-like* neck, whose resistance scales with ℓ/A_tk. Near the percolation threshold (Coarse: D_eff ~10⁻³, τ_TF up to 210) transport runs through such necks. This is my physical interpretation, not a measured separation. It is consistent with the area-weighted volume split (where a link's conductance scales ∝ A_tk through its volume share) cutting the Coarse constriction factor from 5.3 to 2.1 (§6.6).

### 6.6 Volume split (`split_volume_equal`)

| Class | short_circuit, equal split: 1/F ratio (C factor) | short_circuit, area-weighted: 1/F ratio (C factor) |
|---|---|---|
| Fine | 0.934 (1.03) | 0.883 (1.00) |
| Medium | 1.397 (1.53) | 1.135 (1.23) |
| Coarse | 2.282 (5.33) | 0.989 (2.13) |

The area-weighted split looks much better on Coarse 1/F. But it still carries a 2.1× constriction error that the over-shortened paths (1.13) and the porosity factor (0.41) happen to cancel. It also sends 82% of Fine throats into the fallback. I left the default at `True` (equal split). Choosing the split per microstructure class would be tuning, not physics. If you want a single change that targets the real problem, the constriction law for narrow necks (§6.5) is where to look.

### 6.7 Smaller effects

- TauFactor itself. My re-solves on `body_array > 0` give D_eff within 0–9% of the CSV values (Coarse_0: 0.00417 vs 0.00384). The difference comes from the CSV runs' `segmented_percolated.npy` input and convergence settings. For the very low-D Coarse samples, the reference is uncertain at roughly that level.
- Network porosity Ω/V differs from TauFactor's ε by ≤0.2% on Fine and ≤5% on Coarse_9, because `body_array` and the percolated segmentation differ slightly.

---

## 7. Not done / caveats

- **The 2-pass streamtube re-solve** from the previous report isn't implemented. That report found it worse than the 1-pass rule on Fine_0–3, and a geometry-only rule avoids an extra solve.
- **Cauchy area**: not used, as you asked. Everything here is `vox_projection`.
- **Previous report's bugs §4.1 (possible infinite loop in `_flow_decomposition`) and §4.3 (a body touching both faces is neither inlet nor outlet)**: I didn't fix these, since they are outside this change. `flow_decomposition` terminated on all 25 + 5 samples here.
- **The short-circuit constants** (weights, the 0.5 offset, clip 0.25, plane floor 0.5) are heuristic. §3.3 shows the clip and floor don't matter. The weighting does, and it was chosen to match the flux-weighted tortuosity, not 1/F.
- **Merging**: the merge code itself is unchanged in behaviour. It is now just called from inside extraction.

---

## 8. Reproducing

Everything is in `Test Run/rn_testbench/`. Run with `/Users/rochan/miniconda3/envs/rbenv/bin/python`.

| Script | Purpose |
|---|---|
| `bench.py <module_dir> <sample_dir> <out.pkl> rn_method=… split_volume_equal=…` | stage-timed run of the whole berg_cc path; saves every intermediate |
| `compare.py base.pkl new.pkl` | bit-level equivalence report (§2) |
| `run_variants.sh <data_root> <tag> key=val…` | runs bench.py over every sample folder |
| `table.py`, `budget.py` (+ `cache_lookup.py`) | tables vs TauFactor and the exact error budget (§4–6.2) |
| `ablation.py` | §3.3 |
| `throat_faces.py` | vox-projection area vs shared-face count (§6.5) |
| `run_tf.py`, `link_conductance.py` | TauFactor field + link-level conductance test (§6.5) |

`results/` holds the JSON outputs, the budget CSVs for all four data_new variants, and the baseline/new timing JSONs. To regenerate the baseline, point `bench.py` at a checkout of `origin/main` (commit `78c06de`, the parent of the speedup commit on the `short_circuit` branch). Its code is the pre-change version used for every "baseline" number here. For example, `git worktree add /tmp/rn_base 78c06de`, then `python bench.py /tmp/rn_base <sample> base.pkl`. The scripts find the repo root relative to their own location, and `run_variants.sh` takes the interpreter from `$PY`.
