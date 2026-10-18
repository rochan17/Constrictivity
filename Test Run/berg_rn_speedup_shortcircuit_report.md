# Berg RN: Speedups, a `vanilla` / `short_circuit` Toggle, and Validation Against TauFactor

Follow-up to [`berg_rn_improvements_report.md`](berg_rn_improvements_report.md). That report proposed the speedups and the "1-Pass Geometric Short-Circuit" from scratch prototypes. This one **implements** them in [`particulate_claude.py`](../particulate_claude.py), [`berg_cc.py`](../berg_cc.py) and [`pipeline.py`](../pipeline.py). I tested them on `TRACE_fine_test` (`Fine_microstructure_0`–`4`, $150^3$ voxels), then ran all **25** microstructures in `data_new`, and then explained the remaining discrepancies against TauFactor with diagnostics built on the TauFactor potential field itself.

**Environment:** conda env `rbenv` (numpy 1.26.4, scipy 1.14.1, shapely 2.1.2, scikit-image 0.20.0, numba 0.59.1). All scripts and raw results are in [`rn_testbench/`](rn_testbench/) (§8).

---

## 0. Summary

| # | Result | Evidence |
| :--- | :--- | :--- |
| 1 | **Speed: $69.1\text{ s} \to 7.3\text{ s}$ per $150^3$ sample ($9.5\times$), with unchanged results** | Vanilla mode is bit-identical through the network arrays on all 5 test samples; metrics agree to $\le 2.5\times10^{-14}$ (§2) |
| 2 | **Toggle:** `rn_method = 'vanilla' \| 'short_circuit'`, default set by one constant | `DEFAULT_RN_METHOD` in `particulate_claude.py`; recorded in the cache hash (§1.2) |
| 3 | **The body merges now run inside the berg_cc extraction step** | `extract_throats_from_bodies_voxel_parallel(merge_bodies=True)` (§1.1) |
| 4 | **Fine media (10):** short-circuit removes the path-length error | $T_I$ error $+24.3\% \to -1.2\%$; $1/F$ error $-38\% \to -6.6\%$ (§5) |
| 5 | **Medium / Coarse media (15):** short-circuit makes $1/F$ worse ($+40\%$, $+128\%$) | Vanilla's agreement there was a **cancellation of errors**; the real error is constriction at narrow necks (§6.4–6.5) |

> [!IMPORTANT]
> **Recommendation:** use `short_circuit` as the default RN. It is the physically correct path-length correction, and it is the variant whose agreement with TauFactor does not depend on cancelling errors. Quote Medium/Coarse $1/F$ values with the caveat in §6. The next modelling target is the **conductance of narrow throats**, not the path length.

---

## 1. What Changed in the Code

### 1.1 Files touched

| File | Function / object | Change |
| :--- | :--- | :--- |
| `particulate_claude.py` | `_LabelGraph` (new) | 6-connected label adjacency graph, built in one $O(N_{\text{vox}})$ pass |
| | `merge_boundary_only_bodies_iterative`, `merge_short_circuit_bodies` | Run on `_LabelGraph` instead of per-label full-volume dilations |
| | `compute_initial_body_props` | Labelled reductions (`bincount`, `ndi.center_of_mass` / `ndi.maximum` with `labels=`) |
| | `extract_throats_from_bodies_voxel_parallel` | Intersection crop per body pair; new `merge_bodies=` option |
| | `get_planar_area` | Vectorised per-voxel hulls (`shapely.convex_hull(shapely.multipoints(...))`) |
| | `compute_short_circuit_distances` (new) | Short-circuited half-link lengths $\tilde d_{ik}$ |
| | `calculate_network_effective_properties` | New `rn_method=` and `direction=` arguments |
| | `_compute_body_side` | New `d_override=` argument |
| | `PoreNetwork` | New fields `flow_axis`, `body_array` (merged labels), `rn_method` |
| `berg_cc.py` | `solve_potential_field` | Dirichlet/pinned system assembled from filtered COO triplets (no CSR mutation, no `SparseEfficiencyWarning`) |
| | `recover_segment_potentials`, `compute_local_iota_squared` | Vectorised |
| | `_prepare_cc_augmented_dag` | Removed the leftover debug `print` of the whole augmented DAG |
| `pipeline.py` | `run_berg_cc` | Merges moved into the extraction call (`merge_bodies=True`); passes `rn_method`, `direction`; uses `network.body_array`; reports `rn_method` in its metrics |
| | `run_microstructure` | `berg_cc_config.setdefault("merge_bodies", True)` and `setdefault("rn_method", DEFAULT_RN_METHOD)`, so both are hashed |

### 1.2 Switching between the two RNs

```python
# one-off, any caller
pc.calculate_network_effective_properties(net, rn_method='vanilla')        # original
pc.calculate_network_effective_properties(net, rn_method='short_circuit')  # new

# global revert: one line in particulate_claude.py
DEFAULT_RN_METHOD = 'vanilla'
```

```bash
# batch runs: per-run override (cached as its own version)
python run_batch.py --data-dir data_new --cores-per-microstructure 4 \
    --berg-cc-config '{"rn_method": "vanilla"}'
```

> [!NOTE]
> Every caller that doesn't pass `rn_method` (for example `analysis/artificial/throat_removal.py`) follows `DEFAULT_RN_METHOD`. `merge_bodies` defaults to `False` in the library function, because `throat_removal.py` reuses the **unmerged** `body_array` labels for its bottleneck classification. It defaults to `True` in the pipeline. The merged labels are always available as `network.body_array`.

### 1.3 Cache consequences

`berg_cc_config` now always contains `merge_bodies` and `rn_method`, so the next `run_batch` run hashes differently for berg_cc and recomputes it. TauFactor and berg_voxel still resolve to their existing cached versions.

> [!WARNING]
> This is deliberate and fixes a latent problem. The merge step added to `run_berg_cc` earlier was **not** part of the hash, so a berg_cc version computed *before* the merges existed could have been served as a cache hit for a config that now merges.

---

## 2. Speedups: $69.1\text{ s} \to 7.3\text{ s}$ ($9.5\times$)

### 2.1 Measured timings

Fine_0–4, $150^3$, `n_jobs=1`, mean over the 5 samples. **Baseline** is a frozen copy of the modules as they were at the start of this work. Both columns were measured with 5 samples running concurrently on a 10-core machine, so they carry the same contention.

| Pipeline stage | Function | Baseline (s) | New (s) | Speedup | Output equivalence |
| :--- | :--- | ---: | ---: | ---: | :--- |
| 1. Boundary merge | `merge_boundary_only_bodies_iterative` | 8.25 | **0.055** | **$150\times$** | Identical label array (same dtype) |
| 2. Short-circuit merge | `merge_short_circuit_bodies` | 28.6 | **0.056** | **$510\times$** | Identical label array |
| 3. Body properties | `compute_initial_body_props` | 9.43 | **0.52** | **$18\times$** | Identical (every field, exact float equality) |
| 4. Extraction total (EDT, boundaries, body props, throats, assembly) | `extract_throats_from_bodies_voxel_parallel` | 22.2 | **6.0** | **$3.7\times$** | Identical throat voxels and areas |
| 5. Effective geometry | `calculate_network_effective_properties` | 0.01 | 0.01 | — | Identical |
| 6. Solve + Berg | `compute_all_berg` | 0.08 | 0.06 | $1.3\times$ | $\le 2.5\times10^{-14}$ relative on every metric |
| **Total** | | **69.1** | **7.3** | **$9.5\times$** | |

Through `pipeline.run_berg_cc`, vanilla Fine_0 gives $1/F = 0.1047$, $\tau_c = 1.6578$, $1/C_c = 0.6302$. These equal the cached `version9` numbers.

### 2.2 Why each change is exact

#### Merges
The old code tested adjacency with `binary_dilation(mask_a) & mask_b`, using the default 6-connected cross. Two labels pass that test **if and only if they share a voxel face**. `_LabelGraph` reads every such face pair in one pass, by comparing each voxel with its $+1$ neighbour along every axis. Merges are applied to the in-memory graph, and one lookup-table relabel `lut[body_array]` is done at the end. The decision logic copies the original exactly:
- the same sorted iteration order;
- the same stale-within-a-pass face/neighbour snapshots;
- sets built with the same expressions, so `max()` ties resolve the same way. CPython set order depends only on hashes and insertion sequence, and `hash(np.int32(k)) == hash(k)`.

#### Body properties
`ndi.center_of_mass` divides an integer-valued index sum by the voxel count. Doing it for all bodies at once with `labels=` performs the same exact integer sums, so the quotient is bit-identical. Max radii come from `ndi.maximum` (exact), and volumes and surface counts from `bincount`.

#### Intersection crop
A throat voxel must lie in $\mathrm{dilate}(b_1) \cap \mathrm{dilate}(b_2)$, so it lies inside **both** padded boxes. A voxel on the edge of the intersection box is, by construction, outside the dilation of the body whose box defines that edge. Every voxel one step further in has all 6 of its neighbours inside the crop. So the dilation is never cut short. This holds for both `pad=2` (`dilate_both`) and `pad=1`.

#### Planar area
`shapely.convex_hull(shapely.multipoints(proj2d))` builds the same GEOS geometries as the old per-voxel `MultiPoint(...).convex_hull` loop, from the same projected coordinates, and the union is the same GEOS `UnaryUnion`. I deliberately did **not** use the previous report's translate-one-unit-hull trick: it changes the area at the $10^{-14}$ level, and I wanted vanilla to stay bit-exact.

#### Solver
Dropping COO entries before assembly gives the same matrix values as zeroing rows and columns of a built CSR matrix. The only difference is the stored explicit zeros, which can change SuperLU's pivot order. That is the source of the $\sim10^{-15}$ potential differences.

### 2.3 What's left

> [!NOTE]
> About $3.4\text{ s}$ of the remaining $\sim6\text{ s}$ extraction is GEOS `union_all` inside the vox-projection area (691 throats on Fine_0). I kept it because the area definition is the one you want. In production the pipeline passes `n_jobs = cores_per_task`, and extraction parallelises over body pairs, so this term divides by the core count. The previous report's "watershed 2.13 s" stage doesn't exist in this code path: `body_array.npy` is precomputed.

---

## 3. The Short-Circuit RN

### 3.1 Why the vanilla RN is too resistive

Each throat $k$ of body $i$ becomes a half-link from the body centroid $\mathbf{c}_i$ to the throat centroid $\mathbf{p}_k$, with length $d_{ik} = \lVert \mathbf{p}_k - \mathbf{c}_i \rVert$ and volume $V_{ik} = V_i / z_i$ (equal split). Its conductance is

$$
G_{ik} = \frac{\sigma A_{ik}}{d_{ik}} = \frac{\sigma V_{ik}}{d_{ik}^{2}} .
$$

Current crossing body $i$ from throat $p$ to throat $r$ is charged the **star** length $d_{ip} + d_{ir}$ instead of something close to the **chord** $\lVert \mathbf{p}_p - \mathbf{p}_r \rVert$. Inlet and outlet bodies are worse: their node is the centroid of the body's *face patch*, so each throat is also charged a transverse detour along the equipotential plane.

If every path is $\lambda$ times too long **at fixed volume**, then

$$
d \to \lambda d, \qquad A = \frac{V}{d} \to \frac{A}{\lambda}, \qquad G = \frac{\sigma V}{d^2} \to \frac{G}{\lambda^{2}} .
$$

The measured shortening below is $\approx 0.77$, i.e. $\lambda \approx 1.30$. That predicts a $1/F$ deficit of $1 - 1/\lambda^2 \approx 41\%$; vanilla on Fine media is at $-38\%$ on average.

### 3.2 The rules (`compute_short_circuit_distances`)

Only $d_{ik}$ changes. $V_{ik}$ is untouched, so $A_{ik} = V_{ik} / \tilde d_{ik}$ and the total pore volume $\Omega$ is **conserved exactly**: $\Omega$ is identical between the two variants on every sample.

**Dirichlet (inlet/outlet) bodies**, classified with the same test as `berg_cc.get_inlet_outlet_indices`. The whole contact patch is one equipotential, so current leaves it perpendicular to the plane:

$$
\tilde d_{ik}^{\,\text{plane}} = \min\!\Big(d_{ik},\; \max\big(\lvert p_{k,x} - x_{\text{plane}} \rvert,\; 0.5\big)\Big).
$$

**Interior bodies with $z_i \ge 2$.** Throat $p$ is paired with every other throat $r$ of the same body:

$$
w_{pr} = A_{t,r}\,\big(\lvert \Delta x_{pr} \rvert + 0.5\big), \qquad
\rho_p = \frac{\sum_{r \ne p} w_{pr}\, \min\!\big(\ell^{\text{chord}}_{pr},\, \ell^{\text{star}}_{pr}\big)}{\sum_{r \ne p} w_{pr}\, \ell^{\text{star}}_{pr}}, \qquad
\tilde d_{ip} = d_{ip}\,\mathrm{clip}\big(\rho_p,\, 0.25,\, 1\big),
$$

with $\ell^{\text{chord}}_{pr} = \lVert \mathbf{p}_p - \mathbf{p}_r \rVert$ and $\ell^{\text{star}}_{pr} = d_{ip} + d_{ir}$. The weight $w_{pr}$ is a geometry-only proxy for how much current $p$ exchanges with $r$: larger, more axially separated partner throats carry more of it.

**Dead-end bodies ($z_i = 1$):** unchanged.

The Berg effective geometry ($\alpha$-formula, `RATIO_THRESHOLD`, uniform-pipe fallback) is then built exactly as before, on the shortened lengths.

### 3.3 Ablation and sensitivity (Fine_0–3, equal volume split)

Mean relative error vs TauFactor (MAE in brackets). $T_I$ is defined in §6.1.

| Variant | $1/F$ | $\tau_c$ (Berg) | $T_I$ (current-weighted) | $1/C_c$ | Fallback throats |
| :--- | ---: | ---: | ---: | ---: | ---: |
| vanilla | $-38.6\%$ (38.6) | $+30.2\%$ | $+23.9\%$ | $+7.7\%$ (7.8) | 531 |
| **short_circuit (shipped)** | **$-8.4\%$ (8.4)** | **$+3.2\%$** | **$-1.0\%$** | **$+1.0\%$ (4.4)** | **455** |
| plane rule only | $-35.6\%$ | $+25.5\%$ | $+19.2\%$ | $+5.0\%$ | 530 |
| chord rule only | $-13.9\%$ | $+8.3\%$ | $+3.6\%$ | $+4.4\%$ | 458 |
| $w_{pr} = 1$ | $+4.1\%$ (5.5) | $-3.3\%$ | $-7.8\%$ | $+0.7\%$ | 441 |
| $w_{pr} = A_{t,r}$ | $-0.5\%$ (4.0) | $-1.5\%$ | $-5.8\%$ | $-0.1\%$ | 448 |
| $w_{pr} = \lvert\Delta x_{pr}\rvert + 0.5$ | $-6.8\%$ | $+2.5\%$ | $-1.8\%$ | $+1.3\%$ | 452 |
| clip at $0$ / $0.5$ | unchanged | unchanged | unchanged | $\pm 0.4\%$ | 455 |
| pure axial projection (lower bound) | $+34.3\%$ | $-19.7\%$ | $-20.7\%$ | $-10.4\%$ | 272 |

> [!TIP]
> **What the ablation shows**
> 1. **The interior chord rule does most of the work** ($-38.6\% \to -13.9\%$ on its own). The plane rule adds the last $\sim5$ points.
> 2. **The clip never binds:** $\le 2$ of $\sim1{,}400$ half-links hit $0.25$. Mean shortening factor: $0.77$ interior, $0.75$ boundary.
> 3. **I didn't pick the variant with the best $1/F$.** $w_{pr} = A_{t,r}$ gives $-0.5\%$, but only because paths become too short ($T_I$ $-5.8\%$): a constriction error compensated by a tortuosity error. The shipped $A_{t,r}(\lvert\Delta x\rvert + 0.5)$ weighting is the one whose **paths** match TauFactor's flux-weighted tortuosity ($T_I$ $-1\%$), which is the quantity the short-circuit is meant to correct.
> 4. The pure axial projection overshoots massively. Real streamlines do move transversely inside bodies.

---

## 4. Results on `TRACE_fine_test` (Testbench)

Config: `min_throat_size=1`, `dilate_both`, `skip_surface_surface`, `vox_projection`, $\alpha = 0.5$, `split_volume_equal=True`, `flow_decomposition` (your cached `version9`). TauFactor values come from the cached runs. Fine_4 has no TauFactor cached in `TRACE_fine_test`; it is covered in §5.

| Microstructure & TauFactor reference | Variant | Solved $1/F$ (rel. err) | $\tau_c$ Berg (rel. err) | $T_I$ (rel. err) | $1/C_c$ (rel. err) | Fallback |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **Fine_0** | vanilla | 0.1047 ($-42.4\%$) | 1.658 ($+30.3\%$) | 1.591 ($+25.1\%$) | 0.630 ($-0.1\%$) | 483 |
| $D^{\text{TF}} = 0.1817,\ \tau^{\text{TF}} = 1.272,\ C^{\text{TF}} = 0.631$ | **short_circuit** | **0.1552 ($-14.6\%$)** | **1.314 ($+3.3\%$)** | **1.270 ($-0.2\%$)** | **0.587 ($-6.9\%$)** | **411** |
| **Fine_1** | vanilla | 0.0922 ($-40.3\%$) | 1.701 ($+31.3\%$) | 1.625 ($+25.4\%$) | 0.656 ($+7.5\%$) | 528 |
| $D^{\text{TF}} = 0.1544,\ \tau^{\text{TF}} = 1.296,\ C^{\text{TF}} = 0.610$ | **short_circuit** | **0.1406 ($-8.9\%$)** | **1.341 ($+3.5\%$)** | **1.290 ($-0.5\%$)** | **0.622 ($+1.9\%$)** | **471** |
| **Fine_2** | vanilla | 0.0745 ($-33.8\%$) | 1.753 ($+29.6\%$) | 1.661 ($+22.8\%$) | 0.589 ($+15.1\%$) | 552 |
| $D^{\text{TF}} = 0.1126,\ \tau^{\text{TF}} = 1.353,\ C^{\text{TF}} = 0.511$ | **short_circuit** | **0.1109 ($-1.5\%$)** | **1.388 ($+2.6\%$)** | **1.328 ($-1.9\%$)** | **0.549 ($+7.5\%$)** | **464** |
| **Fine_3** | vanilla | 0.0735 ($-38.1\%$) | 1.746 ($+29.8\%$) | 1.647 ($+22.4\%$) | 0.602 ($+8.3\%$) | 561 |
| $D^{\text{TF}} = 0.1187,\ \tau^{\text{TF}} = 1.345,\ C^{\text{TF}} = 0.556$ | **short_circuit** | **0.1086 ($-8.5\%$)** | **1.391 ($+3.4\%$)** | **1.324 ($-1.6\%$)** | **0.564 ($+1.5\%$)** | **475** |

These match the previous report's "1-Pass Geometric SC" rows to the third decimal. The small differences come from using the exact `get_inlet_outlet_indices` thresholds for the plane rule.

---

## 5. Results on `data_new` (25 Samples)

TauFactor references come from `data_new/all_runs_versions.csv` (latest version per sample). Same berg_cc config as §4. All 25 samples ran with no errors in all four variants, in about $40\text{ s}$ wall time per variant for the whole set (6 in parallel).

### 5.1 Per class

Geometric-mean ratio $\dfrac{1/F_{\text{RN}}}{D^{\text{TF}}}$; arithmetic mean of the $T_I$ error.

| Class | Variant | $\dfrac{1/F_{\text{RN}}}{D^{\text{TF}}}$ | $T_I$ error | Fallback throats | Mean TF $C$ | Mean bodies |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **Fine (10)** | vanilla | 0.618 | $+24.3\%$ | 68% | 0.570 | 357 |
| | **short_circuit** | **0.934** | **$-1.2\%$** | **58%** | | |
| **Medium (10)** | vanilla | 0.968 | $+12.1\%$ | 56% | 0.231 | 476 |
| | short_circuit | 1.397 | $-10.7\%$ | 46% | | |
| **Coarse (5)** | vanilla | 1.620 | $+8.5\%$ | 59% | 0.033 | 144 |
| | short_circuit | 2.282 | $-14.4\%$ | 48% | | |

### 5.2 Per sample

| Sample | TF $D_{\text{eff}}$ | Vanilla $1/F$ | Short-circuit $1/F$ | $T_I$ err (van) | $T_I$ err (SC) | TF $C$ | $1/C_c$ (van) | $1/C_c$ (SC) | $\phi_c/\varepsilon$ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Coarse 0 | 0.0038 | 0.0061 ($+60\%$) | 0.0087 ($+127\%$) | $+12.6\%$ | $-11.9\%$ | 0.039 | 0.192 | 0.166 | 0.52 |
| Coarse 1 | 0.0024 | 0.0042 ($+73\%$) | 0.0055 ($+128\%$) | $+4.5\%$ | $-12.6\%$ | 0.018 | 0.129 | 0.112 | 0.52 |
| Coarse 2 | 0.0026 | 0.0035 ($+32\%$) | 0.0053 ($+101\%$) | $+17.5\%$ | $-12.3\%$ | 0.054 | 0.289 | 0.245 | 0.37 |
| Coarse 7 | 0.0040 | 0.0059 ($+48\%$) | 0.0081 ($+105\%$) | $+11.8\%$ | $-14.2\%$ | 0.038 | 0.185 | 0.148 | 0.43 |
| Coarse 9 | 0.0006 | 0.0013 ($+106\%$) | 0.0018 ($+190\%$) | $-3.8\%$ | $-21.1\%$ | 0.018 | 0.175 | 0.165 | 0.20 |
| Fine 0 | 0.1817 | 0.1047 ($-42\%$) | **0.1552 ($-15\%$)** | $+25.1\%$ | **$-0.2\%$** | 0.631 | 0.630 | 0.587 | 0.98 |
| Fine 1 | 0.1544 | 0.0922 ($-40\%$) | **0.1406 ($-9\%$)** | $+25.4\%$ | **$-0.5\%$** | 0.610 | 0.656 | 0.622 | 0.96 |
| Fine 2 | 0.1126 | 0.0745 ($-34\%$) | **0.1109 ($-1\%$)** | $+22.8\%$ | **$-1.9\%$** | 0.511 | 0.589 | 0.549 | 0.96 |
| Fine 3 | 0.1187 | 0.0735 ($-38\%$) | **0.1086 ($-8\%$)** | $+22.4\%$ | **$-1.6\%$** | 0.556 | 0.602 | 0.564 | 0.96 |
| Fine 4 | 0.1498 | 0.0899 ($-40\%$) | **0.1363 ($-9\%$)** | $+26.6\%$ | **$+0.7\%$** | 0.597 | 0.646 | 0.608 | 0.99 |
| Fine 5 | 0.1258 | 0.0788 ($-37\%$) | **0.1213 ($-4\%$)** | $+25.5\%$ | **$-1.5\%$** | 0.564 | 0.639 | 0.601 | 0.95 |
| Fine 6 | 0.1407 | 0.0870 ($-38\%$) | **0.1333 ($-5\%$)** | $+24.8\%$ | **$-0.7\%$** | 0.575 | 0.634 | 0.602 | 0.98 |
| Fine 7 | 0.1191 | 0.0767 ($-36\%$) | **0.1154 ($-3\%$)** | $+22.0\%$ | **$-3.3\%$** | 0.555 | 0.597 | 0.560 | 0.97 |
| Fine 8 | 0.1198 | 0.0756 ($-37\%$) | **0.1181 ($-1\%$)** | $+25.5\%$ | **$-1.5\%$** | 0.536 | 0.621 | 0.589 | 0.96 |
| Fine 9 | 0.1254 | 0.0768 ($-39\%$) | **0.1140 ($-9\%$)** | $+22.7\%$ | **$-1.2\%$** | 0.566 | 0.610 | 0.580 | 0.95 |
| Medium 0 | 0.0749 | 0.0554 ($-26\%$) | 0.0807 ($+8\%$) | $+21.0\%$ | $-3.8\%$ | 0.351 | 0.479 | 0.438 | 0.91 |
| Medium 1 | 0.0395 | 0.0375 ($-5\%$) | 0.0534 ($+35\%$) | $+12.0\%$ | $-10.1\%$ | 0.224 | 0.381 | 0.340 | 0.82 |
| Medium 2 | 0.0257 | 0.0247 ($-4\%$) | 0.0364 ($+41\%$) | $+10.3\%$ | $-13.1\%$ | 0.208 | 0.375 | 0.339 | 0.75 |
| Medium 3 | 0.1181 | 0.0808 ($-32\%$) | 0.1167 ($-1\%$) | $+19.9\%$ | $-3.0\%$ | 0.445 | 0.526 | 0.488 | 0.91 |
| Medium 4 | 0.0255 | 0.0240 ($-6\%$) | 0.0345 ($+36\%$) | $+8.6\%$ | $-13.3\%$ | 0.191 | 0.311 | 0.283 | 0.81 |
| Medium 5 | 0.0251 | 0.0279 ($+11\%$) | 0.0406 ($+62\%$) | $+8.5\%$ | $-14.1\%$ | 0.163 | 0.305 | 0.285 | 0.81 |
| Medium 6 | 0.0208 | 0.0236 ($+14\%$) | 0.0332 ($+60\%$) | $+8.0\%$ | $-13.6\%$ | 0.178 | 0.337 | 0.298 | 0.79 |
| Medium 7 | 0.0359 | 0.0326 ($-9\%$) | 0.0467 ($+30\%$) | $+13.2\%$ | $-9.2\%$ | 0.241 | 0.363 | 0.326 | 0.87 |
| Medium 8 | 0.0165 | 0.0211 ($+27\%$) | 0.0311 ($+88\%$) | $+10.1\%$ | $-13.6\%$ | 0.128 | 0.284 | 0.258 | 0.76 |
| Medium 9 | 0.0271 | 0.0309 ($+14\%$) | 0.0439 ($+62\%$) | $+9.0\%$ | $-13.0\%$ | 0.179 | 0.353 | 0.308 | 0.81 |

> [!NOTE]
> Medium_0 and Medium_3 have the highest TauFactor constrictivity of their class ($C^{\text{TF}} \approx 0.35$ and $0.45$), and they behave like Fine samples (short-circuit $1/F$ within $8\%$). The split is about **how constricted the pore space is**, not about the class label.

---

## 6. Why the Discrepancies Exist

### 6.1 The quantities are not defined the same way

#### Tortuosity: $T_I$ vs Berg's $\tau_c$
TauFactor's `tau_reconciled` is

$$
\tau^{\text{TF}} = \frac{\langle \lvert \mathbf{J} \rvert \rangle}{\lvert \langle J_x \rangle \rvert}
\quad \text{(over pore voxels)} .
$$

For a field split into streamtubes $\Gamma$ carrying current $I_\Gamma$ along length $L_\Gamma$, $\int \lvert\mathbf{J}\rvert\, dV = \sum_\Gamma I_\Gamma L_\Gamma$ and $\int J_x\, dV = \sum_\Gamma I_\Gamma\, \Delta s$. So $\tau^{\text{TF}}$ is the **current-weighted mean of $L_\Gamma / \Delta s$**. The like-for-like network quantity, reported by the testbench, is

$$
T_I = \frac{\sum_\Gamma I_\Gamma L_\Gamma}{I_{\text{tot}}\, \Delta s} .
$$

Berg's tortuosity is instead a **volume-weighted mean of $(\Delta s / L_\Gamma)^2$**:

$$
\tau_c^{2} = \frac{1}{\Omega_c} \sum_\Gamma V_\Gamma \left(\frac{\Delta s}{L_\Gamma}\right)^{2},
\qquad \tau_{c,B} \equiv \left(\tau_c^{2}\right)^{-1/2} \ge T_I ,
$$

since volume weighting emphasises slow, long tubes. The gap is $+3\text{–}5\%$ on Fine and $+6\text{–}13\%$ on Medium/Coarse (e.g. Coarse_0 SC: $1.631$ vs $1.444$), where path lengths are more widely spread. Comparing $\tau_c$ directly with $\tau^{\text{TF}}$ therefore mixes a definitional offset into the error.

#### Constrictivity: total vs conducting porosity
TauFactor's constrictivity is defined to close

$$
D^{\text{TF}} = \frac{\varepsilon\, C^{\text{TF}}}{\left(\tau^{\text{TF}}\right)^{2}}
$$

with the **total** porosity $\varepsilon$, so it absorbs everything not explained by $\tau^{\text{TF}}$, including dead-end and non-conducting pore volume. Berg's $1/C_c$ closes $1/F = \phi_c\, \tau_c^2 / C_c$ with the **conducting** porosity $\phi_c$. The two agree only when $\phi_c \approx \varepsilon$: true for Fine ($0.97$), not for Medium ($0.82$) or Coarse ($0.39$).

### 6.2 An exact error budget

On the network, $1/F = \phi_c\, \tau_c^{2} / C_c$ holds exactly (Berg 2012, Eq. 17). Combined with the TauFactor definition above:

$$
\boxed{\;
\frac{1/F_{\text{RN}}}{D^{\text{TF}}}
= \underbrace{\frac{\phi_c}{\varepsilon}}_{\text{conducting porosity}}
\times \underbrace{\frac{1/C_c}{C^{\text{TF}}}}_{\text{constriction}}
\times \underbrace{\left(\frac{\tau^{\text{TF}}}{\tau_{c,B}}\right)^{2}}_{\text{tortuosity}}
\;}
$$

The identity holds on every sample to $\le 0.5\%$. The residual comes from $\Omega_c$ being counted from streamtubes in one place and from currents in the other.

| Class | Variant | $1/F$ ratio | $= \phi_c/\varepsilon$ | $\times$ constriction | $\times$ tortuosity |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **Fine** | vanilla | 0.618 | 0.967 | 1.093 | **0.585** |
| | short_circuit | **0.934** | 0.967 | 1.029 | **0.939** |
| **Medium** | vanilla | 0.968 | 0.824 | **1.688** | **0.696** |
| | short_circuit | 1.397 | 0.824 | 1.528 | 1.109 |
| **Coarse** | vanilla | 1.620 | 0.386 | **6.172** | **0.679** |
| | short_circuit | 2.282 | 0.386 | 5.334 | 1.108 |

### 6.3 Fine media: only the geometry was wrong

The vanilla error is almost entirely the tortuosity factor: $0.585 \approx 1/1.31^2$, as predicted in §3.1. Short-circuit moves it to $0.94$, and the constriction and porosity factors stay within a few percent of 1. The $-6.6\%$ average $1/F$ deficit that remains is split between:

1. **the Berg $\tau_c$ definitional offset** (§6.1), which by itself costs $\approx (1/1.035)^2 \approx -7\%$. This is not a flaw in the paths: $T_I$ is within $1.2\%$;
2. **$\phi_c/\varepsilon = 0.97$**: the RN counts $\sim3\%$ of the pore volume as non-conducting, while TauFactor's constrictivity absorbs that;
3. **sample-to-sample constriction scatter** of $\pm7\%$ (Fine_0 $-7\%$, Fine_8 $+10\%$).

A $\sim0.7\%$ systematic also comes from `delta_s = shape[axis] = 150`, while the RN's Dirichlet nodes sit at voxel centres $0$ and $149$. I left this unchanged so vanilla stays exact.

### 6.4 Medium / Coarse: vanilla's agreement was a cancellation of errors

> [!WARNING]
> For Medium, vanilla's $1/F$ ratio of $0.968$ looks good, but it is $0.824 \times 1.688 \times 0.696$: **paths $20\%$ too long cancel a constriction that is $69\%$ too weak.** For Coarse the constriction factor is $6.2\times$ and swamps everything. Short-circuit removes most of the tortuosity error, so the constriction error becomes visible. That is why short-circuit $1/F$ is worse there. No path-length model can fix a constriction error. Keeping vanilla because its Medium mean looks better would be tuning to the wrong physics.

Short-circuit also **over-shortens** paths in these samples: $T_I$ is $-11\%$ to $-14\%$ (tortuosity factor $1.11$). The chord between two throats is a lower bound on the streamline length only inside a **convex** body. In the larger, more irregular bodies of Medium/Coarse samples (Coarse: $144$ bodies per $150^3$ vs $\sim357$ for Fine), the chord can cut through solid. The truth lies between the star (vanilla, $+8\text{–}12\%$) and the chord (short-circuit, $-11\text{–}14\%$). This is a limit of the geometric heuristic: it uses no information about body shape.

### 6.5 Where the constriction is lost: link-level test against the TauFactor field

To find **which** links are wrong, I compared every interior link's RN conductance $G^{\text{RN}}_{ij}$ with the conductance the TauFactor field gives it:

$$
I^{\text{TF}}_{ij} = \sum_{\text{faces } (u,v):\, u\in i,\ v\in j} (\phi_u - \phi_v), \qquad
\Delta\Phi^{\text{TF}}_{ij} = \bar\phi_i - \bar\phi_j, \qquad
G^{\text{TF}}_{ij} = \frac{I^{\text{TF}}_{ij}}{\Delta\Phi^{\text{TF}}_{ij}} ,
$$

with unit conductance per voxel face (the same normalisation as the RN with $\sigma = 1$) and $\bar\phi_i$ the body-mean potential. The decomposition is **power-exact**: by a discrete Green identity, $\sum_{\text{links}} I_{ij}\,\Delta\Phi_{ij}$ plus the Dirichlet-boundary terms equals the total dissipation, so no dissipation is hidden inside bodies. Ratios are aggregated as dissipation-weighted geometric means, $w_{ij} = I^{\text{TF}}_{ij}\,\Delta\Phi^{\text{TF}}_{ij} > 0$. TauFactor was solved on `body_array > 0`; Fine_0 uses the field cached in `TRACE_fine_test/.../version1`.

| Sample (TF $C$) | Variant | $G^{\text{RN}}/G^{\text{TF}}$: all links | fallback links | Berg links | narrow ($\le 16$ faces) | wide ($> 16$ faces) | Dissipation share in narrow links |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Fine_0** (0.63) | vanilla | 0.47 | 0.39 | 0.82 | 1.38 | 0.44 | **5%** |
| | short_circuit | 0.70 | 0.58 | 1.16 | 2.01 | 0.66 | |
| **Medium_0** (0.35) | vanilla | 0.66 | 0.53 | 1.02 | 1.16 | 0.56 | **23%** |
| | short_circuit | 0.96 | 0.73 | 1.30 | 1.57 | 0.82 | |
| **Medium_8** (0.13) | vanilla | 1.15 | 0.52 | 1.51 | 2.17 | 0.70 | **44%** |
| | short_circuit | 1.73 | 0.81 | 2.19 | 3.07 | 1.10 | |
| **Coarse_0** (0.04) | vanilla | 1.21 | 1.16 | 1.23 | 2.28 | 1.01 | **22%** |
| | short_circuit | 1.94 | 1.78 | 2.01 | 3.59 | 1.64 | |
| **Coarse_9** (0.02) | vanilla | 2.63 | 7.83 | 2.24 | 2.23 | 3.79 | **69%** |
| | short_circuit | 3.57 | 12.07 | 3.01 | 3.06 | 5.05 | |

Body-mean potentials are not exactly the RN's node potentials, so read these as **relative** comparisons. The "all links" level shouldn't be compared directly with the global $1/F$. The trends are robust, though:

> [!IMPORTANT]
> 1. **Narrow necks are always over-conductive in the RN ($1.4\text{–}3.6\times$), and their share of the dissipation grows from $5\%$ (Fine) to $22\text{–}69\%$ (Medium_8, Coarse).** That is what turns a small, harmless bias in Fine media into a large $1/F$ overestimate in constricted media.
> 2. In Fine media the opposite bias dominates: the uniform-pipe **fallback** links ($72\text{–}77\%$ of the dissipation) are **under**-conductive, and short-circuit partly corrects them, consistent with §6.3.

Two mechanisms make narrow links too conductive, and both can be identified:

**(a) Area.** Compare the vox-projection area $A_t$ with the number of voxel faces $n_f$ the two bodies actually share (the cross-section a finite-volume solve conducts through):

| Class | Median $A_t/n_f$ | Current-weighted $A_t/n_f$ | $n_f \le 4$ | $5 \le n_f \le 16$ | $17 \le n_f \le 64$ | $n_f > 64$ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fine | 0.88 | 1.00 | 2.96 | 1.49 | 0.99 | 0.79 |
| Medium | 0.97 | 1.09 | 2.00 | 1.40 | 0.98 | 0.80 |
| Coarse | 0.96 | 1.07 | 2.04 | 1.40 | 0.99 | 0.77 |

Narrow necks get $2\text{–}3\times$ too much area, but weighted by current the excess over a whole sample is only $0\text{–}9\%$. So area alone cannot explain a $2\text{–}5\times$ constriction miss; its correlation with the constriction factor across the 25 samples is $r = 0.46$.

**(b) Constriction scaling.** In the Berg segment model, the throat resistance per side is

$$
R_{t,i} = \frac{\alpha\, d_{ik}}{\sigma \sqrt{A_{ik} A_{t}}} \quad\Longrightarrow\quad G \propto \sqrt{A_t} .
$$

That is the right scaling for a short **orifice** in a wide body. The Holm/Maxwell spreading resistance $R \approx 1/(4\sigma a)$ per side, with $a = \sqrt{A_t/\pi}$, matches the Berg term to within $\sim10\%$ for compact bodies ($d_{ik} \approx \sqrt{A_{ik}}$). It is too weak for a **channel-like** neck, whose resistance scales as $\ell / (\sigma A_t)$. Near the percolation threshold (Coarse: $D_{\text{eff}} \sim 10^{-3}$, $\tau_{\text{solver}}$ up to $210$) transport runs through such necks.

> [!NOTE]
> (b) is my physical interpretation, not a measured separation. It is consistent with the area-weighted volume split, where a link's conductance scales $\propto A_t$ through its volume share, cutting the Coarse constriction factor from $5.3$ to $2.1$ (§6.6).

### 6.6 Volume split (`split_volume_equal`)

| Class | Short-circuit, equal split: $1/F$ ratio | (constriction factor) | Short-circuit, area-weighted: $1/F$ ratio | (constriction factor) | Area-weighted fallback |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Fine | 0.934 | 1.03 | 0.883 | 1.00 | 82% |
| Medium | 1.397 | 1.53 | 1.135 | 1.23 | 65% |
| Coarse | 2.282 | 5.33 | **0.989** | 2.13 | 62% |

The area-weighted split looks much better on Coarse $1/F$, but it still carries a $2.1\times$ constriction error. The over-shortened paths ($1.13$) and the porosity factor ($0.41$) happen to cancel it. It also sends $82\%$ of Fine throats into the fallback.

> [!TIP]
> I left the default at `split_volume_equal=True`. Choosing the split per microstructure class would be tuning, not physics. A single change that targets the real problem is the constriction law for narrow necks (§6.5).

### 6.7 Smaller effects

| Effect | Size | Comment |
| :--- | ---: | :--- |
| TauFactor reference itself | $0\text{–}9\%$ in $D_{\text{eff}}$ | My re-solves on `body_array > 0` vs the CSV runs (`segmented_percolated.npy`, different convergence), e.g. Coarse_0: $0.00417$ vs $0.00384$ |
| Network porosity $\Omega/V$ vs $\varepsilon$ | $\le 0.2\%$ (Fine), $\le 5\%$ (Coarse_9) | `body_array` and the percolated segmentation differ slightly |
| $\Delta s = 150$ vs node spacing $149$ | $\approx 0.7\%$ | Kept so vanilla stays exact |

---

## 7. Not Done / Caveats

| Item | Status |
| :--- | :--- |
| 2-pass streamtube re-solve (previous report, Method 2) | Not implemented. That report found it worse than the 1-pass rule on Fine_0–3, and a geometry-only rule avoids an extra solve |
| Cauchy area | Not used, as you asked. Everything here is `vox_projection` |
| Previous report §4.1: possible infinite loop in `_flow_decomposition` | Not fixed (outside this change). `flow_decomposition` terminated on all $25 + 5$ samples here |
| Previous report §4.3: a body touching both faces is neither inlet nor outlet | Not fixed (outside this change) |
| Short-circuit constants (weights, $+0.5$ offset, clip $0.25$, plane floor $0.5$) | Heuristic. §3.3 shows the clip and floor don't matter; the weighting does, and it was chosen to match the flux-weighted tortuosity, **not** $1/F$ |
| Merging | Behaviour unchanged; it is now called from inside extraction |

---

## 8. Reproducing

Everything is in [`rn_testbench/`](rn_testbench/). Run with `/Users/rochan/miniconda3/envs/rbenv/bin/python` (or set `$PY` for `run_variants.sh`). The scripts find the repo root relative to their own location.

| Script | Purpose | Report section |
| :--- | :--- | :--- |
| `bench.py <module_dir> <sample_dir> <out.pkl> rn_method=… split_volume_equal=…` | Stage-timed run of the whole berg_cc path; saves every intermediate | §2, §4, §5 |
| `compare.py base.pkl new.pkl` | Bit-level equivalence report | §2 |
| `run_variants.sh <data_root> <tag> key=val…` | Runs `bench.py` over every sample folder | §4, §5 |
| `table.py`, `budget.py` (+ `cache_lookup.py`) | Tables vs TauFactor and the exact error budget | §4–§6.2 |
| `ablation.py` | Ablation and sensitivity | §3.3 |
| `throat_faces.py` | Vox-projection area vs shared-face count | §6.5 (a) |
| `run_tf.py`, `link_conductance.py` | TauFactor field + link-level conductance test | §6.5 |

`results/` holds the JSON outputs, the budget CSVs for all four `data_new` variants, and the baseline/new timing JSONs.

To regenerate the baseline, point `bench.py` at a checkout of `origin/main` (commit `78c06de`, the parent of the speedup commit on the `short_circuit` branch). Its code is the pre-change version used for every "baseline" number here:

```bash
git worktree add /tmp/rn_base 78c06de
python bench.py /tmp/rn_base <sample_dir> base.pkl rn_method=vanilla
```
