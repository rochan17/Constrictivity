# Berg RN: Speedups, a `vanilla` / `short_circuit` Toggle, and Validation Against TauFactor

Follow-up to [`berg_rn_improvements_report.md`](berg_rn_improvements_report.md). That report proposed the speedups and the "1-Pass Geometric Short-Circuit" from scratch prototypes. This one **implements** them in [`particulate_claude.py`](../particulate_claude.py), [`berg_cc.py`](../berg_cc.py) and [`pipeline.py`](../pipeline.py). I tested them on `TRACE_fine_test` (`Fine_microstructure_0`–`4`, $150^3$ voxels), then on the **10 Fine** microstructures in `data_new`, and explained the remaining discrepancies against TauFactor with diagnostics built on the TauFactor potential field itself. From this revision on the report covers **Fine media only**; the earlier Medium/Coarse analysis is in git history.

**Environment:** conda env `rbenv` (numpy 1.26.4, scipy 1.14.1, shapely 2.1.2, scikit-image 0.20.0, numba 0.59.1). All scripts and raw results are in [`rn_testbench/`](rn_testbench/) (§8).

---

## 0. Summary

> [!IMPORTANT]
> **Revision (2026-10-18): the short-circuit rule.** The centroid stays where it is, and each half-link is only **shortened along its own ray**. The lengths are the least-resistance ones for which every pair of throats of a body is charged between its chord and its vanilla route through the centroid:
> $$\lVert\mathbf p_p-\mathbf p_r\rVert\ \le\ \tilde d_{ip}+\tilde d_{ir}\ \le\ d_{ip}+d_{ir}\qquad\text{for all throat pairs }(p,r).$$
> Both bounds hold by construction. Two earlier rules are superseded (git history):
> - the weighted-ratio rule, which charged $47\%$ of throat pairs less than their chord;
> - a single-junction rule, which respected the chord but kept a full star detour.

| # | Result | Evidence |
| :--- | :--- | :--- |
| 1 | **Speed: $69.1\text{ s} \to 7.3\text{ s}$ per $150^3$ sample ($9.5\times$), with unchanged results** | Vanilla mode is bit-identical through the network arrays on all 5 test samples; metrics agree to $\le 2.5\times10^{-14}$ (§2) |
| 2 | **Toggle:** `rn_method = 'vanilla' \| 'short_circuit'`, default set by one constant | `DEFAULT_RN_METHOD` in `particulate_claude.py`; recorded in the cache hash (§1.2) |
| 3 | **The body merges now run inside the berg_cc extraction step** | `extract_throats_from_bodies_voxel_parallel(merge_bodies=True)` (§1.1) |
| 4 | **Both bounds hold exactly:** on Fine_0, $0$ of $3{,}886$ interior throat pairs fall below the chord and $0$ are longer than vanilla; every 2-throat body sits exactly on its chord | §3.3 |
| 5 | **Fine $1/F$:** $0.618 \to$ **$0.731$** of TauFactor (geometric mean, 10 samples); mean absolute error $38.1\% \to 26.8\%$ | §5 |
| 6 | **Path length:** $T_I$ error $+24.3\% \to$ **$+12.2\%$**; porosity and constriction factors stay within $5\%$ of 1 | §5, §6.2 |
| 7 | **What is left is almost all in-body excess over the chords:** $T_I/\tau^{\text{TF}} = 1.122 = \underbrace{1.101}_{\text{in-body excess}} \times \underbrace{1.019}_{\text{throat-centroid polyline}}$ | §6.3 |

> [!TIP]
> **Default:** `DEFAULT_RN_METHOD = 'vanilla'`, so every existing result and caller is unchanged; `short_circuit` is opt-in (`rn_method='short_circuit'`). On Fine media it is the largest correction you can make with one length per half-link without ever charging a path shorter than its chord or longer than vanilla. It costs $0.13\text{ s}$ per sample. The remaining $\approx 10\%$ path excess comes from half-links shared by several throat pairs (§6.3). Removing it needs lengths that depend on the throat *pair*, for example direct throat-to-throat links inside each body.

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
| | `compute_short_circuit_distances`, `_chord_bounded_lengths` (new) | Short-circuit half-link lengths $\tilde d_{ik}$, chord-bounded per body (§3.2) |
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

# global switch: one line in particulate_claude.py (default is 'vanilla')
DEFAULT_RN_METHOD = 'short_circuit'
```

```bash
# batch runs: per-run override (cached as its own version)
python run_batch.py --data-dir data_new --cores-per-microstructure 4 \
    --berg-cc-config '{"rn_method": "short_circuit"}'
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

Current that enters body $i$ through throat $p$ and leaves through throat $r$ is charged the route through the centroid, $d_{ip} + d_{ir}$. The shortest possible route is the **chord** $\lVert \mathbf{p}_p - \mathbf{p}_r \rVert$, and a real streamline in a compact body runs close to it. Inlet and outlet bodies are worse: their node is the centroid of the body's *face patch*, so each throat is also charged a transverse detour along the equipotential face. If every path is $\lambda$ times too long **at fixed volume**, then

$$
d \to \lambda d, \qquad A = \frac{V}{d} \to \frac{A}{\lambda}, \qquad G = \frac{\sigma V}{d^2} \to \frac{G}{\lambda^{2}} .
$$

On Fine_0 the current-weighted ratio of path to chord through interior bodies is $\lambda = 1.270$ for vanilla.

### 3.2 The rule (`compute_short_circuit_distances`)

Only $d_{ik}$ changes. $V_{ik}$ is untouched, so $A_{ik} = V_{ik} / \tilde d_{ik}$ and the total pore volume $\Omega$ is **conserved exactly**. The centroid $\mathbf c_i$ is **not moved**: each half-link keeps its ray from $\mathbf p_k$ towards $\mathbf c_i$, and only its length can shrink, $\tilde d_{ik} \le d_{ik}$.

The solve does not know in advance which throats of a body are inflow and which are outflow, so one set of lengths has to serve **every** throat pair at once. For each pair, the path must lie between the shortest possible route (the chord) and the vanilla route (through the centroid):

$$
\lVert\mathbf p_p-\mathbf p_r\rVert\ \le\ \tilde d_{ip}+\tilde d_{ir}\ \le\ d_{ip}+d_{ir}.
$$

The upper bound follows from $\tilde d_{ik} \le d_{ik}$. Within these bounds the code takes the lengths that give the body the **least half-link resistance**:

$$
\boxed{\;
\min_{\tilde d}\ \sum_k \frac{\tilde d_{ik}^{\,2}}{\sigma V_{ik}}
\quad\text{s.t.}\quad
\tilde d_{ip}+\tilde d_{ir}\ \ge\ \lVert\mathbf p_p-\mathbf p_r\rVert\ \ \forall\,p\ne r,
\qquad
\min(0.5,\,d_{ik})\ \le\ \tilde d_{ik}\ \le\ d_{ik}
\;}
$$

| Case | Result | Why |
| :--- | :--- | :--- |
| **Interior, $z_i = 2$** | $\tilde d_{i1}+\tilde d_{i2} = \lVert\mathbf p_1-\mathbf p_2\rVert$ exactly, split $\propto V_{ik}$ (equal halves for the equal split) | Closed form. The two half-links then have exactly the resistance of **one uniform straight tube** of the body's volume between the two throats, $R = \ell^2/(\sigma\,2V_{ik})$ |
| **Interior, $z_i \ge 3$** | Lengths between the chords and the centroid route; at least one pair per throat is usually *at* its chord | A half-link shared by several pairs must be long enough for the pair that needs it most |
| **Inlet / outlet** (Dirichlet, same test as `berg_cc.get_inlet_outlet_indices`) | $\tilde d_{ik} = \lvert p_{k,x} - c_{i,x}\rvert \le d_{ik}$ | The node is on the equipotential face, so throat $k$ is reached straight from the face: the transverse detour along the face is dropped |
| **Dead end, $z_i = 1$** | Unchanged | No pair to bound |

**Why it is always solvable, and how it is solved.** The vanilla lengths always satisfy the constraints, because $d_{ip}+d_{ir} \ge \lVert\mathbf p_p-\mathbf p_r\rVert$ by the triangle inequality through $\mathbf c_i$. The problem is a small convex QP per body (at most a few dozen variables).
- **Two throats:** closed form.
- **Three or more throats** (`_chord_bounded_lengths`): SLSQP started from the vanilla lengths. Its result is then made **exactly** feasible by moving it towards the vanilla lengths just far enough. The feasible set is convex and contains the vanilla lengths, so this keeps every bound.

> [!NOTE]
> Throat pairs of an **inlet/outlet** body are not chord-bounded. Current passing from one throat to another through a Dirichlet node goes through the equipotential electrode, which is a real short circuit.

The Berg effective geometry ($\alpha$-formula, `RATIO_THRESHOLD`, uniform-pipe fallback) is then built exactly as before, on the new lengths.

### 3.3 Checks (Fine_0, equal volume split)

| Check | Result |
| :--- | :--- |
| Interior throat pairs below the chord | **$0$ of $3{,}886$** (smallest ratio $1.000000$) |
| Half-links longer than vanilla | **$0$** |
| Two-throat bodies: $(\tilde d_{i1}+\tilde d_{i2})/\text{chord}$ | **$1.0000$** (median) |
| Streamtube current through a body on a path shorter than its chord | **$0\%$** |
| Current-weighted path/chord through interior bodies, $\lambda$ | vanilla $1.270 \to$ **$1.171$** |
| Run time of `compute_short_circuit_distances` | $0.13\text{ s}$ |

| Rule | Pairs below chord | Pairs above vanilla | $1/F$ (TF $0.1817$) | $T_I$ (TF $1.272$) | Status |
| :--- | ---: | ---: | ---: | ---: | :--- |
| vanilla (centroid route) | $0$ | — | 0.1047 ($-42.4\%$) | 1.591 | baseline |
| weighted ratio $\tilde d = \rho_p d$ | $1{,}841$ | $0$ | 0.1552 ($-14.6\%$) | 1.270 | superseded: **violates the chord bound** |
| single junction per body | $0$ | — | 0.1099 ($-39.5\%$) | 1.503 | superseded: moved the centre, kept the full star detour |
| **chord-bounded per throat (shipped)** | **$0$** | **$0$** | **0.1226 ($-32.5\%$)** | **1.438** | |

> [!WARNING]
> The weighted-ratio rule of the earlier report agreed with TauFactor much better on $1/F$. It did so by crediting current with paths through bodies shorter than the straight line between the throats, which is impossible. The shipped rule gives up that agreement to stay physically admissible.

---

## 4. Results on `TRACE_fine_test` (Testbench)

Config: `min_throat_size=1`, `dilate_both`, `skip_surface_surface`, `vox_projection`, $\alpha = 0.5$, `split_volume_equal=True`, `flow_decomposition` (your cached `version9`). TauFactor values come from the cached runs; Fine_4 is covered in §5.

| Microstructure & TauFactor reference | Variant | Solved $1/F$ (rel. err) | $\tau_c$ Berg (rel. err) | $T_I$ (rel. err) | $1/C_c$ (rel. err) | Fallback |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **Fine_0** | vanilla | 0.1047 ($-42.4\%$) | 1.658 ($+30.3\%$) | 1.591 ($+25.1\%$) | 0.630 ($-0.1\%$) | 483 |
| $D^{\text{TF}} = 0.1817,\ \tau^{\text{TF}} = 1.272,\ C^{\text{TF}} = 0.631$ | **short_circuit** | **0.1226 ($-32.5\%$)** | **1.492 ($+17.2\%$)** | **1.437 ($+13.0\%$)** | **0.597 ($-5.3\%$)** | **453** |
| **Fine_1** | vanilla | 0.0922 ($-40.3\%$) | 1.701 ($+31.3\%$) | 1.625 ($+25.4\%$) | 0.656 ($+7.5\%$) | 528 |
| $D^{\text{TF}} = 0.1544,\ \tau^{\text{TF}} = 1.296,\ C^{\text{TF}} = 0.610$ | **short_circuit** | **0.1106 ($-28.4\%$)** | **1.525 ($+17.7\%$)** | **1.456 ($+12.4\%$)** | **0.632 ($+3.6\%$)** | **496** |
| **Fine_2** | vanilla | 0.0745 ($-33.8\%$) | 1.753 ($+29.6\%$) | 1.661 ($+22.8\%$) | 0.589 ($+15.1\%$) | 552 |
| $D^{\text{TF}} = 0.1126,\ \tau^{\text{TF}} = 1.353,\ C^{\text{TF}} = 0.511$ | **short_circuit** | **0.0868 ($-22.9\%$)** | **1.585 ($+17.1\%$)** | **1.512 ($+11.7\%$)** | **0.560 ($+9.5\%$)** | **514** |
| **Fine_3** | vanilla | 0.0735 ($-38.1\%$) | 1.746 ($+29.8\%$) | 1.647 ($+22.4\%$) | 0.602 ($+8.3\%$) | 561 |
| $D^{\text{TF}} = 0.1187,\ \tau^{\text{TF}} = 1.345,\ C^{\text{TF}} = 0.556$ | **short_circuit** | **0.0867 ($-27.0\%$)** | **1.575 ($+17.1\%$)** | **1.488 ($+10.6\%$)** | **0.578 ($+4.0\%$)** | **521** |

---

## 5. Results on the 10 Fine Samples of `data_new`

TauFactor references come from `data_new/all_runs_versions.csv` (latest version per sample). Same berg_cc config as §4. All 10 samples ran with no errors.

### 5.1 Summary

| Variant | $\dfrac{1/F_{\text{RN}}}{D^{\text{TF}}}$ (geo. mean) | Range | Mean $\lvert$error$\rvert$ | $T_I$ error | $\tau_{c,B}$ error | Fallback throats |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| vanilla | 0.618 | 0.576 – 0.662 | $38.1\%$ | $+24.3\%$ | $+30.7\%$ | 68% |
| **short_circuit** | **0.731** | **0.675 – 0.771** | **$26.8\%$** | **$+12.2\%$** | **$+17.7\%$** | **64%** |

### 5.2 Per sample

| Sample | TF $D_{\text{eff}}$ | TF $\tau$ | TF $C$ | Vanilla $1/F$ | Short-circuit $1/F$ | $T_I$ err (van) | $T_I$ err (SC) | $1/C_c$ (van) | $1/C_c$ (SC) | Fallback (van / SC) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fine 0 | 0.1817 | 1.272 | 0.631 | 0.1047 ($-42\%$) | **0.1226 ($-33\%$)** | $+25.1\%$ | **$+13.0\%$** | 0.630 | 0.597 | 483 / 453 |
| Fine 1 | 0.1544 | 1.296 | 0.610 | 0.0922 ($-40\%$) | **0.1106 ($-28\%$)** | $+25.4\%$ | **$+12.4\%$** | 0.656 | 0.632 | 528 / 496 |
| Fine 2 | 0.1126 | 1.353 | 0.511 | 0.0745 ($-34\%$) | **0.0868 ($-23\%$)** | $+22.8\%$ | **$+11.7\%$** | 0.589 | 0.560 | 552 / 514 |
| Fine 3 | 0.1187 | 1.345 | 0.556 | 0.0735 ($-38\%$) | **0.0867 ($-27\%$)** | $+22.4\%$ | **$+10.6\%$** | 0.602 | 0.578 | 561 / 521 |
| Fine 4 | 0.1498 | 1.306 | 0.597 | 0.0899 ($-40\%$) | **0.1071 ($-29\%$)** | $+26.6\%$ | **$+14.7\%$** | 0.646 | 0.625 | 542 / 515 |
| Fine 5 | 0.1258 | 1.321 | 0.564 | 0.0788 ($-37\%$) | **0.0952 ($-24\%$)** | $+25.5\%$ | **$+11.3\%$** | 0.639 | 0.613 | 527 / 480 |
| Fine 6 | 0.1407 | 1.305 | 0.575 | 0.0870 ($-38\%$) | **0.1028 ($-27\%$)** | $+24.8\%$ | **$+13.6\%$** | 0.634 | 0.617 | 544 / 513 |
| Fine 7 | 0.1191 | 1.367 | 0.555 | 0.0767 ($-36\%$) | **0.0906 ($-24\%$)** | $+22.0\%$ | **$+9.7\%$** | 0.597 | 0.566 | 522 / 499 |
| Fine 8 | 0.1198 | 1.340 | 0.536 | 0.0756 ($-37\%$) | **0.0900 ($-25\%$)** | $+25.5\%$ | **$+13.2\%$** | 0.621 | 0.594 | 504 / 469 |
| Fine 9 | 0.1254 | 1.360 | 0.566 | 0.0768 ($-39\%$) | **0.0889 ($-29\%$)** | $+22.7\%$ | **$+11.8\%$** | 0.610 | 0.582 | 545 / 515 |

The improvement is uniform: every sample gains $11\text{–}19$ points of $1/F$ and halves its $T_I$ error.

---

## 6. Why the Discrepancies Exist

### 6.1 The quantities are not defined the same way

#### Tortuosity: $T_I$ vs Berg's $\tau_c$
TauFactor's `tau_reconciled` is

$$
\tau^{\text{TF}} = \frac{\langle \lvert \mathbf{J} \rvert \rangle}{\lvert \langle J_x \rangle \rvert}
\quad \text{(over pore voxels)} .
$$

For a field split into streamtubes $\Gamma$ carrying current $I_\Gamma$ along length $L_\Gamma$, $\int \lvert\mathbf{J}\rvert\, dV = \sum_\Gamma I_\Gamma L_\Gamma$ and $\int J_x\, dV = \sum_\Gamma I_\Gamma\, \Delta s$. So $\tau^{\text{TF}}$ is the **current-weighted mean of $L_\Gamma / \Delta s$**. The like-for-like network quantity is

$$
T_I = \frac{\sum_\Gamma I_\Gamma L_\Gamma}{I_{\text{tot}}\, \Delta s} .
$$

Berg's tortuosity is instead a **volume-weighted mean of $(\Delta s / L_\Gamma)^2$**:

$$
\tau_c^{2} = \frac{1}{\Omega_c} \sum_\Gamma V_\Gamma \left(\frac{\Delta s}{L_\Gamma}\right)^{2},
\qquad \tau_{c,B} \equiv \left(\tau_c^{2}\right)^{-1/2} \ge T_I ,
$$

since volume weighting emphasises slow, long tubes. On Fine media $\tau_{c,B}$ is $\approx 5\%$ above $T_I$ (Fine_0: $1.492$ vs $1.437$). Comparing $\tau_c$ directly with $\tau^{\text{TF}}$ therefore mixes a definitional offset into the error.

#### Constrictivity: total vs conducting porosity
TauFactor's constrictivity is defined to close $D^{\text{TF}} = \varepsilon\, C^{\text{TF}} / (\tau^{\text{TF}})^{2}$ with the **total** porosity $\varepsilon$. Berg's $1/C_c$ closes $1/F = \phi_c\, \tau_c^2 / C_c$ with the **conducting** porosity $\phi_c$. On Fine media $\phi_c/\varepsilon = 0.97$, so the two are directly comparable.

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

The identity holds on every sample to $\le 10^{-4}$.

| Variant (10 Fine, geometric means) | $1/F$ ratio | $= \phi_c/\varepsilon$ | $\times$ constriction | $\times$ tortuosity |
| :--- | ---: | ---: | ---: | ---: |
| vanilla | 0.618 | 0.967 | 1.093 | **0.585** |
| **short_circuit** | **0.731** | 0.967 | 1.047 | **0.722** |

> [!IMPORTANT]
> On Fine media the error is almost entirely **path length**. Porosity and constriction are within $5\%$ of TauFactor in both variants. The short circuit moves the tortuosity factor from $0.585$ to $0.722$; the rest of the gap is the next section.

### 6.3 Where the remaining path-length error sits

Replace each streamtube's RN length by the **shortest length any network through throat centroids could give it**: straight lines from the inlet face to the first throat centroid, between consecutive throat centroids, and to the outlet face. Current-weighting these like $T_I$ gives $T_{\text{chord}}$, and the error splits **exactly** into two factors:

$$
\frac{T_I}{\tau^{\text{TF}}}
= \underbrace{\frac{T_I}{T_{\text{chord}}}}_{\text{in-body excess over the chords}}
\times \underbrace{\frac{T_{\text{chord}}}{\tau^{\text{TF}}}}_{\text{throat-centroid polyline vs real paths}} .
$$

| Variant (10 Fine, geometric means) | $T_I/\tau^{\text{TF}}$ | $=$ in-body excess $T_I/T_{\text{chord}}$ | $\times$ polyline $T_{\text{chord}}/\tau^{\text{TF}}$ |
| :--- | ---: | ---: | ---: |
| vanilla | 1.243 | **1.219** | 1.020 |
| **short_circuit** | 1.122 | **1.101** | 1.019 |

1. **The throat-centroid polyline is only $2\%$ longer than TauFactor's current paths.** Throat centroids are a good geometric input on Fine media. The small excess is current cutting across wide throats off-centre.
2. **The in-body excess drops from $22\%$ to $10\%$.** The remaining $10\%$ is the cost of **one length per half-link**. A throat's half-link is shared by all pairs through that throat, so it must stay long enough for the pair that needs it most, even when the current uses another pair.

> [!NOTE]
> **Example (Fine_0, body 201, $z = 4$; the guide's section 3).** $55\%$ of the streamtube current through this body goes T325 → T486, yet that pair is charged $1.39\times$ its chord. T486's half-link cannot shrink, because the pair T420 ↔ T486, which carries only $5.6\%$ of the current, is already exactly at its chord ($23.20$ vox).
>
> Removing this needs lengths that depend on the **pair**, not just on the throat. Two options:
> - direct throat-to-throat links inside each body (each body becomes a small clique sharing its volume);
> - a second pass that re-weights pairs by the solved current.

### 6.4 Link-level test against the TauFactor field (Fine_0)

Every interior link's RN conductance $G^{\text{RN}}_{ij}$ is compared with the conductance the TauFactor field gives it:

$$
I^{\text{TF}}_{ij} = \sum_{\text{faces } (u,v):\, u\in i,\ v\in j} (\phi_u - \phi_v), \qquad
\Delta\Phi^{\text{TF}}_{ij} = \bar\phi_i - \bar\phi_j, \qquad
G^{\text{TF}}_{ij} = \frac{I^{\text{TF}}_{ij}}{\Delta\Phi^{\text{TF}}_{ij}} ,
$$

with unit conductance per voxel face, $\bar\phi_i$ the body-mean potential, and dissipation-weighted geometric means ($w_{ij} = I^{\text{TF}}_{ij}\,\Delta\Phi^{\text{TF}}_{ij}$).

| Variant | $G^{\text{RN}}/G^{\text{TF}}$: all links | fallback links | Berg links | narrow ($\le 16$ faces) | wide ($> 16$ faces) | Dissipation in fallback / narrow links |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| vanilla | 0.47 | 0.39 | 0.82 | 1.38 | 0.44 | 77% / 5% |
| **short_circuit** | **0.53** | **0.44** | **0.92** | 1.63 | **0.50** | 75% / 5% |

Body-mean potentials are not exactly the RN's node potentials, so read these as **relative** comparisons.

> [!IMPORTANT]
> The wide, uniform-pipe **fallback** links carry three quarters of the dissipation and are the most under-conductive ($0.44$). Their pipe length is $\tilde d_{1k}+\tilde d_{2k}$, so the in-body excess of §6.3 shows up directly here. The narrow necks are over-conductive, but they carry only $5\%$ of the dissipation on Fine media, so they barely affect $1/F$. Vox-projection areas of narrow necks are $2\text{–}3\times$ the shared-face count, while current-weighted over the sample the excess is $\approx 0\%$ (`throat_faces.py`).

### 6.5 Volume split (`split_volume_equal`)

| Variant (10 Fine) | $1/F$ ratio | $T_I$ error | Constriction factor | Fallback |
| :--- | ---: | ---: | ---: | ---: |
| vanilla, equal split | 0.618 | $+24.3\%$ | 1.093 | 68% |
| **short_circuit, equal split** | **0.731** | **$+12.2\%$** | 1.047 | 64% |
| vanilla, area-weighted | 0.572 | $+27.1\%$ | 1.028 | 94% |
| short_circuit, area-weighted | 0.682 | $+15.3\%$ | 0.999 | 90% |

With $V_{ik}\propto A_{t,k}$ the objective weights large throats' half-links less ($1/V_{ik}$), and $90\%$ of throats fall back to the uniform pipe. Keep `split_volume_equal=True` (the default).

### 6.6 Smaller effects

| Effect | Size | Comment |
| :--- | ---: | :--- |
| Berg $\tau_{c,B}$ vs $T_I$ | $\approx (1/1.05)^2 \approx -9\%$ on $1/F$ | Definitional (§6.1); not a path error |
| $\phi_c/\varepsilon$ | $0.97$ | $\sim3\%$ of the pore volume carries no current in the RN |
| TauFactor reference itself | $0\text{–}9\%$ in $D_{\text{eff}}$ | Re-solves on `body_array > 0` vs the CSV runs |
| $\Delta s = 150$ vs node spacing $149$ | $\approx 0.7\%$ | Kept so vanilla stays exact |

---

## 7. Not Done / Caveats

| Item | Status |
| :--- | :--- |
| Pair-dependent lengths (throat-to-throat links inside bodies, or a current-weighted second pass) | **Not implemented.** §6.3 identifies this as what removes the remaining $\approx 10\%$ path excess; it changes the network topology or needs a second solve |
| Medium / Coarse media | Out of scope from this revision. The earlier analysis (constriction-dominated errors) is in git history |
| Earlier short-circuit rules | Removed: weighted ratio (violated the chord bound for $47\%$ of pairs) and single junction. Results in `rn_testbench/results/*_v1` and `*_junction` |
| Cauchy area | Not used, as you asked. Everything here is `vox_projection` |
| Previous report §4.1: possible infinite loop in `_flow_decomposition` | Not fixed (outside this change). It terminated on every sample here |
| Previous report §4.3: a body touching both faces is neither inlet nor outlet | Not fixed (outside this change) |
| Short-circuit constants | Only the $0.5$-voxel lower bound on a length (never above vanilla) |

---

## 8. Reproducing

Everything is in [`rn_testbench/`](rn_testbench/). Run with `/Users/rochan/miniconda3/envs/rbenv/bin/python` (or set `$PY` for `run_variants.sh`). The scripts find the repo root relative to their own location.

| Script | Purpose | Report section |
| :--- | :--- | :--- |
| `bench.py <module_dir> <sample_dir> <out.pkl> rn_method=… split_volume_equal=…` | Stage-timed run of the whole berg_cc path; saves every intermediate | §2, §4, §5 |
| `compare.py base.pkl new.pkl` | Bit-level equivalence report | §2 |
| `run_variants.sh <data_root> <tag> key=val…` | Runs `bench.py` over every sample folder (for Fine only, point it at a folder of links to `data_new/Fine_*`) | §4, §5 |
| `table.py`, `budget.py` (+ `cache_lookup.py`) | Tables vs TauFactor and the exact error budget | §4–§6.2 |
| `ablation.py` | Ablation of the superseded weighted-ratio rule (historical) | — |
| `throat_faces.py` | Vox-projection area vs shared-face count | §6.5 (a) |
| `run_tf.py`, `link_conductance.py` | TauFactor field + link-level conductance test | §6.5 |

`results/` holds the JSON outputs and budget CSVs (`new_van`, `new_sc_eq`, `new_sc_aw` are current; `*_v1` and `*_junction` are the superseded rules), plus the baseline/new timing JSONs. The $T_{\text{chord}}$ decomposition (§6.3) re-walks each streamtube's throat sequence. The page `berg_network_formation_guide.html` (built by `build_guide.py`) shows it, and every other Fine_0 number here, interactively.

To regenerate the baseline, point `bench.py` at a checkout of `origin/main` (commit `78c06de`, the parent of the speedup commit on the `short_circuit` branch). Its code is the pre-change version used for every "baseline" number here:

```bash
git worktree add /tmp/rn_base 78c06de
python bench.py /tmp/rn_base <sample_dir> base.pkl rn_method=vanilla
```
