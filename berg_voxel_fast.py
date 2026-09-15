"""
berg_voxel_fast.py

Fast, configurable implementation of the voxel-network diagnostics used to
compute Berg's (2012) tortuosity, constriction, and conductance-reduction
quantities from a segmented 3D image.  This version preserves the original KCL/
PyAMG workflow while adding compiled greedy and topological flow decompositions
and the vectorised volume/chord post-processing paths.

WHAT THIS MODULE IS FOR
------------------------
Berg's paper reformulates Archie's empirical F = phi^-m relationship in terms
of physically defined quantities computed on the pore microstructure itself:

    - conductance reduction factor (iota^2): how much a local piece of pore
      space underperforms an "optimal" straight, constant-area tube carrying
      the same current for the same potential drop.
    - tortuosity (tau_c): how much longer the current's actual path is
      compared to the straight-line sample size, i.e. how much of the loss
      of conductance is because current has to travel further than it
      "needs" to geometrically.
    - constriction (C_c): how much conductance is lost purely from the
      cross-sectional area changing along the current's path (bottlenecks
      and expansions), independent of path length.

These three are related through F = C_c / (tau_c^2 * phi_c), which is the
central result of the paper: the empirical cementation exponent m can be
replaced by two descriptive, geometry-derived numbers instead of one fitted
one.

This module computes all of these numerically on a voxel image by:
    1. Building a resistor network where each pore voxel is a node and each
       face-sharing pair of pore voxels is a resistor (a discretisation of
       the continuum "electric field line" picture in the paper).
    2. Solving Kirchhoff's current law (KCL) for the electrical potential at
       every voxel, given a potential difference applied between the inlet
       and outlet faces of the sample.
    3. Decomposing the resulting current flow into individual "streamtubes"
       (discrete analogues of Berg's electric field lines: bundles of
       current that can be followed from inlet to outlet without splitting).
    4. Attributing a share of the pore volume to each streamtube, in
       proportion to how much current passes through the voxels it visits.
    5. Computing tau_c^2, C_c and iota_g^2 as volume- or current-weighted
       averages of the per-streamtube values, exactly mirroring Berg's
       definitions but with each continuum integral replaced by a discrete
       sum over voxels/streamtubes.

WHY HALF-CELL BOUNDARIES
-------------------------
Berg's sample is a cube of side length Delta_S with a potential difference
applied across two OUTER FACES of the cube, not across the centres of the
first and last layer of pore voxels. If the inlet/outlet voxel centres were
simply pinned to the applied potential, the effective sample length used in
every downstream formula would be (N-1)*dx (voxel-centre to voxel-centre)
instead of the physical N*dx, systematically shrinking the sample and
biasing every quantity that depends on comparing a path length to the
sample size (tortuosity in particular).

To avoid this, every inlet/outlet voxel here is connected to a *virtual*
external Dirichlet plane through a half-length resistor (length dx/2,
double the internal conductance of a full dx-long link). The inlet/outlet
voxel potentials remain unknowns solved for by KCL, exactly like every
other pore voxel; they simply have one extra resistor connecting them to
the fixed external boundary condition. This makes the electrical distance
between the two external planes exactly N*dx, matching Berg's Delta_S.

CONFIGURABLE MODELLING CHOICES
--------------------------------
Berg's continuum derivation leaves several discretisation choices open when
you move to a voxel grid: what conductance to assign a link, what length to
associate with it, how to split a voxel's volume among the streamtubes that
pass through it. These are exposed through BergPNConfig so the same pipeline
can be re-run under different conventions without duplicating code:

    conductance_mode : how internal-edge conductance G is computed
        "physical"       G = sigma * dx  (face area dx^2, length dx --
                          G = sigma*A/l = sigma*dx^2/dx = sigma*dx)
        "unit_dx"        G = sigma * dx^2  (face area dx^2, but the link
                          length is collapsed to 1 voxel unit rather than
                          the physical dx; isolates the effect of treating
                          voxel adjacency as unit-length hops while still
                          scaling with the physical face area)
        "topological_G1" G = 1 for every internal edge, regardless of sigma
                          or dx; a purely topological limit used to check
                          how much of a result comes from network
                          connectivity alone versus real geometry/physics

    length_mode : what length l is associated with an internal edge for
        path-length bookkeeping (tortuosity, constriction integrals)
        "geometric_dx"   l = dx (the physical voxel pitch)
        "unit_1"         l = 1 (每 link counts as one topological step;
                          only meaningful paired with topological_G1)

    volume_mode : how much pore volume is attributed to each voxel node
        "node_unsplit"   the full dx^3 voxel volume is attributed to its
                          own node (the only mode implemented here; the
                          field exists so future splitting schemes -- e.g.
                          apportioning a voxel's volume across its incident
                          edges -- can be added without touching the rest
                          of the pipeline)

    current_split_mode : how a voxel's volume is divided among the several
        streamtubes that may pass through it
        "flux_weighted"  a streamtube receives a share of each visited
                          voxel's volume in proportion to the streamtube's
                          own current divided by the total current passing
                          through that voxel (the only mode implemented)

    tortuosity_mode : whether tortuosity is computed from the raw graph
        path length ("raw_graph") or from a locally smoothed ("chord_
        corrected") path length that removes some of the Manhattan-grid
        staircasing artefact inherent to a 6-connected voxel graph. In
        chord-corrected mode, C_c is deliberately left untouched -- only
        tau_c is recomputed -- so the two modes isolate exactly how much
        of the raw tortuosity was a voxel-grid artefact rather than real
        pore-scale geometry.

CONSERVATION GUARANTEE
------------------------
Regardless of any configuration choice above, streamtube decomposition
(Stage 4) always traces current from the inlet boundary to the outlet
boundary and always accounts for it to within the configured current
tolerances (rel_current_tol, abs_current_tol). This is checked explicitly
and reported in the returned decomposition-info dictionary
(I_total_direct vs I_decomp, current_loss_rel) so any silent loss of
current is visible rather than hidden.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

try:
    import pyamg
    HAS_PYAMG = True
except Exception:
    HAS_PYAMG = False

try:
    from numba import njit
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False
    njit = None


# =============================================================================
# Small helpers
# =============================================================================

def _fmt(x: Any) -> str:
    """Format a number for human-readable diagnostic printing.

    Falls back to str() for anything that can't be cast to float, so this
    is safe to use on both numeric results and stray non-numeric values
    without crashing the verbose print statements.
    """
    try:
        return f"{float(x):.12g}"
    except Exception:
        return str(x)


def _degree_stats(deg: np.ndarray) -> Dict[int, int]:
    """Summarise a per-node degree array as {degree: count of nodes}.

    Used only for the verbose network-build report, to give a quick sanity
    check that the pore network looks like a plausible 6-connected voxel
    graph (e.g. no unexpectedly large number of isolated or degree-1 nodes).
    """
    vals, counts = np.unique(deg, return_counts=True)
    return dict(zip(vals.tolist(), counts.tolist()))


# =============================================================================
# 0. Configuration
# =============================================================================

@dataclass
class BergPNConfig:
    """All modelling and solver choices for one run of the pipeline.

    Grouping every choice into one object means the same nine-stage
    pipeline (build -> solve -> transport -> decompose -> split volume ->
    Berg quantities -> optional chord correction -> orchestrate -> test)
    can be re-run under different discretisation conventions just by
    constructing a different config, without any branching logic living
    outside of build_network() and the small number of places noted below.

    Physical / geometric parameters
    --------------------------------
    axis : int
        Which array axis (0, 1, or 2) the applied potential difference
        acts along, i.e. the direction Berg's Delta_S is measured in.
        Uses the same 0='x'/1='y'/2='z' convention as berg_cc's `direction`
        and particulate_claude's `direction`/`surface_axis`
        (particulate_claude.AXIS_TO_DIRECTION) -- there is no separate
        `direction` string here since this dataclass has no other axis
        parameter to go out of sync with.
    dx : float
        Physical voxel side length (a single number since voxels are
        assumed cubic). This is Berg's basic length scale for every
        internal edge under "physical" conductance/length modes.
    sigma : float
        Electrolyte conductivity, sigma in Berg's notation -- the
        conductivity of the fluid that would occupy an "optimal" tube of
        the same porosity. Sets the overall scale of every conductance in
        the network; it never varies from pore voxel to pore voxel here
        (Berg's assumption of constant-conductivity electrolyte, no
        surface conductance).
    phase_id : int
        Label in the segmented array identifying the conducting phase
        (Berg's pore space Omega) to build the network from.
    inlet_value, outlet_value : float
        The two potentials Delta_Phi is measured between, applied at the
        external Dirichlet planes on either side of the half-cell boundary
        resistors (Berg's ``Delta Phi`` across the two opposite faces of
        the cube).

    Discretisation-convention parameters (see module docstring for detail)
    ------------------------------------------------------------------------
    conductance_mode, length_mode, volume_mode, current_split_mode : str

    Tortuosity-correction parameters
    ----------------------------------
    tortuosity_mode : str
        "raw_graph" or "chord_corrected" (see module docstring).
    bend_radius_vox : float
        Only used when tortuosity_mode == "chord_corrected". The number of
        voxels on either side of a path bend that get replaced by a
        straight chord instead of the raw right-angle staircase step; a
        local smoothing radius, not a global path simplification.
    require_chord_inside_phase : bool
        If True, a chord replacement is only accepted if the straight
        segment it introduces stays entirely within the conducting phase
        (checked by dense sampling); chords that would cut through solid
        matrix are rejected and that bend is left uncorrected.
    chord_samples_per_dx : int
        Sampling density (samples per unit dx) used for the phase-
        containment check above; higher values are a more conservative
        (stricter) check at higher computational cost.

    Solver parameters
    -------------------
    use_pyamg : bool
        Whether to attempt an algebraic-multigrid solve (fast, approximate,
        iterative) before falling back to a direct sparse solve. Multigrid
        is normally much faster on the large, well-conditioned Laplacian
        systems this pipeline produces, but a direct solve is used as a
        fallback (and always used if pyamg is not installed) since it
        cannot fail to converge the way an iterative solver can.
    kcl_tol, kcl_maxiter : float, int
        Convergence tolerance and iteration cap for the iterative
        (pyamg) KCL solve. Not used if a direct solve is performed.
    rel_current_tol, abs_current_tol : float
        Thresholds below which a current is treated as numerical noise
        rather than a real current-carrying edge/path, used throughout
        Stage 4 (streamtube decomposition) to decide when all current has
        been accounted for and tracing can stop.
    progress_every : int
        How often (in number of streamtubes traced) to print a progress
        line during decomposition, for tracking long-running solves.
    verbose : bool
        Whether each stage should print a diagnostic summary block.
    """

    # Physical / geometric
    axis: int = 0
    dx: float = 1.0
    sigma: float = 1.0
    phase_id: int = 1
    inlet_value: float = 1.0
    outlet_value: float = 0.0

    # Discretisation conventions
    conductance_mode: str = "physical"        # "physical" | "unit_dx" | "topological_G1"
    length_mode: str = "geometric_dx"         # "geometric_dx" | "unit_1"
    volume_mode: str = "node_unsplit"         # "node_unsplit"
    current_split_mode: str = "flux_weighted"  # "flux_weighted"

    # Tortuosity correction
    tortuosity_mode: str = "raw_graph"        # "raw_graph" | "chord_corrected"
    bend_radius_vox: float = 3.0
    require_chord_inside_phase: bool = True
    chord_samples_per_dx: int = 24

    # Solver
    use_pyamg: bool = True
    kcl_tol: float = 1e-12
    kcl_maxiter: int = 300
    rel_current_tol: float = 1e-10
    abs_current_tol: float = 1e-14
    progress_every: int = 5000
    verbose: bool = True

    # Fast streamtube decomposition
    decomposition_method: str = "greedy_fast"   # "greedy_fast" | "topological"
    use_numba_decomposition: bool = True
    store_legacy_path_arrays: bool = True

    # Fast post-processing
    volume_chunk: int = 20000
    chord_chunk: int = 20000

    def __post_init__(self):
        valid_G = {"physical", "unit_dx", "topological_G1"}
        valid_L = {"geometric_dx", "unit_1"}
        valid_V = {"node_unsplit"}
        valid_split = {"flux_weighted"}
        valid_tau = {"raw_graph", "chord_corrected"}
        valid_decomp = {"greedy_fast", "topological"}

        if self.conductance_mode not in valid_G:
            raise ValueError(f"conductance_mode must be one of {valid_G}")
        if self.length_mode not in valid_L:
            raise ValueError(f"length_mode must be one of {valid_L}")
        if self.volume_mode not in valid_V:
            raise ValueError(f"volume_mode must be one of {valid_V}")
        if self.current_split_mode not in valid_split:
            raise ValueError(f"current_split_mode must be one of {valid_split}")
        if self.tortuosity_mode not in valid_tau:
            raise ValueError(f"tortuosity_mode must be one of {valid_tau}")
        if self.decomposition_method not in valid_decomp:
            raise ValueError(f"decomposition_method must be one of {valid_decomp}")
        if self.volume_chunk < 0 or self.chord_chunk < 0:
            raise ValueError("volume_chunk and chord_chunk must be non-negative")
        if self.axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        if self.dx <= 0 or self.sigma <= 0:
            raise ValueError("dx and sigma must be positive")


def _internal_edge_A_l(config: BergPNConfig) -> Tuple[float, float]:
    """Return (cross-sectional area A, length l) for one internal edge.

    Every internal-edge conductance in this pipeline is computed as
    G = sigma * A / l, mirroring Berg's continuum definition of the
    conductance of a tube segment (Section I: "Gr = sigma / integral
    1/A(x) dx", specialised to a constant-A, constant-l segment). The
    "topological_G1" mode is the one exception -- it bypasses this formula
    entirely and fixes G = 1 directly, since its purpose is to strip out
    all physical scaling and look at graph topology alone.

    Returns
    -------
    (A, l) : tuple of float
        A is the face area shared between two adjacent pore voxels, l is
        the centre-to-centre distance used for that edge, both according
        to config.conductance_mode / config.length_mode.
    """
    dx = config.dx
    if config.conductance_mode == "physical":
        A = dx**2
        l = dx if config.length_mode == "geometric_dx" else 1.0
    elif config.conductance_mode == "unit_dx":
        # Face area kept physical, but the edge length is collapsed to a
        # single topological unit rather than the physical dx.
        A = dx**2
        l = 1.0
    elif config.conductance_mode == "topological_G1":
        # G is fixed to 1 directly below; A, l are only used for length
        # bookkeeping (tortuosity/constriction integrals), so give them a
        # consistent unit-length convention.
        A = 1.0
        l = 1.0
    else:  # pragma: no cover - guarded by __post_init__
        raise ValueError(config.conductance_mode)
    return A, l


# =============================================================================
# 1. Network construction
# =============================================================================

def build_network(segmented: np.ndarray, config: BergPNConfig) -> Dict[str, Any]:
    """Build the 6-connected voxel resistor network with half-cell boundaries.

    This is the discrete analogue of Berg's continuum pore space Omega: every
    voxel of the conducting phase becomes a network node (a discretised
    piece of pore volume), and every pair of face-adjacent conducting
    voxels becomes a resistor (a discretised segment of an electric field
    line, Berg Fig. 1). Internal-edge conductance/length follow
    config.conductance_mode / config.length_mode (see BergPNConfig and
    _internal_edge_A_l for the exact formulas).

    Half-cell boundary resistors
    ------------------------------
    Every voxel touching the inlet face (axis-coordinate 0) or the outlet
    face (axis-coordinate shape[axis]-1) additionally gets ONE resistor
    connecting it to a virtual external node held at the applied boundary
    potential. That resistor has half the length of an internal edge and
    (since resistance is proportional to length) double the conductance of
    an internal edge computed under the same conductance_mode/length_mode.
    This is what makes the electrical distance between the two external
    planes exactly N*dx (N = number of voxel layers along the flow axis),
    matching Berg's macroscopic sample size Delta_S, rather than
    (N-1)*dx if only voxel centres were used as the boundary. See the
    module docstring for the full justification.

    Unlike a construction that deletes in-plane (lateral) edges at the
    boundary, this network keeps ALL internal edges, including those
    connecting two neighbouring inlet-face (or outlet-face) voxels to each
    other. Those voxels' potentials are still unknowns to be solved for --
    only the *external* virtual nodes are fixed -- so lateral coupling on
    the boundary plane is physically real and must remain in the network.

    Parameters
    ----------
    segmented : np.ndarray, 3D
        Segmented volume in (axis0, axis1, axis2) index order. Any voxel
        equal to config.phase_id is treated as conducting pore space;
        everything else is treated as insulating matrix.
    config : BergPNConfig
        Modelling choices; see BergPNConfig docstring. Uses phase_id, dx,
        sigma, axis, conductance_mode, length_mode.

    Returns
    -------
    net : dict
        Network description consumed by every later stage. Keys include:
        shape, mask, coords (voxel index coordinates of each node, in
        array order), id_grid (voxel index coords -> node id, -1 if not
        conducting), u/v (edge endpoint node ids), edge_axis (which array
        axis each internal edge runs along), degree, G/l_e/A_e (per-edge
        conductance/length/area), node_volume (per-node pore volume),
        phase_voxel_volume (total pore volume Omega), inlet/outlet (node
        ids of voxels touching each external face), G_boundary and
        boundary_half_length (the half-cell resistor conductance/length,
        shared by every boundary voxel), and a few provenance strings
        recording which config choices were used to build it.
    """
    segmented = np.asarray(segmented)
    if segmented.ndim != 3:
        raise ValueError("segmented must be a 3D array.")

    axis = config.axis
    dx = float(config.dx)
    sigma = float(config.sigma)

    mask = segmented == config.phase_id
    shape = mask.shape
    coords = np.argwhere(mask)
    n_nodes = len(coords)
    if n_nodes == 0:
        raise ValueError(f"No voxels found for phase_id={config.phase_id}.")

    id_grid = -np.ones(shape, dtype=np.int64)
    id_grid[mask] = np.arange(n_nodes, dtype=np.int64)

    # ---- internal face-sharing edges, one axis at a time ----
    u_parts, v_parts, axis_parts = [], [], []
    for ax in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[ax] = slice(0, shape[ax] - 1)
        sl1[ax] = slice(1, shape[ax])
        pair = mask[tuple(sl0)] & mask[tuple(sl1)]
        if np.any(pair):
            uu = id_grid[tuple(sl0)][pair]
            vv = id_grid[tuple(sl1)][pair]
            u_parts.append(uu)
            v_parts.append(vv)
            axis_parts.append(np.full(len(uu), ax, dtype=np.int8))

    if u_parts:
        u = np.concatenate(u_parts)
        v = np.concatenate(v_parts)
        edge_axis = np.concatenate(axis_parts)
    else:
        u = np.empty(0, dtype=np.int64)
        v = np.empty(0, dtype=np.int64)
        edge_axis = np.empty(0, dtype=np.int8)

    degree = np.zeros(n_nodes, dtype=np.int64)
    if len(u):
        np.add.at(degree, u, 1)
        np.add.at(degree, v, 1)

    # ---- internal-edge conductance / length / area ----
    if config.conductance_mode == "topological_G1":
        A_face, l_internal = _internal_edge_A_l(config)
        G_internal = 1.0
    else:
        A_face, l_internal = _internal_edge_A_l(config)
        G_internal = sigma * A_face / l_internal

    G = np.full(len(u), G_internal, dtype=float)
    l_e = np.full(len(u), l_internal, dtype=float)
    A_e = np.full(len(u), A_face, dtype=float)

    # ---- node volume (Berg's local pore-volume element) ----
    # Only "node_unsplit" is implemented: the full physical voxel volume
    # dx^3 is attributed to its own node. This is independent of
    # conductance_mode/length_mode, which only affect electrical/length
    # bookkeeping, not the physical volume the voxel actually occupies.
    if config.volume_mode != "node_unsplit":  # pragma: no cover - guarded above
        raise ValueError(config.volume_mode)
    node_volume = np.full(n_nodes, dx**3, dtype=float)

    # ---- half-cell boundary resistors ----
    inlet = np.flatnonzero(coords[:, axis] == 0)
    outlet = np.flatnonzero(coords[:, axis] == shape[axis] - 1)
    if len(inlet) == 0 or len(outlet) == 0:
        raise ValueError("Selected phase does not touch both external transport faces.")

    boundary_half_length = 0.5 * l_internal
    G_boundary = G_internal * (l_internal / boundary_half_length)  # = 2*G_internal

    net = {
        "shape": shape,
        "phase_id": config.phase_id,
        "mask": mask,
        "coords": coords,
        "id_grid": id_grid,
        "u": u,
        "v": v,
        "edge_axis": edge_axis,
        "degree": degree,
        "G": G,
        "l_e": l_e,
        "A_e": A_e,
        "node_volume": node_volume,
        "phase_voxel_volume": float(n_nodes * dx**3),
        "dx": dx,
        "sigma": sigma,
        "axis": int(axis),
        "inlet": inlet,
        "outlet": outlet,
        "G_boundary": float(G_boundary),
        "boundary_half_length": float(boundary_half_length),
        "conductance_mode": config.conductance_mode,
        "length_mode": config.length_mode,
        "volume_mode": config.volume_mode,
    }

    if config.verbose:
        print("\n" + "=" * 100)
        print("BUILD NETWORK -- HALF-CELL EXTERNAL BOUNDARIES")
        print("=" * 100)
        print(f"shape                         : {shape}")
        print(f"phase_id                      : {config.phase_id}")
        print(f"conductance_mode / length_mode: {config.conductance_mode} / {config.length_mode}")
        print(f"active voxels / nodes         : {n_nodes:,}")
        print(f"face-sharing internal edges   : {len(u):,}")
        print(f"degree breakdown              : {_degree_stats(degree)}")
        print(f"inlet / outlet voxels         : {len(inlet):,} / {len(outlet):,}")
        print(f"internal l / G                : {_fmt(l_internal)} / {_fmt(G_internal)}")
        print(f"boundary l / G                : {_fmt(boundary_half_length)} / {_fmt(G_boundary)}")
        print(f"phase voxel volume Omega      : {_fmt(net['phase_voxel_volume'])}")

    return net


def _assemble_laplacian(net: Dict[str, Any]) -> sp.csr_matrix:
    """Assemble the symmetric graph Laplacian of the internal-edge network.

    For an edge (u, v) with conductance G, Kirchhoff's current law
    contributes +G to the (u,u) and (v,v) diagonal entries and -G to both
    the (u,v) and (v,u) off-diagonal entries (current out of u into v must
    equal current out of v into u with opposite sign). Both directions are
    added explicitly here so the resulting matrix is guaranteed symmetric
    regardless of which endpoint happens to be listed first in net["u"]/
    net["v"] -- this is what makes the KCL solve in solve_kcl well-posed.

    This Laplacian only includes internal pore-to-pore edges; the external
    half-cell boundary resistors are added separately in solve_kcl, since
    they connect to virtual nodes outside the pore network itself.
    """
    n = len(net["coords"])
    u, v, G = net["u"], net["v"], net["G"]
    rows = np.concatenate([u, v, u, v])
    cols = np.concatenate([v, u, u, v])
    data = np.concatenate([-G, -G, G, G])
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


# =============================================================================
# 2. KCL solve
# =============================================================================

def solve_kcl(
    net: Dict[str, Any],
    config: BergPNConfig,
    warm_start_field: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Solve Kirchhoff's current law for the potential at every pore voxel.

    Physically this finds the electrical potential Phi(x) inside the pore
    space that satisfies charge conservation everywhere, given a fixed
    potential difference applied across the two external planes via the
    half-cell boundary resistors built in build_network. This is the
    discrete equivalent of solving Laplace's equation for Phi in Berg's
    continuum picture (Section II), from which the drift velocity field
    -mu*grad(Phi) and hence the conductance-reduction factor iota are
    derived in later stages.

    Handling of disconnected pore regions
    ----------------------------------------
    A segmented image can contain pore voxels that are not connected to
    both the inlet and outlet (dead-end pores, or clusters cut off
    entirely from the flow path). These carry no current and would make a
    single global linear system singular if included naively. This
    function first splits the network into connected components and
    solves each one separately:
        - a component touching the inlet and/or outlet boundary is solved
          for normally (its half-cell boundary resistors act as the
          Dirichlet-like coupling to the fixed external potential);
        - a component touching NEITHER boundary is "floating": it carries
          no current in the steady state, so its nodes are simply assigned
          a constant potential (using warm_start_field's local average if
          available, otherwise the midpoint of inlet_value/outlet_value)
          rather than being run through the solver at all.
    This means the returned potential field is always fully defined and
    the KCL residual is always checkable, even on segmentations with a
    lot of disconnected porosity.

    warm_start_field as an initial guess only
    --------------------------------------------
    If provided, warm_start_field is used ONLY to seed the iterative
    solver's initial guess (x0) -- for example, an approximate potential
    field already produced by a faster/less exact solver such as
    TauFactor. It never replaces the solve: the full KCL system is always
    assembled and solved to config.kcl_tol here, so the returned field is
    always this pipeline's own converged solution, just potentially found
    faster by starting from a good guess. A good warm start typically
    reduces the number of multigrid iterations needed; a poor or
    mismatched one only costs a little wasted solver effort, since the
    solve still runs to convergence regardless.

    Parameters
    ----------
    net : dict
        Output of build_network.
    config : BergPNConfig
        Uses inlet_value, outlet_value, use_pyamg, kcl_tol, kcl_maxiter,
        verbose.
    warm_start_field : np.ndarray or None
        Optional 3D array, same shape as the segmented volume, giving an
        approximate potential at every voxel (values outside the pore
        mask are ignored). Used purely as an iterative-solver initial
        guess as described above.

    Returns
    -------
    sol : dict
        phi (potential at every pore node), L (internal-edge Laplacian),
        inlet/outlet/is_inlet/is_outlet, component_labels/component_info/
        n_components (connected-component bookkeeping described above),
        inlet_value/outlet_value, and kcl_residual_full (the full KCL
        residual at every node, including the external half-cell
        contributions -- should be close to zero everywhere; a useful
        correctness check).
    """
    n = len(net["coords"])
    u, v = net["u"], net["v"]
    inlet = net["inlet"]
    outlet = net["outlet"]
    Gb = float(net["G_boundary"])
    L = _assemble_laplacian(net)

    is_inlet = np.zeros(n, dtype=bool)
    is_outlet = np.zeros(n, dtype=bool)
    is_inlet[inlet] = True
    is_outlet[outlet] = True

    if len(u):
        adj = sp.coo_matrix(
            (np.ones(2 * len(u), dtype=np.int8),
             (np.concatenate([u, v]), np.concatenate([v, u]))),
            shape=(n, n),
        ).tocsr()
        ncomp, labels = connected_components(adj, directed=False)
    else:
        ncomp, labels = n, np.arange(n, dtype=int)

    raw_phi = None
    if warm_start_field is not None:
        arr = np.asarray(warm_start_field)
        if arr.shape != tuple(net["shape"]):
            raise ValueError("warm_start_field must have the same 3D shape as segmented.")
        raw_phi = arr[net["mask"]].astype(float)

    phi = np.full(n, np.nan, dtype=float)
    comp_info: List[Dict[str, Any]] = []
    inlet_value = float(config.inlet_value)
    outlet_value = float(config.outlet_value)

    for c in range(ncomp):
        nodes = np.flatnonzero(labels == c)
        loc_in = is_inlet[nodes]
        loc_out = is_outlet[nodes]
        has_in = bool(np.any(loc_in))
        has_out = bool(np.any(loc_out))

        if has_in or has_out:
            A = L[nodes][:, nodes].tocsr().copy()
            gext = np.zeros(len(nodes), dtype=float)
            rhs = np.zeros(len(nodes), dtype=float)

            if has_in:
                gext[loc_in] += Gb
                rhs[loc_in] += Gb * inlet_value
            if has_out:
                gext[loc_out] += Gb
                rhs[loc_out] += Gb * outlet_value

            A = A + sp.diags(gext, format="csr")
            used_solver = "spsolve"

            if config.use_pyamg and HAS_PYAMG and A.shape[0] >= 10:
                try:
                    ml = pyamg.ruge_stuben_solver(A)
                    x0 = raw_phi[nodes] if raw_phi is not None else None
                    x = ml.solve(rhs, x0=x0, tol=config.kcl_tol, maxiter=config.kcl_maxiter)
                    used_solver = "pyamg.ruge_stuben"
                except Exception as exc:
                    if config.verbose:
                        print(f"pyamg failed on component {c}; using spsolve: {exc!r}")
                    x = spla.spsolve(A, rhs)
                    used_solver = "spsolve_after_pyamg_failure"
            else:
                x = spla.spsolve(A, rhs)

            phi[nodes] = x
            if has_in and has_out:
                kind = "spanning"
            elif has_in:
                kind = "inlet_only"
            else:
                kind = "outlet_only"
        else:
            if raw_phi is not None:
                vals = raw_phi[nodes]
                vals = vals[np.isfinite(vals)]
                value = float(np.mean(vals)) if len(vals) else 0.5 * (inlet_value + outlet_value)
            else:
                value = 0.5 * (inlet_value + outlet_value)
            phi[nodes] = value
            kind = "floating_disconnected"
            used_solver = "constant"

        comp_info.append({
            "component": int(c),
            "n_nodes": int(len(nodes)),
            "has_inlet": has_in,
            "has_outlet": has_out,
            "kind": kind,
            "solver": used_solver,
        })

    if np.any(~np.isfinite(phi)):
        raise RuntimeError("Some node potentials were not assigned.")

    residual = L @ phi
    residual[inlet] += Gb * (phi[inlet] - inlet_value)
    residual[outlet] += Gb * (phi[outlet] - outlet_value)

    if config.verbose:
        kinds: Dict[str, int] = {}
        for info in comp_info:
            kinds[info["kind"]] = kinds.get(info["kind"], 0) + 1
        print("\n" + "=" * 100)
        print("KCL SOLVE")
        print("=" * 100)
        print(f"inlet / outlet potential       : {inlet_value} / {outlet_value}")
        print(f"connected components           : {ncomp:,}")
        print(f"component kinds                : {kinds}")
        print(f"pyamg available / requested    : {HAS_PYAMG} / {config.use_pyamg}")
        print(f"warm start field provided      : {warm_start_field is not None}")
        print(f"max |full KCL residual|        : {_fmt(np.max(np.abs(residual)))}")
        print(f"RMS full KCL residual          : {_fmt(np.sqrt(np.mean(residual**2)))}")

    return {
        "phi": phi,
        "L": L,
        "inlet": inlet,
        "outlet": outlet,
        "is_inlet": is_inlet,
        "is_outlet": is_outlet,
        "component_labels": labels,
        "component_info": comp_info,
        "n_components": int(ncomp),
        "inlet_value": inlet_value,
        "outlet_value": outlet_value,
        "kcl_residual_full": residual,
    }


def _compute_edge_currents(net: Dict[str, Any], phi: np.ndarray) -> np.ndarray:
    """Current through every internal edge, I = G * (phi[u] - phi[v]).

    Sign convention: positive means current flows from node u to node v
    (i.e. from the higher-potential end to the lower one, as in Ohm's law
    for a resistor with no internal EMF).
    """
    return net["G"] * (phi[net["u"]] - phi[net["v"]])


# =============================================================================
# 3. Effective transport quantities
# =============================================================================

def effective_transport(
    net: Dict[str, Any],
    sol: Dict[str, Any],
    config: BergPNConfig,
) -> Dict[str, Any]:
    """Compute the sample-scale conductance and formation resistivity factor.

    This is the discrete version of Berg's formation resistivity factor
    F = R_o/R_w (Section I), obtained here from the total current the
    solved potential field drives through the half-cell boundary
    resistors, divided by the applied potential difference, and compared
    to the conductance an "optimal" (Berg's term: a set of straight,
    constant-area tubes filling the whole domain) geometry of the same
    total volume would give.

    Because L_sample = shape[axis]*dx exactly under the half-cell boundary
    convention (see build_network), this reproduces Berg's macroscopic
    Delta_S precisely rather than being off by one voxel.

    Parameters
    ----------
    net : dict
        Output of build_network.
    sol : dict
        Output of solve_kcl.
    config : BergPNConfig
        Uses axis (defaults to net["axis"] if not overridden by the caller
        elsewhere), dx, sigma.

    Returns
    -------
    transport : dict
        edge_current (per internal edge), inlet_boundary_current /
        outlet_boundary_current (current through each boundary resistor,
        should sum to the same total on both sides at convergence), I_in /
        I_out / I_total, delta_phi, G_eff (effective sample conductance),
        L_sample (= shape[axis]*dx), V_domain (total sample volume,
        including matrix), G0_Berg (the "optimal geometry" conductance an
        equivalent-volume set of straight tubes would give, Berg Eq. 4),
        one_over_F and F (Berg's formation resistivity factor and its
        reciprocal), A_cross_geom / sigma_eff_geom / relative_conductivity
        (geometric cross-section based effective conductivity, an
        alternative sanity-check route to the same F), and energy_internal
        / energy_boundary / energy_total / power_boundary_I_dV (power
        dissipation bookkeeping, useful for verifying energy conservation
        of the solved field).
    """
    axis = net["axis"]
    shape = net["shape"]
    dx = float(net["dx"])
    sigma = float(net["sigma"])
    Gb = float(net["G_boundary"])
    phi = sol["phi"]

    L_sample = shape[axis] * dx
    V_domain = float(np.prod(shape)) * dx**3

    Vin = float(sol["inlet_value"])
    Vout = float(sol["outlet_value"])
    delta_phi = abs(Vin - Vout)

    edge_current = _compute_edge_currents(net, phi)
    inlet_boundary_current = Gb * (Vin - phi[sol["inlet"]])
    outlet_boundary_current = Gb * (phi[sol["outlet"]] - Vout)

    I_in = float(np.sum(inlet_boundary_current))
    I_out = float(np.sum(outlet_boundary_current))
    I_total = 0.5 * (I_in + I_out)
    G_eff = I_total / delta_phi

    G0 = sigma * V_domain / L_sample**2
    one_over_F = G_eff / G0
    F = np.inf if one_over_F <= 0 else 1.0 / one_over_F

    other = [0, 1, 2]
    other.remove(axis)
    A_cross = shape[other[0]] * shape[other[1]] * dx**2
    sigma_eff_geom = G_eff * L_sample / A_cross

    dphi_e = phi[net["u"]] - phi[net["v"]]
    energy_internal = float(np.sum(net["G"] * dphi_e**2))
    energy_boundary = float(
        np.sum(Gb * (Vin - phi[sol["inlet"]])**2)
        + np.sum(Gb * (phi[sol["outlet"]] - Vout)**2)
    )
    energy_total = energy_internal + energy_boundary
    power_boundary = I_total * delta_phi

    out = {
        "edge_current": edge_current,
        "inlet_boundary_current": inlet_boundary_current,
        "outlet_boundary_current": outlet_boundary_current,
        "I_in": I_in,
        "I_out": I_out,
        "I_total": float(I_total),
        "delta_phi": float(delta_phi),
        "G_eff": float(G_eff),
        "L_sample": float(L_sample),
        "V_domain": float(V_domain),
        "G0_Berg": float(G0),
        "one_over_F": float(one_over_F),
        "F": float(F),
        "A_cross_geom": float(A_cross),
        "sigma_eff_geom": float(sigma_eff_geom),
        "relative_conductivity_geom": float(sigma_eff_geom / sigma),
        "energy_internal": energy_internal,
        "energy_boundary": energy_boundary,
        "energy_total": energy_total,
        "power_boundary_I_dV": float(power_boundary),
    }

    if config.verbose:
        print("\n" + "=" * 100)
        print("EFFECTIVE TRANSPORT")
        print("=" * 100)
        print(f"I_in / I_out                  : {_fmt(I_in)} / {_fmt(I_out)}")
        print(f"I_total                       : {_fmt(I_total)}")
        print(f"DeltaPhi external             : {_fmt(delta_phi)}")
        print(f"L_sample = N*dx               : {_fmt(L_sample)}")
        print(f"G_eff                         : {_fmt(G_eff)}")
        print(f"1/F                           : {_fmt(one_over_F)}")
        print(f"F                             : {_fmt(F)}")
        print(f"sigma_eff/sigma (geom)        : {_fmt(sigma_eff_geom / sigma)}")
        print(f"energy internal               : {_fmt(energy_internal)}")
        print(f"energy boundary half-cells    : {_fmt(energy_boundary)}")
        print(f"energy total                  : {_fmt(energy_total)}")
        print(f"I_total*DeltaPhi              : {_fmt(power_boundary)}")

    return out


# =============================================================================
# 4. Streamtube decomposition (conservative, boundary-mode independent)
# =============================================================================

def _prepare_augmented_flow_dag(
    net: Dict[str, Any],
    sol: Dict[str, Any],
    transport: Dict[str, Any],
    config: BergPNConfig,
) -> Dict[str, Any]:
    """Build the same thresholded/pruned augmented current DAG used by Stage 4.

    The physical directed edges are oriented strictly down potential, so after
    removing zero/tiny-current links the graph is acyclic.  A virtual source S
    injects the solved inlet half-cell currents and a virtual sink T receives
    the solved outlet half-cell currents.
    """
    phi = sol["phi"]
    u, v = net["u"], net["v"]
    edge_current = transport["edge_current"]
    n_phys = len(phi)
    S = n_phys
    T = n_phys + 1
    n_aug = n_phys + 2

    rel_tol = float(config.rel_current_tol)
    abs_tol = float(config.abs_current_tol)
    I_scale = max(
        float(np.max(np.abs(edge_current))) if len(edge_current) else 0.0,
        float(np.max(np.abs(transport["inlet_boundary_current"])))
        if len(transport["inlet_boundary_current"]) else 0.0,
        abs_tol,
    )
    I_threshold = max(abs_tol, rel_tol * I_scale)

    pos = edge_current > I_threshold
    neg = edge_current < -I_threshold
    up_int = np.concatenate([u[pos], v[neg]]).astype(np.int64, copy=False)
    dn_int = np.concatenate([v[pos], u[neg]]).astype(np.int64, copy=False)
    eidx_int = np.concatenate([np.flatnonzero(pos), np.flatnonzero(neg)]).astype(np.int64, copy=False)
    flow_int = np.concatenate([edge_current[pos], -edge_current[neg]]).astype(float, copy=False)
    dphi_int = np.abs(phi[up_int] - phi[dn_int])

    inlet_nodes_all = np.asarray(sol["inlet"], dtype=np.int64)
    inlet_I_all = np.asarray(transport["inlet_boundary_current"], dtype=float)
    keep_in = inlet_I_all > I_threshold
    inlet_nodes = inlet_nodes_all[keep_in]
    inlet_I = inlet_I_all[keep_in]
    up_in = np.full(len(inlet_nodes), S, dtype=np.int64)
    dn_in = inlet_nodes
    dphi_in = sol["inlet_value"] - phi[inlet_nodes]

    outlet_nodes_all = np.asarray(sol["outlet"], dtype=np.int64)
    outlet_I_all = np.asarray(transport["outlet_boundary_current"], dtype=float)
    keep_out = outlet_I_all > I_threshold
    outlet_nodes = outlet_nodes_all[keep_out]
    outlet_I = outlet_I_all[keep_out]
    up_out = outlet_nodes
    dn_out = np.full(len(outlet_nodes), T, dtype=np.int64)
    dphi_out = phi[outlet_nodes] - sol["outlet_value"]

    up = np.concatenate([up_int, up_in, up_out])
    dn = np.concatenate([dn_int, dn_in, dn_out])
    flow = np.concatenate([flow_int, inlet_I, outlet_I])
    dphi = np.concatenate([dphi_int, dphi_in, dphi_out])
    kind = np.concatenate([
        np.zeros(len(flow_int), dtype=np.int8),
        np.ones(len(inlet_I), dtype=np.int8),
        np.full(len(outlet_I), 2, dtype=np.int8),
    ])
    orig_edge = np.concatenate([
        eidx_int,
        np.full(len(inlet_I), -1, dtype=np.int64),
        np.full(len(outlet_I), -1, dtype=np.int64),
    ])
    # Keep legacy Stage-4 convention exactly: internal bookkeeping uses dx.
    seg_len = np.concatenate([
        np.full(len(flow_int), net["dx"], dtype=float),
        np.full(len(inlet_I), net["boundary_half_length"], dtype=float),
        np.full(len(outlet_I), net["boundary_half_length"], dtype=float),
    ])

    valid = (flow > I_threshold) & (dphi > 0)
    up, dn, flow, dphi, kind, orig_edge, seg_len = (
        arr[valid] for arr in (up, dn, flow, dphi, kind, orig_edge, seg_len)
    )
    if len(flow) == 0:
        raise RuntimeError("No current-carrying augmented edges above threshold.")

    # Reverse reachability from T, preserving the original implementation's
    # pruning semantics but paying this cost only once.
    order_dn = np.argsort(dn, kind="mergesort")
    dn_sorted = dn[order_dn]
    counts_dn = np.bincount(dn_sorted, minlength=n_aug)
    ptr_dn = np.empty(n_aug + 1, dtype=np.int64)
    ptr_dn[0] = 0
    np.cumsum(counts_dn, out=ptr_dn[1:])

    reachable = np.zeros(n_aug, dtype=bool)
    reachable[T] = True
    stack = [T]
    while stack:
        node = stack.pop()
        a, b = int(ptr_dn[node]), int(ptr_dn[node + 1])
        for q in range(a, b):
            ee = int(order_dn[q])
            prev = int(up[ee])
            if not reachable[prev]:
                reachable[prev] = True
                stack.append(prev)

    keep = reachable[dn]
    n_pruned = int(np.count_nonzero(~keep))
    up, dn, flow, dphi, kind, orig_edge, seg_len = (
        arr[keep] for arr in (up, dn, flow, dphi, kind, orig_edge, seg_len)
    )
    if not reachable[S]:
        raise RuntimeError("Virtual source cannot reach virtual sink through current-carrying edges.")

    order = np.argsort(up, kind="mergesort")
    up_s, dn_s = up[order], dn[order]
    flow_s, dphi_s = flow[order], dphi[order]
    kind_s, edge_s, len_s = kind[order], orig_edge[order], seg_len[order]

    counts = np.bincount(up_s, minlength=n_aug)
    ptr = np.empty(n_aug + 1, dtype=np.int64)
    ptr[0] = 0
    np.cumsum(counts, out=ptr[1:])

    return {
        "S": int(S), "T": int(T), "n_aug": int(n_aug),
        "I_threshold": float(I_threshold), "n_pruned": n_pruned,
        "up": up_s, "dn": dn_s, "flow": flow_s, "dphi": dphi_s,
        "kind": kind_s, "edge": edge_s, "seg_len": len_s, "ptr": ptr,
    }


def _greedy_fast_core_python(ptr, dn, flow, threshold, S, T):
    """Fallback greedy core.  Uses a source max-heap and DAG path tracing."""
    import heapq

    remaining = flow.copy()
    aS, bS = int(ptr[S]), int(ptr[S + 1])
    versions = np.zeros(len(remaining), dtype=np.int64)
    heap = [(-float(remaining[p]), int(p), 0) for p in range(aS, bS)]
    heapq.heapify(heap)
    source_total = float(np.sum(remaining[aS:bS]))

    flat = []
    offsets = [0]
    gammas = []
    n_stuck = 0
    stuck_removed = 0.0

    def top_source():
        while heap:
            negf, p, ver = heap[0]
            if ver != versions[p] or remaining[p] <= threshold or -negf != float(remaining[p]):
                heapq.heappop(heap)
                continue
            return p
        return -1

    while source_total > threshold:
        sp = top_source()
        if sp < 0:
            break
        source_before = float(remaining[sp])
        path = [sp]
        node = int(dn[sp])
        stuck = False

        while node != T:
            a, b = int(ptr[node]), int(ptr[node + 1])
            best = -1
            best_flow = threshold
            for p in range(a, b):
                f = float(remaining[p])
                if f > best_flow:
                    best_flow = f
                    best = p
            if best < 0:
                stuck = True
                break
            path.append(best)
            node = int(dn[best])

        if stuck or len(path) < 2:
            n_stuck += 1
            removed = float(remaining[sp])
            remaining[sp] = 0.0
            source_total -= removed
            stuck_removed += max(removed, 0.0)
            versions[sp] += 1
            continue

        I_gamma = min(float(remaining[p]) for p in path)
        if I_gamma <= threshold:
            break
        for p in path:
            r = float(remaining[p]) - I_gamma
            remaining[p] = 0.0 if r < threshold else r

        source_after = float(remaining[sp])
        source_total -= max(0.0, source_before - source_after)
        versions[sp] += 1
        if source_after > threshold:
            heapq.heappush(heap, (-source_after, int(sp), int(versions[sp])))

        flat.extend(path)
        offsets.append(len(flat))
        gammas.append(I_gamma)

    return (
        np.asarray(flat, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(gammas, dtype=float),
        int(n_stuck), float(stuck_removed), float(max(source_total, 0.0)),
    )


if HAS_NUMBA:
    @njit
    def _heap_better(p, q, remaining):
        fp = remaining[p]
        fq = remaining[q]
        return (fp > fq) or (fp == fq and p < q)

    @njit
    def _heap_sift_down(heap, size, i, remaining):
        while True:
            l = 2 * i + 1
            r = l + 1
            best = i
            if l < size and _heap_better(heap[l], heap[best], remaining):
                best = l
            if r < size and _heap_better(heap[r], heap[best], remaining):
                best = r
            if best == i:
                break
            tmp = heap[i]
            heap[i] = heap[best]
            heap[best] = tmp
            i = best

    @njit
    def _heapify(heap, size, remaining):
        if size <= 1:
            return
        for i in range(size // 2 - 1, -1, -1):
            _heap_sift_down(heap, size, i, remaining)

    @njit
    def _greedy_fast_core_numba(ptr, dn, flow, threshold, S, T, n_aug):
        remaining = flow.copy()
        aS = ptr[S]
        bS = ptr[S + 1]
        heap_size = bS - aS
        heap = np.empty(heap_size, dtype=np.int64)
        for i in range(heap_size):
            heap[i] = aS + i
        _heapify(heap, heap_size, remaining)

        source_total = 0.0
        for p in range(aS, bS):
            source_total += remaining[p]

        max_paths = len(flow) + 1
        offsets = np.empty(max_paths + 1, dtype=np.int64)
        gammas = np.empty(max_paths, dtype=np.float64)
        offsets[0] = 0
        path_count = 0

        flat_cap = max(1024, len(flow) * 2)
        flat = np.empty(flat_cap, dtype=np.int64)
        flat_n = 0
        temp = np.empty(n_aug + 1, dtype=np.int64)

        n_stuck = 0
        stuck_removed = 0.0

        while source_total > threshold and heap_size > 0:
            sp = heap[0]
            if remaining[sp] <= threshold:
                break
            source_before = remaining[sp]
            plen = 1
            temp[0] = sp
            node = dn[sp]
            stuck = False

            while node != T:
                a = ptr[node]
                b = ptr[node + 1]
                best = -1
                best_flow = threshold
                for p in range(a, b):
                    f = remaining[p]
                    if f > best_flow:
                        best_flow = f
                        best = p
                if best < 0:
                    stuck = True
                    break
                temp[plen] = best
                plen += 1
                node = dn[best]

            if stuck or plen < 2:
                n_stuck += 1
                removed = remaining[sp]
                remaining[sp] = 0.0
                source_total -= removed
                if removed > 0.0:
                    stuck_removed += removed
                _heap_sift_down(heap, heap_size, 0, remaining)
                continue

            I_gamma = remaining[temp[0]]
            for j in range(1, plen):
                f = remaining[temp[j]]
                if f < I_gamma:
                    I_gamma = f
            if I_gamma <= threshold:
                break

            for j in range(plen):
                p = temp[j]
                r = remaining[p] - I_gamma
                if r < threshold:
                    r = 0.0
                remaining[p] = r

            source_after = remaining[sp]
            source_total -= source_before - source_after
            if source_total < 0.0 and source_total > -1e-14:
                source_total = 0.0
            _heap_sift_down(heap, heap_size, 0, remaining)

            need = flat_n + plen
            if need > flat_cap:
                newcap = flat_cap
                while newcap < need:
                    newcap = int(newcap * 1.7) + 1024
                tmpflat = np.empty(newcap, dtype=np.int64)
                tmpflat[:flat_n] = flat[:flat_n]
                flat = tmpflat
                flat_cap = newcap

            flat[flat_n:flat_n + plen] = temp[:plen]
            flat_n += plen
            gammas[path_count] = I_gamma
            path_count += 1
            offsets[path_count] = flat_n

        return (flat[:flat_n].copy(), offsets[:path_count + 1].copy(),
                gammas[:path_count].copy(), n_stuck, stuck_removed,
                max(source_total, 0.0))

    @njit
    def _grow_i64(a, newcap, used):
        b = np.empty(newcap, dtype=np.int64)
        b[:used] = a[:used]
        return b

    @njit
    def _grow_f64(a, newcap, used):
        b = np.empty(newcap, dtype=np.float64)
        b[:used] = a[:used]
        return b

    @njit
    def _topological_core_numba(ptr, dn, flow, threshold, S, T, topo_nodes):
        """One-pass local packet matching on a DAG; no source-to-sink path search."""
        n_aug = len(ptr) - 1
        remaining = flow.copy()
        head = np.full(n_aug, -1, dtype=np.int64)
        tail = np.full(n_aug, -1, dtype=np.int64)

        cap = max(2048, len(flow) * 2)
        parent = np.empty(cap, dtype=np.int64)
        edge_pos = np.empty(cap, dtype=np.int64)
        amount = np.empty(cap, dtype=np.float64)
        next_rec = np.empty(cap, dtype=np.int64)
        nrec = 0

        term_cap = max(1024, len(flow))
        term_rec = np.empty(term_cap, dtype=np.int64)
        term_amt = np.empty(term_cap, dtype=np.float64)
        nterm = 0

        source_injected = 0.0
        threshold_discarded = 0.0
        unmatched_current = 0.0

        # Every source edge creates one initial packet at its inlet voxel.
        for p in range(ptr[S], ptr[S + 1]):
            f = remaining[p]
            if f <= threshold:
                continue
            if nrec >= cap:
                newcap = int(cap * 1.7) + 2048
                parent = _grow_i64(parent, newcap, nrec)
                edge_pos = _grow_i64(edge_pos, newcap, nrec)
                amount = _grow_f64(amount, newcap, nrec)
                next_rec = _grow_i64(next_rec, newcap, nrec)
                cap = newcap
            rid = nrec
            nrec += 1
            parent[rid] = -1
            edge_pos[rid] = p
            amount[rid] = f
            next_rec[rid] = -1
            node = dn[p]
            if head[node] < 0:
                head[node] = rid
                tail[node] = rid
            else:
                next_rec[tail[node]] = rid
                tail[node] = rid
            source_injected += f
            remaining[p] = 0.0

        # Process physical nodes once in strict high->low potential order.
        for kk in range(len(topo_nodes)):
            node = topo_nodes[kk]
            rid = head[node]
            if rid < 0:
                continue

            p = ptr[node]
            b = ptr[node + 1]
            while p < b and remaining[p] <= threshold:
                p += 1

            while rid >= 0:
                rem = amount[rid]
                while rem > threshold:
                    while p < b and remaining[p] <= threshold:
                        p += 1
                    if p >= b:
                        unmatched_current += rem
                        rem = 0.0
                        break

                    avail = remaining[p]
                    q = rem if rem < avail else avail
                    if q <= threshold:
                        threshold_discarded += q
                        rem -= q
                        remaining[p] -= q
                        continue

                    if nrec >= cap:
                        newcap = int(cap * 1.7) + 2048
                        parent = _grow_i64(parent, newcap, nrec)
                        edge_pos = _grow_i64(edge_pos, newcap, nrec)
                        amount = _grow_f64(amount, newcap, nrec)
                        next_rec = _grow_i64(next_rec, newcap, nrec)
                        cap = newcap
                    newrid = nrec
                    nrec += 1
                    parent[newrid] = rid
                    edge_pos[newrid] = p
                    amount[newrid] = q
                    next_rec[newrid] = -1

                    nxt = dn[p]
                    if nxt == T:
                        if nterm >= term_cap:
                            newcap_t = int(term_cap * 1.7) + 1024
                            term_rec = _grow_i64(term_rec, newcap_t, nterm)
                            term_amt = _grow_f64(term_amt, newcap_t, nterm)
                            term_cap = newcap_t
                        term_rec[nterm] = newrid
                        term_amt[nterm] = q
                        nterm += 1
                    else:
                        if head[nxt] < 0:
                            head[nxt] = newrid
                            tail[nxt] = newrid
                        else:
                            next_rec[tail[nxt]] = newrid
                            tail[nxt] = newrid

                    rem -= q
                    remaining[p] -= q
                    if remaining[p] < threshold:
                        if remaining[p] > 0.0:
                            threshold_discarded += remaining[p]
                        remaining[p] = 0.0
                        p += 1
                    if rem < threshold:
                        if rem > 0.0:
                            threshold_discarded += rem
                        rem = 0.0

                rid = next_rec[rid]

        # Any source current not delivered to T is visible in the conservation report.
        return (parent[:nrec].copy(), edge_pos[:nrec].copy(),
                term_rec[:nterm].copy(), term_amt[:nterm].copy(),
                source_injected, threshold_discarded, unmatched_current)

    @njit
    def _chains_to_flat_numba(parent, edge_pos, term_rec):
        n = len(term_rec)
        offsets = np.empty(n + 1, dtype=np.int64)
        offsets[0] = 0
        for k in range(n):
            r = term_rec[k]
            m = 0
            while r >= 0:
                m += 1
                r = parent[r]
            offsets[k + 1] = offsets[k] + m
        flat = np.empty(offsets[n], dtype=np.int64)
        for k in range(n):
            r = term_rec[k]
            j = offsets[k + 1] - 1
            while r >= 0:
                flat[j] = edge_pos[r]
                j -= 1
                r = parent[r]
        return flat, offsets


def _topological_core_python(ptr, dn, flow, threshold, S, T, topo_nodes):
    """Pure-Python fallback for the local topological packet matcher."""
    from collections import deque
    remaining = flow.copy()
    queues = [deque() for _ in range(len(ptr) - 1)]
    records = []  # (parent_record, edge_position, amount)
    terminals = []
    source_injected = 0.0
    threshold_discarded = 0.0
    unmatched_current = 0.0

    for p in range(int(ptr[S]), int(ptr[S + 1])):
        f = float(remaining[p])
        if f <= threshold:
            continue
        rid = len(records)
        records.append((-1, int(p), f))
        queues[int(dn[p])].append(rid)
        source_injected += f
        remaining[p] = 0.0

    for node in topo_nodes:
        node = int(node)
        if not queues[node]:
            continue
        p = int(ptr[node]); b = int(ptr[node + 1])
        while p < b and remaining[p] <= threshold:
            p += 1
        while queues[node]:
            rid = queues[node].popleft()
            rem = float(records[rid][2])
            while rem > threshold:
                while p < b and remaining[p] <= threshold:
                    p += 1
                if p >= b:
                    unmatched_current += rem
                    rem = 0.0
                    break
                avail = float(remaining[p])
                q = min(rem, avail)
                nr = len(records)
                records.append((rid, p, q))
                nxt = int(dn[p])
                if nxt == T:
                    terminals.append((nr, q))
                else:
                    queues[nxt].append(nr)
                rem -= q
                remaining[p] -= q
                if remaining[p] < threshold:
                    threshold_discarded += max(float(remaining[p]), 0.0)
                    remaining[p] = 0.0
                    p += 1
                if rem < threshold:
                    threshold_discarded += max(rem, 0.0)
                    rem = 0.0

    offsets = [0]
    flat = []
    gammas = []
    for rid, q in terminals:
        rev = []
        r = rid
        while r >= 0:
            pr, ep, _ = records[r]
            rev.append(ep)
            r = pr
        flat.extend(reversed(rev))
        offsets.append(len(flat))
        gammas.append(q)
    return (np.asarray(flat, np.int64), np.asarray(offsets, np.int64),
            np.asarray(gammas, float), source_injected,
            threshold_discarded, unmatched_current)


def _streamtubes_from_augmented_paths(flat_pos, offsets, gammas, dag, net, sol,
                                      store_legacy_path_arrays=True):
    """Convert compact augmented-edge paths to the legacy streamtube dictionaries."""
    dn_s = dag["dn"]
    kind_s = dag["kind"]
    edge_s = dag["edge"]
    dphi_s = dag["dphi"]
    len_s = dag["seg_len"]
    DeltaPhi = abs(float(sol["inlet_value"] - sol["outlet_value"]))
    l_internal_default = float(net["l_e"][0]) if len(net["l_e"]) else float(net["dx"])

    streamtubes: List[Dict[str, Any]] = []
    append = streamtubes.append
    npaths = len(gammas)
    for k in range(npaths):
        a, b = int(offsets[k]), int(offsets[k + 1])
        pp = flat_pos[a:b]
        if len(pp) < 2:
            continue
        kinds = kind_s[pp]
        if kinds[0] != 1 or kinds[-1] != 2:
            raise RuntimeError("Malformed augmented streamtube boundary segments.")

        # Downstream nodes of every edge except the final sink edge are exactly
        # the ordered physical voxel nodes of the streamtube.
        physical_nodes = dn_s[pp[:-1]].astype(np.int64, copy=True)
        segment_lengths = len_s[pp]
        segment_dphi = np.maximum(dphi_s[pp], 1e-300)
        L_raw = float(np.sum(segment_lengths))
        sum_l2_over_dphi = float(np.sum(segment_lengths * segment_lengths / segment_dphi))

        st = {
            "nodes": physical_nodes,
            "I_gamma": float(gammas[k]),
            "L_gamma_raw": L_raw,
            "L_gamma": L_raw,
            "DeltaPhi_gamma": DeltaPhi,
            "_sum_l2_over_dphi": sum_l2_over_dphi,
        }

        if store_legacy_path_arrays:
            internal_mask = kinds == 0
            internal_edges = edge_s[pp][internal_mask]
            internal_dphi = dphi_s[pp][internal_mask]
            st.update({
                "edges": internal_edges.copy(),
                "segment_lengths_raw": segment_lengths.copy(),
                "segment_dphi_raw": segment_dphi.copy(),
                "segment_kind": kinds.copy(),
                "dphi_edges": internal_dphi.copy(),
                "l_edges": np.full(len(internal_edges), l_internal_default, dtype=float),
            })
        append(st)
    return streamtubes


def decompose_streamtubes(
    net: Dict[str, Any],
    sol: Dict[str, Any],
    transport: Dict[str, Any],
    config: BergPNConfig,
    method: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fast conservative streamtube decomposition of the solved current DAG.

    Parameters
    ----------
    method : {"greedy_fast", "topological"} or None
        None uses config.decomposition_method.

        greedy_fast
            Same greedy rule as the legacy voxel implementation: begin with
            the largest remaining source current, then at every physical node
            take the largest remaining outgoing edge and strip the path's
            bottleneck.  The expensive repeated source scan is replaced by a
            max-heap and the path loop is Numba-compiled when available.  Since
            all retained edges have strictly positive potential drop, the graph
            is a DAG and no visited-set cycle check is required.

        topological
            No repeated source-to-sink path search.  Nodes are visited once in
            decreasing potential order and incoming current packets are locally
            paired with outgoing edge capacities.  This is a deterministic
            local flow decomposition; it conserves the solved DAG flow to the
            configured threshold but generally produces a different valid set
            of individual streamtubes than greedy_fast.
    """
    t0 = time.perf_counter()
    method = config.decomposition_method if method is None else method
    if method not in {"greedy_fast", "topological"}:
        raise ValueError("method must be 'greedy_fast' or 'topological'")

    dag = _prepare_augmented_flow_dag(net, sol, transport, config)
    ptr, dn, flow = dag["ptr"], dag["dn"], dag["flow"]
    S, T = dag["S"], dag["T"]
    thr = dag["I_threshold"]
    use_numba = bool(config.use_numba_decomposition and HAS_NUMBA)

    if method == "greedy_fast":
        if use_numba:
            flat, offsets, gammas, n_stuck, stuck_removed, source_residual = \
                _greedy_fast_core_numba(ptr, dn, flow, thr, S, T, dag["n_aug"])
            engine = "numba"
        else:
            flat, offsets, gammas, n_stuck, stuck_removed, source_residual = \
                _greedy_fast_core_python(ptr, dn, flow, thr, S, T)
            engine = "python_heap"

        extra = {
            "n_stuck": int(n_stuck),
            "stuck_current_removed": float(stuck_removed),
            "remaining_source_current": float(source_residual),
        }
    else:
        # Strict dphi>0 makes descending potential a valid topological ordering.
        topo_nodes = np.argsort(-np.asarray(sol["phi"]), kind="mergesort").astype(np.int64)
        if use_numba:
            parent, edge_pos, term_rec, gammas, source_injected, dropped, unmatched = \
                _topological_core_numba(ptr, dn, flow, thr, S, T, topo_nodes)
            flat, offsets = _chains_to_flat_numba(parent, edge_pos, term_rec)
            engine = "numba"
        else:
            flat, offsets, gammas, source_injected, dropped, unmatched = \
                _topological_core_python(ptr, dn, flow, thr, S, T, topo_nodes)
            engine = "python"
        extra = {
            "n_stuck": 0,
            "stuck_current_removed": 0.0,
            "remaining_source_current": 0.0,
            "source_current_injected": float(source_injected),
            "threshold_current_discarded": float(dropped),
            "unmatched_internal_current": float(unmatched),
        }

    streamtubes = _streamtubes_from_augmented_paths(
        flat, offsets, gammas, dag, net, sol,
        store_legacy_path_arrays=bool(config.store_legacy_path_arrays),
    )

    I_decomp = float(np.sum(gammas)) if len(gammas) else 0.0
    I_direct = float(transport["I_total"])
    info = {
        "method": method,
        "engine": engine,
        "current_threshold": float(thr),
        "n_paths": int(len(streamtubes)),
        "n_pruned_unreachable_edges": int(dag["n_pruned"]),
        "I_total_direct": I_direct,
        "I_decomp": I_decomp,
        "current_loss_rel": float(abs(I_direct - I_decomp) / max(abs(I_direct), 1e-300)),
        "elapsed_s": float(time.perf_counter() - t0),
        **extra,
    }

    if config.verbose:
        print("\n" + "=" * 100)
        print(f"STREAMTUBE DECOMPOSITION -- {method.upper()} ({engine})")
        print("=" * 100)
        print(f"streamtubes                    : {len(streamtubes):,}")
        print(f"unreachable directed edges cut : {dag['n_pruned']:,}")
        if method == "greedy_fast":
            print(f"stuck traces                   : {info['n_stuck']:,}")
        else:
            print(f"threshold current discarded    : {_fmt(info['threshold_current_discarded'])}")
            print(f"unmatched internal current     : {_fmt(info['unmatched_internal_current'])}")
        print(f"I_total direct                 : {_fmt(I_direct)}")
        print(f"I_decomp                       : {_fmt(I_decomp)}")
        print(f"relative current loss          : {_fmt(info['current_loss_rel'])}")
        print(f"elapsed                        : {info['elapsed_s']:.3f} s")

    return streamtubes, info


# =============================================================================
# 5. Node-volume -> streamtube current split
# =============================================================================

def assign_node_volume(
    net: Dict[str, Any],
    streamtubes: List[Dict[str, Any]],
    config: BergPNConfig,
) -> Dict[str, Any]:
    """Attribute a share of each visited voxel's volume to every streamtube.

    Berg's global conductance reduction factor is a VOLUME-weighted average
    of a local quantity (Eq. 5: integral over Omega of iota^2, divided by
    Omega). To compute that average discretely from a set of streamtubes
    rather than a continuum field, each streamtube needs its own effective
    volume V_gamma -- the portion of pore space "responsible for" carrying
    that streamtube's current.

    Because several streamtubes can pass through the same voxel, a voxel's
    volume has to be split among them. Under config.current_split_mode ==
    "flux_weighted" (the only mode implemented), a voxel's volume is
    divided among the streamtubes passing through it in proportion to how
    much current each one contributes there: a streamtube carrying twice
    the current of another through the same voxel is attributed twice the
    share of that voxel's volume. Formally, if node i is visited by
    streamtubes with currents I_1, I_2, ... summing to I_total_i, then
    streamtube k receives (I_k / I_total_i) * node_volume[i] from that
    voxel; summing over every voxel a streamtube visits gives its V_gamma.

    Parameters
    ----------
    net : dict
        Output of build_network (uses node_volume).
    streamtubes : list of dict
        Output of decompose_streamtubes. Each entry is mutated in place to
        add "V_gamma" (total attributed volume), "volume_nodes" (the
        unique node ids visited, in path order) and
        "node_volume_shares" (this streamtube's volume share at each of
        those nodes).
    config : BergPNConfig
        Uses current_split_mode (validated to be "flux_weighted"),
        verbose.

    Returns
    -------
    info : dict
        Omega_c (sum of all V_gamma -- the discrete equivalent of Berg's
        "conducting" pore volume), Omega_c_from_nodes (sum of node_volume
        over every voxel touched by at least one streamtube, as an
        independent cross-check that should closely match Omega_c),
        n_nodes_touched, elapsed_s.
    """
    if config.current_split_mode != "flux_weighted":  # pragma: no cover - guarded above
        raise ValueError(config.current_split_mode)

    t0 = time.perf_counter()
    node_volume = net["node_volume"]
    n_nodes = len(node_volume)
    current_through_node = np.zeros(n_nodes, dtype=float)

    for st in streamtubes:
        for i in np.unique(st["nodes"]):
            current_through_node[i] += st["I_gamma"]

    touched = current_through_node > 0
    Omega_c = 0.0
    for st in streamtubes:
        path_nodes = st["nodes"]
        _, first_idx = np.unique(path_nodes, return_index=True)
        nodes = path_nodes[np.sort(first_idx)]
        denom = current_through_node[nodes]
        shares = (st["I_gamma"] / denom) * node_volume[nodes]
        st["V_gamma"] = float(np.sum(shares))
        st["volume_nodes"] = nodes
        st["node_volume_shares"] = shares
        Omega_c += st["V_gamma"]

    Omega_nodes = float(np.sum(node_volume[touched]))
    info = {
        "Omega_c": float(Omega_c),
        "Omega_c_from_nodes": Omega_nodes,
        "n_nodes_touched": int(np.count_nonzero(touched)),
        "elapsed_s": float(time.perf_counter() - t0),
    }

    if config.verbose:
        print("\n" + "=" * 100)
        print("NODE VOLUME -> STREAMTUBE CURRENT SPLIT")
        print("=" * 100)
        print(f"nodes touched                   : {info['n_nodes_touched']:,} / {n_nodes:,}")
        print(f"sum V_gamma                    : {_fmt(Omega_c)}")
        print(f"sum touched node volume        : {_fmt(Omega_nodes)}")

    return info


# =============================================================================
# 6. Berg quantities: tortuosity, constriction, conductance reduction factor
# =============================================================================

def compute_berg_quantities(
    net: Dict[str, Any],
    sol: Dict[str, Any],
    transport: Dict[str, Any],
    streamtubes: List[Dict[str, Any]],
    config: BergPNConfig,
) -> Dict[str, Any]:
    """Compute Berg quantities with vectorised streamtube scalar arithmetic.

    Numerically this is the same calculation as the legacy routine.  Fast
    decomposition stores sum(l_j^2/dphi_j) while building each path, so C_gamma
    does not need a second traversal over every segment.  Legacy streamtubes
    lacking that cached scalar are still supported.
    """
    L_sample = float(transport["L_sample"])
    V_domain = float(transport["V_domain"])
    Omega = float(net["phase_voxel_volume"])
    n = len(streamtubes)
    if n == 0:
        raise RuntimeError("No streamtubes available for Berg averaging.")

    Vg = np.fromiter((float(st["V_gamma"]) for st in streamtubes), dtype=float, count=n)
    Ig = np.fromiter((float(st["I_gamma"]) for st in streamtubes), dtype=float, count=n)
    Lg = np.fromiter((float(st["L_gamma_raw"]) for st in streamtubes), dtype=float, count=n)
    dV = np.fromiter((float(st["DeltaPhi_gamma"]) for st in streamtubes), dtype=float, count=n)

    shape_sum = np.empty(n, dtype=float)
    for k, st in enumerate(streamtubes):
        if "_sum_l2_over_dphi" in st:
            shape_sum[k] = float(st["_sum_l2_over_dphi"])
        else:
            lengths = np.asarray(st["segment_lengths_raw"], dtype=float)
            dphi = np.asarray(st["segment_dphi_raw"], dtype=float)
            shape_sum[k] = float(np.sum(lengths * lengths / dphi))

    tau_sq = (L_sample / Lg) ** 2
    Cg = dV * shape_sum / (Lg ** 2)

    for k, st in enumerate(streamtubes):
        st["tau_sq_gamma_raw"] = float(tau_sq[k])
        st["C_gamma"] = float(Cg[k])
        st["user_tortuosity_gamma_raw"] = float(Lg[k] / L_sample)

    Omega_c = float(np.sum(Vg))
    I_decomp = float(np.sum(Ig))
    tau_sq_c = float(np.sum(Vg * tau_sq) / Omega_c)
    C_c = float(np.sum(Ig * Cg) / I_decomp)
    phi = Omega / V_domain
    phi_c = Omega_c / V_domain
    one_over_F = float(transport["one_over_F"])
    C_user = 1.0 / C_c

    T_inferred_phi_c = np.sqrt(phi_c * C_user / one_over_F)
    T_inferred_phi = np.sqrt(phi * C_user / one_over_F)

    out = {
        "n_paths": n,
        "Omega": float(Omega),
        "Omega_c": Omega_c,
        "phi": float(phi),
        "phi_c": float(phi_c),
        "I_decomp": I_decomp,
        "I_total_direct": float(transport["I_total"]),
        "current_loss_rel": float(abs(I_decomp - transport["I_total"]) / max(abs(transport["I_total"]), 1e-300)),
        "tau_sq_c": tau_sq_c,
        "user_tortuosity": float(np.sqrt(1.0 / tau_sq_c)),
        "C_c": C_c,
        "one_over_C_c": float(C_user),
        "one_over_F_direct": one_over_F,
        "F_direct": float(transport["F"]),
        "user_tortuosity_inferred_from_C_F_phi_c": float(T_inferred_phi_c),
        "user_tortuosity_inferred_from_C_F_phi": float(T_inferred_phi),
        "streamtubes": streamtubes,
    }

    if config.verbose:
        print("\n" + "=" * 100)
        print("BERG QUANTITIES -- TORTUOSITY, CONSTRICTION, CONDUCTANCE REDUCTION")
        print("=" * 100)
        print(f"tau_sq_c (Berg <=1 convention) : {_fmt(tau_sq_c)}")
        print(f"user tortuosity sqrt(1/tau²)   : {_fmt(out['user_tortuosity'])}")
        print(f"C_c                            : {_fmt(C_c)}")
        print(f"1/C_c                          : {_fmt(C_user)}")
        print(f"1/F direct                     : {_fmt(one_over_F)}")
        print(f"phi / phi_c                    : {_fmt(phi)} / {_fmt(phi_c)}")
        print(f"T inferred from C,1/F,phi_c    : {_fmt(T_inferred_phi_c)}")
        print(f"T inferred from C,1/F,phi      : {_fmt(T_inferred_phi)}")

    return out


# =============================================================================
# 7. Chord correction (tortuosity only)
# =============================================================================

def _runs_from_path_nodes(coords: np.ndarray, nodes: np.ndarray) -> List[Dict[str, Any]]:
    """Split an ordered 6-neighbour voxel path into straight directional runs.

    A streamtube's node path is a sequence of unit face-to-face steps on
    the voxel grid, each step pointing along +/-x, +/-y, or +/-z. This
    groups consecutive steps that point the same direction into a single
    "run" -- a straight segment of the Manhattan-style path -- which is
    the basic unit chord_correct_streamtube_length operates on: replacing
    the sharp corner between two runs with a smooth chord.

    Returns a list of dicts, each with "direction" (unit step vector),
    "n" (number of voxel steps in this run), and "step_start"/"step_end"
    (index range into the path's step array, step_end exclusive).
    """
    if len(nodes) < 2:
        return []
    xyz = coords[nodes].astype(float)
    steps = np.diff(xyz, axis=0)
    manhattan = np.sum(np.abs(steps), axis=1)
    if not np.allclose(manhattan, 1.0):
        raise ValueError("Path contains a non-face-neighbour physical step.")

    runs = []
    start = 0
    for k in range(1, len(steps) + 1):
        if k == len(steps) or not np.array_equal(steps[k], steps[start]):
            runs.append({
                "direction": steps[start].copy(),
                "n": int(k - start),
                "step_start": int(start),
                "step_end": int(k),
            })
            start = k
    return runs


def _segment_inside_phase(mask: np.ndarray, A: np.ndarray, B: np.ndarray, samples_per_dx: int = 24) -> bool:
    """Check (by dense sampling) whether the straight segment A->B stays inside the conducting phase.

    A and B are given in voxel-index coordinates (not physical units).
    The segment is sampled at approximately samples_per_dx points per unit
    of index distance; each sample point is rounded to the nearest voxel
    and that voxel is checked against the phase mask. This is a
    conservative, easy-to-reason-about check rather than an exact
    supercover/line-traversal algorithm -- appropriate for a diagnostic
    correction where false rejections (treating a valid chord as invalid)
    are much less costly than false acceptances (letting a chord cut
    through solid matrix).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    length = float(np.linalg.norm(B - A))
    ns = max(2, int(np.ceil(length * samples_per_dx)) + 1)
    t = np.linspace(0.0, 1.0, ns)
    pts = A[None, :] + t[:, None] * (B - A)[None, :]
    ijk = np.floor(pts + 0.5).astype(int)

    valid = np.ones(len(ijk), dtype=bool)
    for ax in range(3):
        valid &= (ijk[:, ax] >= 0) & (ijk[:, ax] < mask.shape[ax])
    if not np.all(valid):
        return False
    ijk = ijk[valid]
    return bool(np.all(mask[ijk[:, 0], ijk[:, 1], ijk[:, 2]]))


def chord_correct_streamtube_length(
    net: Dict[str, Any],
    st: Dict[str, Any],
    config: BergPNConfig,
) -> Dict[str, Any]:
    """Replace local right-angle staircasing in one streamtube's path with straight chords.

    A path on a 6-connected voxel grid can only move axis-aligned one
    voxel at a time, so even a physically fairly straight channel produces
    a "staircase" of right-angle turns if it isn't perfectly aligned with
    the grid. This systematically inflates the raw graph path length (and
    hence tau_gamma) relative to the true pore-scale geometry. This
    function corrects for that by locally replacing the two straight runs
    on either side of a bend with a single straight chord between them,
    within a limited influence radius (config.bend_radius_vox), so that
    only genuinely local staircasing is removed -- not large-scale
    detours the current actually has to take around obstacles.

    Bend allocation
    -----------------
    For a bend between run j (incoming) and run j+1 (outgoing), each run
    contributes up to config.bend_radius_vox voxels of length to be
    "rounded off" at that bend, but never more than half of a run's own
    length if that run is shared between two bends (so two adjacent bends
    on a short run cannot double-claim the same voxels). The first and
    last runs in the path only have one adjacent bend each, so they may
    contribute up to the full config.bend_radius_vox (or the whole run,
    if shorter) to that single bend.

    For each bend, if the incoming run contributes length a and the
    outgoing run contributes length b, the raw Manhattan length at that
    bend (a+b)*dx is replaced by the Euclidean chord length between the
    two points a*dx and b*dx away from the bend along each run's
    direction. If config.require_chord_inside_phase is True, this
    replacement is only accepted when the straight chord stays entirely
    within the conducting phase (checked via _segment_inside_phase);
    otherwise that bend is left uncorrected and contributes its raw
    length unchanged.

    Only the INTERNAL voxel-to-voxel path is ever modified here -- the two
    physical half-cell boundary segments (dx/2 each) at the very start and
    end of the streamtube are never touched, since they are a real
    physical distance to an external plane, not a voxel-grid artefact.

    Parameters
    ----------
    net : dict
        Output of build_network (uses coords, mask, dx).
    st : dict
        One streamtube entry from decompose_streamtubes (uses nodes,
        L_gamma_raw).
    config : BergPNConfig
        Uses bend_radius_vox, require_chord_inside_phase,
        chord_samples_per_dx.

    Returns
    -------
    result : dict
        L_gamma_chord (corrected total path length), length_removed
        (total length subtracted across all accepted bends), n_bends
        (total bends in this path), n_bends_accepted/n_bends_rejected,
        bend_details (per-bend record: position, allocation, raw/chord
        local length, whether accepted, length removed -- useful for
        inspecting individual corrections).
    """
    R = float(config.bend_radius_vox)
    if R < 0:
        raise ValueError("bend_radius_vox must be non-negative.")

    nodes = st["nodes"]
    runs = _runs_from_path_nodes(net["coords"], nodes)
    raw = float(st["L_gamma_raw"])
    if len(runs) <= 1 or R == 0:
        return {
            "L_gamma_chord": raw,
            "length_removed": 0.0,
            "n_bends": max(0, len(runs) - 1),
            "n_bends_accepted": 0,
            "n_bends_rejected": 0,
            "bend_details": [],
        }

    nr = len(runs)
    left_share = np.zeros(nr, dtype=float)
    right_share = np.zeros(nr, dtype=float)

    for j, run in enumerate(runs):
        n = float(run["n"])
        if j == 0:
            right_share[j] = min(R, n)
        elif j == nr - 1:
            left_share[j] = min(R, n)
        else:
            s = min(R, 0.5 * n)
            left_share[j] = s
            right_share[j] = s

    coords = net["coords"]
    dx = float(net["dx"])
    total_removed = 0.0
    accepted = 0
    rejected = 0
    details = []

    cumulative_steps = 0
    for j in range(nr - 1):
        cumulative_steps += runs[j]["n"]
        bend_node = int(nodes[cumulative_steps])
        P = coords[bend_node].astype(float)
        d1 = runs[j]["direction"].astype(float)
        d2 = runs[j + 1]["direction"].astype(float)
        a = float(right_share[j])
        b = float(left_share[j + 1])

        if a <= 0 or b <= 0:
            continue

        A = P - a * d1
        B = P + b * d2
        raw_local = (a + b) * dx
        chord_local = float(np.linalg.norm(B - A) * dx)
        delta = max(0.0, raw_local - chord_local)

        phase_ok = True
        if config.require_chord_inside_phase:
            phase_ok = _segment_inside_phase(net["mask"], A, B, samples_per_dx=config.chord_samples_per_dx)

        if phase_ok and delta > 0:
            total_removed += delta
            accepted += 1
        else:
            delta = 0.0
            rejected += 1

        details.append({
            "bend_index": j,
            "bend_node": bend_node,
            "a_vox": a,
            "b_vox": b,
            "raw_local_length": raw_local,
            "chord_local_length": chord_local,
            "accepted": bool(phase_ok and raw_local > chord_local),
            "length_removed": delta,
        })

    corrected = raw - total_removed
    return {
        "L_gamma_chord": float(corrected),
        "length_removed": float(total_removed),
        "n_bends": int(nr - 1),
        "n_bends_accepted": int(accepted),
        "n_bends_rejected": int(rejected),
        "bend_details": details,
    }


def chord_correct_tortuosity(
    net: Dict[str, Any],
    transport: Dict[str, Any],
    berg: Dict[str, Any],
    config: BergPNConfig,
) -> Dict[str, Any]:
    """Recompute tau_c using chord-corrected path lengths, leaving C_c untouched.

    This applies chord_correct_streamtube_length to every streamtube and
    recomputes the volume-weighted tau_sq_c exactly as in
    compute_berg_quantities, but using each streamtube's corrected length
    L_gamma_chord in place of L_gamma_raw. C_c is deliberately NOT
    recomputed here -- it is copied through unchanged from berg -- so
    comparing this function's tortuosity against the raw one isolates
    exactly how much of the raw tortuosity was caused by voxel-grid
    staircasing versus genuine pore-scale path length, while keeping every
    other quantity (constriction, porosity, F) fixed as a control.

    Parameters
    ----------
    net : dict
        Output of build_network.
    transport : dict
        Output of effective_transport (uses L_sample).
    berg : dict
        Output of compute_berg_quantities (uses streamtubes, Omega_c, and
        the unchanged C_c/one_over_F/user_tortuosity for comparison).
    config : BergPNConfig
        Uses bend_radius_vox, require_chord_inside_phase,
        chord_samples_per_dx, verbose.

    Returns
    -------
    chord : dict
        bend_radius_vox, require_chord_inside_phase, tau_sq_c_chord,
        user_tortuosity_chord, user_tortuosity_raw (copied from berg for
        convenience), C_c_unchanged/one_over_C_c_unchanged/
        one_over_F_direct (copied through unchanged, see above),
        user_tortuosity_inferred_from_C_F_phi_c/_phi (copied through
        unchanged), total_bends/accepted_bends/rejected_bends (summed
        across all streamtubes), n_paths_corrected_below_L_sample (a
        sanity flag: number of streamtubes whose corrected length came
        out shorter than the macroscopic sample length, which should
        essentially never happen for a physically sensible correction),
        volume_weighted_length_removed, streamtubes (same list, now with
        per-streamtube L_gamma_chord/tau_sq_gamma_chord/
        user_tortuosity_gamma_chord/chord_info fields added).
    """
    streamtubes = berg["streamtubes"]
    L_sample = float(transport["L_sample"])
    Omega_c = float(berg["Omega_c"])

    tau_num = 0.0
    removed_weighted = 0.0
    total_bends = accepted = rejected = 0
    n_too_short = 0

    for st in streamtubes:
        corr = chord_correct_streamtube_length(net, st, config)
        Lc = float(corr["L_gamma_chord"])
        if Lc + 1e-12 < L_sample:
            n_too_short += 1
        tau_sq = (L_sample / Lc)**2
        st["L_gamma_chord"] = Lc
        st["tau_sq_gamma_chord"] = float(tau_sq)
        st["user_tortuosity_gamma_chord"] = float(Lc / L_sample)
        st["chord_info"] = corr

        Vg = float(st["V_gamma"])
        tau_num += Vg * tau_sq
        removed_weighted += Vg * corr["length_removed"]
        total_bends += corr["n_bends"]
        accepted += corr["n_bends_accepted"]
        rejected += corr["n_bends_rejected"]

    tau_sq_c = tau_num / Omega_c
    T_chord = np.sqrt(1.0 / tau_sq_c)
    out = {
        "bend_radius_vox": float(config.bend_radius_vox),
        "require_chord_inside_phase": bool(config.require_chord_inside_phase),
        "tau_sq_c_chord": float(tau_sq_c),
        "user_tortuosity_chord": float(T_chord),
        "user_tortuosity_raw": float(berg["user_tortuosity"]),
        "C_c_unchanged": float(berg["C_c"]),
        "one_over_C_c_unchanged": float(berg["one_over_C_c"]),
        "one_over_F_direct": float(berg["one_over_F_direct"]),
        "user_tortuosity_inferred_from_C_F_phi_c": float(berg["user_tortuosity_inferred_from_C_F_phi_c"]),
        "user_tortuosity_inferred_from_C_F_phi": float(berg["user_tortuosity_inferred_from_C_F_phi"]),
        "total_bends": int(total_bends),
        "accepted_bends": int(accepted),
        "rejected_bends": int(rejected),
        "n_paths_corrected_below_L_sample": int(n_too_short),
        "volume_weighted_length_removed": float(removed_weighted / Omega_c),
        "streamtubes": streamtubes,
    }

    if config.verbose:
        print("\n" + "=" * 100)
        print("CHORD-CORRECTED TORTUOSITY ONLY -- C_c UNCHANGED")
        print("=" * 100)
        print(f"bend influence radius R        : {_fmt(config.bend_radius_vox)} voxels")
        print(f"raw user tortuosity            : {_fmt(out['user_tortuosity_raw'])}")
        print(f"chord user tortuosity          : {_fmt(T_chord)}")
        print(f"C_c unchanged                  : {_fmt(out['C_c_unchanged'])}")
        print(f"1/C_c unchanged                : {_fmt(out['one_over_C_c_unchanged'])}")
        print(f"1/F direct                     : {_fmt(out['one_over_F_direct'])}")
        print(f"T inferred C+1/F using phi_c   : {_fmt(out['user_tortuosity_inferred_from_C_F_phi_c'])}")
        print(f"T inferred C+1/F using phi     : {_fmt(out['user_tortuosity_inferred_from_C_F_phi'])}")
        print(f"bends total/accepted/rejected  : {total_bends:,} / {accepted:,} / {rejected:,}")
        print(f"mean V-weighted length removed : {_fmt(out['volume_weighted_length_removed'])}")
        print(f"paths with L_chord < L_sample  : {n_too_short:,}")

    return out


# =============================================================================
# 8. Orchestrator
# =============================================================================

def run_berg_pipeline(
    segmented: np.ndarray,
    config: Optional[BergPNConfig] = None,
    warm_start_field: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run the full fast Berg voxel pipeline.

    The KCL/PyAMG and transport stages are unchanged.  Stage 4 dispatches to
    config.decomposition_method ("greedy_fast" or "topological"), node-volume
    attribution uses the chunked vectorised routine, and chord-corrected mode
    uses compute_chord_tortuosity_only_fast.
    """
    if config is None:
        config = BergPNConfig()

    net = build_network(segmented, config)
    sol = solve_kcl(net, config, warm_start_field=warm_start_field)
    transport = effective_transport(net, sol, config)
    streamtubes, decomp = decompose_streamtubes(net, sol, transport, config)
    split = assign_node_volume_to_streamtubes_fast(
        net, streamtubes, chunk=config.volume_chunk, verbose=config.verbose
    )
    berg = compute_berg_quantities(net, sol, transport, streamtubes, config)

    result = {
        "net": net,
        "sol": sol,
        "transport": transport,
        "decomposition": decomp,
        "split": split,
        "berg": berg,
    }

    if config.tortuosity_mode == "chord_corrected":
        chord = compute_chord_tortuosity_only_fast(
            net, transport, berg,
            bend_radius_vox=config.bend_radius_vox,
            require_chord_inside_phase=config.require_chord_inside_phase,
            chunk=config.chord_chunk,
            verbose=config.verbose,
        )
        result["chord"] = chord

    return result


# =============================================================================
# 9. Straight-channel unit test
# =============================================================================

def straight_cuboid_unit_test(
    N: int = 32,
    width: int = 10,
    config: Optional[BergPNConfig] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Validate the pipeline against Berg's exact idealised straight-tube result.

    Builds a width x width square channel of conducting phase running the
    full length of an N x N x N cube along config.axis (default axis 0),
    matching Berg's Section IV-A idealised porous medium: a single
    straight, constant-cross-section tube connecting two opposite faces of
    a cube. For this exact geometry, Berg's own algebra gives tau = 1 (the
    path is already perfectly straight and axis-aligned, so it should be
    entirely unaffected by chord correction too), C = 1 (no cross-section
    variation along the tube), and F = 1/phi with phi = (width/N)^2 (the
    fraction of the cube's cross-section occupied by the channel).

    Since this geometry has zero staircasing, chord correction should
    leave the tortuosity unchanged; comparing raw and chord-corrected
    tortuosity here is a good check that the chord-correction logic isn't
    introducing spurious corrections on a path that has no bends to
    correct.

    Parameters
    ----------
    N : int
        Cube side length in voxels.
    width : int
        Side length of the square channel cross-section in voxels; must
        be <= N.
    config : BergPNConfig or None
        If provided, used as a base configuration (phase_id, axis, dx,
        sigma, tortuosity_mode etc. are taken from it); note phase_id will
        be forced to 1 to match the synthetic segmentation built here
        regardless of what was set in the passed-in config, since this
        test always builds its own single-phase volume.

    Returns
    -------
    result : dict
        Full output of run_berg_pipeline on the synthetic geometry.
    summary : dict
        expected_L_sample/computed_L_sample, expected_phi/
        computed_one_over_F, raw_user_tortuosity, chord_user_tortuosity
        (only meaningful if config.tortuosity_mode == "chord_corrected"),
        C_c -- a compact comparison against the known analytic answer.
    """
    if width > N:
        raise ValueError("width must be <= N")

    if config is None:
        config = BergPNConfig()
    # This test always builds and labels its own synthetic volume with
    # phase_id=1; force the config to match regardless of what the caller
    # passed in, so the test is self-consistent even if reused across a
    # sweep of otherwise-different configs.
    test_config = BergPNConfig(**{**config.__dict__, "phase_id": 1})

    axis = test_config.axis
    dx = test_config.dx
    seg = np.zeros((N, N, N), dtype=np.uint8)
    a = (N - width) // 2
    if axis == 0:
        seg[:, a:a + width, a:a + width] = 1
    elif axis == 1:
        seg[a:a + width, :, a:a + width] = 1
    else:
        seg[a:a + width, a:a + width, :] = 1

    res = run_berg_pipeline(seg, config=test_config, warm_start_field=None)

    expected_phi = float(seg.mean())
    summary = {
        "expected_L_sample": N * dx,
        "computed_L_sample": res["transport"]["L_sample"],
        "expected_phi": expected_phi,
        "computed_one_over_F": res["transport"]["one_over_F"],
        "raw_user_tortuosity": res["berg"]["user_tortuosity"],
        "chord_user_tortuosity": res["chord"]["user_tortuosity_chord"] if "chord" in res else None,
        "C_c": res["berg"]["C_c"],
    }
    if test_config.verbose:
        print("\n" + "=" * 100)
        print("STRAIGHT CUBOID UNIT-TEST SUMMARY")
        print("=" * 100)
        for k, v in summary.items():
            print(f"{k:32s}: {_fmt(v)}")
    return res, summary


"""
Drop-in fast replacements for the two slow stages in
berg_voxel_half_boundary_chord.py:

    assign_node_volume_to_streamtubes  ->  assign_node_volume_to_streamtubes_fast
    compute_chord_tortuosity_only      ->  compute_chord_tortuosity_only_fast

Everything is vectorised across ALL streamtubes at once. The per-bend line
sampling is gone entirely (see chord_correct_all_fast docstring).
"""


# =============================================================================
# 1. Node-volume -> streamtube split, vectorised
# =============================================================================
def assign_node_volume_to_streamtubes_fast(net, streamtubes, chunk=20000, verbose=True):
    """Chunked vectorised node-volume split for DAG streamtubes.

    A valid directed streamtube cannot visit the same node twice because every
    retained edge strictly decreases potential, so the legacy per-path
    np.unique calls are unnecessary.  Chunking bounds temporary memory while
    preserving exactly the same flux-weighted volume split.
    """
    t0 = time.perf_counter()
    node_volume = np.asarray(net["node_volume"], dtype=float)
    n_nodes = len(node_volume)
    npaths = len(streamtubes)
    if npaths == 0:
        return {"Omega_c": 0.0, "Omega_c_from_nodes": 0.0,
                "n_nodes_touched": 0, "elapsed_s": 0.0}
    if not chunk or chunk <= 0:
        chunk = npaths

    current_through_node = np.zeros(n_nodes, dtype=float)

    # Pass 1: total decomposed current through every voxel.
    for s0 in range(0, npaths, chunk):
        ss = streamtubes[s0:s0 + chunk]
        lens = np.fromiter((len(st["nodes"]) for st in ss), np.int64, len(ss))
        if not len(lens) or int(lens.sum()) == 0:
            continue
        flat = np.concatenate([st["nodes"] for st in ss])
        Ig = np.fromiter((st["I_gamma"] for st in ss), float, len(ss))
        weights = np.repeat(Ig, lens)
        current_through_node += np.bincount(flat, weights=weights, minlength=n_nodes)

    # Pass 2: assign each voxel's volume to tubes in proportion to I_gamma.
    Vg_all = np.zeros(npaths, dtype=float)
    for s0 in range(0, npaths, chunk):
        ss = streamtubes[s0:s0 + chunk]
        lens = np.fromiter((len(st["nodes"]) for st in ss), np.int64, len(ss))
        if not len(lens) or int(lens.sum()) == 0:
            continue
        flat = np.concatenate([st["nodes"] for st in ss])
        Ig = np.fromiter((st["I_gamma"] for st in ss), float, len(ss))
        Ig_flat = np.repeat(Ig, lens)
        denom = current_through_node[flat]
        shares_flat = (Ig_flat / denom) * node_volume[flat]
        off = np.empty(len(ss) + 1, dtype=np.int64)
        off[0] = 0
        np.cumsum(lens, out=off[1:])
        Vg = np.add.reduceat(shares_flat, off[:-1])
        Vg_all[s0:s0 + len(ss)] = Vg
        for k, st in enumerate(ss):
            a, b = int(off[k]), int(off[k + 1])
            st["V_gamma"] = float(Vg[k])
            st["volume_nodes"] = st["nodes"]
            # copy keeps each streamtube independent and avoids retaining a
            # whole temporary chunk through a tiny view.
            st["node_volume_shares"] = shares_flat[a:b].copy()

    touched = current_through_node > 0
    info = {
        "Omega_c": float(Vg_all.sum()),
        "Omega_c_from_nodes": float(node_volume[touched].sum()),
        "n_nodes_touched": int(np.count_nonzero(touched)),
        "elapsed_s": float(time.perf_counter() - t0),
    }
    if verbose:
        print("\n" + "=" * 100)
        print("NODE VOLUME -> STREAMTUBE CURRENT SPLIT  (fast/chunked)")
        print("=" * 100)
        print(f"nodes touched                  : {info['n_nodes_touched']:,} / {n_nodes:,}")
        print(f"sum V_gamma                    : {info['Omega_c']:.12g}")
        print(f"sum touched node volume        : {info['Omega_c_from_nodes']:.12g}")
        print(f"elapsed                        : {info['elapsed_s']:.2f} s")
    return info


def _integral_mask_3d(mask):
    """3-D summed-area table, built once per full chord-correction call."""
    S = np.asarray(mask, dtype=np.int32).cumsum(0).cumsum(1).cumsum(2)
    return np.pad(S, ((1, 0), (1, 0), (1, 0)))


def _chord_correct_chunk_fast(net, streamtubes, bend_radius_vox,
                              require_chord_inside_phase, sat=None):
    R = float(bend_radius_vox)
    dx = float(net["dx"])
    coords = net["coords"]
    npaths = len(streamtubes)
    if npaths == 0:
        z = np.empty(0, dtype=float)
        return dict(L_raw=z, L_chord=z.copy(), removed=z.copy(),
                    n_bends=0, n_accepted=0, n_rejected=0)

    lens = np.fromiter((len(st["nodes"]) for st in streamtubes), np.int64, npaths)
    L_raw = np.fromiter((st["L_gamma_raw"] for st in streamtubes), float, npaths)
    if int(lens.sum()) == 0 or R == 0:
        L_chord = L_raw.copy()
        for k, st in enumerate(streamtubes):
            st["L_gamma_chord"] = float(L_chord[k])
        return dict(L_raw=L_raw, L_chord=L_chord, removed=np.zeros(npaths),
                    n_bends=0, n_accepted=0, n_rejected=0)

    flat = np.concatenate([st["nodes"] for st in streamtubes])
    pid = np.repeat(np.arange(npaths, dtype=np.int64), lens)
    xyz = coords[flat].astype(np.int32, copy=False)

    if len(flat) < 2:
        L_chord = L_raw.copy()
        return dict(L_raw=L_raw, L_chord=L_chord, removed=np.zeros(npaths),
                    n_bends=0, n_accepted=0, n_rejected=0)

    step_all = xyz[1:] - xyz[:-1]
    same = pid[1:] == pid[:-1]
    if not np.any(same):
        L_chord = L_raw.copy()
        for k, st in enumerate(streamtubes):
            st["L_gamma_chord"] = float(L_chord[k])
        return dict(L_raw=L_raw, L_chord=L_chord, removed=np.zeros(npaths),
                    n_bends=0, n_accepted=0, n_rejected=0)

    step_pid = pid[1:][same]
    step = step_all[same]
    ax = np.argmax(np.abs(step), axis=1)
    sgn = step[np.arange(len(step)), ax] > 0
    code = (ax * 2 + sgn).astype(np.int8)

    newrun = np.empty(len(code), dtype=bool)
    newrun[0] = True
    if len(code) > 1:
        newrun[1:] = (code[1:] != code[:-1]) | (step_pid[1:] != step_pid[:-1])
    run_start = np.flatnonzero(newrun)
    run_len = np.diff(np.append(run_start, len(code))).astype(float)
    run_pid = step_pid[run_start]
    n_runs = len(run_start)

    is_first = np.empty(n_runs, dtype=bool)
    is_last = np.empty(n_runs, dtype=bool)
    is_first[0] = True
    if n_runs > 1:
        is_first[1:] = run_pid[1:] != run_pid[:-1]
    is_last[-1] = True
    if n_runs > 1:
        is_last[:-1] = run_pid[:-1] != run_pid[1:]

    half = np.minimum(R, 0.5 * run_len)
    full = np.minimum(R, run_len)
    left = np.where(is_first, 0.0, np.where(is_last, full, half))
    right = np.where(is_last, 0.0, np.where(is_first, full, half))

    if n_runs <= 1:
        bend = np.empty(0, dtype=np.int64)
    else:
        bend = np.flatnonzero((~is_last[:-1]) & (run_pid[:-1] == run_pid[1:]))

    if len(bend) == 0:
        L_chord = L_raw.copy()
        for k, st in enumerate(streamtubes):
            st["L_gamma_chord"] = float(L_chord[k])
        return dict(L_raw=L_raw, L_chord=L_chord, removed=np.zeros(npaths),
                    n_bends=0, n_accepted=0, n_rejected=0)

    a = right[bend]
    b = left[bend + 1]
    bend_pid = run_pid[bend]
    c1b, c2b = code[run_start[bend]], code[run_start[bend + 1]]
    dot = np.where((c1b // 2) == (c2b // 2), -1.0, 0.0)
    chord = np.sqrt(np.maximum(a * a + b * b + 2.0 * a * b * dot, 0.0))
    delta = np.maximum(a + b - chord, 0.0) * dx
    ok = (a > 0) & (b > 0) & (delta > 0)

    n_rej = 0
    if require_chord_inside_phase:
        if sat is None:
            sat = _integral_mask_3d(net["mask"])
        mshape = np.asarray(net["mask"].shape, dtype=np.int64)

        # compressed step index -> starting node index in flat
        step_to_node = np.flatnonzero(same)
        step_index_of_bend = run_start[bend + 1]
        node_at_bend = flat[step_to_node[step_index_of_bend]]
        P = coords[node_at_bend].astype(np.int64)

        d1 = np.zeros((len(bend), 3), dtype=np.int64)
        d2 = np.zeros((len(bend), 3), dtype=np.int64)
        c1 = code[run_start[bend]]
        c2 = code[run_start[bend + 1]]
        d1[np.arange(len(bend)), c1 // 2] = np.where(c1 % 2 == 1, 1, -1)
        d2[np.arange(len(bend)), c2 // 2] = np.where(c2 % 2 == 1, 1, -1)

        A = P - np.ceil(a)[:, None].astype(np.int64) * d1
        B = P + np.ceil(b)[:, None].astype(np.int64) * d2
        lo = np.minimum(A, B)
        hi = np.maximum(A, B) + 1
        inside = np.all((lo >= 0) & (hi <= mshape), axis=1)
        lo_c = np.maximum(lo, 0)
        hi_c = np.minimum(hi, mshape)

        i0, j0, k0 = lo_c[:, 0], lo_c[:, 1], lo_c[:, 2]
        i1, j1, k1 = hi_c[:, 0], hi_c[:, 1], hi_c[:, 2]
        vol = np.prod(hi_c - lo_c, axis=1)
        tot = (sat[i1, j1, k1] - sat[i0, j1, k1]
               - sat[i1, j0, k1] - sat[i1, j1, k0]
               + sat[i0, j0, k1] + sat[i0, j1, k0]
               + sat[i1, j0, k0] - sat[i0, j0, k0])
        phase_ok = inside & (tot == vol)
        n_rej = int(np.count_nonzero(ok & ~phase_ok))
        ok &= phase_ok

    removed = np.bincount(bend_pid[ok], weights=delta[ok], minlength=npaths)
    L_chord = L_raw - removed
    for k, st in enumerate(streamtubes):
        st["L_gamma_chord"] = float(L_chord[k])

    return dict(L_raw=L_raw, L_chord=L_chord, removed=removed,
                n_bends=int(len(bend)), n_accepted=int(np.count_nonzero(ok)),
                n_rejected=n_rej)


def chord_correct_all_fast(net, streamtubes, bend_radius_vox=3.0,
                           require_chord_inside_phase=True, chunk=20000,
                           verbose=True):
    """Vectorised chord correction with one cached 3-D phase integral image.

    This retains the fast conservative *bounding-box* phase test introduced in
    berg_voxel_final(3).py.  It is intentionally stricter than the original
    dense line sampler: a chord is accepted only if the entire axis-aligned box
    spanning its two arms is conducting.  Therefore it is fast and cannot admit
    a chord through solid, but can reject some chords the dense sampler accepted.
    """
    t0 = time.perf_counter()
    npaths = len(streamtubes)
    if not chunk or chunk <= 0:
        chunk = max(1, npaths)

    sat = _integral_mask_3d(net["mask"]) if require_chord_inside_phase else None
    Lr, Lc, rem = [], [], []
    nb = na = nr = 0
    for i in range(0, npaths, chunk):
        c = _chord_correct_chunk_fast(
            net, streamtubes[i:i + chunk], bend_radius_vox,
            require_chord_inside_phase, sat=sat,
        )
        Lr.append(c["L_raw"]); Lc.append(c["L_chord"]); rem.append(c["removed"])
        nb += c["n_bends"]; na += c["n_accepted"]; nr += c["n_rejected"]

    z = np.empty(0, dtype=float)
    out = {
        "L_raw": np.concatenate(Lr) if Lr else z,
        "L_chord": np.concatenate(Lc) if Lc else z.copy(),
        "removed": np.concatenate(rem) if rem else z.copy(),
        "n_bends": int(nb), "n_accepted": int(na), "n_rejected": int(nr),
        "elapsed_s": float(time.perf_counter() - t0),
    }
    if verbose:
        print(f"  chord correction: {nb:,} bends, {na:,} accepted, "
              f"{nr:,} rejected by phase box, {out['elapsed_s']:.2f} s")
    return out


def compute_chord_tortuosity_only_fast(net, transport, berg_raw,
                                       bend_radius_vox=3.0,
                                       require_chord_inside_phase=True,
                                       chunk=20000, verbose=True):
    streamtubes = berg_raw["streamtubes"]
    L_sample = float(transport["L_sample"])
    Omega_c = float(berg_raw["Omega_c"])

    cc = chord_correct_all_fast(
        net, streamtubes,
        bend_radius_vox=bend_radius_vox,
        require_chord_inside_phase=require_chord_inside_phase,
        chunk=chunk, verbose=verbose,
    )
    Lc = cc["L_chord"]
    Vg = np.fromiter((st["V_gamma"] for st in streamtubes), float, len(streamtubes))
    tau_sq = (L_sample / Lc) ** 2
    for k, st in enumerate(streamtubes):
        st["tau_sq_gamma_chord"] = float(tau_sq[k])
        st["user_tortuosity_gamma_chord"] = float(Lc[k] / L_sample)

    tau_sq_c = float(np.sum(Vg * tau_sq) / Omega_c)
    T_chord = float(np.sqrt(1.0 / tau_sq_c))

    out = {
        "method": "half_boundary_chord_tau_only_fast",
        "bend_radius_vox": float(bend_radius_vox),
        "tau_sq_c_chord": tau_sq_c,
        "user_tortuosity_chord": T_chord,
        "user_tortuosity_raw": float(berg_raw["user_tortuosity"]),
        "C_c_unchanged": float(berg_raw["C_c"]),
        "one_over_C_c_unchanged": float(berg_raw["one_over_C_c"]),
        "one_over_F_direct": float(berg_raw["one_over_F_direct"]),
        "user_tortuosity_inferred_from_C_F_phi_c":
            float(berg_raw["user_tortuosity_inferred_from_C_F_phi_c"]),
        "total_bends": cc["n_bends"],
        "accepted_bends": cc["n_accepted"],
        "rejected_bends": cc["n_rejected"],
        "n_paths_corrected_below_L_sample": int(np.count_nonzero(Lc < L_sample - 1e-12)),
        "volume_weighted_length_removed": float(np.sum(Vg * cc["removed"]) / Omega_c),
        "streamtubes": streamtubes,
    }
    if verbose:
        print("\n" + "=" * 100)
        print("CHORD-CORRECTED TORTUOSITY ONLY (fast) -- C_c UNCHANGED")
        print("=" * 100)
        print(f"bend influence radius R        : {bend_radius_vox}")
        print(f"raw  user tortuosity           : {out['user_tortuosity_raw']:.12g}")
        print(f"chord user tortuosity          : {T_chord:.12g}")
        print(f"C_c / 1/C_c   (unchanged)      : {out['C_c_unchanged']:.12g} / "
              f"{out['one_over_C_c_unchanged']:.12g}")
        print(f"T inferred from C,1/F,phi_c    : "
              f"{out['user_tortuosity_inferred_from_C_F_phi_c']:.12g}")
        print(f"mean V-weighted length removed : "
              f"{out['volume_weighted_length_removed']:.12g}")
        print(f"paths with L_chord < L_sample  : {out['n_paths_corrected_below_L_sample']:,}")
    return out

