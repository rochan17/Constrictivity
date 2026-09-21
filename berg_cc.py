import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
import networkx as nx

# Reuse the Numba-accelerated flow-decomposition cores from the voxel-network
# pipeline. They operate on generic CSR-style (ptr, dn, flow) augmented-DAG
# arrays with no voxel-specific assumptions, so the same compiled kernels
# work unmodified on the body/throat network built in this module.
from berg_voxel_fast import (
    HAS_NUMBA,
    _greedy_fast_core_python,
    _topological_core_python,
)
# Single source of truth for axis<->direction, shared with particulate_claude
# and berg_voxel_fast (0='x', 1='y', 2='z' everywhere in this codebase).
from particulate_claude import DIRECTION_TO_AXIS
if HAS_NUMBA:
    from berg_voxel_fast import (
        _greedy_fast_core_numba,
        _topological_core_numba,
        _chains_to_flat_numba,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Function 1 — build_network_arrays
# ──────────────────────────────────────────────────────────────────────────────

def build_network_arrays(network, area_method='vox_projection'):
    """
    Convert a PoreNetwork (from particulate.py) into flat numpy arrays
    ready for conductance calculation and linear solve.

    Requires that `calculate_network_effective_properties` has already been
    called on the network so that BodyThroatConnection objects exist on each
    body, and effective lengths are set on each throat.

    Parameters
    ----------
    network : PoreNetwork
        Fully initialised network with effective properties calculated.
    area_method : str
        Throat area method passed to PoreThroat.get_area().
        One of 'naive', 'naive_projected', 'vox_projection'.

    Returns
    -------
    arrays : dict with keys
        'body_ids'      – original body IDs in index order          (N_bodies,)
        'id_to_idx'     – dict mapping body_id → integer index
        'coords'        – body centroids (x, y, z) in voxels        (N_bodies, 3)
        'volumes'       – body volumes in voxels                     (N_bodies,)
        'is_surface'    – bool, True if body touches a domain face   (N_bodies,)
        'conns'         – [body1_idx, body2_idx] per throat          (N_throats, 2)
        'throat_ids'    – original throat IDs in index order         (N_throats,)
        'throat_area'   – throat cross-sectional area in voxels²     (N_throats,)
        'throat_volume' – throat effective volume in voxels³         (N_throats,)
        'L_throat'      – throat segment length in voxels            (N_throats,)
        'L_body1'       – body-1 effective segment length in voxels  (N_throats,)
        'L_body2'       – body-2 effective segment length in voxels  (N_throats,)
        'A_body1'       – body-1 effective segment area in voxels²   (N_throats,)
        'A_body2'       – body-2 effective segment area in voxels²   (N_throats,)
        'V_body1'       – body-1 effective segment volume in voxels³ (N_throats,)
        'V_body2'       – body-2 effective segment volume in voxels³ (N_throats,)
        'is_fallback'   – bool, True if throat used uniform-pipe fallback (N_throats,)
        'geometric_throat_area' – original geometric throat area     (N_throats,)
    """

    # ── 1. Build a stable body ordering and index map ──────────────────────
    body_ids  = sorted(network.bodies.keys())
    id_to_idx = {bid: i for i, bid in enumerate(body_ids)}
    N_bodies  = len(body_ids)

    coords     = np.zeros((N_bodies, 3), dtype=float)
    volumes    = np.zeros(N_bodies,      dtype=float)
    is_surface = np.zeros(N_bodies,      dtype=bool)

    for bid in body_ids:
        body            = network.bodies[bid]
        idx             = id_to_idx[bid]
        coords[idx]     = body.centroid
        volumes[idx]    = body.volume_voxels
        is_surface[idx] = body.is_surface

    # ── 2. Build per-throat arrays ─────────────────────────────────────────
    throat_ids = sorted(network.throats.keys())
    N_throats  = len(throat_ids)

    conns                = np.full((N_throats, 2), -1, dtype=int)
    throat_area          = np.zeros(N_throats, dtype=float)
    geometric_throat_area = np.zeros(N_throats, dtype=float)
    throat_volume        = np.zeros(N_throats, dtype=float)
    L_throat             = np.zeros(N_throats, dtype=float)
    L_body1              = np.zeros(N_throats, dtype=float)
    L_body2              = np.zeros(N_throats, dtype=float)
    A_body1              = np.zeros(N_throats, dtype=float)
    A_body2              = np.zeros(N_throats, dtype=float)
    is_fallback          = np.zeros(N_throats, dtype=bool)

    skipped = 0
    for tidx, tid in enumerate(throat_ids):
        throat = network.throats[tid]
        b1_id  = throat.body1_id
        b2_id  = throat.body2_id

        # ── Body index lookup ──────────────────────────────────────────────
        if b1_id not in id_to_idx or b2_id not in id_to_idx:
            skipped += 1
            continue

        body1 = network.bodies[b1_id]
        body2 = network.bodies[b2_id]

        # ── Connection lookup ──────────────────────────────────────────────
        conn1 = body1.connections_to_throats.get(tid)
        conn2 = body2.connections_to_throats.get(tid)

        if conn1 is None or conn2 is None:
            skipped += 1
            continue

        # ── Store body indices ─────────────────────────────────────────────
        conns[tidx, 0] = id_to_idx[b1_id]
        conns[tidx, 1] = id_to_idx[b2_id]

        # ── Throat-segment properties ──────────────────────────────────────
        # For fallback throats use A_uniform; for normal throats use geometric area.
        geometric_throat_area[tidx] = throat.geometric_throat_area
        if throat.is_fallback:
            throat_area[tidx] = throat.fallback_area
        else:
            throat_area[tidx] = throat.get_area(area_method)

        throat_volume[tidx] = throat.effective_volume
        L_throat[tidx]      = throat.effective_length_total
        is_fallback[tidx]   = throat.is_fallback

        # ── Body-segment properties ────────────────────────────────────────
        L_body1[tidx] = conn1.effective_distance
        A_body1[tidx] = conn1.effective_area
        L_body2[tidx] = conn2.effective_distance
        A_body2[tidx] = conn2.effective_area

    if skipped:
        print(f"[build_network_arrays] WARNING: {skipped} throat(s) skipped "
              f"(missing body or connection).")

    n_fallback = is_fallback.sum()
    if n_fallback:
        print(f"[build_network_arrays] {n_fallback} throat(s) used uniform-pipe fallback.")

    print(f"[build_network_arrays] {N_bodies} bodies, {N_throats} throats indexed.")

    return {
        # Body arrays
        'body_ids':             np.array(body_ids),
        'id_to_idx':            id_to_idx,
        'coords':               coords,
        'volumes':              volumes,
        'is_surface':           is_surface,
        # Throat arrays
        'throat_ids':           np.array(throat_ids),
        'conns':                conns,
        'throat_area':          throat_area,
        'geometric_throat_area': geometric_throat_area,
        'throat_volume':        throat_volume,
        'L_throat':             L_throat,
        'L_body1':              L_body1,
        'L_body2':              L_body2,
        'A_body1':              A_body1,
        'A_body2':              A_body2,
        'V_body1':              A_body1 * L_body1,
        'V_body2':              A_body2 * L_body2,
        'is_fallback':          is_fallback,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Function 2 — calculate_conductances
# ──────────────────────────────────────────────────────────────────────────────

def calculate_conductances(arrays, sigma=1.0, min_length=1e-8, verbose=True):
    """
    Calculate per-throat electrical conductances using Berg's 3-segment
    series-resistance model (Berg 2012, Eq. 23).

    Each throat is treated as three resistors in series:
        body-1 segment  |  throat channel  |  body-2 segment

    Resistance of each segment:  R = L / (sigma * A)
    Total conductance:            G = 1 / (R1 + R_throat + R2)

    Parameters
    ----------
    arrays : dict
        Output of build_network_arrays().
    sigma : float
        Electrical conductivity of the pore-filling electrolyte [S/m or
        dimensionless if working in voxel units].
    min_length : float
        Floor applied to all segment lengths to avoid division by zero.
    verbose : bool
        Print a summary table.

    Returns
    -------
    G : ndarray, shape (N_throats,)
        Conductance of each throat [S, or consistent units].
    conductance_data : dict
        Per-throat breakdown: R1, R_throat, R2, R_total, G.
    """

    L1       = np.maximum(arrays['L_body1'],  min_length)
    L_throat = np.maximum(arrays['L_throat'], min_length)
    L2       = np.maximum(arrays['L_body2'],  min_length)

    A1       = np.maximum(arrays['A_body1'],       min_length**2)
    A_throat = np.maximum(arrays['throat_area'],   min_length**2)
    A2       = np.maximum(arrays['A_body2'],       min_length**2)

    R1       = L1       / (sigma * A1)
    R_throat = L_throat / (sigma * A_throat)
    R2       = L2       / (sigma * A2)
    R_total  = R1 + R_throat + R2

    G = 1.0 / R_total

    conductance_data = {
        'G':        G,
        'R1':       R1,
        'R_throat': R_throat,
        'R2':       R2,
        'R_total':  R_total,
        'L1':       L1,
        'L_throat': L_throat,
        'L2':       L2,
        'A1':       A1,
        'A_throat': A_throat,
        'A2':       A2,
        'sigma':    sigma,
    }

    if verbose:
        N = len(G)
        print(f"\n{'='*60}")
        print(f"{'BERG CONDUCTANCE CALCULATION':^60}")
        print(f"{'='*60}")
        print(f"  Throats:                {N}")
        print(f"  Electrolyte sigma:      {sigma}")
        print(f"\n  Conductance [G]:")
        print(f"    min  = {G.min():.4g}")
        print(f"    max  = {G.max():.4g}")
        print(f"    mean = {G.mean():.4g}")
        frac1  = R1.mean()      / R_total.mean() * 100
        frac_t = R_throat.mean()/ R_total.mean() * 100
        frac2  = R2.mean()      / R_total.mean() * 100
        print(f"\n  Mean resistance fractions:")
        print(f"    Body-1  : {frac1:.1f}%")
        print(f"    Throat  : {frac_t:.1f}%")
        print(f"    Body-2  : {frac2:.1f}%")
        print(f"{'='*60}\n")

    return G, conductance_data

# 
# Get inlet and outlet indices based on surface bodies touching domain faces
def get_inlet_outlet_indices(arrays, body_array, direction='x'):
    """
    Identify inlet and outlet body indices from surface bodies,
    based on which face of the domain they touch in the flow direction.

    Parameters
    ----------
    arrays : dict
        Output of build_network_arrays().
    body_array : ndarray
        Original labeled body array. Array axis 0/1/2 correspond to x/y/z
        respectively (DIRECTION_TO_AXIS), matching berg_voxel_fast's `axis`
        and particulate_claude's `surface_axis` convention -- there is no
        (Z,Y,X) reordering anywhere in this codebase.
    direction : str
        Flow direction: 'x', 'y', or 'z'.

    Returns
    -------
    inlet_indices  : ndarray of int   — bodies on the low face
    outlet_indices : ndarray of int   — bodies on the high face
    """

    # Map direction to axis index and domain extent
    axis = DIRECTION_TO_AXIS[direction]
    domain_max = body_array.shape[axis] - 1

    coords    = arrays['coords']        # (N_bodies, 3), coords[:, axis] indexed by the same 0=x/1=y/2=z convention
    is_surface = arrays['is_surface']   # (N_bodies,) bool

    surface_indices = np.where(is_surface)[0]

    inlet_indices  = []
    outlet_indices = []

    for idx in surface_indices:
        coord_in_axis = coords[idx, axis]
        if coord_in_axis <= 1:
            inlet_indices.append(idx)
        elif coord_in_axis >= domain_max - 1:
            outlet_indices.append(idx)
        # bodies touching other faces are surface but neither inlet nor outlet
        # they are just insulating walls — leave them as interior for the solve

    return np.array(inlet_indices, dtype=int), np.array(outlet_indices, dtype=int)


def filter_nodes_by_boundary_connectivity(arrays, G_conductance,
                                           inlet_indices, outlet_indices,
                                           verbose=True):
    """
    Pre-solve filter: keep only nodes that satisfy BOTH conditions:

    1. At least 2 active conducting throat connections.
    2. Among those connections, at least one neighbour has a path to
       an inlet AND at least one neighbour has a path to an outlet.
       These can be different neighbours — enforcing true flow-through.

    Boundary nodes (inlet/outlet) are exempt from both conditions.

    Iterated until convergence because removing one node can cause
    its neighbours to fail the conditions in the next round.

    Parameters
    ----------
    arrays : dict
        Output of build_network_arrays().
    G_conductance : ndarray, shape (N_throats,)
        Per-throat conductances from calculate_conductances().
    inlet_indices : array-like of int
        Body indices of inlet boundary.
    outlet_indices : array-like of int
        Body indices of outlet boundary.
    verbose : bool

    Returns
    -------
    pre_mask : ndarray of bool, shape (N_bodies,)
        True for nodes satisfying both conditions.
    pre_dead_indices : ndarray of int
        Indices of removed nodes.
    pre_dead_volume : float
        Total volume of removed nodes [vox³].
    removal_log : list of dict
        Per-iteration removal log.
    """

    conns          = arrays['conns']
    volumes        = arrays['volumes']
    N_bodies       = len(arrays['body_ids'])
    inlet_indices  = np.asarray(inlet_indices,  dtype=int)
    outlet_indices = np.asarray(outlet_indices, dtype=int)
    inlet_set      = set(inlet_indices.tolist())
    outlet_set     = set(outlet_indices.tolist())
    boundary_set   = inlet_set | outlet_set

    active      = np.ones(N_bodies, dtype=bool)
    removal_log = []
    iteration   = 0

    while True:
        iteration += 1

        # ── Build adjacency of active nodes ───────────────────────────────
        # neighbours[node] = set of active neighbours via conducting throats
        neighbours = {n: set() for n in range(N_bodies) if active[n]}

        for t_idx in range(len(G_conductance)):
            i, j = conns[t_idx, 0], conns[t_idx, 1]
            if i < 0 or j < 0:
                continue
            if G_conductance[t_idx] < 1e-30:
                continue
            if active[i] and active[j]:
                neighbours[i].add(j)
                neighbours[j].add(i)

        # ── Build undirected graph for reachability queries ────────────────
        G = nx.Graph()
        for node, nbrs in neighbours.items():
            for nbr in nbrs:
                G.add_edge(node, nbr)

        # Precompute connected components for fast reachability
        # A node can reach inlet if it is in the same component as
        # any inlet node, and similarly for outlet
        node_to_component = {}
        for comp in nx.connected_components(G):
            for node in comp:
                node_to_component[node] = comp

        # ── Check both conditions for every active non-boundary node ───────
        removed_this_round = []

        # Replace the condition check section inside the while loop:

        for node in range(N_bodies):
            if not active[node]:
                continue

            nbrs = neighbours.get(node, set())

            if node in inlet_set:
                # Inlet nodes only need at least 1 neighbour 
                # that can reach an outlet
                if len(nbrs) < 1:
                    active[node] = False
                    removed_this_round.append((node, 'inlet_no_outlet_neighbour'))
                    continue
                nbr_reaches_outlet = any(
                    outlet_set & node_to_component.get(nbr, set())
                    for nbr in nbrs
                )
                if not nbr_reaches_outlet:
                    active[node] = False
                    removed_this_round.append((node, 'inlet_no_outlet_neighbour'))

            elif node in outlet_set:
                # Outlet nodes only need at least 1 neighbour
                # that can reach an inlet
                if len(nbrs) < 1:
                    active[node] = False
                    removed_this_round.append((node, 'outlet_no_inlet_neighbour'))
                    continue
                nbr_reaches_inlet = any(
                    inlet_set & node_to_component.get(nbr, set())
                    for nbr in nbrs
                )
                if not nbr_reaches_inlet:
                    active[node] = False
                    removed_this_round.append((node, 'outlet_no_inlet_neighbour'))

            else:
                # Interior nodes need at least 2 connections,
                # one reaching inlet AND one reaching outlet
                if len(nbrs) < 2:
                    active[node] = False
                    removed_this_round.append((node, 'coord<2'))
                    continue

                nbr_reaches_inlet  = any(
                    inlet_set & node_to_component.get(nbr, set())
                    for nbr in nbrs
                )
                nbr_reaches_outlet = any(
                    outlet_set & node_to_component.get(nbr, set())
                    for nbr in nbrs
                )
                if not nbr_reaches_inlet or not nbr_reaches_outlet:
                    active[node] = False
                    reason = ('no_inlet_neighbour'  if not nbr_reaches_inlet
                            else 'no_outlet_neighbour')
                    removed_this_round.append((node, reason))

        removal_log.append({
            'iteration':        iteration,
            'n_removed':        len(removed_this_round),
            'removals':         removed_this_round,
        })

        if not removed_this_round:
            break     # converged

    # ── Build output ───────────────────────────────────────────────────────
    pre_dead_indices = np.where(~active)[0]
    pre_dead_volume  = float(volumes[pre_dead_indices].sum())

    if verbose:
        total_removed = len(pre_dead_indices)
        print(f"\n{'='*60}")
        print(f"{'PRE-SOLVE: BOUNDARY CONNECTIVITY FILTER':^60}")
        print(f"{'='*60}")
        print(f"  Total bodies              : {N_bodies}")
        print(f"  Inlet bodies              : {len(inlet_indices)}")
        print(f"  Outlet bodies             : {len(outlet_indices)}")
        print(f"  Iterations to converge    : {iteration}")

        for log in removal_log:
            if log['n_removed'] > 0:
                by_reason = {}
                for node, reason in log['removals']:
                    by_reason.setdefault(reason, []).append(node)
                print(f"\n    Iteration {log['iteration']:2d} "
                      f"— {log['n_removed']} removed:")
                for reason, nodes in by_reason.items():
                    print(f"      {reason:25s}: "
                          f"{len(nodes)} node(s) {nodes[:5]}"
                          f"{'...' if len(nodes) > 5 else ''}")

        print(f"\n  Total removed             : {total_removed}")
        print(f"  Volume removed            : {pre_dead_volume:.4g} vox³")
        print(f"  Surviving bodies          : {int(active.sum())}")
        if 0 < total_removed <= 20:
            print(f"  Removed indices           : "
                  f"{pre_dead_indices.tolist()}")
        print(f"{'='*60}\n")

    return active.copy(), pre_dead_indices, pre_dead_volume, removal_log

# ──────────────────────────────────────────────────────────────────────────────
# Function 3 — solve_potential_field
# ──────────────────────────────────────────────────────────────────────────────

def solve_potential_field(arrays, G, inlet_indices, outlet_indices,
                          conducting_mask=None,
                          delta_V=1.0, verbose=True):
    """
    Solve for the steady-state electrical potential at every pore body
    using Kirchhoff's current law (sparse linear system).

    The system enforces:
        sum_j  G_ij * (V_i - V_j) = 0   for all interior bodies
        V_i = delta_V                     for inlet bodies
        V_i = 0                           for outlet bodies

    Parameters
    ----------
    arrays : dict
        Output of build_network_arrays().
    G : ndarray, shape (N_throats,)
        Per-throat conductances from calculate_conductances().
    inlet_indices : array-like of int
        Integer indices (into arrays['body_ids']) of inlet boundary bodies.
    outlet_indices : array-like of int
        Integer indices of outlet boundary bodies.
    delta_V : float
        Potential difference applied across the sample.
    verbose : bool
        Print solution summary.

    Returns
    -------
    potential : ndarray, shape (N_bodies,)
        Electric potential at each body [V].
    current : ndarray, shape (N_throats,)
        Current through each throat: I = G * (V_i - V_j) [A].
    solution_data : dict
        'total_current', 'effective_conductance', 'inlet_indices',
        'outlet_indices', 'delta_V'.
    """

    conns   = arrays['conns']
    N_pores = len(arrays['body_ids'])

    inlet_indices  = np.asarray(inlet_indices,  dtype=int)
    outlet_indices = np.asarray(outlet_indices, dtype=int)
    #boundary_set   = set(inlet_indices.tolist() + outlet_indices.tolist())

    # ── Pin dead-end nodes before building Laplacian ──────────────────────
    # Dead-end nodes are not isolated (diagonal nonzero) but have no
    # through-path. They make the Laplacian singular if not handled.
    # We pin them to V=0 — they carry no current so this is physically correct.
    dead_end_pin_set = set()
    if conducting_mask is not None:
        dead_end_indices = np.where(~conducting_mask)[0]
        dead_end_pin_set = set(dead_end_indices.tolist())


    # ── Build sparse Laplacian ────────────────────────────────────────────
    # Skip throats that were flagged as invalid (conns == -1)
    valid = (conns[:, 0] >= 0) & (conns[:, 1] >= 0)
    i_idx = conns[valid, 0]
    j_idx = conns[valid, 1]
    g_val = G[valid]

    # Off-diagonal: -G_ij
    rows = np.concatenate([i_idx, j_idx])
    cols = np.concatenate([j_idx, i_idx])
    data = np.concatenate([-g_val, -g_val])

    # Diagonal: sum of conductances at each node
    diag_vals = np.bincount(i_idx, weights=g_val, minlength=N_pores) + \
                np.bincount(j_idx, weights=g_val, minlength=N_pores)

    rows = np.concatenate([rows, np.arange(N_pores)])
    cols = np.concatenate([cols, np.arange(N_pores)])
    data = np.concatenate([data, diag_vals])

    A = sps.csr_matrix((data, (rows, cols)), shape=(N_pores, N_pores))
    b = np.zeros(N_pores)

    # ── Pin isolated nodes (diagonal == 0, not a boundary) ───────────────
    # Bodies with no valid throat connections produce a zero row in the
    # Laplacian which makes the matrix singular. Pin them to V=0 so the
    # system is always non-singular regardless of network connectivity.
    # ── Pin isolated nodes AND dead-end nodes ─────────────────────────────
    # Build list of all nodes to pin (isolated + dead-end)
    pin_set = set()

    # Isolated nodes (diagonal == 0)
    diag_array = np.array(A.diagonal())
    isolated   = np.where(
        (diag_array == 0) &
        (~np.isin(np.arange(N_pores),
                np.concatenate([inlet_indices, outlet_indices])))
    )[0]
    pin_set.update(isolated.tolist())

    # Dead-end nodes from conducting_mask
    if conducting_mask is not None:
        dead_end_indices = np.where(~conducting_mask)[0]
        pin_set.update(dead_end_indices.tolist())

    # Remove inlet/outlet from pin set — boundaries override everything
    pin_set -= set(inlet_indices.tolist())
    pin_set -= set(outlet_indices.tolist())

    # For dead-end nodes: zero out ENTIRE row AND column, then pin
    # This prevents their connections from distorting neighbour potentials
    for p in pin_set:
        # Zero the column too — removes influence on neighbours
        A[:, p] = 0.0
        A[p, :] = 0.0
        A[p, p] = 1.0
        b[p]    = 0.0

    # ── Apply Dirichlet boundary conditions ───────────────────────────────
    for p in inlet_indices:
        A[p, :] = 0.0
        A[p, p] = 1.0
        b[p]    = delta_V
    for p in outlet_indices:
        A[p, :] = 0.0
        A[p, p] = 1.0
        b[p]    = 0.0
    A = A.tocsr()

    # ── Solve ─────────────────────────────────────────────────────────────
    potential = spsolve(A, b)

    # ── Per-throat current ────────────────────────────────────────────────
    current         = np.zeros(len(G))
    V_i             = potential[conns[valid, 0]]
    V_j             = potential[conns[valid, 1]]
    current[valid]  = G[valid] * (V_i - V_j)

    # Total current = sum of absolute current leaving inlet bodies
    inlet_mask    = np.isin(conns[:, 0], inlet_indices) | \
                    np.isin(conns[:, 1], inlet_indices)
    total_current = np.abs(current[inlet_mask]).sum()

    # Effective conductance of whole sample: G_eff = I_total / delta_V
    G_eff = total_current / delta_V if delta_V != 0 else 0.0

    solution_data = {
        'total_current':          total_current,
        'effective_conductance':  G_eff,
        'inlet_indices':          inlet_indices,
        'outlet_indices':         outlet_indices,
        'delta_V':                delta_V,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"{'POTENTIAL FIELD SOLUTION':^60}")
        print(f"{'='*60}")
        print(f"  Inlet bodies:           {len(inlet_indices)}")
        print(f"  Outlet bodies:          {len(outlet_indices)}")
        print(f"  Applied delta_V:        {delta_V}")
        print(f"  Potential range:        [{potential.min():.4f}, {potential.max():.4f}]")
        print(f"  Total current:          {total_current:.6g}")
        print(f"  Effective conductance:  {G_eff:.6g}")
        print(f"{'='*60}\n")

    return potential, current, solution_data

def filter_nonconducting_nodes_directed(arrays, current, solution_data,
                                         coord_mask,
                                         I_threshold_factor=1e-15,
                                         verbose=True):
    """
    Post-solve directed reachability filter.

    Using the solved current to assign edge directions, checks that
    every surviving node (after coordination filter) lies on a complete
    directed path from inlet to outlet. Nodes that fail are removed
    and their volume tracked separately.

    Only nodes passing coord_mask are considered — this avoids
    re-examining already-removed nodes.

    Parameters
    ----------
    arrays : dict
        Output of build_network_arrays().
    current : ndarray, shape (N_throats,)
        Signed per-throat current from solve_potential_field().
    solution_data : dict
        Output of solve_potential_field(). Required keys:
            'inlet_indices', 'outlet_indices', 'total_current'.
    coord_mask : ndarray of bool, shape (N_bodies,)
        Output of filter_low_coordination_nodes().
        Only nodes where coord_mask=True are checked.
    I_threshold_factor : float
        Throats with |current| < I_threshold_factor * I_total are
        treated as numerical noise. Default 1e-15.
    verbose : bool

    Returns
    -------
    directed_mask : ndarray of bool, shape (N_bodies,)
        True for nodes on a complete directed inlet → outlet path.
        This is the final conducting mask to pass to build_streamtubes.
    directed_dead_indices : ndarray of int
        Nodes that passed coord filter but failed directed check.
    directed_dead_volume : float
        Volume of nodes failing directed check [vox³].
    I_threshold : float
        Absolute current threshold used, for passing to build_streamtubes.
    """

    conns          = arrays['conns']
    volumes        = arrays['volumes']
    N_bodies       = len(arrays['body_ids'])
    inlet_indices  = np.asarray(solution_data['inlet_indices'], dtype=int)
    outlet_indices = np.asarray(solution_data['outlet_indices'], dtype=int)
    I_total        = float(solution_data['total_current'])
    I_threshold    = I_total * I_threshold_factor

    # ── Build directed graph from solved current ───────────────────────────
    # Only include nodes that survived coord filter
    # Only include throats above noise threshold
    G_dir = nx.DiGraph()
    G_dir.add_nodes_from(np.where(coord_mask)[0].tolist())

    for t_idx in range(len(current)):
        i, j = conns[t_idx, 0], conns[t_idx, 1]
        if i < 0 or j < 0:
            continue
        if not coord_mask[i] or not coord_mask[j]:
            continue                       # already removed pre-solve
        abs_I = abs(float(current[t_idx]))
        if abs_I < I_threshold:
            continue                       # numerical noise
        up, dn = (i, j) if current[t_idx] > 0 else (j, i)
        G_dir.add_edge(up, dn)

    # ── Directed reachability ──────────────────────────────────────────────
    SUPER_INLET  = N_bodies
    SUPER_OUTLET = N_bodies + 1
    G_dir.add_node(SUPER_INLET)
    G_dir.add_node(SUPER_OUTLET)

    for idx in inlet_indices:
        if idx < N_bodies and coord_mask[idx]:
            G_dir.add_edge(SUPER_INLET, idx)
    for idx in outlet_indices:
        if idx < N_bodies and coord_mask[idx]:
            G_dir.add_edge(idx, SUPER_OUTLET)

    # Forward: reachable from super-inlet following directed edges
    reachable_from_inlet = nx.descendants(G_dir, SUPER_INLET)

    # Backward: can reach super-outlet
    G_rev               = G_dir.reverse(copy=True)
    reachable_to_outlet = nx.descendants(G_rev, SUPER_OUTLET)

    conducting_set = (reachable_from_inlet & reachable_to_outlet) - \
                     {SUPER_INLET, SUPER_OUTLET}

    # ── Build output ───────────────────────────────────────────────────────
    directed_mask = np.zeros(N_bodies, dtype=bool)
    for idx in conducting_set:
        if idx < N_bodies:
            directed_mask[idx] = True

    # Directed dead-ends = survived coord filter but failed directed check
    directed_dead_indices = np.where(coord_mask & ~directed_mask)[0]
    directed_dead_volume  = float(volumes[directed_dead_indices].sum())

    if verbose:
        n_coord_surviving = int(coord_mask.sum())
        n_directed        = int(directed_mask.sum())
        n_dead            = len(directed_dead_indices)
        print(f"\n{'='*60}")
        print(f"{'POST-SOLVE: DIRECTED REACHABILITY FILTER':^60}")
        print(f"{'='*60}")
        print(f"  I_total                   : {I_total:.6g}")
        print(f"  I_threshold               : {I_threshold:.4e}")
        print(f"  Nodes entering (coord OK) : {n_coord_surviving}")
        print(f"  Nodes passing directed    : {n_directed}")
        print(f"  Nodes removed             : {n_dead}")
        print(f"  Volume removed            : {directed_dead_volume:.2f} vox³")
        if 0 < n_dead <= 20:
            print(f"  Removed indices           : "
                  f"{directed_dead_indices.tolist()}")
        elif n_dead > 20:
            print(f"  Removed indices           : "
                  f"{directed_dead_indices[:10].tolist()} "
                  f"... ({n_dead} total)")
        print(f"{'='*60}\n")

    return directed_mask, directed_dead_indices, directed_dead_volume, I_threshold

def _prepare_cc_augmented_dag(arrays, current, potential, inlet_indices, outlet_indices,
                              I_threshold):
    """
    Build the source→sink augmented flow DAG for the body/throat network, in
    the same (ptr, dn, flow, kind, edge) CSR layout the fast voxel-network
    decomposers expect (see berg_voxel_fast._prepare_augmented_flow_dag).

    Physical directed edges come from throats, oriented strictly down
    potential (so after dropping sub-threshold current the graph is acyclic).
    A virtual source S injects current at each inlet body node; a virtual
    sink T absorbs it at each outlet body node. Since inlet/outlet nodes here
    are ordinary Dirichlet-pinned network nodes (no half-cell boundary
    resistors, unlike the voxel pipeline), the S→inlet and outlet→T edge
    capacities are simply the node's total outgoing / incoming throat
    current — by construction never a bottleneck relative to the internal
    throat edges.

    Returns a dict with keys: S, T, n_aug, up, dn, flow, kind, edge, ptr,
    I_threshold, n_pruned. `kind` is 0=internal throat edge, 1=source edge,
    2=sink edge; `edge` gives the original throat index for kind==0 edges
    and -1 otherwise.
    """
    conns = arrays['conns']
    N_bodies = len(arrays['body_ids'])
    S = N_bodies
    T = N_bodies + 1
    n_aug = N_bodies + 2

    valid = (conns[:, 0] >= 0) & (conns[:, 1] >= 0)
    i_idx = conns[:, 0]
    j_idx = conns[:, 1]

    pos = valid & (current > I_threshold)
    neg = valid & (current < -I_threshold)

    up_int = np.concatenate([i_idx[pos], j_idx[neg]]).astype(np.int64, copy=False)
    dn_int = np.concatenate([j_idx[pos], i_idx[neg]]).astype(np.int64, copy=False)
    eidx_int = np.concatenate([np.flatnonzero(pos), np.flatnonzero(neg)]).astype(np.int64, copy=False)
    flow_int = np.concatenate([current[pos], -current[neg]]).astype(float, copy=False)

    inlet_indices = np.asarray(inlet_indices, dtype=np.int64)
    outlet_indices = np.asarray(outlet_indices, dtype=np.int64)

    out_current = np.bincount(up_int, weights=flow_int, minlength=N_bodies)
    in_current = np.bincount(dn_int, weights=flow_int, minlength=N_bodies)

    inlet_I_all = out_current[inlet_indices]
    keep_in = inlet_I_all > I_threshold
    inlet_nodes = inlet_indices[keep_in]
    inlet_I = inlet_I_all[keep_in]

    outlet_I_all = in_current[outlet_indices]
    keep_out = outlet_I_all > I_threshold
    outlet_nodes = outlet_indices[keep_out]
    outlet_I = outlet_I_all[keep_out]

    up = np.concatenate([up_int, np.full(len(inlet_nodes), S, dtype=np.int64), outlet_nodes])
    dn = np.concatenate([dn_int, inlet_nodes, np.full(len(outlet_nodes), T, dtype=np.int64)])
    flow = np.concatenate([flow_int, inlet_I, outlet_I])
    kind = np.concatenate([
        np.zeros(len(flow_int), dtype=np.int8),
        np.ones(len(inlet_I), dtype=np.int8),
        np.full(len(outlet_I), 2, dtype=np.int8),
    ])
    edge = np.concatenate([
        eidx_int,
        np.full(len(inlet_I), -1, dtype=np.int64),
        np.full(len(outlet_I), -1, dtype=np.int64),
    ])

    keep = flow > I_threshold
    up, dn, flow, kind, edge = (arr[keep] for arr in (up, dn, flow, kind, edge))
    if len(flow) == 0:
        raise RuntimeError("No current-carrying augmented edges above threshold.")

    # ── Prune to edges that can reach T (reverse reachability from sink) ───
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
    up, dn, flow, kind, edge = (arr[keep] for arr in (up, dn, flow, kind, edge))
    if not reachable[S]:
        raise RuntimeError("Virtual source cannot reach virtual sink through current-carrying edges.")

    order = np.argsort(up, kind="mergesort")
    up_s, dn_s = up[order], dn[order]
    flow_s, kind_s, edge_s = flow[order], kind[order], edge[order]

    counts = np.bincount(up_s, minlength=n_aug)
    ptr = np.empty(n_aug + 1, dtype=np.int64)
    ptr[0] = 0
    np.cumsum(counts, out=ptr[1:])



    print({
        "S": int(S), "T": int(T), "n_aug": int(n_aug),
        "I_threshold": float(I_threshold), "n_pruned": n_pruned,
        "up": up_s, "dn": dn_s, "flow": flow_s,
        "kind": kind_s, "edge": edge_s, "ptr": ptr,
    })


    return {
        "S": int(S), "T": int(T), "n_aug": int(n_aug),
        "I_threshold": float(I_threshold), "n_pruned": n_pruned,
        "up": up_s, "dn": dn_s, "flow": flow_s,
        "kind": kind_s, "edge": edge_s, "ptr": ptr,
    }


def build_streamtubes(arrays, current, potential, solution_data,I_threshold,
                      method='flow_decomposition', verbose=True):
    """
    Discretise the conducting pore volume Ωc into a disjoint union of
    streamtubes Γ following Berg (2012) Section V.

    Four methods are available via the `method` toggle:

    'all_simple_paths' (exhaustive)
        Uses nx.all_simple_paths to find every simple path from each inlet
        to each outlet. Full-mixing weights are applied at each node to
        compute I_Γ. Produces the maximum number of streamtubes. Can be
        slow for large or well-connected networks.

    'flow_decomposition' (greedy, pure Python / networkx)
        Decomposes the flow into a minimum set of path flows. At each step,
        greedily follows the highest-remaining-flow edge from an inlet to
        an outlet, records I_Γ as the bottleneck flow on that path, then
        subtracts it from all edges. Repeats until all flow is accounted
        for. Produces at most E streamtubes (E = number of conducting
        throats), consistent with Berg's disjoint union requirement.

    'greedy_fast' (recommended)
        Same greedy rule as 'flow_decomposition', but built on the
        Numba-compiled max-heap/path-trace core shared with the voxel-network
        pipeline (berg_voxel_fast.decompose_streamtubes) instead of
        networkx — orders of magnitude faster on large networks. Falls back
        to an equivalent pure-Python heap implementation if numba is not
        installed.

    'topological'
        No repeated source-to-sink path search. Nodes are visited once in
        decreasing potential order (a valid topological order, since every
        retained edge has strictly positive potential drop) and incoming
        current packets are locally paired with outgoing edge capacities.
        Deterministic, single-pass, and generally the fastest option on
        large networks; produces a different (but still conservative) set of
        streamtubes than the greedy methods.

    Volume conservation:   Σ_Γ V_Γ = Ωc
    Current conservation:  Σ_Γ I_Γ = I_total

    Parameters
    ----------
    arrays : dict
        Output of build_network_arrays(). Required keys:
            'conns'         – (N_throats, 2) body index pairs
            'throat_volume' – (N_throats,)   throat channel volumes [vox³]
            'V_body1'       – (N_throats,)   body-1 segment volumes [vox³]
            'V_body2'       – (N_throats,)   body-2 segment volumes [vox³]
            'L_body1'       – (N_throats,)   body-1 segment lengths [vox]
            'L_body2'       – (N_throats,)   body-2 segment lengths [vox]
            'L_throat'      – (N_throats,)   throat segment lengths [vox]
    current : ndarray, shape (N_throats,)
        Signed per-throat current from solve_potential_field().
        Positive = flow conns[:,0] → conns[:,1].
    potential : ndarray, shape (N_bodies,)
        Solved electric potential at each body.
    solution_data : dict
        Output of solve_potential_field(). Required keys:
            'inlet_indices'  – body indices of inlet boundary
            'outlet_indices' – body indices of outlet boundary
            'total_current'  – scalar I_total
    method : str
        'all_simple_paths', 'flow_decomposition', 'greedy_fast', or
        'topological'. Default: 'flow_decomposition'.
    verbose : bool
        Print conservation checks and summary.

    Returns
    -------
    streamtubes : list of dict
        One dict per streamtube Γ. Keys:
            'node_path'      – list of body indices inlet → outlet
            'throat_path'    – list of throat indices in series order
            'I_gamma'        – absolute current carried [A]
            'f_gamma'        – fractional current I_Γ / I_total
            'volume_shares'  – per-throat volume fraction for this streamtube [vox³]
            'V_gamma'        – total streamtube volume Σ volume_shares [vox³]
            'L_gamma'        – total path length Σ(L_body1+L_throat+L_body2) [vox]
    Omega_c : float
        Total conducting network volume Ωc [vox³].
    """

    # ── Unpack inputs ─────────────────────────────────────────────────────
    conns    = arrays['conns']
    V_body1  = arrays['V_body1']
    V_body2  = arrays['V_body2']
    V_throat = arrays['throat_volume']
    L_body1  = arrays['L_body1']
    L_body2  = arrays['L_body2']
    L_throat = arrays['L_throat']

    inlet_indices  = np.asarray(solution_data['inlet_indices'],  dtype=int)
    outlet_indices = np.asarray(solution_data['outlet_indices'], dtype=int)
    I_total        = float(solution_data['total_current'])
    I_threshold    = I_threshold if I_threshold is not None else I_total * 1e-10
    inlet_set      = set(inlet_indices.tolist())
    outlet_set     = set(outlet_indices.tolist())

    # ── 1. Build directed graph from current signs ────────────────────────
    # Edge direction: upstream (high V) → downstream (low V).
    # Only conducting throats (|current| > 0) are included — these form Ωc.
    # Each edge stores:
    #   throat_idx  : index into arrays for geometry lookup
    #   abs_current : |current[t]|  used for splitting and decomposition

    G = nx.DiGraph()
    edge_to_throat = {}   # (up, dn) → throat array index

    for t_idx in range(len(current)):
        i, j = conns[t_idx, 0], conns[t_idx, 1]
        if i < 0 or j < 0:
            continue                          # invalid throat
        abs_I = abs(float(current[t_idx]))
        if abs_I < I_threshold:
            continue                          # non-conducting: exclude from Ωc

        up, dn = (i, j) if current[t_idx] > 0 else (j, i)
        G.add_edge(up, dn, throat_idx=t_idx, abs_current=abs_I)
        edge_to_throat[(up, dn)] = t_idx

    # ── 2. Compute Ωc over all conducting throats ─────────────────────────
    conducting = [d['throat_idx'] for _, _, d in G.edges(data=True)]
    Omega_c    = float(np.sum(
        V_body1[conducting] + V_throat[conducting] + V_body2[conducting]
    ))

    active_inlets  = [n for n in inlet_indices  if n in G]  
    active_outlets = [n for n in outlet_indices if n in G]

    # ── Helper: convert node path → throat path ───────────────────────────
    def node_path_to_throat_path(node_path):
        throat_path = []
        for k in range(len(node_path) - 1):
            key = (node_path[k], node_path[k + 1])
            if key not in edge_to_throat:
                return None
            throat_path.append(edge_to_throat[key])
        return throat_path if throat_path else None

    # ── Helper: compute volume shares and L_gamma for a throat path ───────
    def compute_path_geometry(throat_path, I_gamma):
        volume_shares = []
        L_gamma = 0.0
        for t_idx in throat_path:
            abs_I_t  = abs(float(current[t_idx]))
            V_total  = V_body1[t_idx] + V_throat[t_idx] + V_body2[t_idx]
            share    = (I_gamma / abs_I_t) * V_total if abs_I_t > I_threshold else 0.0
            volume_shares.append(float(share))
            L_gamma += L_body1[t_idx] + L_throat[t_idx] + L_body2[t_idx]
        return volume_shares, float(L_gamma)

    # ════════════════════════════════════════════════════════════════════════
    # METHOD A — all_simple_paths
    # ════════════════════════════════════════════════════════════════════════
    def _trace_all_simple_paths():
        """
        Full-mixing weight applied at every intermediate node.
        I_Γ = |current[first_throat]| × Π weights along path.
        """
        # Precompute full-mixing split weights at every node:
        # weight[(u,v)] = |current[(u,v)]| / Σ|current[outgoing from u]|
        node_out_weights = {}
        for node in G.nodes():
            out_edges = list(G.out_edges(node, data=True))
            if not out_edges:
                continue
            total_out = sum(d['abs_current'] for _, _, d in out_edges)
            if total_out < 1e-30:
                continue
            node_out_weights[node] = {
                (u, v): d['abs_current'] / total_out
                for u, v, d in out_edges
            }

        results = []
        for src in active_inlets:
            for tgt in active_outlets:
                if src == tgt:
                    continue
                for node_path in nx.all_simple_paths(G, source=src, target=tgt):
                    throat_path = node_path_to_throat_path(node_path)
                    if throat_path is None:
                        continue

                    # I_Γ: start with first edge current, multiply mixing weights
                    I_gamma = G[node_path[0]][node_path[1]]['abs_current']
                    for k in range(1, len(node_path) - 1):
                        u, v    = node_path[k], node_path[k + 1]
                        weights = node_out_weights.get(node_path[k], {})
                        I_gamma *= weights.get((u, v), 0.0)

                    volume_shares, L_gamma = compute_path_geometry(throat_path, I_gamma)
                    results.append({
                        'node_path':     list(node_path),
                        'throat_path':   throat_path,
                        'I_gamma':       I_gamma,
                        'f_gamma':       I_gamma / I_total if I_total > 1e-30 else 0.0,
                        'volume_shares': volume_shares,
                        'V_gamma':       float(np.sum(volume_shares)),
                        'L_gamma':       L_gamma,
                    })
        return results,0.0

    # ════════════════════════════════════════════════════════════════════════
    # METHOD B — greedy flow decomposition
    # ════════════════════════════════════════════════════════════════════════
    def _flow_decomposition():
        """
        Greedy path decomposition of the flow.
        At each step:
          1. Pick the inlet node with the most remaining outgoing flow.
          2. Greedily follow the highest-remaining-flow outgoing edge at
             each node until an outlet is reached.
          3. I_Γ = minimum remaining flow along the path (bottleneck).
          4. Subtract I_Γ from every edge in the path.
          5. Remove exhausted edges (remaining flow < threshold).
          6. Repeat until no flow remains at inlets.
        Produces at most E streamtubes.
        """
        EPS = I_threshold   # threshold for considering an edge exhausted

        # Working copy of remaining flow on each directed edge
        remaining = {(u, v): d['abs_current']
                     for u, v, d in G.edges(data=True)}

        results = []
        V_stuck_total = 0.0   # track volume of flow that gets "stuck" due to dead-ends

        while True:
            # Find an inlet node that still has outgoing flow
            src = None
            best_flow = EPS
            for n in active_inlets:
                out_flow = sum(
                    remaining.get((n, v), 0.0)
                    for v in G.successors(n)
                )
                if out_flow > best_flow:
                    best_flow = out_flow
                    src = n
            if src is None:
                break   # all inlet flow exhausted

            # Greedy path: always follow the highest-remaining-flow edge
            node_path = [src]
            visited   = {src}

            stuck = False
            while node_path[-1] not in outlet_set:
                node = node_path[-1]
                # Outgoing edges with remaining flow, excluding visited nodes
                candidates = [
                    (v, remaining.get((node, v), 0.0))
                    for v in G.successors(node)
                    if v not in visited and remaining.get((node, v), 0.0) > EPS
                ]
                if not candidates:
                    stuck = True
                    break
                # Pick highest remaining flow
                next_node = max(candidates, key=lambda x: x[1])[0]
                node_path.append(next_node)
                visited.add(next_node)

            stuck_path_lengths = []
            if stuck or node_path[-1] not in outlet_set:
                stuck_path_lengths.append(len(node_path))
                if len(node_path) >= 2:
                    key          = (node_path[0], node_path[1])
                    drained_flow = remaining.get(key, 0.0)   # use remaining flow directly
                    remaining[key] = 0.0

                    t_idx = edge_to_throat.get(key)
                    if t_idx is not None and drained_flow > EPS:
                        abs_I_t = abs(float(current[t_idx]))
                        if abs_I_t > I_threshold:
                            V_t = (V_body1[t_idx] +
                                V_throat[t_idx] +
                                V_body2[t_idx])
                            V_stuck_total += (drained_flow / abs_I_t) * V_t
                continue

            # I_Γ = bottleneck (minimum remaining flow along path)
            path_edges = [(node_path[k], node_path[k+1])
                          for k in range(len(node_path) - 1)]
            I_gamma    = min(remaining.get(e, 0.0) for e in path_edges)

            if I_gamma < EPS:
                for e in path_edges:
                    t_idx = edge_to_throat.get(e)
                    drained = remaining.get(e, 0.0)
                    remaining[e] = 0.0
                    if t_idx is not None and drained > EPS:
                        abs_I_t = abs(float(current[t_idx]))
                        if abs_I_t > I_threshold:
                            V_t = V_body1[t_idx] + V_throat[t_idx] + V_body2[t_idx]
                            V_stuck_total += (drained / abs_I_t) * V_t
                continue

            # Subtract I_Γ from all edges in path
            for e in path_edges:
                remaining[e] = max(remaining.get(e, 0.0) - I_gamma, 0.0)

            # Build throat path and geometry
            throat_path = node_path_to_throat_path(node_path)
            if throat_path is None:
                continue

            volume_shares, L_gamma = compute_path_geometry(throat_path, I_gamma)
            results.append({
                'node_path':     list(node_path),
                'throat_path':   throat_path,
                'I_gamma':       I_gamma,
                'f_gamma':       I_gamma / I_total if I_total > 1e-30 else 0.0,
                'volume_shares': volume_shares,
                'V_gamma':       float(np.sum(volume_shares)),
                'L_gamma':       L_gamma,
            })

        return results, V_stuck_total

    # ════════════════════════════════════════════════════════════════════════
    # METHOD C/D — Numba-accelerated greedy / topological decomposition
    # ════════════════════════════════════════════════════════════════════════
    def _decompose_fast(fast_method):
        """
        Build the augmented source→sink DAG for the body/throat network and
        run it through the shared Numba cores (see module docstring / import
        header). Converts the compact augmented-edge paths they return back
        into this module's streamtube dict format.
        """
        dag = _prepare_cc_augmented_dag(
            arrays, current, potential, inlet_indices, outlet_indices, I_threshold)
        ptr, dn_arr, flow_arr = dag['ptr'], dag['dn'], dag['flow']
        S_node, T_node = dag['S'], dag['T']
        thr = dag['I_threshold']
        use_numba = HAS_NUMBA

        if fast_method == 'greedy_fast':
            if use_numba:
                flat, offsets, gammas, *_ = _greedy_fast_core_numba(
                    ptr, dn_arr, flow_arr, thr, S_node, T_node, dag['n_aug'])
            else:
                flat, offsets, gammas, *_ = _greedy_fast_core_python(
                    ptr, dn_arr, flow_arr, thr, S_node, T_node)
        else:  # 'topological'
            topo_nodes = np.argsort(-potential, kind='mergesort').astype(np.int64)
            if use_numba:
                parent, edge_pos, term_rec, gammas, *_ = _topological_core_numba(
                    ptr, dn_arr, flow_arr, thr, S_node, T_node, topo_nodes)
                flat, offsets = _chains_to_flat_numba(parent, edge_pos, term_rec)
            else:
                flat, offsets, gammas, *_ = _topological_core_python(
                    ptr, dn_arr, flow_arr, thr, S_node, T_node, topo_nodes)

        dn_s, kind_s, edge_s = dag['dn'], dag['kind'], dag['edge']
        results = []
        for k in range(len(gammas)):
            a, b = int(offsets[k]), int(offsets[k + 1])
            pp = flat[a:b]
            if len(pp) < 2:
                continue
            kinds = kind_s[pp]
            if kinds[0] != 1 or kinds[-1] != 2:
                raise RuntimeError("Malformed augmented streamtube boundary segments.")

            node_path   = dn_s[pp[:-1]].tolist()          # inlet node ... outlet node
            throat_path = edge_s[pp[1:-1]].tolist()        # internal edges only
            I_gamma     = float(gammas[k])

            volume_shares, L_gamma = compute_path_geometry(throat_path, I_gamma)
            results.append({
                'node_path':     node_path,
                'throat_path':   throat_path,
                'I_gamma':       I_gamma,
                'f_gamma':       I_gamma / I_total if I_total > 1e-30 else 0.0,
                'volume_shares': volume_shares,
                'V_gamma':       float(np.sum(volume_shares)),
                'L_gamma':       L_gamma,
            })

        return results, 0.0

    # ── 3. Dispatch to chosen method ──────────────────────────────────────
    if method == 'all_simple_paths':
        streamtubes, V_stuck_total = _trace_all_simple_paths()
    elif method == 'flow_decomposition':
        streamtubes, V_stuck_total = _flow_decomposition()
    elif method in ('greedy_fast', 'topological'):
        streamtubes, V_stuck_total = _decompose_fast(method)
    else:
        raise ValueError(f"method must be one of 'all_simple_paths', "
                         f"'flow_decomposition', 'greedy_fast', 'topological', "
                         f"got '{method}'")

    # ── 4. Conservation checks (both methods) ─────────────────────────────
    if verbose:
        I_check = sum(st['I_gamma'] for st in streamtubes)
        V_check = sum(st['V_gamma'] for st in streamtubes)
        I_err   = abs(I_check - I_total) / I_total   if I_total  > 1e-30 else 0.0
        V_err   = abs(V_check - Omega_c) / Omega_c   if Omega_c  > 1e-30 else 0.0

        print(f"\n{'='*60}")
        print(f"{'STREAMTUBE DECOMPOSITION  [' + method + ']':^60}")
        print(f"{'='*60}")
        print(f"  Conducting throats in graph : {G.number_of_edges()}")
        print(f"  Total streamtubes traced    : {len(streamtubes)}")
        print(f"  Ωc (conducting volume)      : {Omega_c:.6g} vox³")
        print(f"\n  Current conservation:")
        print(f"    I_total (from solve)      : {I_total:.6g}")
        print(f"    Σ I_Γ   (streamtubes)     : {I_check:.6g}")
        print(f"    Relative error            : {I_err:.2e}")
        print(f"\n  Volume conservation:")
        print(f"    Ωc                        : {Omega_c:.6g}")
        print(f"    Σ V_Γ  (streamtubes)      : {V_check:.6g}")
        print(f"    Relative error            : {V_err:.2e}")
        if I_err > 1e-3:
            print(f"\n  *** WARNING: current error > 0.1% — check inlet/outlet "
                  f"indices or network connectivity ***")
        if V_err > 1e-3:
            print(f"  *** WARNING: volume error > 0.1% — check V_body1/V_body2 "
                  f"arrays ***")
        print(f"{'='*60}\n")

    return streamtubes, Omega_c, V_stuck_total,G, edge_to_throat

def recover_segment_potentials(arrays, potential, current, conductance_data):
    """
    Recover the two interface potentials Φ_1t and Φ_2t at the throat-body
    interfaces for every throat, using current conservation across the three
    sub-segments in series (Berg 2012, Section V).

    For a throat t connecting body nodes i and j, the three sub-segments are:
        body-1 segment : Φ_i   → Φ_1t   conductance G1 = σ*A1/L1
        throat channel : Φ_1t  → Φ_2t   conductance Gt = σ*At/Lt
        body-2 segment : Φ_2t  → Φ_j    conductance G2 = σ*A2/L2

    Since the same current flows through all three in series:
        I = G1*(Φ_i - Φ_1t) = Gt*(Φ_1t - Φ_2t) = G2*(Φ_2t - Φ_j)

    Rearranging:
        Φ_1t = Φ_i - I/G1
        Φ_2t = Φ_j + I/G2

    where I = current[t], with sign convention positive = flow i→j.

    Parameters
    ----------
    arrays : dict
        Output of build_network_arrays(). Required keys:
            'conns'   – (N_throats, 2) body index pairs
    potential : ndarray, shape (N_bodies,)
        Solved electric potential at each body node.
    current : ndarray, shape (N_throats,)
        Signed per-throat current from solve_potential_field().
    conductance_data : dict
        Output of calculate_conductances(). Required keys:
            'G' – per-throat total conductance  (N_throats,)  [not used directly]
            'R1', 'R_throat', 'R2' – per-segment resistances  (N_throats,)

    Returns
    -------
    Phi_1t : ndarray, shape (N_throats,)
        Interface potential at body-1 / throat boundary [V].
    Phi_2t : ndarray, shape (N_throats,)
        Interface potential at throat / body-2 boundary [V].

    Notes
    -----
    For invalid throats (conns == -1) or non-conducting throats (current ≈ 0),
    both interface potentials are set to NaN.
    """

    conns   = arrays['conns']
    R1      = conductance_data['R1']        # L1 / (sigma * A1)
    R2      = conductance_data['R2']        # L2 / (sigma * A2)
    N       = len(current)

    Phi_1t  = np.full(N, np.nan)
    Phi_2t  = np.full(N, np.nan)

    for t in range(N):
        i, j = conns[t, 0], conns[t, 1]
        if i < 0 or j < 0:
            continue                         # invalid throat

        I_t = float(current[t])
        if abs(I_t) < 1e-30:
            continue                         # non-conducting throat

        Phi_i = potential[i]
        Phi_j = potential[j]

        # current sign convention: positive = flow from node i → node j
        # body-1 segment is always on the i side, body-2 on the j side
        # regardless of flow direction — signs handle direction automatically
        Phi_1t[t] = Phi_i - I_t * R1[t]    # Φ_1t = Φ_i - I * R1
        Phi_2t[t] = Phi_j + I_t * R2[t]    # Φ_2t = Φ_j + I * R2

    return Phi_1t, Phi_2t

def compute_local_iota_squared(arrays, potential, Phi_1t, Phi_2t,
                                delta_V, delta_s, current=None,
                                current_threshold=1e-30,
                                verbose=True):
    """
    Compute local conductance reduction factor ι² for each three-segment
    throat and compute both:

        ι²_g : global Berg conductance reduction factor over Ω
        ι²_c : conducting-volume conductance reduction factor over Ωc

    Important distinction
    ---------------------
    Ω  = total effective network volume:
         Σ_t (V_body1[t] + V_throat[t] + V_body2[t])

    Ωc = conducting effective network volume:
         same sum, but only over throats with valid potential drops/current.

    Nonconducting throats contribute volume to Ω but contribute zero to the
    numerator of ι²_g. This is the correction relative to your previous version.

    Parameters
    ----------
    arrays : dict
        Output of build_network_arrays().
    potential : ndarray
        Solved node potentials.
    Phi_1t, Phi_2t : ndarray
        Interface potentials from recover_segment_potentials().
        For nonconducting throats these may be NaN.
    delta_V : float
        Applied potential difference ΔΦ.
    delta_s : float
        Sample length Δs in flow direction.
    current : ndarray or None
        Optional signed current array. If provided, Ωc is determined using
        abs(current) > current_threshold. If None, Ωc is determined from
        finite Phi_1t and Phi_2t.
    current_threshold : float
        Threshold for identifying conducting throats.
    verbose : bool
        Print summary.

    Returns
    -------
    iota_sq : ndarray, shape (N_throats, 3)
        Local ι² values for:
            column 0 = body-1 segment
            column 1 = throat segment
            column 2 = body-2 segment

        Nonconducting or invalid throats are assigned zero ι².

    iota_sq_g : float
        Global Berg ι²_g over Ω.

    Omega : float
        Total effective network volume Ω.

    extra : dict
        Additional quantities:
            'Omega_c'      : conducting effective volume
            'iota_sq_c'    : conducting-volume average
            'numerator_g'  : Σ V_t ι²_t over conducting parts
            'conducting'   : boolean mask of conducting throats
            'valid_throat' : boolean mask of valid throats
    """

    import numpy as np

    conns = arrays['conns']

    V_body1 = np.asarray(arrays['V_body1'], dtype=float)
    V_body2 = np.asarray(arrays['V_body2'], dtype=float)
    V_throat = np.asarray(arrays['throat_volume'], dtype=float)

    L_body1 = np.asarray(arrays['L_body1'], dtype=float)
    L_body2 = np.asarray(arrays['L_body2'], dtype=float)
    L_throat = np.asarray(arrays['L_throat'], dtype=float)

    N = len(V_throat)

    iota_sq = np.zeros((N, 3), dtype=float)

    V_seg = np.column_stack([V_body1, V_throat, V_body2])
    V_total_per_throat = V_body1 + V_throat + V_body2

    # Valid throat means it exists in the arrays and has positive volume.
    valid_throat = (
        (conns[:, 0] >= 0) &
        (conns[:, 1] >= 0) &
        np.isfinite(V_total_per_throat) &
        (V_total_per_throat > 0.0)
    )

    # Ω must include all valid effective network volume, not only conducting volume.
    Omega = float(np.sum(V_total_per_throat[valid_throat]))

    # Conducting mask.
    # Prefer current if supplied; otherwise use finite recovered interface potentials.
    if current is not None:
        current = np.asarray(current, dtype=float)
        conducting = valid_throat & np.isfinite(current) & (np.abs(current) > current_threshold)
    else:
        conducting = (
            valid_throat &
            np.isfinite(Phi_1t) &
            np.isfinite(Phi_2t)
        )

    Omega_c = float(np.sum(V_total_per_throat[conducting]))

    if abs(delta_V) < 1e-30:
        raise ValueError("delta_V is too close to zero; cannot compute ι².")

    scale = (float(delta_s) / float(delta_V)) ** 2

    numerator = 0.0

    for t in np.where(conducting)[0]:
        i, j = conns[t, 0], conns[t, 1]

        Phi_i = float(potential[i])
        Phi_j = float(potential[j])
        Phi_a = float(Phi_1t[t])
        Phi_b = float(Phi_2t[t])

        # Skip if interface potentials are invalid.
        # This keeps nonconducting/ill-defined throats at ι² = 0.
        if not (np.isfinite(Phi_a) and np.isfinite(Phi_b)):
            conducting[t] = False
            continue

        # Segment potential drops.
        # Use absolute values because ι² depends on |∇Φ|².
        dPhi_1 = abs(Phi_i - Phi_a)
        dPhi_3 = abs(Phi_a - Phi_b)
        dPhi_2 = abs(Phi_b - Phi_j)

        # Segment lengths.
        l_1 = max(float(L_body1[t]), 1e-8)
        l_3 = max(float(L_throat[t]), 1e-8)
        l_2 = max(float(L_body2[t]), 1e-8)

        # Local Berg factor:
        # ι² = (ΔΦ_segment / length_segment)^2 * (Δs / ΔΦ_sample)^2
        iota_sq[t, 0] = (dPhi_1 / l_1) ** 2 * scale
        iota_sq[t, 1] = (dPhi_3 / l_3) ** 2 * scale
        iota_sq[t, 2] = (dPhi_2 / l_2) ** 2 * scale

        numerator += (
            V_body1[t]   * iota_sq[t, 0] +
            V_throat[t]  * iota_sq[t, 1] +
            V_body2[t]   * iota_sq[t, 2]
        )

    # If any throats were removed because Phi_1t/Phi_2t were invalid,
    # recompute Ωc consistently.
    Omega_c = float(np.sum(V_total_per_throat[conducting]))

    # Berg global average over Ω.
    iota_sq_g = numerator / Omega if Omega > 1e-30 else 0.0

    # Conducting-volume average over Ωc.
    iota_sq_c = numerator / Omega_c if Omega_c > 1e-30 else 0.0

    extra = {
        'Omega': Omega,
        'Omega_c': Omega_c,
        'iota_sq_g': iota_sq_g,
        'iota_sq_c': iota_sq_c,
        'numerator_g': numerator,
        'conducting': conducting,
        'valid_throat': valid_throat,
        'V_total_per_throat': V_total_per_throat,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"{'LOCAL IOTA²: GLOBAL Ω AND CONDUCTING Ωc':^60}")
        print(f"{'='*60}")
        print(f"  Valid throats              : {int(valid_throat.sum())}")
        print(f"  Conducting throats         : {int(conducting.sum())}")
        print(f"  Ω  (effective volume)      : {Omega:.6g} vox³")
        print(f"  Ωc (conducting volume)     : {Omega_c:.6g} vox³")
        print(f"  Ω - Ωc                     : {Omega - Omega_c:.6g} vox³")
        print(f"  ι²_g over Ω                : {iota_sq_g:.6f}")
        print(f"  ι²_c over Ωc               : {iota_sq_c:.6f}")

        if Omega > 1e-30 and Omega_c > 1e-30:
            print(f"  Check ι²_g × Ω             : {iota_sq_g * Omega:.6g}")
            print(f"  Check ι²_c × Ωc            : {iota_sq_c * Omega_c:.6g}")

        print(f"\n  Per-segment ι² statistics over conducting throats:")
        if conducting.any():
            for col, name in enumerate(['body-1', 'throat', 'body-2']):
                vals = iota_sq[conducting, col]
                print(f"    {name:8s}  min={vals.min():.4g}  "
                      f"max={vals.max():.4g}  mean={vals.mean():.4g}")
        else:
            print("    No conducting throats found.")

        print(f"{'='*60}\n")

    return iota_sq, iota_sq_g, Omega, extra


def compute_streamtube_properties(streamtubes, arrays, iota_sq, Phi_1t, Phi_2t,
                                   potential, delta_V, delta_s,
                                   I_total, Omega_c, verbose=True):
    """
    Compute per-streamtube ι²(Γ), τ(Γ), C(Γ) and global Berg quantities
    ι²_g, ι²_c, τ²_c, C_c (Berg 2012, Eqs. 29-33).

    Per-streamtube quantities
    -------------------------
    ι²(Γ)  – Berg Eq. 29: volume-weighted average of local ι² along Γ
    τ(Γ)   – Berg Eq. 30: Δs / L_Γ  (sample length / path length)
    C(Γ)   – Berg Eq. 31: constriction factor along Γ

    Global quantities
    -----------------
    ι²_g   – Berg Eq. 25: volume-weighted average over all throats
              (cross-check against compute_local_iota_squared output)
    ι²_c   – Berg Eq. 15: ι²_g × (Ω / Ωc)
    τ²_c   – Berg Eq. 32: (1/Ωc) × Σ_Γ V_Γ × τ(Γ)²
    C_c    – Berg Eq. 33: (1/I_t) × Σ_Γ I_Γ × C(Γ)

    Consistency check: τ²_c / C_c ≈ ι²_c  (Berg Eq. 17)

    Parameters
    ----------
    streamtubes : list of dict
        Output of build_streamtubes(). Each dict must contain:
            'throat_path'   – list of throat indices
            'volume_shares' – per-throat volume share for this streamtube
            'V_gamma'       – total streamtube volume
            'L_gamma'       – total path length [vox]
            'I_gamma'       – streamtube current
    arrays : dict
        Output of build_network_arrays(). Required keys:
            'conns', 'L_body1', 'L_body2', 'L_throat',
            'V_body1', 'V_body2', 'throat_volume'
    iota_sq : ndarray, shape (N_throats, 3)
        Local ι² per sub-segment from compute_local_iota_squared().
    Phi_1t : ndarray, shape (N_throats,)
        Interface potentials from recover_segment_potentials().
    Phi_2t : ndarray, shape (N_throats,)
        Interface potentials from recover_segment_potentials().
    potential : ndarray, shape (N_bodies,)
        Solved electric potential at each body node.
    delta_V : float
        Applied potential difference ΔΦ [V].
    delta_s : float
        Sample length in flow direction Δs [vox].
    I_total : float
        Total current through sample [A].
    Omega_c : float
        Conducting volume Ωc from build_streamtubes() [vox³].
    verbose : bool
        Print global Berg quantities and consistency check.

    Returns
    -------
    per_streamtube : list of dict
        One dict per streamtube Γ with keys:
            'iota_sq_gamma' – ι²(Γ)  Berg Eq. 29
            'tau_gamma'     – τ(Γ)   Berg Eq. 30
            'C_gamma'       – C(Γ)   Berg Eq. 31
            'tau_sq_gamma'  – τ(Γ)²
            'I_gamma'       – streamtube current (carried from input)
            'V_gamma'       – streamtube volume  (carried from input)
    global_berg : dict
        Keys:
            'iota_sq_g'   – ι²_g  (volume-weighted, Eq. 25 cross-check)
            'iota_sq_c'   – ι²_c  (Eq. 15)
            'tau_sq_c'    – τ²_c  (Eq. 32)
            'C_c'         – C_c   (Eq. 33)
            'F'           – formation factor 1/(ι²_g × φ) — φ must be supplied
            'consistency' – τ²_c / C_c, should equal ι²_c
    """

    conns    = arrays['conns']
    L_body1  = arrays['L_body1']
    L_body2  = arrays['L_body2']
    L_throat = arrays['L_throat']
    V_body1  = arrays['V_body1']
    V_body2  = arrays['V_body2']
    V_throat = arrays['throat_volume']

    per_streamtube = []

    for st in streamtubes:
        throat_path   = st['throat_path']
        volume_shares = st['volume_shares']   # one share per throat in path
        V_gamma       = st['V_gamma']
        L_gamma       = st['L_gamma']
        I_gamma       = st['I_gamma']

        # ── τ(Γ) — Berg Eq. 30 ───────────────────────────────────────────
        # τ(Γ) = Δs / L_Γ  where L_Γ = Σ_t (l_t1 + l_t3 + l_t2)
        tau_gamma = delta_s / L_gamma if L_gamma > 1e-30 else 0.0

        # ── ι²(Γ) — Berg Eq. 29 ──────────────────────────────────────────
        # ι²(Γ) = (1/V_Γ) × Σ_t (V_share_t1·ι²_t1
        #                        + V_share_t3·ι²_t3
        #                        + V_share_t2·ι²_t2)
        # V_share for each sub-segment is proportional to its volume
        # fraction of the throat's total volume, scaled by the streamtube share.
        iota_sq_num = 0.0
        C_num       = 0.0     # accumulator for C(Γ) numerator (Berg Eq. 31)

        for k, t_idx in enumerate(throat_path):
            i, j = conns[t_idx, 0], conns[t_idx, 1]
            if np.isnan(Phi_1t[t_idx]):
                continue

            # Volume share of this throat belonging to streamtube Γ
            V_share = volume_shares[k]

            # Total volume of this throat (all three segments)
            V_total_t = (float(V_body1[t_idx]) +
                         float(V_throat[t_idx]) +
                         float(V_body2[t_idx]))

            if V_total_t < 1e-30:
                continue

            # Split the streamtube volume share among the three sub-segments
            # proportional to each sub-segment's volume
            frac_t1 = float(V_body1[t_idx])  / V_total_t
            frac_t3 = float(V_throat[t_idx]) / V_total_t
            frac_t2 = float(V_body2[t_idx])  / V_total_t

            V_share_t1 = V_share * frac_t1
            V_share_t3 = V_share * frac_t3
            V_share_t2 = V_share * frac_t2

            # ι²(Γ) numerator contribution from this throat
            iota_sq_num += (V_share_t1 * iota_sq[t_idx, 0] +
                            V_share_t3 * iota_sq[t_idx, 1] +
                            V_share_t2 * iota_sq[t_idx, 2])

            # ── C(Γ) — Berg Eq. 31 ───────────────────────────────────────
            # C(Γ) = ΔΦ / L_Γ² × Σ_t (l²_t1/|ΔΦ_t1|
            #                         + l²_t2/|ΔΦ_t2|
            #                         + l²_t3/|ΔΦ_t3|)
            dPhi_t1 = abs(potential[i]  - Phi_1t[t_idx])
            dPhi_t3 = abs(Phi_1t[t_idx] - Phi_2t[t_idx])
            dPhi_t2 = abs(Phi_2t[t_idx] - potential[j])

            l_t1 = max(float(L_body1[t_idx]),  1e-8)
            l_t3 = max(float(L_throat[t_idx]), 1e-8)
            l_t2 = max(float(L_body2[t_idx]),  1e-8)

            # Guard against zero potential drops (uniform potential segment)
            if dPhi_t1 > 1e-30:
                C_num += l_t1 ** 2 / dPhi_t1
            if dPhi_t3 > 1e-30:
                C_num += l_t3 ** 2 / dPhi_t3
            if dPhi_t2 > 1e-30:
                C_num += l_t2 ** 2 / dPhi_t2

        # Finalise ι²(Γ)
        iota_sq_gamma = iota_sq_num / V_gamma if V_gamma > 1e-30 else 0.0

        # Finalise C(Γ) — Berg Eq. 31
        C_gamma = (delta_V / L_gamma ** 2) * C_num if L_gamma > 1e-30 else 0.0

        per_streamtube.append({
            'iota_sq_gamma': iota_sq_gamma,
            'tau_gamma':     tau_gamma,
            'tau_sq_gamma':  tau_gamma ** 2,
            'C_gamma':       C_gamma,
            'I_gamma':       I_gamma,
            'V_gamma':       V_gamma,
        })

    # ── Global Berg quantities ─────────────────────────────────────────────

    # ι²_c — Berg Eq. 14: volume-weighted average of ι²(Γ) over Ωc
    iota_sq_c = (sum(st['V_gamma'] * st['iota_sq_gamma']
                     for st in per_streamtube) / Omega_c
                 if Omega_c > 1e-30 else 0.0)

    # τ²_c — Berg Eq. 32: (1/Ωc) × Σ_Γ V_Γ × τ(Γ)²
    tau_sq_c = (sum(st['V_gamma'] * st['tau_sq_gamma']
                    for st in per_streamtube) / Omega_c
                if Omega_c > 1e-30 else 0.0)

    # C_c — Berg Eq. 33: (1/I_t) × Σ_Γ I_Γ × C(Γ)
    C_c = (sum(st['I_gamma'] * st['C_gamma']
               for st in per_streamtube) / I_total
           if I_total > 1e-30 else 0.0)

    # Consistency: τ²_c / C_c should equal ι²_c (Berg Eq. 17)
    consistency = tau_sq_c / C_c if C_c > 1e-30 else float('nan')

    global_berg = {
        'iota_sq_c':   iota_sq_c,
        'tau_sq_c':    tau_sq_c,
        'C_c':         C_c,
        'consistency': consistency,   # should ≈ iota_sq_c
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"{'BERG (2012) GLOBAL QUANTITIES':^60}")
        print(f"{'='*60}")
        print(f"  Streamtubes               : {len(per_streamtube)}")
        print(f"  Ωc                        : {Omega_c:.6g} vox³")
        print(f"\n  ι²_c                      : {iota_sq_c:.6f}")
        print(f"  τ²_c  (tortuosity²)       : {tau_sq_c:.6f}")
        print(f"  C_c   (constriction)      : {C_c:.6f}")
        print(f"\n  Consistency check (Berg Eq. 17):")
        print(f"    τ²_c / C_c              : {consistency:.6f}")
        print(f"    ι²_c                    : {iota_sq_c:.6f}")
        print(f"    Relative error          : "
              f"{abs(consistency - iota_sq_c)/iota_sq_c:.2e}"
              if iota_sq_c > 1e-30 else "    ι²_c too small to check")
        print(f"{'='*60}\n")

    return per_streamtube, global_berg

def compute_all_berg(
    network,
    body_array,
    *,
    area_method='vox_projection',
    sigma=1.0,
    direction='x',
    delta_V=1.0,
    voxel_size=1.0,
    streamtube_method='flow_decomposition',
    verbose=True,
):
    """
    Run conductances -> solve -> streamtubes -> iota^2 -> Berg quantities and
    return everything in one dict.  `network` must already have effective
    properties calculated.
    """

    # ── 1. Flat arrays ───────────────────────────────────────────────────────
    arrays = build_network_arrays(network, area_method=area_method)

    # ── 2. Conductances ──────────────────────────────────────────────────────
    G, conductance_data = calculate_conductances(
        arrays, sigma=sigma, verbose=verbose)

    # ── 3. Inlet / outlet bodies ─────────────────────────────────────────────
    inlet_indices, outlet_indices = get_inlet_outlet_indices(
        arrays, body_array, direction=direction)
    if verbose:
        print(f"Inlet  bodies : {len(inlet_indices)}")
        print(f"Outlet bodies : {len(outlet_indices)}")

    # ── 4. Solve potential field ─────────────────────────────────────────────
    potential, current, solution_data = solve_potential_field(
        arrays, G,
        inlet_indices=inlet_indices,
        outlet_indices=outlet_indices,
        conducting_mask=None,
        delta_V=delta_V,
        verbose=verbose)

    # ── 4b. Current threshold (same recipe as your driver) ───────────────────
    I_total = abs(solution_data['total_current'])
    I_max = np.max(np.abs(current))
    I_threshold = min(1e-10 * I_total, 1e-10 * I_max,1e-10)
    print(f"Current threshold I_threshold = {I_threshold:.3e}")

    # ── 5. Sample length in flow direction ───────────────────────────────────
    delta_s = body_array.shape[DIRECTION_TO_AXIS[direction]] * voxel_size
    if verbose:
        print(f"Sample length delta_s = {delta_s}")

    # ── 6. Streamtubes ───────────────────────────────────────────────────────
    streamtubes, Omega_c, V_stuck_total, G_directed, edge_to_throat = \
        build_streamtubes(
            arrays, current, potential, solution_data,
            I_threshold=I_threshold, method=streamtube_method, verbose=verbose)

    # ── 7. Segment interface potentials ──────────────────────────────────────
    Phi_1t, Phi_2t = recover_segment_potentials(
        arrays, potential, current, conductance_data)

    # ── 8. Local iota^2 and global iota^2_g ──────────────────────────────────
    iota_sq, iota_sq_g, Omega, iota_extra = compute_local_iota_squared(
        arrays, potential, Phi_1t, Phi_2t,
        delta_V=delta_V, delta_s=delta_s,
        current=current, current_threshold=I_threshold, verbose=verbose)

    Omega_c_direct = iota_extra['Omega_c']
    iota_sq_c_direct = iota_extra['iota_sq_c']

    # ── 9. Per-streamtube and global Berg quantities ─────────────────────────
    per_streamtube, global_berg = compute_streamtube_properties(
        streamtubes, arrays, iota_sq, Phi_1t, Phi_2t, potential,
        delta_V=delta_V, delta_s=delta_s,
        I_total=solution_data['total_current'], Omega_c=Omega_c, verbose=verbose)

    # ── 10. Final summary ────────────────────────────────────────────────────
    V_total = float(body_array.shape[0] * body_array.shape[1] * body_array.shape[2])
    phi_total = Omega / V_total
    phi_c = Omega_c_direct / V_total
    F_total = 1.0 / (iota_sq_g * phi_total)
    F_c = 1.0 / (iota_sq_c_direct * phi_c)

    if verbose:
        w = 60
        print(f"\n{'='*w}")
        print(f"{'FINAL BERG RESULTS':^{w}}")
        print(f"{'='*w}")
        print(f"  V_total                     : {V_total:.0f} vox^3")
        print(f"  Omega (total network vol)   : {Omega:.6g} vox^3")
        print(f"  Omega_c (conducting vol)    : {Omega_c:.6g} vox^3")
        print(f"  phi (total porosity)        : {phi_total:.4f}")
        print(f"  phi_c (conducting porosity) : {phi_c:.4f}")
        print(f"  iota^2_g (global)           : {iota_sq_g:.6f}")
        print(f"  iota^2_c (conducting)       : {global_berg['iota_sq_c']:.6f}")
        print(f"  tau^2_c (tortuosity^2)      : {global_berg['tau_sq_c']:.6f}")
        print(f"  tau_c   (tortuosity)        : {np.sqrt(global_berg['tau_sq_c']):.6f}")
        print(f"  C_c     (constriction)      : {global_berg['C_c']:.6f}")
        print(f"  F = 1/(iota^2_g * phi)      : {F_total:.4f}")
        print(f"  F = 1/(iota^2_c * phi_c)    : {F_c:.4f}")
        print(f"  check tau^2_c/C_c ~ iota^2_c: {global_berg['consistency']:.6f}")
        print(f"{'='*w}")

    return {
        'arrays':            arrays,
        'G':                 G,
        'conductance_data':  conductance_data,
        'inlet_indices':     inlet_indices,
        'outlet_indices':    outlet_indices,
        'potential':         potential,
        'current':           current,
        'solution_data':     solution_data,
        'I_threshold':       I_threshold,
        'delta_s':           delta_s,
        'streamtubes':       streamtubes,
        'Omega_c':           Omega_c,
        'V_stuck_total':     V_stuck_total,
        'edge_to_throat':    edge_to_throat,
        'Phi_1t':            Phi_1t,
        'Phi_2t':            Phi_2t,
        'iota_sq':           iota_sq,
        'iota_sq_g':         iota_sq_g,
        'Omega':             Omega,
        'iota_extra':        iota_extra,
        'per_streamtube':    per_streamtube,
        'global_berg':       global_berg,
        'phi_total':         phi_total,
        'phi_c':             phi_c,
        'F_total':           F_total,
        'F_c':               F_c,
    }