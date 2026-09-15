"""
particulate_claude.py
======================

Improved drop-in replacement for particulate.py.

Public API preserved (so berg_cc.py and your driver script need no changes):

    extract_throats_from_bodies_voxel_parallel(body_array, min_throat_size=10,
                                               dilate_both=True, n_jobs=1, ...)
    extract_throats_from_bodies_voxel(body_array, ...)        # serial alias
    calculate_network_effective_properties(network, area_method='vox_projection',
                                           alpha=0.5, split_volume_equal=True)
    PoreBody, PoreThroat, BodyThroatConnection, PoreNetwork

Key improvements over the original
----------------------------------
1.  Throat extraction is no longer O(N_bodies^2) over the full volume.
    * Candidate body pairs are pruned with a vectorised bounding-box
      (AABB) overlap test  -> non-adjacent pairs are discarded with zero
      array work.
    * Each surviving pair is processed on a *cropped* sub-volume (the union
      of the two padded bounding boxes) instead of the full domain, so the
      dilation / boolean ANDs run on ~20^3 voxels instead of ~100^3.
    The voxel-projection area method is preserved exactly.

2.  Shapely "LinearRing" crash fixed.  `Polygon(point_cloud)` (wrong API)
    is replaced by `MultiPoint(point_cloud).convex_hull` *per voxel*.  The
    overall footprint is still a `unary_union` of per-voxel polygons, so
    non-convex / staircase footprints are measured correctly (NOT a global
    convex hull).  If the union itself ever raises, the fallback is the sum
    of per-voxel areas, not a global hull.

3.  `estimate_plane_normal` uses an O(M) SVD instead of O(M^2) pairwise
    cross products.

4.  Local-area sampling (unused downstream) removed -> no repeated
    `unary_union` calls per throat.

5.  `get_new_axis` degenerate-basis bug fixed (abs() guard) -> no NaN areas
    for normals pointing along -x.

6.  Serial and parallel extraction share one code path and one network
    assembler (no more duplicated 150-line blocks).

7.  `alpha` is now actually used inside the body-side effective-length
    formula (was silently hard-coded to 0.5).

8.  n_jobs control:  -1 = all cores, 1 = single core (default),
    k>1 = k cores (clamped to cpu_count).

9.  compute_surface_mask takes an explicit `axis` so surface detection can
    follow the chosen flow direction.
"""

import time
import multiprocessing as mp
from itertools import combinations  # kept for backward-compat imports; not used internally
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import find_boundaries
from shapely import MultiPoint, unary_union
import networkx as nx

# Single source of truth for the flow-axis <-> direction-letter convention
# shared by particulate_claude.py, berg_cc.py and berg_voxel_fast.py: axis 0
# is x, 1 is y, 2 is z. Every function in this module that used to take a
# bare `surface_axis`/`axis` int now takes `direction` and derives the axis
# from this mapping internally, so the same call gives the same result
# whether it's made directly or through pipeline.run_microstructure.
AXIS_TO_DIRECTION = {0: 'x', 1: 'y', 2: 'z'}
DIRECTION_TO_AXIS = {v: k for k, v in AXIS_TO_DIRECTION.items()}


def _resolve_axis(direction=None, axis=None):
    """Resolve a single flow axis (int) from `direction` and/or `axis`.

    `direction` is the primary input. `axis` is accepted only for backward
    compatibility / explicit override; if both are given they must agree.
    """
    if direction is None and axis is None:
        return 0
    if direction is not None:
        if direction not in DIRECTION_TO_AXIS:
            raise ValueError(f"direction must be one of {sorted(DIRECTION_TO_AXIS)}, got {direction!r}")
        resolved = DIRECTION_TO_AXIS[direction]
        if axis is not None and axis != resolved:
            raise ValueError(
                f"direction={direction!r} (axis {resolved}) conflicts with "
                f"explicit axis={axis}; pass only one, or matching values.")
        return resolved
    if axis not in AXIS_TO_DIRECTION:
        raise ValueError(f"axis must be one of {sorted(AXIS_TO_DIRECTION)}, got {axis!r}")
    return axis

# scipy's binary_dilation with its default structure is identical to
# skimage.morphology.binary_dilation's default footprint (connectivity-1
# cross), but has a stable API (skimage's is deprecated in 0.26).
_binary_dilation = ndi.binary_dilation

try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except Exception:                                    # pragma: no cover
    _HAVE_TQDM = False

    def tqdm(it, **kwargs):                           # type: ignore
        return it


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SPHERE_VOLUME_CONSTANT = 3.0 / (4.0 * np.pi)
DEFAULT_ALPHA = 0.5
# A_tk / A_itk above this triggers the uniform-pipe (equal-thirds) fallback.
# NOTE: Berg's effective-length formula d_eff = d*(1 - alpha*sqrt(A_tk/A_itk))
# only goes negative for ratio > (1/alpha)^2 (= 4 when alpha = 0.5), so this
# default (~1.587) is conservative.  Raise it toward (1/alpha)^2 if you want
# fewer throats on the fallback path.
RATIO_THRESHOLD = 4 ** (1 / 3)

# 8 corner offsets of a unit voxel, used by the voxel-projection area method.
_CUBE_CORNERS = np.array(
    [[dx, dy, dz] for dx in (0.0, 1.0) for dy in (0.0, 1.0) for dz in (0.0, 1.0)],
    dtype=float,
)


# ──────────────────────────────────────────────────────────────────────────────
# StepTimer
# ──────────────────────────────────────────────────────────────────────────────
class StepTimer:
    def __init__(self, logger=None, enabled=True):
        self.enabled = enabled
        self.logger = logger
        self.t_start = time.perf_counter()
        self.t_prev = self.t_start

    def log(self, msg: str):
        if not self.enabled:
            return
        t_now = time.perf_counter()
        dt = t_now - self.t_prev
        t_cum = t_now - self.t_start
        self.t_prev = t_now
        formatted = f"[+{dt:8.3f} s | Total: {t_cum:8.3f} s] {msg}"
        print(formatted, flush=True)
        if self.logger is not None:
            self.logger.info(formatted)


# ──────────────────────────────────────────────────────────────────────────────
# Basic voxel fields
# ──────────────────────────────────────────────────────────────────────────────
def compute_pore_mask(body_array):
    """Boolean mask, True where pore voxels are (labels > 0)."""
    return body_array > 0


def compute_distance_transform(pore_mask):
    """EDT distance transform (in voxels)."""
    return ndi.distance_transform_edt(pore_mask)


def compute_boundaries(body_array):
    """Inner boundaries between labelled regions (candidate throat sites)."""
    return find_boundaries(body_array, mode='inner')


def get_unique_bodies(body_array):
    """Sorted unique body labels (> 0)."""
    labels = np.unique(body_array)
    return labels[labels > 0]


def compute_surface_mask(shape, axis=0):
    """
    Boolean mask, True on the two external faces orthogonal to `axis`.

    Parameters
    ----------
    shape : tuple
    axis : int
        Flow axis (0, 1 or 2).  The two faces at index 0 and -1 along this
        axis are marked as the external (inlet/outlet) surfaces.
    """
    surface_mask = np.zeros(shape, dtype=bool)
    sl_lo = [slice(None)] * len(shape)
    sl_hi = [slice(None)] * len(shape)
    sl_lo[axis] = 0
    sl_hi[axis] = shape[axis] - 1
    surface_mask[tuple(sl_lo)] = True
    surface_mask[tuple(sl_hi)] = True
    return surface_mask

def merge_boundary_only_bodies(body_array, axis=0, verbose=True):
    """
    After SNOW watershed, some bodies touch the inlet or outlet face but
    have no voxel neighbours that are interior bodies — they only neighbour
    other face bodies. These will become dead ends in the network because
    the Dirichlet BC short-circuits all face bodies to the same potential.

    This function relabels those bodies' voxels into their largest
    boundary neighbour, so they become part of one bigger inlet/outlet
    body instead of isolated dead ends.

    Parameters
    ----------
    body_array : 3D int array
        Labeled watershed output from SNOW. 0 = non-pore.
    axis : int
        Flow axis (0=x in your convention). Inlet face = index 0,
        outlet face = index -1 along this axis.
    verbose : bool

    Returns
    -------
    merged_array : 3D int array
        Relabeled body array with boundary-only bodies merged.
    n_merged : int
        Number of bodies that were merged.
    merged_ids : list
        Original label IDs that were merged away.
    """
    from skimage import measure
    merged_array = body_array.copy()

    # ── Face masks ───────────────────────────────────────────────────────────
    sl_inlet  = [slice(None)] * 3;  sl_inlet[axis]  = 0
    sl_outlet = [slice(None)] * 3;  sl_outlet[axis] = -1
    inlet_face_mask  = np.zeros(body_array.shape, dtype=bool)
    outlet_face_mask = np.zeros(body_array.shape, dtype=bool)
    inlet_face_mask[tuple(sl_inlet)]   = True
    outlet_face_mask[tuple(sl_outlet)] = True
    face_mask = inlet_face_mask | outlet_face_mask

    unique_labels = np.unique(body_array)
    unique_labels = unique_labels[unique_labels > 0]

    # ── Identify face-touching bodies ────────────────────────────────────────
    inlet_bodies  = set()
    outlet_bodies = set()
    for lab in unique_labels:
        body_mask = (body_array == lab)
        if np.any(body_mask & inlet_face_mask):
            inlet_bodies.add(lab)
        if np.any(body_mask & outlet_face_mask):
            outlet_bodies.add(lab)

    face_bodies = inlet_bodies | outlet_bodies

    # ── For each face body, check if it has any interior voxel neighbours ────
    # A voxel neighbour is interior if it belongs to a body NOT in face_bodies.
    # We use 6-connectivity dilation to find voxel-level neighbours.
    import scipy.ndimage as ndi
    struct = ndi.generate_binary_structure(3, 1)   # 6-connectivity

    boundary_only = []   # bodies to merge away
    for lab in face_bodies:
        body_mask = (body_array == lab)
        # Dilate by 1 voxel to find all neighbouring voxels
        dilated   = ndi.binary_dilation(body_mask, structure=struct)
        neighbour_voxels = dilated & ~body_mask & (body_array > 0)
        neighbour_labels = set(np.unique(body_array[neighbour_voxels])) - {0, lab}
        # Interior neighbours = neighbours not on any face
        interior_neighbours = neighbour_labels - face_bodies
        if len(interior_neighbours) == 0:
            boundary_only.append(lab)

    if verbose:
        print(f"\nBoundary-only body detection (axis={axis}):")
        print(f"  Total face bodies        : {len(face_bodies)}")
        print(f"  Inlet face bodies        : {len(inlet_bodies)}")
        print(f"  Outlet face bodies       : {len(outlet_bodies)}")
        print(f"  Boundary-only (no interior neighbour): {len(boundary_only)}")

    if len(boundary_only) == 0:
        if verbose:
            print("  Nothing to merge.")
        return merged_array, 0, []

    # ── Merge each boundary-only body into its largest face neighbour ─────────
    merged_ids = []
    for lab in boundary_only:
        body_mask = (body_array == lab)
        dilated   = ndi.binary_dilation(body_mask, structure=struct)
        neighbour_voxels  = dilated & ~body_mask & (body_array > 0)
        neighbour_labels  = set(np.unique(body_array[neighbour_voxels])) - {0, lab}
        face_neighbours   = neighbour_labels & face_bodies - {lab}

        if len(face_neighbours) == 0:
            # Completely isolated on the face — skip, will be caught as
            # isolated body later
            continue

        # Merge into the largest face neighbour by voxel count
        target = max(face_neighbours,
                     key=lambda l: int(np.sum(merged_array == l)))
        merged_array[body_mask] = target
        merged_ids.append(lab)

    if verbose:
        print(f"  Bodies merged            : {len(merged_ids)}")
        v_before = int(np.sum(body_array > 0))
        v_after  = int(np.sum(merged_array > 0))
        print(f"  Pore voxels before       : {v_before}")
        print(f"  Pore voxels after        : {v_after}  (should be same)")
        # Report new label counts
        n_before = len(unique_labels)
        n_after  = len(np.unique(merged_array)) - 1  # exclude 0
        print(f"  Distinct bodies before   : {n_before}")
        print(f"  Distinct bodies after    : {n_after}")

    return merged_array, len(merged_ids), merged_ids


def merge_boundary_only_bodies_iterative(body_array, axis=0, verbose=True):
    """
    Iterative version: repeat merging until no more boundary-only bodies exist.
    Each pass can expose new boundary-only bodies (chains of dead ends).
    """
    import scipy.ndimage as ndi
    struct = ndi.generate_binary_structure(3, 1)

    merged_array  = body_array.copy()
    total_merged  = 0
    all_merged_ids = []
    iteration     = 0

    while True:
        iteration += 1

        sl_inlet  = [slice(None)] * 3; sl_inlet[axis]  = 0
        sl_outlet = [slice(None)] * 3; sl_outlet[axis] = -1
        inlet_face_mask  = np.zeros(merged_array.shape, dtype=bool)
        outlet_face_mask = np.zeros(merged_array.shape, dtype=bool)
        inlet_face_mask[tuple(sl_inlet)]   = True
        outlet_face_mask[tuple(sl_outlet)] = True

        unique_labels = np.unique(merged_array)
        unique_labels = unique_labels[unique_labels > 0]

        inlet_bodies  = set()
        outlet_bodies = set()
        for lab in unique_labels:
            body_mask = (merged_array == lab)
            if np.any(body_mask & inlet_face_mask):
                inlet_bodies.add(lab)
            if np.any(body_mask & outlet_face_mask):
                outlet_bodies.add(lab)
        face_bodies = inlet_bodies | outlet_bodies

        boundary_only = []
        for lab in face_bodies:
            body_mask        = (merged_array == lab)
            dilated          = ndi.binary_dilation(body_mask, structure=struct)
            neighbour_voxels = dilated & ~body_mask & (merged_array > 0)
            neighbour_labels = set(np.unique(merged_array[neighbour_voxels])) - {0, lab}
            interior_neighbours = neighbour_labels - face_bodies
            if len(interior_neighbours) == 0:
                boundary_only.append(lab)

        if verbose:
            print(f"  Iteration {iteration}: {len(boundary_only)} boundary-only bodies")

        if len(boundary_only) == 0:
            break

        merged_this_pass = []
        for lab in boundary_only:
            body_mask        = (merged_array == lab)
            dilated          = ndi.binary_dilation(body_mask, structure=struct)
            neighbour_voxels = dilated & ~body_mask & (merged_array > 0)
            neighbour_labels = set(np.unique(merged_array[neighbour_voxels])) - {0, lab}
            face_neighbours  = neighbour_labels & face_bodies - {lab}

            if len(face_neighbours) == 0:
                continue

            target = max(face_neighbours,
                         key=lambda l: int(np.sum(merged_array == l)))
            merged_array[body_mask] = target
            merged_this_pass.append(lab)

        total_merged   += len(merged_this_pass)
        all_merged_ids += merged_this_pass

    if verbose:
        n_before = len(np.unique(body_array))  - 1
        n_after  = len(np.unique(merged_array)) - 1
        print(f"\nIterative boundary merge complete:")
        print(f"  Total iterations         : {iteration-1}")
        print(f"  Total bodies merged      : {total_merged}")
        print(f"  Distinct bodies before   : {n_before}")
        print(f"  Distinct bodies after    : {n_after}")
        v_before = int(np.sum(body_array   > 0))
        v_after  = int(np.sum(merged_array > 0))
        print(f"  Pore voxels before       : {v_before}")
        print(f"  Pore voxels after        : {v_after}  (should be same)")

    return merged_array, total_merged, all_merged_ids

def merge_short_circuit_bodies(body_array, axis=0, verbose=True):
    """
    After the surface merge, find any body that:
      - connects to 2+ inlet bodies (or 2+ outlet bodies)
      - has NO interior neighbours
    
    These bodies will always carry zero current (short-circuited between
    same-potential nodes). Merge them into their largest boundary neighbour.
    
    Run iteratively because merging one body can expose another.
    """
    import scipy.ndimage as ndi
    struct = ndi.generate_binary_structure(3, 1)

    merged_array = body_array.copy()
    total_merged = 0

    # ── Face masks ────────────────────────────────────────────────────────────
    sl_inlet  = [slice(None)] * 3; sl_inlet[axis]  = 0
    sl_outlet = [slice(None)] * 3; sl_outlet[axis] = -1
    inlet_face_mask  = np.zeros(body_array.shape, dtype=bool)
    outlet_face_mask = np.zeros(body_array.shape, dtype=bool)
    inlet_face_mask[tuple(sl_inlet)]   = True
    outlet_face_mask[tuple(sl_outlet)] = True

    changed   = True
    iteration = 0

    while changed:
        changed   = False
        iteration += 1

        unique_labels = np.unique(merged_array)
        unique_labels = unique_labels[unique_labels > 0]

        # Recompute face bodies and neighbours each iteration
        # since merging changes the array
        inlet_bodies  = set()
        outlet_bodies = set()
        for lab in unique_labels:
            body_mask = (merged_array == lab)
            if np.any(body_mask & inlet_face_mask):
                inlet_bodies.add(lab)
            if np.any(body_mask & outlet_face_mask):
                outlet_bodies.add(lab)
        boundary_bodies = inlet_bodies | outlet_bodies

        neighbours = {}
        for lab in unique_labels:
            body_mask        = (merged_array == lab)
            dilated          = ndi.binary_dilation(body_mask, structure=struct)
            nbr_vox          = dilated & ~body_mask & (merged_array > 0)
            neighbours[lab]  = set(np.unique(merged_array[nbr_vox])) - {0, lab}

        merged_this_iter = []

        for lab in unique_labels:
            if lab in boundary_bodies:
                continue

            nbrs          = neighbours[lab]
            inlet_nbrs    = nbrs & inlet_bodies
            outlet_nbrs   = nbrs & outlet_bodies
            interior_nbrs = nbrs - boundary_bodies

            # Short-circuit: connected to 2+ inlet bodies only
            # OR connected to 2+ outlet bodies only
            # OR connected to 1+ inlet AND 1+ outlet (also zero net current)
            # AND has no interior neighbours
            is_short_circuit = (
                len(interior_nbrs) == 0 and (
                    len(inlet_nbrs)  >= 2 or
                    len(outlet_nbrs) >= 2 or
                    (len(inlet_nbrs) >= 1 and len(outlet_nbrs) >= 1)
                )
            )

            # Also catch: connected to only 1 boundary body with no interior
            # (these are the chain cases — one hop from boundary, no other exit)
            is_dead_end = (
                len(interior_nbrs) == 0 and
                len(nbrs) > 0 and
                len(nbrs) == len(boundary_bodies & nbrs)
            )

            if is_short_circuit or is_dead_end:
                # Merge into largest boundary neighbour
                boundary_nbrs = nbrs & boundary_bodies
                if len(boundary_nbrs) == 0:
                    continue
                target = max(boundary_nbrs,
                             key=lambda l: int(np.sum(merged_array == l)))
                merged_array[merged_array == lab] = target
                merged_this_iter.append(lab)
                changed = True

        total_merged += len(merged_this_iter)

        if verbose and len(merged_this_iter) > 0:
            print(f"  Iteration {iteration}: merged {len(merged_this_iter)} bodies "
                  f"→ {merged_this_iter}")

    if verbose:
        n_before = len(np.unique(body_array))   - 1
        n_after  = len(np.unique(merged_array)) - 1
        v_before = int(np.sum(body_array   > 0))
        v_after  = int(np.sum(merged_array > 0))
        print(f"\nShort-circuit merge complete:")
        print(f"  Total iterations       : {iteration - 1}")
        print(f"  Total bodies merged    : {total_merged}")
        print(f"  Distinct bodies before : {n_before}")
        print(f"  Distinct bodies after  : {n_after}")
        print(f"  Pore voxels before     : {v_before}")
        print(f"  Pore voxels after      : {v_after}  (should be same)")

    return merged_array, total_merged



def compute_initial_body_props(body_array, distance, unique_bodies, direction=None, surface_axis=None):
    """
    Raw per-body properties (volume, centroids, max inscribed radius, surface
    info).  Volumes in voxel counts, centroids in voxel (array-index) order.

    `direction` ('x'/'y'/'z') is the flow direction; the two faces
    orthogonal to it are the external (inlet/outlet) surfaces used to flag
    is_surface. `surface_axis` (int, 0/1/2) is accepted only as a legacy
    override and must agree with `direction` if both are given -- both
    default to axis 0 / 'x' if neither is passed.
    """
    axis = _resolve_axis(direction, surface_axis)
    surface_mask = compute_surface_mask(body_array.shape, axis=axis)

    body_props = {}
    for body_id in unique_bodies:
        body_mask = (body_array == body_id)
        volume = int(np.sum(body_mask))
        if volume == 0:
            continue

        volume_centroid = ndi.center_of_mass(body_mask)
        max_radius = float(np.max(distance[body_mask]))
        equiv_radius = (SPHERE_VOLUME_CONSTANT * volume) ** (1.0 / 3.0)

        surface_overlap = body_mask & surface_mask
        surface_area_voxels = int(surface_overlap.sum())
        is_surface = surface_area_voxels > 0

        if is_surface:
            surface_centroid = ndi.center_of_mass(surface_overlap)
            nodal_state = 1
            centroid = surface_centroid
        else:
            surface_centroid = None
            nodal_state = 0
            centroid = volume_centroid

        body_props[int(body_id)] = {
            'nodal_state':         nodal_state,
            'body_id':             int(body_id),
            'volume_voxels':       volume,
            'equivalent_radius':   equiv_radius,
            'max_radius':          max_radius,
            'centroid':            centroid,
            'volume_centroid':     volume_centroid,
            'surface_centroid':    surface_centroid,
            'is_surface':          is_surface,
            'surface_area_voxels': surface_area_voxels,
            'connections':         [],
        }

    return body_props


# ──────────────────────────────────────────────────────────────────────────────
# Geometry: plane normal, projection, planar area
# ──────────────────────────────────────────────────────────────────────────────
def estimate_plane_normal(throat_coords, centroid=None):
    """
    Best-fit plane normal of a voxel cloud via SVD (O(M)).

    The normal is the singular vector with the smallest singular value of the
    mean-centred coordinates.  Replaces the original O(M^2) pairwise
    cross-product scheme.
    """
    coords = np.asarray(throat_coords, dtype=float)
    if len(coords) < 3:
        return np.array([0.0, 0.0, 1.0])

    c = coords.mean(axis=0) if centroid is None else np.asarray(centroid, dtype=float)
    r = coords - c
    # full_matrices=False keeps it cheap for tall (M,3) matrices
    _, _, vt = np.linalg.svd(r, full_matrices=False)
    normal = vt[-1]
    nrm = np.linalg.norm(normal)
    if nrm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return normal / nrm


def get_new_axis(plane_normal):
    """
    Orthonormal in-plane basis (3x2 matrix, columns e1, e2) for the plane with
    the given normal.  Uses abs() so a normal pointing along -x does not
    collapse the basis (the original `< 0.9` test produced a zero vector).
    """
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    if abs(np.dot(plane_normal, np.array([1.0, 0.0, 0.0]))) < 0.9:
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = np.array([0.0, 1.0, 0.0])

    e1 = a - np.dot(a, plane_normal) * plane_normal
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(plane_normal, e1)
    e2 = e2 / np.linalg.norm(e2)
    return np.column_stack([e1, e2])          # (3, 2)


def get_projected_coords(coords, plane_normal, centroid):
    """Project 3-D coords onto the plane through `centroid` with `plane_normal`."""
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    n_nT = np.outer(plane_normal, plane_normal)
    projection_matrix = np.eye(3) - n_nT
    return coords @ projection_matrix.T + n_nT @ np.asarray(centroid, dtype=float)


def get_planar_area(coords, plane_normal, centroid):
    """
    In-plane footprint area of a set of voxels, projected onto the throat
    plane.

    Each voxel contributes the convex hull of its 8 projected corners (always
    convex), and the footprint is the `unary_union` of those per-voxel hulls.
    This measures NON-CONVEX / staircase footprints correctly — it is not a
    single global convex hull.

    Fallback (only if the union raises): sum of per-voxel hull areas — a tight
    upper bound that ignores the small overlaps between neighbouring squares.
    """
    coords = np.unique(np.asarray(coords, dtype=float), axis=0)
    n = len(coords)
    if n == 0:
        return 0.0

    new_axis = get_new_axis(plane_normal)                       # (3, 2)

    # Vectorised projection of all 8*n corners at once.
    ends = coords[:, None, :] + _CUBE_CORNERS[None, :, :]        # (n, 8, 3)
    flat = ends.reshape(-1, 3)                                  # (8n, 3)
    proj = get_projected_coords(flat, plane_normal, centroid)   # (8n, 3)
    proj2d = (proj @ new_axis).reshape(n, 8, 2)                 # (n, 8, 2)

    polys = [MultiPoint(proj2d[k]).convex_hull for k in range(n)]
    try:
        return float(unary_union(polys).area)
    except Exception:
        return float(sum(p.area for p in polys))


def compute_throat_areas(throat_coords, throat_centroid, flow_direction, voxel_size=1.0):
    """
    Throat area metrics.

    Returns a dict with:
        naive_area           : voxel count * face area
        naive_projected      : naive_area projected onto the flow direction
        vox_projection_area  : in-plane footprint area (voxel-projection method)
        plane_normal         : best-fit throat-plane normal
        num_voxels, cos_theta
    """
    num_voxels = len(throat_coords)
    face_area = voxel_size ** 2

    if num_voxels < 3:
        return {
            'naive_area':          num_voxels * face_area,
            'naive_projected':     0.0,
            'vox_projection_area': 0.0,
            'plane_normal':        np.array([0.0, 0.0, 1.0]),
            'num_voxels':          num_voxels,
            'cos_theta':           0.0,
        }

    coords = np.asarray(throat_coords, dtype=float) * voxel_size
    throat_normal = estimate_plane_normal(coords, centroid=throat_centroid)

    flow_norm = flow_direction / np.linalg.norm(flow_direction)
    naive_area = num_voxels * face_area
    cos_theta = float(np.abs(np.dot(throat_normal, flow_norm)))
    naive_projected = naive_area * cos_theta
    vox_projection_area = get_planar_area(coords, throat_normal, throat_centroid)

    return {
        'naive_area':          naive_area,
        'naive_projected':     naive_projected,
        'vox_projection_area': vox_projection_area,
        'plane_normal':        throat_normal,
        'num_voxels':          num_voxels,
        'cos_theta':           cos_theta,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PoreBody:
    """A single pore body. Lengths in voxels, volumes in voxel counts,
    centroids in voxel (array-index) order."""
    body_id: int
    nodal_state: int                       # 0 = interior, 1 = surface, 2 = throat
    volume_voxels: int
    equivalent_radius: float
    max_radius: float
    centroid: Tuple[float, float, float]
    volume_centroid: Tuple[float, float, float]
    surface_centroid: Optional[Tuple[float, float, float]] = None
    is_surface: bool = False
    surface_area_voxels: int = 0
    connected_bodies: List[int] = field(default_factory=list)
    connected_throats: List[int] = field(default_factory=list)
    connections_to_throats: Dict[int, Any] = field(default_factory=dict)

    def __repr__(self):
        return (f"PoreBody(id={self.body_id}, volume={self.volume_voxels}, "
                f"connections={len(self.connected_bodies)}, surface={self.is_surface})")

    @property
    def coordination_number(self) -> int:
        return len(self.connected_bodies)

    @property
    def num_throats(self) -> int:
        return len(self.connected_throats)

    def get_connection_to_throat(self, throat_id: int) -> Optional[Any]:
        return self.connections_to_throats.get(throat_id)

    def has_effective_properties(self) -> bool:
        return len(self.connections_to_throats) > 0


@dataclass
class PoreThroat:
    """A single pore throat connecting two bodies. Lengths in voxels,
    volumes in voxel counts, areas in voxel^2."""
    throat_id: int
    connects: Tuple[int, int]
    volume_voxels: int
    center: Optional[Tuple[float, float, float]] = None
    coords: Optional[np.ndarray] = None
    length_cc_voxels: float = 0.0
    length_via_ct_voxels: float = 0.0
    length_body1_to_ct_voxels: float = 0.0
    length_ct_to_body2_voxels: float = 0.0
    curvature_ratio: float = 1.0
    plane_normal: Optional[np.ndarray] = None
    max_radius_voxels: float = 0.0

    # Areas
    area_naive_voxels2: float = 0.0
    area_naive_projected_voxels2: float = 0.0
    area_vox_projection_voxels2: float = 0.0

    # Effective properties (Berg)
    effective_length_to_body1: float = 0.0
    effective_length_to_body2: float = 0.0
    effective_length_total: float = 0.0
    effective_volume: float = 0.0
    is_fallback: bool = False
    geometric_throat_area: float = 0.0
    fallback_area: float = 0.0

    def __repr__(self):
        return (f"PoreThroat(id={self.throat_id}, connects={self.connects}, "
                f"volume={self.volume_voxels}, length={self.length_cc_voxels:.2f})")

    @property
    def body1_id(self) -> int:
        return self.connects[0]

    @property
    def body2_id(self) -> int:
        return self.connects[1]

    @property
    def has_curvature(self) -> bool:
        return self.curvature_ratio > 1.01

    @property
    def equivalent_radius(self) -> float:
        if self.length_cc_voxels > 0:
            return float(np.sqrt(self.volume_voxels / (np.pi * self.length_cc_voxels)))
        return 0.0

    @property
    def aspect_ratio(self) -> float:
        equiv_diameter = 2 * self.equivalent_radius
        return self.length_cc_voxels / equiv_diameter if equiv_diameter > 0 else 0.0

    def get_area(self, method: str = 'vox_projection') -> float:
        method_map = {
            'naive':           self.area_naive_voxels2,
            'naive_projected': self.area_naive_projected_voxels2,
            'vox_projection':  self.area_vox_projection_voxels2,
        }
        if method not in method_map:
            raise ValueError(f"Unknown method '{method}'. "
                             f"Choose from: {list(method_map.keys())}")
        return method_map[method]

    def has_effective_properties(self) -> bool:
        return self.effective_length_total > 0


@dataclass
class BodyThroatConnection:
    """Properties of the connection between a body and one of its throats.
    Implements Berg's effective geometry. All measurements in voxel units."""
    body_id: int
    throat_id: int
    distance: float                 # d_itk
    body_path_area: float           # A_itk
    throat_area: float              # A_tk
    effective_distance: float       # d_eff_itk
    effective_area: float           # A_eff_itk
    effective_volume: float         # V_eff_itk
    is_fallback: bool = False
    geometric_throat_area: float = 0.0
    alpha: float = DEFAULT_ALPHA

    def __repr__(self):
        return (f"Connection(body={self.body_id}, throat={self.throat_id}, "
                f"dist={self.distance:.2f}, area={self.body_path_area:.2f})")

    @property
    def area_ratio(self) -> float:
        return self.throat_area / self.body_path_area if self.body_path_area > 0 else 0.0

    @property
    def length_reduction_ratio(self) -> float:
        return self.effective_distance / self.distance if self.distance > 0 else 1.0

    def recalculate_effective_properties(self, alpha: float = DEFAULT_ALPHA):
        self.alpha = alpha
        if self.body_path_area > 0:
            sqrt_ratio = np.sqrt(self.throat_area / self.body_path_area)
            self.effective_distance = self.distance * (1.0 - alpha * sqrt_ratio)
        else:
            self.effective_distance = self.distance
        if self.effective_distance > 0:
            numerator = (self.body_path_area * self.distance -
                         self.throat_area * (self.distance - self.effective_distance))
            self.effective_area = numerator / self.effective_distance
        else:
            self.effective_area = self.body_path_area
        self.effective_volume = self.effective_distance * self.effective_area


@dataclass
class PoreNetwork:
    """Container for all bodies, throats and metadata."""
    bodies: Dict[int, PoreBody] = field(default_factory=dict)
    throats: Dict[int, PoreThroat] = field(default_factory=dict)
    throat_array: Optional[np.ndarray] = None
    conservation_info: Dict[str, Any] = field(default_factory=dict)
    shape: Optional[Tuple[int, int, int]] = None
    voxel_size: float = 1.0

    def __repr__(self):
        return (f"PoreNetwork(bodies={len(self.bodies)}, "
                f"throats={len(self.throats)}, "
                f"total_volume={self.total_pore_volume} voxels, shape={self.shape})")

    @property
    def num_bodies(self) -> int:
        return len(self.bodies)

    @property
    def num_throats(self) -> int:
        return len(self.throats)

    @property
    def total_pore_volume(self) -> int:
        if self.conservation_info:
            return self.conservation_info.get('total_pore_voxels', 0)
        return sum(b.volume_voxels for b in self.bodies.values())

    @property
    def total_throat_volume(self) -> int:
        if self.conservation_info:
            return self.conservation_info.get('total_throat_voxels', 0)
        return sum(t.volume_voxels for t in self.throats.values())

    @property
    def porosity(self) -> float:
        if self.shape is not None:
            return self.total_pore_volume / float(np.prod(self.shape))
        return 0.0

    def get_body_property(self, property_name: str) -> np.ndarray:
        if not self.bodies:
            return np.array([])
        body_ids = sorted(self.bodies.keys())
        values = [getattr(self.bodies[bid], property_name) for bid in body_ids]
        return np.array(values)

    @property
    def body_ids(self) -> np.ndarray:
        return np.array(sorted(self.bodies.keys()))

    @property
    def body_volumes(self) -> np.ndarray:
        return self.get_body_property('volume_voxels')

    @property
    def body_centroids(self) -> np.ndarray:
        return self.get_body_property('centroid')

    @property
    def body_coordination_numbers(self) -> np.ndarray:
        body_ids = sorted(self.bodies.keys())
        return np.array([self.bodies[bid].coordination_number for bid in body_ids])

    def get_throat_property(self, property_name: str) -> np.ndarray:
        if not self.throats:
            return np.array([])
        throat_ids = sorted(self.throats.keys())
        values = [getattr(self.throats[tid], property_name) for tid in throat_ids]
        return np.array(values)


# ──────────────────────────────────────────────────────────────────────────────
# Candidate pair pruning  (bounding-box / AABB overlap)
# ──────────────────────────────────────────────────────────────────────────────
def find_candidate_pairs(body_array, unique_bodies, pad):
    """
    Return only the body pairs whose (padded) bounding boxes overlap — a
    superset of all pairs that can possibly share a throat.

    This replaces the O(N^2) full-volume pair scan with a single
    `ndi.find_objects` pass plus a vectorised AABB test.  Non-adjacent pairs
    are discarded with zero array work.

    Returns
    -------
    pairs : list[(int, int)]   body-label pairs with b1 < b2
    boxes : dict[int -> (lo, hi)]   padded bounding box per body label
    """
    objects = ndi.find_objects(body_array)
    shape = body_array.shape

    labels, los, his = [], [], []
    for lbl in unique_bodies:
        lbl = int(lbl)
        sl = objects[lbl - 1] if (lbl - 1) < len(objects) else None
        if sl is None:
            continue
        lo = [max(s.start - pad, 0) for s in sl]
        hi = [min(s.stop + pad, shape[d]) for d, s in enumerate(sl)]
        labels.append(lbl)
        los.append(lo)
        his.append(hi)

    labels = np.array(labels, dtype=int)
    los = np.array(los, dtype=int)
    his = np.array(his, dtype=int)
    M = len(labels)

    boxes = {int(labels[k]): (los[k], his[k]) for k in range(M)}
    if M < 2:
        return [], boxes

    # AABB overlap on every axis: lo_i < hi_j  AND  lo_j < hi_i.
    # lo_lt_hi[i, j] == all_axes(los[i] < his[j])
    # Memory: (M, M, 3) bool transient.  Fine up to a few thousand bodies.
    lo_lt_hi = (los[:, None, :] < his[None, :, :]).all(axis=2)     # (M, M)
    overlap = lo_lt_hi & lo_lt_hi.T

    iu = np.triu_indices(M, k=1)
    keep = overlap[iu]
    pi = labels[iu[0][keep]]
    pj = labels[iu[1][keep]]
    pairs = list(zip(pi.tolist(), pj.tolist()))
    return pairs, boxes

def pool_surface_volumes(body_props, candidate_pairs, verbose=True):
    """
    Redistributes the volume of isolated surface bodies into the 
    surface 'feeder' bodies that actively connect to the interior.
    """
    surface_to_surface_pairs = []
    feeders = set()
    
    # 1. Categorize connections
    for (b1, b2) in candidate_pairs:
        is_b1_surf = body_props[b1]['is_surface']
        is_b2_surf = body_props[b2]['is_surface']
        
        if is_b1_surf and is_b2_surf:
            surface_to_surface_pairs.append((b1, b2))
        else:
            if is_b1_surf: feeders.add(b1)
            if is_b2_surf: feeders.add(b2)

    # 2. Build graph and find puddles
    G_surface = nx.Graph()
    G_surface.add_edges_from(surface_to_surface_pairs)
    puddles = list(nx.connected_components(G_surface))
    
    # Tracking variables for the log
    total_rescued_volume = 0.0
    total_dead_ends = 0
    total_feeders_benefited = 0
    
    # 3. Redistribute volume
    for puddle in puddles:
        puddle_feeders = [b for b in puddle if b in feeders]
        puddle_dead_ends = [b for b in puddle if b not in feeders]
        
        if not puddle_feeders:
            continue
            
        pooled_volume = sum(body_props[b]['volume_voxels'] for b in puddle_dead_ends)
        
        if pooled_volume > 0:
            # Track stats
            total_rescued_volume += pooled_volume
            total_dead_ends += len(puddle_dead_ends)
            total_feeders_benefited += len(puddle_feeders)
            
            share = pooled_volume / len(puddle_feeders)
            
            for feeder in puddle_feeders:
                body_props[feeder]['volume_voxels'] += share
                
            for dead_end in puddle_dead_ends:
                body_props[dead_end]['volume_voxels'] = 0
                
    # 4. Print summary block
    if verbose and total_rescued_volume > 0:
        print(f"\n{'='*60}")
        print(f"{'SURFACE VOLUME POOLING (STRUCTURAL HEALING)':^60}")
        print(f"{'='*60}")
        print(f"  Dead-end surface bodies rescued : {total_dead_ends}")
        print(f"  Feeder surface bodies boosted   : {total_feeders_benefited}")
        print(f"  Total volume rescued            : {total_rescued_volume:.1f} vox³")
        print(f"{'='*60}\n")

    # 5. Return filtered pairs
    valid_interior_pairs = [pair for pair in candidate_pairs 
                            if pair not in surface_to_surface_pairs]
                            
    return body_props, valid_interior_pairs


# ──────────────────────────────────────────────────────────────────────────────
# Per-pair throat extraction (used by both serial and parallel paths)
# ──────────────────────────────────────────────────────────────────────────────
def _extract_one_pair(b1, b2, lo, hi, c1, c2,
                      body_array, distance, boundaries, pore_mask,
                      min_throat_size, dilate_both):
    """
    Extract the throat (if any) between bodies b1 and b2, working only on the
    cropped sub-volume defined by (lo, hi).  Coordinates are shifted back to
    the global frame so they line up with body centroids.

    Returns a throat-property dict, None (no throat), or
    ('__error__', b1, b2, msg) on failure.
    """
    try:
        sl = tuple(slice(int(lo[d]), int(hi[d])) for d in range(3))
        sub = body_array[sl]

        m1 = _binary_dilation(sub == b1)        # default footprint == original code
        sub_b = boundaries[sl]
        sub_p = pore_mask[sl]
        if dilate_both:
            throat = m1 & _binary_dilation(sub == b2) & sub_b & sub_p
        else:
            throat = m1 & (sub == b2) & sub_b & sub_p

        num_voxels = int(throat.sum())
        if num_voxels < min_throat_size:
            return None

        # Local -> global coordinates
        coords = np.argwhere(throat) + np.asarray(lo, dtype=int)
        throat_centroid = coords.mean(axis=0)

        flow_direction = np.asarray(c2, float) - np.asarray(c1, float)
        length_cc = float(np.linalg.norm(flow_direction))
        L1_ct = float(np.linalg.norm(throat_centroid - np.asarray(c1, float)))
        L2_ct = float(np.linalg.norm(np.asarray(c2, float) - throat_centroid))
        length_via_ct = L1_ct + L2_ct
        curvature_ratio = length_via_ct / length_cc if length_cc > 0 else 1.0

        area_results = compute_throat_areas(
            throat_coords=coords,
            throat_centroid=throat_centroid,
            flow_direction=flow_direction,
            voxel_size=1.0,
        )

        max_radius = float(distance[sl][throat].max())

        return {
            'connects':                   (int(b1), int(b2)),
            'volume_voxels':              num_voxels,
            'coords':                     coords,
            'center':                     throat_centroid,
            'length_cc_voxels':           length_cc,
            'length_via_ct_voxels':       length_via_ct,
            'length_body1_to_ct_voxels':  L1_ct,
            'length_ct_to_body2_voxels':  L2_ct,
            'curvature_ratio':            curvature_ratio,
            'max_radius_voxels':          max_radius,
            'area_naive_voxels2':         area_results['naive_area'],
            'area_naive_projected_voxels2': area_results['naive_projected'],
            'area_vox_projection_voxels2':  area_results['vox_projection_area'],
            'plane_normal':               area_results['plane_normal'],
        }
    except Exception as e:                       # surfaced as a count, not silent
        return ('__error__', int(b1), int(b2), repr(e))


# Worker-process state (populated once per worker via the initializer, so the
# large read-only arrays are pickled once per process — not once per task).
_WORKER_STATE: Dict[str, Any] = {}


def _init_worker(body_array, distance, boundaries, pore_mask,
                 min_throat_size, dilate_both):
    _WORKER_STATE['body_array'] = body_array
    _WORKER_STATE['distance'] = distance
    _WORKER_STATE['boundaries'] = boundaries
    _WORKER_STATE['pore_mask'] = pore_mask
    _WORKER_STATE['min_throat_size'] = min_throat_size
    _WORKER_STATE['dilate_both'] = dilate_both


def _extract_pair_task(task):
    b1, b2, lo, hi, c1, c2 = task
    S = _WORKER_STATE
    return _extract_one_pair(
        b1, b2, lo, hi, c1, c2,
        S['body_array'], S['distance'], S['boundaries'], S['pore_mask'],
        S['min_throat_size'], S['dilate_both'],
    )


def _resolve_n_jobs(n_jobs):
    """-1 -> all cores; 1 (or None/<1) -> single core; k -> min(k, cpu_count)."""
    n_cpu = mp.cpu_count()
    if n_jobs == -1:
        return n_cpu
    if n_jobs is None or n_jobs < 1:
        return 1
    return min(int(n_jobs), n_cpu)


# ──────────────────────────────────────────────────────────────────────────────
# Network assembly (shared by serial and parallel extraction)
# ──────────────────────────────────────────────────────────────────────────────
def _assemble_network(body_props, throat_props, body_array,
                      voxel_size=1.0, verbose=True):
    """Build a PoreNetwork from raw body_props and throat_props dicts."""
    # Connectivity
    for tid, p in throat_props.items():
        b1, b2 = p['connects']
        body_props[b1]['connections'].append(int(b2))
        body_props[b2]['connections'].append(int(b1))

    # Labelled throat array (paint in throat-id order)
    throat_array = np.zeros(body_array.shape, dtype=np.int32)
    overlap_voxels = 0
    for tid, p in throat_props.items():
        idx = tuple(np.asarray(p['coords']).T)
        overlap_voxels += int(np.count_nonzero(throat_array[idx]))
        throat_array[idx] = tid
    if verbose and overlap_voxels:
        print(f"[assemble] {overlap_voxels} throat voxel(s) shared between "
              f"throats (later id wins in the label array).")

    # Bodies
    bodies_dict = {}
    for bid, props in body_props.items():
        bodies_dict[int(bid)] = PoreBody(
            body_id=int(props['body_id']),
            nodal_state=int(props['nodal_state']),
            volume_voxels=int(props['volume_voxels']),
            equivalent_radius=float(props['equivalent_radius']),
            max_radius=float(props['max_radius']),
            centroid=tuple(props['centroid']),
            volume_centroid=tuple(props['volume_centroid']),
            surface_centroid=(tuple(props['surface_centroid'])
                              if props['surface_centroid'] is not None else None),
            is_surface=bool(props['is_surface']),
            surface_area_voxels=int(props['surface_area_voxels']),
            connected_bodies=list(props['connections']),
            connected_throats=[],
        )

    # Throats
    throats_dict = {}
    for tid, p in throat_props.items():
        throat = PoreThroat(
            throat_id=int(tid),
            connects=tuple(p['connects']),
            coords=p['coords'],
            plane_normal=p.get('plane_normal'),
            volume_voxels=int(p['volume_voxels']),
            center=tuple(p['center']) if p['center'] is not None else None,
            length_cc_voxels=float(p['length_cc_voxels']),
            length_via_ct_voxels=float(p['length_via_ct_voxels']),
            length_body1_to_ct_voxels=float(p['length_body1_to_ct_voxels']),
            length_ct_to_body2_voxels=float(p['length_ct_to_body2_voxels']),
            curvature_ratio=float(p['curvature_ratio']),
            max_radius_voxels=float(p['max_radius_voxels']),
            area_naive_voxels2=float(p['area_naive_voxels2']),
            area_naive_projected_voxels2=float(p['area_naive_projected_voxels2']),
            area_vox_projection_voxels2=float(p['area_vox_projection_voxels2']),
        )
        throats_dict[int(tid)] = throat
        b1, b2 = p['connects']
        bodies_dict[int(b1)].connected_throats.append(int(tid))
        bodies_dict[int(b2)].connected_throats.append(int(tid))

    total_body_voxels = int(sum(props['volume_voxels'] for props in body_props.values()))
    total_throat_voxels = int(sum(p['volume_voxels'] for p in throat_props.values()))
    conservation_info = {
        # Throats are a sub-partition of the body voxels (carved from
        # boundaries), NOT additive with bodies.  We therefore report them
        # separately rather than as a (meaningless) additive residual.
        'total_pore_voxels':   total_body_voxels,
        'total_body_voxels':   total_body_voxels,
        'total_throat_voxels': total_throat_voxels,
    }

    return PoreNetwork(
        bodies=bodies_dict,
        throats=throats_dict,
        throat_array=throat_array,
        conservation_info=conservation_info,
        shape=tuple(body_array.shape),
        voxel_size=voxel_size,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Unified throat extraction (serial when n_jobs == 1, else parallel)
# ──────────────────────────────────────────────────────────────────────────────
def extract_throats_from_bodies_voxel_parallel(
    body_array,
    min_throat_size=10,
    dilate_both=True,
    n_jobs=1,
    direction=None,
    surface_axis=None,
    skip_surface_surface=True,
    verbose=True,
):
    """
    Extract throats between pre-labelled pore bodies.

    Parameters
    ----------
    body_array : ndarray[int]
        Labelled bodies (0 = background, >0 = body label).
    min_throat_size : int
        Minimum throat voxel count to keep.
    dilate_both : bool
        If True, both bodies are dilated before intersecting (allows a small
        gap between bodies); if False, only body 1 is dilated.
    n_jobs : int
        -1 = all cores, 1 = single core (default), k = k cores (clamped to
        cpu_count).
    direction : str, optional
        Flow direction ('x'/'y'/'z'). The two faces orthogonal to it define
        the external (inlet/outlet) surfaces used for is_surface detection.
        This is the single source of truth for flow direction in this
        function; defaults to 'x' if neither this nor `surface_axis` is set.
    surface_axis : int, optional
        Legacy alias for `direction` as an int (0/1/2). Accepted only as an
        explicit override and must agree with `direction` if both are given.
    skip_surface_surface : bool
        If True, skip pairs where BOTH bodies touch a surface.  The old
        parallel code did this implicitly; the default here is False (do not
        skip), which is the more conservative / correct behaviour.  Set True
        to reproduce prior parallel results exactly.
    verbose : bool
        Print timing / progress.

    Returns
    -------
    PoreNetwork
    """
    timer = StepTimer(enabled=verbose)

    pore_mask = compute_pore_mask(body_array)
    distance = compute_distance_transform(pore_mask)
    boundaries = compute_boundaries(body_array)
    unique_bodies = get_unique_bodies(body_array)
    timer.log(f"Identified {len(unique_bodies)} unique bodies.")

    body_props = compute_initial_body_props(
        body_array, distance, unique_bodies, direction=direction, surface_axis=surface_axis)
    timer.log("Computed initial body properties.")

    # ── Candidate pairs via bounding-box overlap ──────────────────────────
    pad = 2 if dilate_both else 1
    pairs, boxes = find_candidate_pairs(body_array, unique_bodies, pad=pad)
    if skip_surface_surface:
        pairs = [(a, b) for (a, b) in pairs
                 if not (body_props[a]['is_surface'] and body_props[b]['is_surface'])]

    n_bodies = len(unique_bodies)
    n_all_pairs = n_bodies * (n_bodies - 1) // 2
    kept_pct = 100.0 * len(pairs) / max(n_all_pairs, 1)
    timer.log(f"Candidate pairs: {len(pairs)} / {n_all_pairs} "
              f"({kept_pct:.1f}% kept after bounding-box filter).")

    # ── Build tiny per-pair tasks (union crop box + centroids) ─────────────
    tasks = []
    for (a, b) in pairs:
        lo = np.minimum(boxes[a][0], boxes[b][0])
        hi = np.maximum(boxes[a][1], boxes[b][1])
        c1 = np.asarray(body_props[a]['centroid'], dtype=float)
        c2 = np.asarray(body_props[b]['centroid'], dtype=float)
        tasks.append((a, b, lo, hi, c1, c2))

    n_workers = _resolve_n_jobs(n_jobs)
    timer.log(f"Extracting throats from {len(tasks)} candidate pairs "
              f"with {n_workers} worker(s)...")

    # ── Execute ────────────────────────────────────────────────────────────
    if n_workers == 1:
        iterator = tqdm(tasks, desc="Extracting throats", unit="pair") \
            if (verbose and _HAVE_TQDM) else tasks
        results = [
            _extract_one_pair(t[0], t[1], t[2], t[3], t[4], t[5],
                              body_array, distance, boundaries, pore_mask,
                              min_throat_size, dilate_both)
            for t in iterator
        ]
    else:
        chunksize = max(1, len(tasks) // (n_workers * 4))
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(body_array, distance, boundaries, pore_mask,
                      min_throat_size, dilate_both),
        ) as ex:
            mapper = ex.map(_extract_pair_task, tasks, chunksize=chunksize)
            if verbose and _HAVE_TQDM:
                mapper = tqdm(mapper, total=len(tasks),
                              desc="Extracting throats", unit="pair")
            results = list(mapper)

    # ── Collect results ─────────────────────────────────────────────────────
    throat_props = {}
    errors = []
    tid = 1
    for r in results:
        if r is None:
            continue
        if isinstance(r, tuple) and len(r) == 4 and r[0] == '__error__':
            errors.append(r[1:])
            continue
        throat_props[tid] = r
        tid += 1

    if errors:
        print(f"[extract] WARNING: {len(errors)} pair(s) raised errors and "
              f"were skipped. First: bodies {errors[0][0]}-{errors[0][1]}: "
              f"{errors[0][2]}")
    timer.log(f"Found {len(throat_props)} throats.")

    # ── Assemble network ─────────────────────────────────────────────────────
    network = _assemble_network(body_props, throat_props, body_array,
                                voxel_size=1.0, verbose=verbose)
    timer.log("Assembled PoreNetwork.")
    return network


def extract_throats_from_bodies_voxel(body_array, min_throat_size=10,
                                      dilate_both=True, direction=None, surface_axis=None,
                                      skip_surface_surface=False, verbose=True):
    """Serial alias (n_jobs = 1) kept for backward compatibility."""
    return extract_throats_from_bodies_voxel_parallel(
        body_array,
        min_throat_size=min_throat_size,
        dilate_both=dilate_both,
        n_jobs=1,
        direction=direction,
        surface_axis=surface_axis,
        skip_surface_surface=skip_surface_surface,
        verbose=verbose,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Effective properties (Berg 3-segment model)
# ──────────────────────────────────────────────────────────────────────────────
def _compute_sum_throat_areas(network, throat_area_cache):
    """Sum of connected throat areas per body (for area-weighted volume split)."""
    return {
        body_id: sum(throat_area_cache.get(tid, 0.0) for tid in body.connected_throats)
        for body_id, body in network.bodies.items()
    }


def _compute_volume_share(body, throat_id, A_tk, sum_throat_areas, split_volume_equal):
    """Volume V_itk that body_i allocates to throat k."""
    if body.coordination_number == 0:
        return 0.0
    if split_volume_equal:
        return body.volume_voxels / body.coordination_number
    total_area = sum_throat_areas.get(body.body_id, 0.0)
    if total_area <= 0:
        return body.volume_voxels / body.coordination_number
    return body.volume_voxels * (A_tk / total_area)


def _compute_body_side(body, throat, A_tk, V_itk, alpha=DEFAULT_ALPHA):
    """
    Effective properties for one body side of a connection.

    Returns a dict (d_itk, A_itk, ratio, d_eff_itk, A_eff_itk, V_eff_itk),
    or None if the body-to-throat distance is zero.

    `alpha` is now honoured here (the original hard-coded 0.5).  d_eff is
    floored at a small positive fraction of d_itk so it can never go
    negative for alpha values near 1 (no effect for the default alpha=0.5,
    where d_eff stays positive throughout the normal regime).
    """
    if throat.center is not None:
        body_center = np.array(body.centroid, dtype=float)
        throat_center = np.array(throat.center, dtype=float)
        d_itk = float(np.linalg.norm(throat_center - body_center))
    else:
        d_itk = (throat.length_body1_to_ct_voxels
                 if throat.body1_id == body.body_id
                 else throat.length_ct_to_body2_voxels)

    if d_itk == 0:
        return None

    A_itk = V_itk / d_itk
    ratio = A_tk / A_itk if A_itk > 0 else np.inf

    d_eff_itk = d_itk * (1.0 - alpha * np.sqrt(A_tk / A_itk)) if A_itk > 0 else d_itk
    d_eff_itk = max(d_eff_itk, 1e-9 * d_itk)          # defensive positivity floor

    numerator = A_itk * d_itk - A_tk * (d_itk - d_eff_itk)
    A_eff_itk = numerator / d_eff_itk if d_eff_itk > 0 else A_itk
    V_eff_itk = d_eff_itk * A_eff_itk

    return {
        'd_itk':     d_itk,
        'A_itk':     A_itk,
        'ratio':     ratio,
        'd_eff_itk': d_eff_itk,
        'A_eff_itk': A_eff_itk,
        'V_eff_itk': V_eff_itk,
    }


def _compute_fallback(side1, side2):
    """
    Uniform-pipe fallback for degenerate connections (A_tk/A_itk above
    threshold).  Treats body1 -> throat -> body2 as one volume-conserving
    tube split into three equal-length segments.
    """
    V_total = side1['V_itk'] + side2['V_itk']
    L_total = side1['d_itk'] + side2['d_itk']

    A_uniform = V_total / L_total if L_total > 0 else 0.0
    L_segment = L_total / 3.0

    seg = {'d_eff': L_segment, 'A_eff': A_uniform, 'V_eff': L_segment * A_uniform}
    return {
        'L_segment': L_segment,
        'A_uniform': A_uniform,
        'side1':     seg,
        'side2':     seg,
        'throat': {
            'd_eff_tk_1': L_segment / 2,
            'd_eff_tk_2': L_segment / 2,
            'd_eff_tk':   L_segment,
            'V_eff_tk':   L_segment * A_uniform,
        },
    }


def _store_body_connection(body, throat_id, d_itk, A_itk, A_tk,
                           d_eff, A_eff, V_eff, is_fallback, alpha):
    body.connections_to_throats[throat_id] = BodyThroatConnection(
        body_id=body.body_id,
        throat_id=throat_id,
        distance=d_itk,
        body_path_area=A_itk,
        throat_area=A_tk,
        effective_distance=d_eff,
        effective_area=A_eff,
        effective_volume=V_eff,
        alpha=alpha,
        is_fallback=is_fallback,
        geometric_throat_area=A_tk,
    )


def _store_throat_properties(throat, d_eff_tk_1, d_eff_tk_2, A_tk,
                             is_fallback, A_uniform=None):
    throat.effective_length_to_body1 = d_eff_tk_1
    throat.effective_length_to_body2 = d_eff_tk_2
    throat.effective_length_total = d_eff_tk_1 + d_eff_tk_2
    throat.is_fallback = is_fallback
    throat.geometric_throat_area = A_tk

    if is_fallback and A_uniform is not None:
        throat.fallback_area = A_uniform
        throat.effective_volume = throat.effective_length_total * A_uniform
    else:
        throat.fallback_area = 0.0
        throat.effective_volume = throat.effective_length_total * A_tk


def calculate_network_effective_properties(
    network,
    area_method='vox_projection',
    alpha=DEFAULT_ALPHA,
    split_volume_equal=True,
    verbose=True,
):
    """
    Compute Berg effective geometry (lengths/areas/volumes) for every
    connection and throat in the network.

    Run serially: each iteration is a handful of scalar numpy ops and mutates
    shared body/throat objects, so process-parallelism would cost far more in
    pickling/merging than it saves.
    """
    warnings = {
        'zero_distance': [],   # (body_id, throat_id)
        'missing_body':  [],   # throat_id
        'fallback':      [],   # throat_id
        'zero_length':   [],   # throat_id
    }

    if verbose:
        print("Starting effective property calculations...")
        print(f"  area_method={area_method}, alpha={alpha}, "
              f"split_volume_equal={split_volume_equal}")

    throat_area_cache = {tid: t.get_area(area_method)
                         for tid, t in network.throats.items()}
    sum_throat_areas = _compute_sum_throat_areas(network, throat_area_cache)

    for body in network.bodies.values():
        body.connections_to_throats.clear()

    for throat_id, throat in network.throats.items():
        body1 = network.bodies.get(throat.body1_id)
        body2 = network.bodies.get(throat.body2_id)
        if body1 is None or body2 is None:
            warnings['missing_body'].append(throat_id)
            continue

        A_tk = throat_area_cache[throat_id]

        V_1tk = _compute_volume_share(body1, throat_id, A_tk, sum_throat_areas, split_volume_equal)
        V_2tk = _compute_volume_share(body2, throat_id, A_tk, sum_throat_areas, split_volume_equal)

        side1 = _compute_body_side(body1, throat, A_tk, V_1tk, alpha=alpha)
        side2 = _compute_body_side(body2, throat, A_tk, V_2tk, alpha=alpha)

        if side1 is None:
            warnings['zero_distance'].append((throat.body1_id, throat_id))
        if side2 is None:
            warnings['zero_distance'].append((throat.body2_id, throat_id))
        if side1 is None or side2 is None:
            continue

        side1['V_itk'] = V_1tk
        side2['V_itk'] = V_2tk

        use_fallback = (side1['ratio'] > RATIO_THRESHOLD or
                        side2['ratio'] > RATIO_THRESHOLD)

        if use_fallback:
            warnings['fallback'].append(throat_id)
            if (side1['d_itk'] + side2['d_itk']) == 0:
                warnings['zero_length'].append(throat_id)
                continue

            fb = _compute_fallback(side1, side2)

            _store_body_connection(
                body1, throat_id,
                d_itk=side1['d_itk'], A_itk=side1['A_itk'], A_tk=A_tk,
                d_eff=fb['side1']['d_eff'], A_eff=fb['side1']['A_eff'],
                V_eff=fb['side1']['V_eff'], is_fallback=True, alpha=alpha)
            _store_body_connection(
                body2, throat_id,
                d_itk=side2['d_itk'], A_itk=side2['A_itk'], A_tk=A_tk,
                d_eff=fb['side2']['d_eff'], A_eff=fb['side2']['A_eff'],
                V_eff=fb['side2']['V_eff'], is_fallback=True, alpha=alpha)
            _store_throat_properties(
                throat,
                d_eff_tk_1=fb['throat']['d_eff_tk_1'],
                d_eff_tk_2=fb['throat']['d_eff_tk_2'],
                A_tk=A_tk, is_fallback=True, A_uniform=fb['A_uniform'])
        else:
            d_eff_tk_1 = side1['d_itk'] - side1['d_eff_itk']
            d_eff_tk_2 = side2['d_itk'] - side2['d_eff_itk']

            _store_body_connection(
                body1, throat_id,
                d_itk=side1['d_itk'], A_itk=side1['A_itk'], A_tk=A_tk,
                d_eff=side1['d_eff_itk'], A_eff=side1['A_eff_itk'],
                V_eff=side1['V_eff_itk'], is_fallback=False, alpha=alpha)
            _store_body_connection(
                body2, throat_id,
                d_itk=side2['d_itk'], A_itk=side2['A_itk'], A_tk=A_tk,
                d_eff=side2['d_eff_itk'], A_eff=side2['A_eff_itk'],
                V_eff=side2['V_eff_itk'], is_fallback=False, alpha=alpha)
            _store_throat_properties(
                throat,
                d_eff_tk_1=d_eff_tk_1, d_eff_tk_2=d_eff_tk_2,
                A_tk=A_tk, is_fallback=False)

    if verbose:
        total_warnings = sum(len(v) for v in warnings.values())
        print(f"Processed {network.num_throats} connections "
              f"({network.num_bodies} bodies)")
        if total_warnings:
            for key, items in warnings.items():
                if items:
                    print(f"  {key}: {len(items)}")

    return network, warnings
