"""Stage-by-stage testbench for the berg_cc body/throat pipeline.

usage: python bench.py <module_dir> <sample_dir> <out.pkl> [key=value ...]

module_dir : directory holding the particulate_claude/berg_cc modules to test
             (the frozen baseline copy or the live repo).
keys       : rn_method, split_volume_equal, streamtube_method, min_throat_size, n_jobs
"""
import contextlib, io, json, pickle, sys, time
from pathlib import Path

mod_dir, sample, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
opts = dict(a.split('=', 1) for a in sys.argv[4:])
sys.path.insert(0, mod_dir)

import numpy as np
import particulate_claude as pc
import berg_cc

rn_method = opts.get('rn_method', 'vanilla')
split_eq = opts.get('split_volume_equal', 'True') == 'True'
st_method = opts.get('streamtube_method', 'flow_decomposition')
mts = int(opts.get('min_throat_size', 1))
n_jobs = int(opts.get('n_jobs', 1))
direction = 'x'
axis = 0

body_array = np.load(sample / 'body_array.npy')
times = {}
log = io.StringIO()
has_new_api = hasattr(pc, 'RN_METHODS')

with contextlib.redirect_stdout(log):
    t = time.perf_counter()
    merged, n_m1, _ = pc.merge_boundary_only_bodies_iterative(body_array, axis=axis, verbose=False)
    times['merge_boundary'] = time.perf_counter() - t
    t = time.perf_counter()
    merged, n_m2 = pc.merge_short_circuit_bodies(merged, axis=axis, verbose=False)
    times['merge_short_circuit'] = time.perf_counter() - t

    # body props timed on its own (it is also called inside extraction)
    t = time.perf_counter()
    pm = pc.compute_pore_mask(merged)
    dist = pc.compute_distance_transform(pm)
    ub = pc.get_unique_bodies(merged)
    times['edt'] = time.perf_counter() - t
    t = time.perf_counter()
    bp = pc.compute_initial_body_props(merged, dist, ub, direction=direction)
    times['body_props'] = time.perf_counter() - t

    t = time.perf_counter()
    kw = dict(min_throat_size=mts, dilate_both=True, n_jobs=n_jobs, direction=direction,
              skip_surface_surface=True, verbose=False)
    if has_new_api:
        kw['merge_bodies'] = False      # already merged above
    net = pc.extract_throats_from_bodies_voxel_parallel(merged, **kw)
    times['extract_total'] = time.perf_counter() - t

    t = time.perf_counter()
    ekw = dict(area_method='vox_projection', alpha=0.5, split_volume_equal=split_eq, verbose=False)
    if has_new_api:
        ekw['rn_method'] = rn_method
    net, warns = pc.calculate_network_effective_properties(net, **ekw)
    times['effective_props'] = time.perf_counter() - t

    t = time.perf_counter()
    res = berg_cc.compute_all_berg(net, merged, area_method='vox_projection', sigma=1.0,
                                   direction=direction, delta_V=1.0, voxel_size=1.0,
                                   streamtube_method=st_method, verbose=False)
    times['compute_all_berg'] = time.perf_counter() - t

times['total'] = sum(v for k, v in times.items())
gb = res['global_berg']
arrays = res['arrays']
sts = res['streamtubes']
I_tot = res['solution_data']['total_current']
delta_s = res['delta_s']
# current-weighted geometric tortuosity: sum I L / (I_tot * delta_s) -- the
# streamtube analogue of TauFactor's <|J|>/<J_x> (see report)
T_flux = sum(s['I_gamma'] * s['L_gamma'] for s in sts) / (I_tot * delta_s)
A_cross = merged.shape[1] * merged.shape[2]
metrics = {
    'n_bodies_raw': int(len(np.unique(body_array)) - 1),
    'n_bodies': net.num_bodies, 'n_throats': net.num_throats,
    'n_fallback': int(sum(t.is_fallback for t in net.throats.values())),
    'Omega': res['Omega'], 'Omega_c': res['Omega_c'],
    'one_over_F': 1.0 / res['F_total'],
    'one_over_F_from_G': res['solution_data']['effective_conductance'] * delta_s / A_cross,
    'tau_c': gb['tau_sq_c'] ** -0.5, 'tau_sq_c': gb['tau_sq_c'],
    'C_c': gb['C_c'], 'inv_C_c': 1.0 / gb['C_c'],
    'T_flux_weighted': T_flux,
    'iota_sq_g': res['iota_sq_g'], 'iota_sq_c': gb['iota_sq_c'], 'consistency': gb['consistency'],
    'phi_total': res['phi_total'], 'phi_c': res['phi_c'],
    'n_streamtubes': len(sts),
}
throats = {tid: dict(connects=t.connects, vol=t.volume_voxels, center=t.center,
                     coords=t.coords, area=t.area_vox_projection_voxels2,
                     L1=t.length_body1_to_ct_voxels, L2=t.length_ct_to_body2_voxels)
           for tid, t in net.throats.items()}
bodies = {bid: dict(vol=b.volume_voxels, centroid=b.centroid, vc=b.volume_centroid,
                    sc=b.surface_centroid, maxr=b.max_radius, surf=b.is_surface,
                    sav=b.surface_area_voxels)
          for bid, b in net.bodies.items()}
with open(out, 'wb') as f:
    pickle.dump(dict(times=times, metrics=metrics, merged=merged, n_merged=(n_m1, n_m2),
                     body_props_direct=bp, throats=throats, bodies=bodies,
                     arrays={k: v for k, v in arrays.items() if k != 'id_to_idx'},
                     potential=res['potential'], current=res['current'],
                     opts=opts), f)
print(json.dumps({'sample': sample.name, 'times': {k: round(v, 3) for k, v in times.items()},
                  'metrics': {k: (round(v, 5) if isinstance(v, float) else v) for k, v in metrics.items()}}))
