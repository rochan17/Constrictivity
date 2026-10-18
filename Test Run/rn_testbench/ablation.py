"""Ablation / sensitivity of the short-circuit RN heuristic.

python ablation.py <sample_dir> <out.json>
Builds the network once, then re-runs effective properties + Berg solve for
several variants of the half-link length rule.
"""
import contextlib, io, json, sys
from pathlib import Path
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root
import numpy as np
import particulate_claude as pc
import berg_cc

sample = Path(sys.argv[1])
body = np.load(sample / 'body_array.npy')
log = io.StringIO()
with contextlib.redirect_stdout(log):
    net = pc.extract_throats_from_bodies_voxel_parallel(
        body, min_throat_size=1, direction='x', merge_bodies=True, n_jobs=1, verbose=False)

orig_sc = pc.compute_short_circuit_distances


def sc_variant(boundary=True, interior=True, weight='area_dx', clip=0.25, interior_rule='weighted'):
    def f(network, axis=None, area_method='vox_projection'):
        axis = 0 if axis is None else axis
        L_max = network.shape[axis] - 1
        out = {}
        for b in network.bodies.values():
            tids = [t for t in b.connected_throats if network.throats[t].center is not None]
            if not tids:
                continue
            c = np.asarray(b.centroid, float)
            P = np.array([network.throats[t].center for t in tids], float)
            d = np.linalg.norm(P - c, axis=1)
            if b.is_surface and (c[axis] <= 1 or c[axis] >= L_max - 1):
                if not boundary:
                    continue
                plane = 0.0 if c[axis] <= 1 else float(L_max)
                d_new = np.minimum(d, np.maximum(np.abs(P[:, axis] - plane), 0.5))
            elif len(tids) >= 2:
                if not interior:
                    continue
                A = np.array([max(network.throats[t].get_area(area_method), 1e-6) for t in tids])
                chord = np.linalg.norm(P[:, None] - P[None], axis=2)
                star = d[:, None] + d[None]
                dx = np.abs(P[:, None, axis] - P[None, :, axis])
                if interior_rule == 'axial':
                    # project out transverse detour: half-link = axial distance to throat
                    d_new = np.maximum(np.abs(P[:, axis] - c[axis]), 0.5)
                    d_new = np.minimum(d, d_new)
                else:
                    w = {'area_dx': A[None] * (dx + 0.5), 'area': np.broadcast_to(A[None], dx.shape).copy(),
                         'uniform': np.ones_like(dx), 'dx': dx + 0.5}[weight]
                    w[(star <= 1e-12) | np.eye(len(tids), dtype=bool)] = 0
                    num = (w * np.minimum(chord, star)).sum(1); den = (w * star).sum(1)
                    r = np.ones(len(tids)); ok = den > 1e-12
                    r[ok] = np.clip(num[ok] / den[ok], clip, 1.0)
                    d_new = d * r
            else:
                continue
            for t, dn in zip(tids, d_new):
                out[(b.body_id, t)] = float(dn)
        return out
    return f


variants = {
    'vanilla': None,
    'sc_full': {},
    'sc_boundary_only': dict(interior=False),
    'sc_interior_only': dict(boundary=False),
    'w_uniform': dict(weight='uniform'),
    'w_area': dict(weight='area'),
    'w_dx': dict(weight='dx'),
    'clip_0': dict(clip=0.0),
    'clip_0.5': dict(clip=0.5),
    'interior_axial': dict(interior_rule='axial'),
}
res = {}
for name, kw in variants.items():
    with contextlib.redirect_stdout(log):
        if kw is None:
            pc.calculate_network_effective_properties(net, rn_method='vanilla', verbose=False)
        else:
            pc.compute_short_circuit_distances = sc_variant(**kw)
            pc.calculate_network_effective_properties(net, rn_method='short_circuit', verbose=False)
            pc.compute_short_circuit_distances = orig_sc
        r = berg_cc.compute_all_berg(net, net.body_array, streamtube_method='flow_decomposition', verbose=False)
    gb = r['global_berg']; sts = r['streamtubes']; I = r['solution_data']['total_current']
    res[name] = dict(one_over_F=1 / r['F_total'], tau_c=gb['tau_sq_c'] ** -0.5, inv_C_c=1 / gb['C_c'],
                     T_I=sum(s['I_gamma'] * s['L_gamma'] for s in sts) / (I * r['delta_s']),
                     phi_c=r['phi_c'], phi=r['phi_total'],
                     n_fallback=int(sum(t.is_fallback for t in net.throats.values())))

# how much each rule shortens: distribution of d~/d over interior and boundary half-links
pc.calculate_network_effective_properties(net, rn_method='vanilla', verbose=False)
d_sc = orig_sc(net, axis=0)
ratios = {'boundary': [], 'interior': []}
for (bid, tid), dn in d_sc.items():
    b = net.bodies[bid]
    d0 = float(np.linalg.norm(np.asarray(net.throats[tid].center) - np.asarray(b.centroid)))
    ratios['boundary' if b.is_surface and (b.centroid[0] <= 1 or b.centroid[0] >= net.shape[0] - 2) else 'interior'].append(dn / d0)
res['_ratio_stats'] = {k: dict(n=len(v), mean=float(np.mean(v)), p10=float(np.percentile(v, 10)),
                               p50=float(np.median(v)), at_clip=int(np.sum(np.isclose(v, 0.25))))
                       for k, v in ratios.items() if v}
json.dump(res, open(sys.argv[2], 'w'), indent=1)
print(sample.name, 'done')
