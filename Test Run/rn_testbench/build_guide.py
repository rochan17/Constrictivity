"""Build the standalone interactive guide `Test Run/berg_network_formation_guide.html`.

    python build_guide.py [--sample TRACE_fine_test/Fine_microstructure_0]
                          [--tf-field TRACE_fine_test/Fine_microstructure_0/version1/taufactor/full_result.pkl]

Every number in the page is computed here from the code in the repo
(particulate_claude.py / berg_cc.py) -- both RN variants -- and from a saved
TauFactor field of the same sample. The output HTML is fully standalone: the
data, KaTeX (JS, CSS and fonts) and all plotting code are inlined, so it needs
no network and no other file.
"""
import argparse, base64, contextlib, csv, io, json, pickle, re, sys, zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import scipy.ndimage as ndi
import particulate_claude as pc
import berg_cc

ap = argparse.ArgumentParser()
ap.add_argument('--sample', default='TRACE_fine_test/Fine_microstructure_0')
ap.add_argument('--tf-field', default='TRACE_fine_test/Fine_microstructure_0/version1/taufactor/full_result.pkl')
ap.add_argument('--out', default=str(HERE.parent / 'berg_network_formation_guide.html'))
args = ap.parse_args()
sample = REPO / args.sample
CASE_BODIES = (221, 201)
CFG = dict(min_throat_size=1, dilate_both=True, skip_surface_surface=True, area_method='vox_projection',
           alpha=0.5, split_volume_equal=True, streamtube_method='flow_decomposition', direction='x')


def r(x, n=4):
    return float(round(float(x), n))


def rl(a, n=4):
    return [round(float(v), n) for v in np.asarray(a, dtype=float).ravel()]


def b64z(arr):
    return base64.b64encode(zlib.compress(np.ascontiguousarray(arr).tobytes(), 9)).decode()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Network (merge -> extract) and both RN solves
# ─────────────────────────────────────────────────────────────────────────────
raw = np.load(sample / 'body_array.npy')
quiet = io.StringIO()
with contextlib.redirect_stdout(quiet):
    net = pc.extract_throats_from_bodies_voxel_parallel(
        raw, min_throat_size=CFG['min_throat_size'], dilate_both=True, direction='x',
        skip_surface_surface=True, merge_bodies=True, verbose=False)
merged = net.body_array
shape = merged.shape
L = shape[0]

per = {}
for m in ('vanilla', 'short_circuit'):
    with contextlib.redirect_stdout(quiet):
        pc.calculate_network_effective_properties(net, rn_method=m, alpha=0.5, split_volume_equal=True, verbose=False)
        res = berg_cc.compute_all_berg(net, merged, streamtube_method=CFG['streamtube_method'], verbose=False)
    conn = {(b.body_id, t): (c.distance, c.body_path_area, c.effective_distance, c.effective_area, c.is_fallback)
            for b in net.bodies.values() for t, c in b.connections_to_throats.items()}
    sts = res['streamtubes']
    I_tot = float(res['solution_data']['total_current'])
    per[m] = dict(res=res, conn=conn,
                  fb={tid: bool(t.is_fallback) for tid, t in net.throats.items()},
                  L_eff={tid: (t.effective_length_to_body1, t.effective_length_to_body2) for tid, t in net.throats.items()},
                  glob=dict(one_over_F=1 / res['F_total'], tau_c=res['global_berg']['tau_sq_c'] ** -0.5,
                            tau_sq_c=res['global_berg']['tau_sq_c'], C_c=res['global_berg']['C_c'],
                            inv_C_c=1 / res['global_berg']['C_c'], phi=res['phi_total'], phi_c=res['phi_c'],
                            iota_sq_g=res['iota_sq_g'], Omega=res['Omega'], Omega_c=res['Omega_c'],
                            T_I=sum(s['I_gamma'] * s['L_gamma'] for s in sts) / (I_tot * res['delta_s']),
                            I_tot=I_tot, n_streamtubes=len(sts),
                            n_fallback=int(sum(t.is_fallback for t in net.throats.values()))))
    print(m, {k: round(v, 4) for k, v in per[m]['glob'].items()})

arr = per['vanilla']['res']['arrays']
body_ids = [int(b) for b in arr['body_ids']]
bidx = {b: i for i, b in enumerate(body_ids)}
throat_ids = [int(t) for t in arr['throat_ids']]
inlet = set(arr['body_ids'][per['vanilla']['res']['inlet_indices']].tolist())
outlet = set(arr['body_ids'][per['vanilla']['res']['outlet_indices']].tolist())
btype = {b: (1 if b in inlet else 2 if b in outlet else 0) for b in body_ids}

# ─────────────────────────────────────────────────────────────────────────────
# 2. TauFactor field: orientation, body means, face fluxes between bodies
# ─────────────────────────────────────────────────────────────────────────────
tf = pickle.load(open(REPO / args.tf_field, 'rb'))
phi = np.asarray(tf['potential_field'], dtype=np.float64)
tf_pore = np.asarray(tf['pore_mask'], dtype=bool)
# TauFactor stores the potential centred on zero (here [-0.5, 0.5], rising
# from inlet to outlet). Map it onto the RN's convention, Phi = 1 at the inlet
# and 0 at the outlet, with the boundary values as end points (a pure offset
# and sign, so gradients and face fluxes keep their scale).
lo_bc, hi_bc = float(phi[tf_pore].min()), float(phi[tf_pore].max())
flip = phi[0][tf_pore[0]].mean() < phi[-1][tf_pore[-1]].mean()
phi = (hi_bc - phi) / (hi_bc - lo_bc) if flip else (phi - lo_bc) / (hi_bc - lo_bc)
phi[~tf_pore] = 0.0
J = np.stack([np.asarray(tf[k], dtype=np.float64) for k in ('J_x', 'J_y', 'J_z')])
if flip:
    J = -J
Jmag = np.sqrt((J ** 2).sum(0))
Jx_mean = float(np.abs(J[0][tf_pore].mean()))

# total TF current through each x-plane (unit face conductance), averaged
pp = tf_pore[:-1] & tf_pore[1:]
I_tf_planes = ((phi[:-1] - phi[1:]) * pp).sum(axis=(1, 2))
I_tf_tot = float(I_tf_planes[10:-10].mean())

M = int(merged.max()) + 1
keys, flux, = [], []
for ax in range(3):
    a = np.moveaxis(merged, ax, 0); p = np.moveaxis(phi, ax, 0)
    lo, hi = a[:-1], a[1:]
    msk = (lo != hi) & (lo > 0) & (hi > 0)
    u, v = lo[msk].astype(np.int64), hi[msk].astype(np.int64)
    f = p[:-1][msk] - p[1:][msk]
    keys.append(np.minimum(u, v) * M + np.maximum(u, v)); flux.append(np.where(u < v, f, -f))
k = np.concatenate(keys); uk, inv = np.unique(k, return_inverse=True)
pair_flux = dict(zip(uk.tolist(), np.bincount(inv, weights=np.concatenate(flux)).tolist()))
pair_faces = dict(zip(uk.tolist(), np.bincount(inv).tolist()))
phi_tf_body = dict(zip(body_ids, ndi.mean(phi, labels=merged, index=body_ids)))

tf_metrics = None
for v in sorted(sample.glob('version*/summary.json'), key=lambda p: int(p.parent.name[7:])):
    s = json.load(open(v)).get('taufactor')
    if s and 'D_eff' in s:
        tf_metrics = {k: s[k] for k in ('porosity', 'D_eff', 'tau_solver', 'tau_reconciled', 'constrictivity')}
print('TF', tf_metrics, 'I_tf_tot', I_tf_tot, 'flip', flip)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Nodes, throats, half-links, interior throat pairs
# ─────────────────────────────────────────────────────────────────────────────
pot = {m: per[m]['res']['potential'] for m in per}
nodes = dict(id=body_ids, type=[btype[b] for b in body_ids],
             V=[int(net.bodies[b].volume_voxels) for b in body_ids],
             z=[int(net.bodies[b].num_throats) for b in body_ids],
             x=rl([net.bodies[b].centroid[0] for b in body_ids], 2),
             y=rl([net.bodies[b].centroid[1] for b in body_ids], 2),
             zc=rl([net.bodies[b].centroid[2] for b in body_ids], 2),
             phi_tf=rl([phi_tf_body[b] for b in body_ids]),
             phi_van=rl(pot['vanilla']), phi_sc=rl(pot['short_circuit']))

cd = {m: per[m]['res']['conductance_data'] for m in per}
cur = {m: per[m]['res']['current'] for m in per}
th = dict(tid=[], b1=[], b2=[], cx=[], cy=[], cz=[], At=[], nvox=[], nfaces=[], I_tf=[])
for m, tag in (('vanilla', 'van'), ('short_circuit', 'sc')):
    for f in ('G', 'R1', 'Rt', 'R2', 'I', 'fb', 'L1', 'Lt', 'L2'):
        th[f'{f}_{tag}'] = []
for t_idx, tid in enumerate(throat_ids):
    t = net.throats[tid]
    b1, b2 = t.connects
    key = min(b1, b2) * M + max(b1, b2)
    th['tid'].append(tid); th['b1'].append(bidx[b1]); th['b2'].append(bidx[b2])
    th['cx'].append(r(t.center[0], 2)); th['cy'].append(r(t.center[1], 2)); th['cz'].append(r(t.center[2], 2))
    th['At'].append(r(t.area_vox_projection_voxels2, 3)); th['nvox'].append(int(t.volume_voxels))
    th['nfaces'].append(int(pair_faces.get(key, 0)))
    th['I_tf'].append(r(pair_flux.get(key, 0.0) * (1 if b1 < b2 else -1), 6))
    for m, tag in (('vanilla', 'van'), ('short_circuit', 'sc')):
        c = cd[m]
        th[f'G_{tag}'].append(r(c['G'][t_idx], 6)); th[f'R1_{tag}'].append(r(c['R1'][t_idx], 6))
        th[f'Rt_{tag}'].append(r(c['R_throat'][t_idx], 6)); th[f'R2_{tag}'].append(r(c['R2'][t_idx], 6))
        th[f'I_{tag}'].append(r(cur[m][t_idx], 6)); th[f'fb_{tag}'].append(int(per[m]['fb'][tid]))
        th[f'L1_{tag}'].append(r(c['L1'][t_idx], 3)); th[f'Lt_{tag}'].append(r(c['L_throat'][t_idx], 3))
        th[f'L2_{tag}'].append(r(c['L2'][t_idx], 3))

hl = dict(body=[], throat=[], type=[], z=[], V=[], At=[], d_van=[], d_sc=[], A_van=[], A_sc=[])
for b in body_ids:
    B = net.bodies[b]
    for tid in B.connected_throats:
        if (b, tid) not in per['vanilla']['conn']:
            continue
        dv, Av = per['vanilla']['conn'][(b, tid)][:2]
        ds, As = per['short_circuit']['conn'][(b, tid)][:2]
        hl['body'].append(bidx[b]); hl['throat'].append(throat_ids.index(tid)); hl['type'].append(btype[b])
        hl['z'].append(B.num_throats); hl['V'].append(r(B.volume_voxels / B.num_throats, 2))
        hl['At'].append(r(net.throats[tid].area_vox_projection_voxels2, 3))
        hl['d_van'].append(r(dv, 3)); hl['d_sc'].append(r(ds, 3)); hl['A_van'].append(r(Av, 3)); hl['A_sc'].append(r(As, 3))

# interior throat pairs (p, r) of the same body: chord vs path through the body
pairs = dict(chord=[], star_van=[], star_sc=[], dx=[])
dmap = {m: {(b, t): v[0] for (b, t), v in per[m]['conn'].items()} for m in per}
for b in body_ids:
    B = net.bodies[b]
    if btype[b] != 0 or B.num_throats < 2:
        continue
    tids = [t for t in B.connected_throats if (b, t) in dmap['vanilla']]
    P = {t: np.asarray(net.throats[t].center, float) for t in tids}
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            p, q = tids[i], tids[j]
            pairs['chord'].append(r(np.linalg.norm(P[p] - P[q]), 3))
            pairs['star_van'].append(r(dmap['vanilla'][(b, p)] + dmap['vanilla'][(b, q)], 3))
            pairs['star_sc'].append(r(dmap['short_circuit'][(b, p)] + dmap['short_circuit'][(b, q)], 3))
            pairs['dx'].append(r(abs(P[p][0] - P[q][0]), 2))
ch = np.array(pairs['chord']); sv = np.array(pairs['star_sc'])
print('pairs', len(ch), 'SC pairs shorter than chord:', int((sv < ch - 1e-9).sum()))

# the pairs current actually uses: consecutive (in, out) throats of every
# streamtube at each interior body it crosses, weighted by I_Gamma. The RN
# charges d(b, t_in) + d(b, t_out) there (Berg segments of a throat sum to d1 + d2).
trav = {}
for m, tag in (('vanilla', 'van'), ('short_circuit', 'sc')):
    rr, ww = [], []
    for st in per[m]['res']['streamtubes']:
        npth = [body_ids[i] for i in st['node_path']]; tpth = [throat_ids[i] for i in st['throat_path']]
        for kk in range(1, len(npth) - 1):
            b = npth[kk]
            if btype[b] != 0:
                continue
            t_in, t_out = tpth[kk - 1], tpth[kk]
            chd = float(np.linalg.norm(np.asarray(net.throats[t_in].center) - np.asarray(net.throats[t_out].center)))
            if chd < 1e-9:
                continue
            rr.append(r((dmap[m][(b, t_in)] + dmap[m][(b, t_out)]) / chd, 4)); ww.append(r(st['I_gamma'], 6))
    trav[tag] = dict(ratio=rr, w=ww)
    a_, w_ = np.array(rr), np.array(ww)
    print(tag, 'traversals', len(a_), 'current-weighted share with path < chord:', round(float(w_[a_ < 1 - 1e-9].sum() / w_.sum()), 4),
          'I-weighted mean ratio', round(float((a_ * w_).sum() / w_.sum()), 4))
pairs['trav'] = trav

# lower bound on any RN whose links run between throat centroids: the polyline
# electrode face -> throat centroids -> electrode face of every streamtube,
# current-weighted like T_I. If this alone exceeds TauFactor's tortuosity,
# no choice of half-link lengths that respects the chords can match it.
T_chord = {}
for m, tag in (('vanilla', 'van'), ('short_circuit', 'sc')):
    num = 0.0
    for st in per[m]['res']['streamtubes']:
        P_ = [np.asarray(net.throats[throat_ids[q]].center, float) for q in st['throat_path']]
        Lp = (P_[0][0] + 0.5) + sum(np.linalg.norm(P_[q + 1] - P_[q]) for q in range(len(P_) - 1)) + (L - 0.5 - P_[-1][0])
        num += st['I_gamma'] * Lp
    T_chord[tag] = num / (per[m]['glob']['I_tot'] * per[m]['res']['delta_s'])
print('T_chord (throat-centroid polyline tortuosity)', T_chord)
pairs['T_chord'] = T_chord

# ─────────────────────────────────────────────────────────────────────────────
# 4. Parity (voxel 2-D histograms and R^2)
# ─────────────────────────────────────────────────────────────────────────────
both = (merged > 0) & tf_pore
lab_v = merged[both]
phi_v = phi[both]
NB = 40
parity = {}
for m, tag in (('vanilla', 'van'), ('short_circuit', 'sc')):
    lut = np.zeros(M); lut[body_ids] = pot[m]
    rn_v = lut[lab_v]
    h, _, _ = np.histogram2d(phi_v, rn_v, bins=NB, range=[[0, 1], [0, 1]])
    ss_res = float(((rn_v - phi_v) ** 2).sum()); ss_tot = float(((phi_v - phi_v.mean()) ** 2).sum())
    parity[tag] = dict(hist=h.astype(int).tolist(), r2_vox=1 - ss_res / ss_tot,
                       rmse_vox=float(np.sqrt(ss_res / len(phi_v))))
parity['n_vox'] = int(both.sum()); parity['bins'] = NB

# ─────────────────────────────────────────────────────────────────────────────
# 5. Volume for slicing (2x nearest downsample) and TF voxel-flux cloud
# ─────────────────────────────────────────────────────────────────────────────
S = 2
sl = (slice(None, None, S),) * 3
mg = merged[sl]; rw = raw[sl]; tfp = tf_pore[sl]
lut_v = np.zeros(M); lut_v[body_ids] = pot['vanilla']
lut_s = np.zeros(M); lut_s[body_ids] = pot['short_circuit']
gone = np.setdiff1d(np.unique(raw), np.unique(merged))
flags = ((mg > 0).astype(np.uint8)
         | (np.isin(rw, gone) & (rw > 0)).astype(np.uint8) << 1
         | np.isin(mg, list(inlet)).astype(np.uint8) << 2
         | np.isin(mg, list(outlet)).astype(np.uint8) << 3
         | tfp.astype(np.uint8) << 4)
q8 = lambda a: np.clip(np.round(a * 255), 0, 255).astype(np.uint8)
JS = 8.0                                          # |J| / <J_x> clipped at JS for display
vol = dict(
    dims=list(mg.shape), step=S,
    flags=b64z(flags),
    label=b64z(mg.astype(np.uint16)),
    rawlabel=b64z(rw.astype(np.uint16)),
    phi_tf=b64z(q8(np.where(tfp, phi[sl], 0))),
    phi_van=b64z(q8(np.where(mg > 0, lut_v[mg], 0))),
    phi_sc=b64z(q8(np.where(mg > 0, lut_s[mg], 0))),
    jmag=b64z(q8(np.sqrt(np.clip(Jmag[sl] / Jx_mean, 0, JS) / JS))),
    jx=b64z(np.clip(np.round(J[0][sl] / Jx_mean * 16), -127, 127).astype(np.int8)),
    jy=b64z(np.clip(np.round(J[1][sl] / Jx_mean * 16), -127, 127).astype(np.int8)),
    jz=b64z(np.clip(np.round(J[2][sl] / Jx_mean * 16), -127, 127).astype(np.int8)),
    jscale=JS, gone=[int(g) for g in gone])

rng = np.random.default_rng(0)
cand = np.flatnonzero(tf_pore.ravel())
w = Jmag.ravel()[cand]; w = w / w.sum()
pick = rng.choice(cand, size=16000, replace=False, p=w)
cx_, cy_, cz_ = np.unravel_index(pick, shape)
cloud = dict(x=b64z(cx_.astype(np.uint8)), y=b64z(cy_.astype(np.uint8)), z=b64z(cz_.astype(np.uint8)),
             j=b64z(q8(np.sqrt(np.clip(Jmag.ravel()[pick] / Jx_mean, 0, JS) / JS))),
             phi=b64z(q8(phi.ravel()[pick])), n=int(len(pick)))

# ─────────────────────────────────────────────────────────────────────────────
# 6. Case-study bodies (exact per-throat numbers from both solves)
# ─────────────────────────────────────────────────────────────────────────────
cases = {}
for b in CASE_BODIES:
    B = net.bodies[b]
    c = np.asarray(B.centroid, float)
    rows = []
    for tid in B.connected_throats:
        t = net.throats[tid]; t_idx = throat_ids.index(tid)
        nb = t.connects[1] if t.connects[0] == b else t.connects[0]
        sgn = 1.0 if t.connects[0] == b else -1.0          # current leaving body b through this throat
        row = dict(tid=tid, nb=nb, nb_type=btype[nb], p=rl(t.center, 2), At=r(t.area_vox_projection_voxels2, 2),
                   I_tf=r(sgn * th['I_tf'][t_idx], 5), phi_nb_tf=r(phi_tf_body[nb]))
        for m, tag in (('vanilla', 'van'), ('short_circuit', 'sc')):
            d, A, de, Ae, fbk = per[m]['conn'][(b, tid)]
            c_ = cd[m]
            own, far = ('R1', 'R2') if t.connects[0] == b else ('R2', 'R1')
            row.update({f'd_{tag}': r(d, 3), f'A_{tag}': r(A, 2), f'de_{tag}': r(de, 3), f'Ae_{tag}': r(Ae, 2),
                        f'fb_{tag}': int(per[m]['fb'][tid]), f'G_{tag}': r(c_['G'][t_idx], 4),
                        f'Rb_{tag}': r(c_[own][t_idx], 4), f'Rt_{tag}': r(c_['R_throat'][t_idx], 4),
                        f'Rn_{tag}': r(c_[far][t_idx], 4),
                        f'I_{tag}': r(sgn * cur[m][t_idx], 5), f'phi_nb_{tag}': r(pot[m][bidx[nb]])})
        rows.append(row)
    # every throat pair of the body: chord vs the length each RN charges, and
    # the streamtube current that actually enters by one and leaves by the other
    tids_ = [rw_['tid'] for rw_ in rows]
    pair_rows = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a_, b_ = rows[i], rows[j]
            pr = dict(i=i, j=j, chord=r(np.linalg.norm(np.subtract(a_['p'], b_['p'])), 3))
            for m, tag in (('vanilla', 'van'), ('short_circuit', 'sc')):
                pr[f'path_{tag}'] = r(a_[f'd_{tag}'] + b_[f'd_{tag}'], 3)
                Iuse = 0.0
                for st in per[m]['res']['streamtubes']:
                    npth = [body_ids[q] for q in st['node_path']]; tpth = [throat_ids[q] for q in st['throat_path']]
                    for kk in range(1, len(npth) - 1):
                        if npth[kk] == b and {tpth[kk - 1], tpth[kk]} == {a_['tid'], b_['tid']}:
                            Iuse += st['I_gamma']
                pr[f'Iuse_{tag}'] = r(Iuse, 5)
            pair_rows.append(pr)
    # 2-D projection for drawing: PCA plane of the throat centroids around c
    P = np.array([rw_['p'] for rw_ in rows]) - c
    _, _, vt = np.linalg.svd(np.vstack([P, -P]), full_matrices=False)
    e1, e2 = vt[0], vt[1]
    if e1[0] < 0: e1 = -e1
    for rw_, pv in zip(rows, P):
        rw_['u'] = r(pv @ e1, 2); rw_['v'] = r(pv @ e2, 2)
    cases[str(b)] = dict(id=b, V=int(B.volume_voxels), z=B.num_throats, c=rl(c, 2), V_half=r(B.volume_voxels / B.num_throats, 2),
                         phi_tf=r(phi_tf_body[b]), phi_van=r(pot['vanilla'][bidx[b]]), phi_sc=r(pot['short_circuit'][bidx[b]]),
                         rows=rows, pairs=pair_rows)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Benchmarks (data_new, from the testbench results) and code map
# ─────────────────────────────────────────────────────────────────────────────
def rows_of(tag):
    return {x['sample']: x for x in csv.DictReader(open(HERE / 'results' / f'budget_{tag}.csv'))}
bv, bs = rows_of('new_van'), rows_of('new_sc_eq')
bench = []
for n in [n for n in bv if n in bs and bv[n]['cls'] == 'Fine']:
    a, s_ = bv[n], bs[n]
    bench.append(dict(sample=n, cls=a['cls'], eps=r(a['eps']), D=float(a['TF_D']), C=r(a['TF_C']),
                      tau=r(float(a['tau_c']) / (1 + float(a['tau_B_err']))),
                      van=dict(F=float(a['one_over_F']), tau=r(a['tau_c']), TI=r(a['T_I']), iC=r(a['inv_C_c']),
                               fphi=r(a['f_phi']), fC=r(a['f_C']), fT=r(a['f_T'])),
                      sc=dict(F=float(s_['one_over_F']), tau=r(s_['tau_c']), TI=r(s_['T_I']), iC=r(s_['inv_C_c']),
                              fphi=r(s_['f_phi']), fC=r(s_['f_C']), fT=r(s_['f_T']))))

code_map = []
for fname in ('particulate_claude.py', 'berg_cc.py', 'pipeline.py'):
    for i, line in enumerate(open(REPO / fname), 1):
        mm = re.match(r'(def|class) (\w+)', line)
        if mm:
            code_map.append(dict(file=fname, name=mm.group(2), line=i))

DATA = dict(
    meta=dict(sample=sample.name, shape=list(shape), n_raw=int(len(np.unique(raw)) - 1), n_bodies=len(body_ids),
              n_throats=len(throat_ids), n_inlet=len(inlet), n_outlet=len(outlet), n_gone=int(len(gone)),
              cfg=CFG, pore_vox=int((merged > 0).sum())),
    tf=dict(**(tf_metrics or {}), I_tot=I_tf_tot, Jx_mean=Jx_mean),
    glob={('van' if m == 'vanilla' else 'sc'): {k: float(v) for k, v in per[m]['glob'].items()} for m in per},
    nodes=nodes, throats=th, halflinks=hl, pairs=pairs, parity=parity, volume=vol, cloud=cloud,
    cases=cases, bench=bench, code=code_map)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Assemble the standalone page
# ─────────────────────────────────────────────────────────────────────────────
K = HERE / 'vendor' / 'katex'
css = (K / 'katex.min.css').read_text()
css = re.sub(r',url\(fonts/[^)]+\.woff\) format\("woff"\)', '', css)
css = re.sub(r',url\(fonts/[^)]+\.ttf\) format\("truetype"\)', '', css)
css = re.sub(r'url\(fonts/([^)]+\.woff2)\)',
             lambda mm: 'url(data:font/woff2;base64,' + base64.b64encode((K / 'fonts' / mm.group(1)).read_bytes()).decode() + ')', css)
assert 'url(fonts/' not in css
page = (HERE / 'guide_template.html').read_text()
for ph, val in (('/*__KATEX_CSS__*/', css), ('/*__KATEX_JS__*/', (K / 'katex.min.js').read_text()),
                ('/*__AUTORENDER_JS__*/', (K / 'auto-render.min.js').read_text()),
                ('/*__DATA__*/', 'const DATA = ' + json.dumps(DATA, separators=(',', ':')) + ';')):
    assert page.count(ph) == 1, ph
    page = page.replace(ph, val.replace('</script', '<\\/script'))
Path(args.out).write_text(page)
print('wrote', args.out, f'{len(page) / 1e6:.2f} MB')
