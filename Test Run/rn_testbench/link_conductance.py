"""Link-level test of the RN against the TauFactor field.

python link_conductance.py <sample_dir> <tf_field.pkl|full_result.pkl> <out.json>

For every throat link (i, j):
  I_ij^TF  = sum of finite-volume face fluxes (phi_u - phi_v) over all voxel
             faces shared by bodies i and j (unit conductance per face, the
             same normalisation as the RN with sigma = 1, voxel units)
  dPhi^TF  = mean(phi | body i) - mean(phi | body j)
  G^TF     = I_ij^TF / dPhi^TF
and compare with the RN link conductance G^RN (3-segment Berg model) for
both rn_methods. Aggregated as a dissipation-weighted (w = I^TF dPhi^TF > 0)
geometric mean of G^RN / G^TF over interior-interior links, split by
fallback flag and by shared-face count.
"""
import contextlib, io, json, pickle, sys
from pathlib import Path
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root
import numpy as np
import scipy.ndimage as ndi
import particulate_claude as pc
import berg_cc

sample, tf_path, out_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
d = pickle.load(open(tf_path, 'rb'))
phi = np.asarray(d.get('phi', d.get('potential_field')), dtype=np.float64)
# orient so the inlet (x = 0) is the high-potential side, as in the RN
if phi[0][phi[0] != 0].mean() < phi[-1][phi[-1] != 0].mean():
    phi = 1.0 - phi

with contextlib.redirect_stdout(io.StringIO()):
    net = pc.extract_throats_from_bodies_voxel_parallel(
        np.load(sample / 'body_array.npy'), min_throat_size=1, direction='x', merge_bodies=True, verbose=False)
B = net.body_array
M = int(B.max()) + 1
pore = B > 0

# face fluxes and face counts between labelled bodies
keys, flux, nfc = [], [], []
for ax in range(3):
    a = np.moveaxis(B, ax, 0); p = np.moveaxis(phi, ax, 0)
    lo, hi = a[:-1], a[1:]
    m = (lo != hi) & (lo > 0) & (hi > 0)
    u, v = lo[m].astype(np.int64), hi[m].astype(np.int64)
    f = (p[:-1][m] - p[1:][m])                      # flux lo -> hi
    sgn = np.where(u < v, 1.0, -1.0)                # orient as min-label -> max-label
    keys.append(np.minimum(u, v) * M + np.maximum(u, v)); flux.append(sgn * f); nfc.append(np.ones_like(f))
k = np.concatenate(keys); uk, inv = np.unique(k, return_inverse=True)
I_pair = dict(zip(uk.tolist(), np.bincount(inv, weights=np.concatenate(flux)).tolist()))
n_pair = dict(zip(uk.tolist(), np.bincount(inv).tolist()))

ids = sorted(net.bodies)
mean_phi = dict(zip(ids, ndi.mean(phi, labels=B, index=ids)))

out = {'sample': sample.name}
for rn in ('vanilla', 'short_circuit'):
    with contextlib.redirect_stdout(io.StringIO()):
        pc.calculate_network_effective_properties(net, rn_method=rn, verbose=False)
        arr = berg_cc.build_network_arrays(net)
        G, _ = berg_cc.calculate_conductances(arr, verbose=False)
        inl, outl = berg_cc.get_inlet_outlet_indices(arr, B, direction='x')
    boundary = set(arr['body_ids'][np.concatenate([inl, outl])].tolist())
    rows = []
    for t_idx, tid in enumerate(arr['throat_ids']):
        t = net.throats[int(tid)]
        b1, b2 = t.connects
        if b1 in boundary or b2 in boundary:
            continue
        key = min(b1, b2) * M + max(b1, b2)
        if key not in I_pair:
            continue
        I = I_pair[key] * (1.0 if b1 < b2 else -1.0)  # flux b1 -> b2
        dP = mean_phi[b1] - mean_phi[b2]
        w = I * dP
        if w <= 0 or abs(dP) < 1e-6:
            continue
        rows.append((G[t_idx] / (I / dP), w, bool(arr['is_fallback'][t_idx]), n_pair[key]))
    r = np.array([x[0] for x in rows]); w = np.array([x[1] for x in rows])
    fb = np.array([x[2] for x in rows]); nf = np.array([x[3] for x in rows])
    gm = lambda s: float(np.exp(np.sum(w[s] * np.log(r[s])) / np.sum(w[s]))) if s.any() else float('nan')
    out[rn] = dict(n_links=int(len(r)), G_ratio=gm(np.ones(len(r), bool)),
                   G_ratio_fallback=gm(fb), G_ratio_berg=gm(~fb),
                   G_ratio_le16faces=gm(nf <= 16), G_ratio_gt16faces=gm(nf > 16),
                   w_frac_fallback=float(w[fb].sum() / w.sum()),
                   w_frac_le16faces=float(w[nf <= 16].sum() / w.sum()))
json.dump(out, open(out_path, 'w'), indent=1)
print(json.dumps(out, indent=1))
