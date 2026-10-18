"""Throat-area diagnostic: vox-projection area vs. shared voxel-face count.

python throat_faces.py <sample_dir> <out.json>

n_faces(b1,b2) = number of 6-neighbour voxel pairs labelled (b1, b2): the
cross-section a finite-volume (TauFactor) solve actually conducts through
between the two bodies. Reported overall and current-weighted (by |I| from
the short-circuit RN solve) and binned by n_faces.
"""
import contextlib, io, json, sys
from pathlib import Path
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))  # repo root
import numpy as np
import particulate_claude as pc
import berg_cc

sample = Path(sys.argv[1])
with contextlib.redirect_stdout(io.StringIO()):
    net = pc.extract_throats_from_bodies_voxel_parallel(
        np.load(sample / 'body_array.npy'), min_throat_size=1, direction='x', merge_bodies=True, verbose=False)
    pc.calculate_network_effective_properties(net, rn_method='short_circuit', verbose=False)
    res = berg_cc.compute_all_berg(net, net.body_array, streamtube_method='greedy_fast', verbose=False)

B = net.body_array
M = int(B.max()) + 1
keys, cnts = [], []
for d in range(3):
    a = np.moveaxis(B, d, 0)
    lo, hi = a[:-1], a[1:]
    m = (lo != hi) & (lo > 0) & (hi > 0)
    u, v = lo[m].astype(np.int64), hi[m].astype(np.int64)
    k, c = np.unique(np.minimum(u, v) * M + np.maximum(u, v), return_counts=True)
    keys.append(k); cnts.append(c)
k = np.concatenate(keys); c = np.concatenate(cnts)
uk, inv = np.unique(k, return_inverse=True)
faces = dict(zip(uk.tolist(), np.bincount(inv, weights=c).tolist()))

arr = res['arrays']
I = np.abs(res['current'])
nf, A, w = [], [], []
for t_idx, tid in enumerate(arr['throat_ids']):
    t = net.throats[int(tid)]
    b1, b2 = sorted(t.connects)
    nf.append(faces.get(b1 * M + b2, 0.0)); A.append(t.area_vox_projection_voxels2); w.append(I[t_idx])
nf, A, w = map(np.asarray, (nf, A, w))
ok = nf > 0
ratio = A[ok] / nf[ok]
out = dict(sample=sample.name, n_throats=int(len(nf)), n_no_shared_face=int((~ok).sum()),
           median_ratio=float(np.median(ratio)),
           current_weighted_ratio=float(np.sum(w[ok] * ratio) / np.sum(w[ok])),
           frac_throats_le4_faces=float(np.mean(nf <= 4)),
           frac_current_le16_faces=float(np.sum(w[nf <= 16]) / np.sum(w)),
           bins={})
for lo_, hi_ in ((1, 4), (5, 16), (17, 64), (65, 10 ** 9)):
    s = ok & (nf >= lo_) & (nf <= hi_)
    if s.any():
        out['bins'][f'{lo_}-{hi_}'] = dict(n=int(s.sum()), median_ratio=float(np.median(A[s] / nf[s])))
json.dump(out, open(sys.argv[2], 'w'), indent=1)
