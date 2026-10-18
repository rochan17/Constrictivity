"""Compare two bench.py outputs field by field: python compare.py base.pkl new.pkl"""
import pickle, sys
import numpy as np

a = pickle.load(open(sys.argv[1], 'rb'))
b = pickle.load(open(sys.argv[2], 'rb'))


def maxrel(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.shape != y.shape:
        return float('inf')
    m = np.isfinite(x) | np.isfinite(y)
    if not m.any():
        return 0.0
    return float(np.max(np.abs(x[m] - y[m]) / np.maximum(np.abs(x[m]), 1e-300)))


rep = {}
rep['merged_identical'] = bool(np.array_equal(a['merged'], b['merged']) and a['merged'].dtype == b['merged'].dtype)
rep['n_merged'] = (a['n_merged'], b['n_merged'])

bp_a, bp_b = a['body_props_direct'], b['body_props_direct']
same_keys = bp_a.keys() == bp_b.keys()
bp_exact = same_keys and all(
    all((np.array_equal(np.asarray(bp_a[k][f], float), np.asarray(bp_b[k][f], float))
         if bp_a[k][f] is not None else bp_b[k][f] is None)
        for f in ('volume_voxels', 'centroid', 'volume_centroid', 'surface_centroid',
                  'max_radius', 'equivalent_radius', 'surface_area_voxels', 'is_surface'))
    for k in bp_a)
rep['body_props_identical'] = bp_exact

ta, tb = a['throats'], b['throats']
rep['throat_ids_same'] = ta.keys() == tb.keys()
if rep['throat_ids_same']:
    rep['throat_voxels_identical'] = all(
        ta[k]['connects'] == tb[k]['connects'] and np.array_equal(ta[k]['coords'], tb[k]['coords'])
        for k in ta)
    rep['throat_area_maxrel'] = maxrel([ta[k]['area'] for k in ta], [tb[k]['area'] for k in ta])
    rep['throat_area_identical'] = all(ta[k]['area'] == tb[k]['area'] for k in ta)

for key in ('L_body1', 'L_body2', 'L_throat', 'A_body1', 'A_body2', 'throat_area', 'throat_volume'):
    rep[f'arr_{key}_maxrel'] = maxrel(a['arrays'][key], b['arrays'][key])
rep['potential_maxabs'] = float(np.max(np.abs(a['potential'] - b['potential'])))
rep['current_maxrel'] = maxrel(a['current'], b['current'])
rep['metrics_maxrel'] = {k: maxrel(a['metrics'][k], b['metrics'][k])
                         for k in a['metrics'] if isinstance(a['metrics'][k], (int, float))}
worst = max(rep['metrics_maxrel'].values())
rep['metrics_worst_rel'] = worst
rep['speedup_total'] = a['times']['total'] / b['times']['total']
rep['times_base'] = {k: round(v, 3) for k, v in a['times'].items()}
rep['times_new'] = {k: round(v, 3) for k, v in b['times'].items()}
for k, v in rep.items():
    print(f'{k:28s}: {v}')
