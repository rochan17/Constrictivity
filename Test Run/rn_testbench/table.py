"""python table.py <data_root> <tag> [<tag> ...]  -> per-sample table vs TauFactor + summary."""
import glob, json, os, pickle, sys
import numpy as np
import cache_lookup

root = sys.argv[1]
tags = sys.argv[2:]
S = os.path.dirname(os.path.abspath(__file__))

rows = []
for sample in sorted(os.listdir(root)):
    tf = cache_lookup.taufactor_metrics(os.path.join(root, sample))
    if tf is None:
        continue
    for tag in tags:
        p = os.path.join(S, tag, sample + '.json')
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            continue
        m = json.load(open(p))['metrics']
        rows.append(dict(sample=sample, tag=tag, tf=tf, m=m,
                         t=json.load(open(p))['times']['total']))


def e(x, ref):
    return 100.0 * (x - ref) / ref


print(f"{'sample':24s} {'variant':12s} {'1/F':>7s} {'err%':>6s} | {'tau_B':>6s} {'T_I':>6s} {'TFtau':>6s} {'errB%':>6s} {'errI%':>6s} | {'1/Cc':>6s} {'TF C':>6s} {'err%':>6s} | {'fb':>4s} {'Omega/OmTF':>9s}")
summ = {t: dict(F=[], tB=[], tI=[], C=[]) for t in tags}
for r in rows:
    tf, m = r['tf'], r['m']
    eF = e(m['one_over_F'], tf['D_eff']); eB = e(m['tau_c'], tf['tau_reconciled'])
    eI = e(m['T_flux_weighted'], tf['tau_reconciled']); eC = e(m['inv_C_c'], tf['constrictivity'])
    s = summ[r['tag']]; s['F'].append(eF); s['tB'].append(eB); s['tI'].append(eI); s['C'].append(eC)
    print(f"{r['sample']:24s} {r['tag']:12s} {m['one_over_F']:7.4f} {eF:+6.1f} | {m['tau_c']:6.3f} {m['T_flux_weighted']:6.3f} "
          f"{tf['tau_reconciled']:6.3f} {eB:+6.1f} {eI:+6.1f} | {m['inv_C_c']:6.3f} {tf['constrictivity']:6.3f} {eC:+6.1f} | "
          f"{m['n_fallback']:4d} {m['phi_total']/tf['porosity']:9.4f}")
print()
print(f"{'variant':12s}  n   1/F: mean err / MAE      tau_B: mean / MAE    T_I: mean / MAE     1/Cc: mean / MAE")
for t, s in summ.items():
    if not s['F']:
        continue
    f = lambda v: f"{np.mean(v):+6.1f} / {np.mean(np.abs(v)):5.1f}"
    print(f"{t:12s} {len(s['F']):2d}   {f(s['F'])}        {f(s['tB'])}      {f(s['tI'])}     {f(s['C'])}")
