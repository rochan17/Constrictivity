"""Exact multiplicative error budget of RN 1/F vs TauFactor D_eff, per class.

(1/F_RN) / D_TF = (phi_c/eps) * ((1/C_c) / C_TF) * (T_TF / tau_B)^2
since 1/F_RN = phi_c * tau_sq_c / C_c (Berg, exact on the network) and
D_TF = eps * C_TF / T_TF^2 (TauFactor's constrictivity definition).

python budget.py <data_root> <tag> [<tag>...]   [writes budget_<tag>.csv]
"""
import json, os, sys, csv
import numpy as np
import cache_lookup

root, tags = sys.argv[1], sys.argv[2:]
S = os.path.dirname(os.path.abspath(__file__))
for tag in tags:
    rows = []
    for sample in sorted(os.listdir(root)):
        tf = cache_lookup.taufactor_metrics(os.path.join(root, sample))
        p = os.path.join(S, tag, sample + '.json')
        if tf is None or not os.path.exists(p):
            continue
        j = json.load(open(p)); m = j['metrics']
        f_phi = m['phi_c'] / tf['porosity']
        f_C = m['inv_C_c'] / tf['constrictivity']
        f_T = (tf['tau_reconciled'] / m['tau_c']) ** 2
        ratio = m['one_over_F'] / tf['D_eff']
        rows.append(dict(sample=sample, cls=sample.split('_')[0], ratio=ratio, f_phi=f_phi, f_C=f_C, f_T=f_T,
                         check=f_phi * f_C * f_T / ratio, fb_frac=m['n_fallback'] / m['n_throats'],
                         n_bodies=m['n_bodies'], n_throats=m['n_throats'],
                         TF_C=tf['constrictivity'], TF_D=tf['D_eff'], eps=tf['porosity'],
                         T_I_err=m['T_flux_weighted'] / tf['tau_reconciled'] - 1,
                         tau_B_err=m['tau_c'] / tf['tau_reconciled'] - 1,
                         one_over_F=m['one_over_F'], tau_c=m['tau_c'], T_I=m['T_flux_weighted'],
                         inv_C_c=m['inv_C_c'], phi_c=m['phi_c'], time=j['times']['total']))
    with open(os.path.join(S, f'budget_{tag}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f'== {tag}   (geometric means; exact-identity check max |1-x| = '
          f'{max(abs(1 - r["check"]) for r in rows):.1e})')
    print(f"{'class':7s} {'n':>2s} {'1/F ratio':>9s} {'= phi_c/eps':>11s} {'x C factor':>10s} {'x T factor':>10s}"
          f" {'T_I err':>8s} {'fallback':>8s} {'TF C':>6s} {'bodies':>6s} {'t[s]':>5s}")
    for cls in ('Fine', 'Medium', 'Coarse'):
        rs = [r for r in rows if r['cls'] == cls]
        if not rs:
            continue
        g = lambda k: float(np.exp(np.mean(np.log([r[k] for r in rs]))))
        a = lambda k: float(np.mean([r[k] for r in rs]))
        print(f"{cls:7s} {len(rs):2d} {g('ratio'):9.3f} {g('f_phi'):11.3f} {g('f_C'):10.3f} {g('f_T'):10.3f}"
              f" {100*a('T_I_err'):+7.1f}% {100*a('fb_frac'):7.0f}% {a('TF_C'):6.3f} {a('n_bodies'):6.0f} {a('time'):5.1f}")
