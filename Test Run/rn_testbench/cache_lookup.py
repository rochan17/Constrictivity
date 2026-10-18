"""Read the TauFactor metrics already cached by pipeline.py for a sample folder."""
import glob, json, os


def taufactor_metrics(folder):
    versions = sorted(glob.glob(os.path.join(folder, 'version*')),
                      key=lambda p: int(p.rsplit('version', 1)[1]))
    for v in reversed(versions):
        summ = os.path.join(v, 'summary.json')
        if os.path.exists(summ):
            tf = json.load(open(summ)).get('taufactor')
            if tf and 'D_eff' in tf:
                return tf
        real = os.path.join(v, 'taufactor', 'metrics.json')
        if os.path.exists(real):
            return json.load(open(real))
    return None


_CSV_CACHE = {}


def _csv_taufactor(folder):
    """Fallback: <root>/all_runs_versions.csv (collect_results.py output), latest version row."""
    import csv
    root, name = os.path.split(os.path.normpath(folder))
    path = os.path.join(root, 'all_runs_versions.csv')
    if not os.path.exists(path):
        return None
    if path not in _CSV_CACHE:
        best = {}
        for row in csv.DictReader(open(path)):
            if not row.get('taufactor.D_eff'):
                continue
            n, v = row['microstructure'], int(row['version'])
            if n not in best or v > best[n][0]:
                best[n] = (v, row)
        _CSV_CACHE[path] = {n: {k.split('.', 1)[1]: float(r[k]) for k in
                                ('taufactor.porosity', 'taufactor.D_eff', 'taufactor.tau_solver',
                                 'taufactor.tau_reconciled', 'taufactor.constrictivity')}
                            for n, (v, r) in best.items()}
    return _CSV_CACHE[path].get(name)


_orig = taufactor_metrics


def taufactor_metrics(folder):  # noqa: F811
    return _orig(folder) or _csv_taufactor(folder)
