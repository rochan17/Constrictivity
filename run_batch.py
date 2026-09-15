"""
run_batch.py
============

Parallel-across-microstructures driver for pipeline.run_microstructure,
sized to a SLURM CPU allocation.

    python run_batch.py --data-dir Data --cores-per-microstructure 16

Given `--cores-per-microstructure N` and a total core budget `M` (taken from
--total-cores, else SLURM_CPUS_PER_TASK / SLURM_CPUS_ON_NODE, else
os.cpu_count()):

    n_workers = max(1, M // N)

microstructures run N-workers-at-a-time in parallel, each pinned to N cores.
If N > M, everything collapses to a single worker using all M cores for one
microstructure at a time (never oversubscribes the allocation).

Per-method, per-version logging: each method's stdout/stderr for an actual
run (not a cached/pointer resolution -- see pipeline.py's run_microstructure)
goes to <data_dir>/<microstructure>/version<N>/<method>/log.txt, written by
pipeline.py itself, not by this module -- so with several parallel workers
you get one readable trail per method per version instead of everything
interleaved into SLURM's shared job log. That shared log still gets
run_batch.py's own orchestration lines ([run_batch] ..., [OK]/[FAIL] per
microstructure). Pass --verbose to make each such log.txt include the full
per-stage diagnostic blocks (network build, KCL solve, streamtube
decomposition, Berg quantities, ...) instead of just the handful of
unconditional summary lines.

This module deliberately imports nothing heavy (numpy/torch/numba/pipeline)
at module scope: pipeline.py is only imported lazily inside each worker
process, after the ProcessPoolExecutor initializer has already set the
BLAS/OpenMP thread-count env vars in that process -- see pipeline.py's
docstring for why import order matters here.
"""

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path


def _resolve_total_cores(explicit):
    if explicit:
        return int(explicit)
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = os.environ.get(var)
        if v:
            return int(v)
    return os.cpu_count() or 1


def _init_worker(cores_per_task: int) -> None:
    """Runs first in every worker process, before pipeline.py (and thus
    numpy/torch/numba) is imported by _worker_entry below."""
    n = str(max(1, int(cores_per_task)))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = n


def _worker_entry(folder_str, cores_per_task, phase_id, axis, segmented_file, body_array_file,
                   taufactor_config, berg_voxel_config, berg_cc_config, save_full, verbose):
    """
    Runs one microstructure. Per-method logging now happens inside
    pipeline.py's run_microstructure itself (version<N>/<method>/log.txt,
    written only when that method actually runs), so this is a direct
    call-and-return -- exceptions propagate to as_completed() in main()
    unmolested, same as the orchestration-level [OK]/[FAIL] bookkeeping
    always relied on.
    """
    from pipeline import run_microstructure
    return run_microstructure(
        folder_str,
        cores_per_task=cores_per_task,
        phase_id=phase_id,
        axis=axis,
        segmented_file=segmented_file,
        body_array_file=body_array_file,
        taufactor_config=taufactor_config,
        berg_voxel_config=berg_voxel_config,
        berg_cc_config=berg_cc_config,
        save_full=save_full,
        verbose=verbose,
    )


def _run_all(microstructures, n_workers, cores_per, submit_args_fn, max_retries):
    """
    Runs _worker_entry for every microstructure, recovering from a worker
    process dying abruptly (BrokenProcessPool -- e.g. OOM-kill or segfault;
    NOT a normal Python exception raised inside run_microstructure) instead
    of letting it poison every other still-pending microstructure in the
    same pool, which is ProcessPoolExecutor's default behaviour.

    Once one worker in a pool dies, every future submitted to that same
    pool -- including ones that hadn't even started yet -- resolves with
    BrokenProcessPool when queried, regardless of whether that particular
    microstructure had anything wrong with it. Those are exactly the
    microstructures re-queued into a fresh pool on the next round; a
    microstructure whose OWN run raises a normal exception (bad data, no
    percolating path, etc.) is recorded as a real, final failure and never
    retried -- retrying it would just reproduce the same genuine error.

    Returns (n_ok, n_fail).
    """
    pending = list(microstructures)
    n_ok, n_fail = 0, 0
    round_num = 0

    while pending and round_num <= max_retries:
        round_num += 1
        if round_num > 1:
            print(f"[run_batch] pool recovery: retry round {round_num - 1}, "
                  f"{len(pending)} microstructure(s) with an unknown outcome remaining")

        next_pending = []
        with ProcessPoolExecutor(max_workers=n_workers,
                                  initializer=_init_worker, initargs=(cores_per,)) as ex:
            futures = {ex.submit(_worker_entry, *submit_args_fn(m)): m for m in pending}
            for fut in as_completed(futures):
                m = futures[fut]
                try:
                    fut.result()
                    n_ok += 1
                    print(f"[OK]   {m.name}")
                except BrokenProcessPool:
                    next_pending.append(m)
                except Exception:
                    n_fail += 1
                    print(f"[FAIL] {m.name}")
                    traceback.print_exc()

        pending = next_pending

    if pending:
        print(f"[run_batch] pool kept dying after {max_retries} retry round(s); "
              f"giving up on {len(pending)} microstructure(s):")
        for m in pending:
            n_fail += 1
            print(f"[FAIL] {m.name} (pool crashed on every attempt)")

    return n_ok, n_fail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True,
                     help="Folder containing one subfolder per microstructure "
                          "(each with segmented.npy and body_array.npy, unless "
                          "overridden by that microstructure's own manifest.json).")
    ap.add_argument("--cores-per-microstructure", type=int, required=True)
    ap.add_argument("--total-cores", type=int, default=None,
                     help="Override the SLURM/os.cpu_count() core budget.")
    ap.add_argument("--phase-id", type=int, default=1,
                     help="Label in segmented.npy identifying the pore phase. "
                          "Default for every microstructure; override per-folder "
                          "via manifest.json (see pipeline._load_manifest).")
    ap.add_argument("--axis", type=int, default=0, choices=[0, 1, 2],
                     help="Flow direction (0/1/2 = x/y/z) shared by all three methods "
                          "by default -- folded into taufactor_config['axis'], "
                          "berg_voxel_config['axis'], and berg_cc_config['direction'] "
                          "automatically (see pipeline.run_microstructure), so you only "
                          "set this once instead of three separate ways. taufactor has "
                          "no native axis parameter (its Solver always solves along "
                          "literal array axis 0); pipeline.py works around that by "
                          "transposing the input before solving and transposing "
                          "save_full outputs back. Override per-folder via manifest.json.")
    ap.add_argument("--segmented-file", default="segmented.npy",
                     help="Filename (within each microstructure folder) of the phase-labelled "
                          "3D array taufactor/berg_voxel read. Default for every microstructure; "
                          "override per-folder via manifest.json.")
    ap.add_argument("--body-array-file", default="body_array.npy",
                     help="Filename (within each microstructure folder) of the watershed-"
                          "labelled body array berg_cc reads. Default for every microstructure; "
                          "override per-folder via manifest.json.")
    ap.add_argument("--taufactor-config", default="{}",
                     help="JSON dict, default for every microstructure "
                          "(per-folder overrides via manifest.json).")
    ap.add_argument("--berg-voxel-config", default="{}", help="JSON dict, see --taufactor-config.")
    ap.add_argument("--berg-cc-config", default="{}", help="JSON dict, see --taufactor-config.")
    ap.add_argument("--save-full", action="store_true",
                     help="Also pickle full result objects alongside metrics.json.")
    ap.add_argument("--verbose", action="store_true",
                     help="Include each stage's diagnostic blocks (network build, KCL solve, "
                          "streamtube decomposition, Berg quantities, ...) in that method's "
                          "version<N>/<method>/log.txt (only written when a method actually "
                          "runs, not for one resolved via pointer), not just the final "
                          "summary. Note: BergPNConfig.progress_every is currently unused (no "
                          "periodic in-decomposition progress line exists in this codebase "
                          "yet), so this gets you full per-stage summaries after each stage "
                          "finishes, not incremental progress during a long decomposition.")
    ap.add_argument("--max-retries", type=int, default=3,
                     help="If a worker process dies abruptly (OOM-kill, segfault -- not a "
                          "normal exception from a microstructure's own run), every other "
                          "microstructure still pending in that same pool would otherwise be "
                          "lost too (BrokenProcessPool). Instead, up to this many times, a "
                          "fresh pool is started for just the microstructures whose outcome "
                          "is still unknown. A microstructure whose own run raises a normal "
                          "exception is never retried -- only ones caught in someone else's "
                          "pool crash. Set to 0 to disable recovery (old behaviour).")
    args = ap.parse_args()

    total_cores = _resolve_total_cores(args.total_cores)
    cores_per = max(1, args.cores_per_microstructure)
    if cores_per > total_cores:
        print(f"[run_batch] cores-per-microstructure ({cores_per}) > total cores "
              f"({total_cores}); using 1 worker with all {total_cores} cores.")
        n_workers = 1
        cores_per = total_cores
    else:
        n_workers = max(1, total_cores // cores_per)

    data_dir = Path(args.data_dir)
    microstructures = sorted(p for p in data_dir.iterdir() if p.is_dir())
    if not microstructures:
        print(f"[run_batch] no microstructure subfolders found under {data_dir}")
        sys.exit(1)

    taufactor_config = json.loads(args.taufactor_config)
    berg_voxel_config = json.loads(args.berg_voxel_config)
    berg_cc_config = json.loads(args.berg_cc_config)

    print(f"[run_batch] total cores: {total_cores} | cores/microstructure: {cores_per} "
          f"| parallel workers: {n_workers} | microstructures: {len(microstructures)}")

    def submit_args_fn(m):
        return (str(m), cores_per, args.phase_id, args.axis,
                args.segmented_file, args.body_array_file,
                taufactor_config, berg_voxel_config, berg_cc_config,
                args.save_full, args.verbose)

    t0 = time.time()
    n_ok, n_fail = _run_all(microstructures, n_workers, cores_per,
                             submit_args_fn, args.max_retries)

    print(f"[run_batch] done in {time.time() - t0:.1f}s | ok={n_ok} fail={n_fail}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
