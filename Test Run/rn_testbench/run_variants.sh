#!/bin/bash
# usage: run_variants.sh <data_root> <tag> <key=val...>  (runs every subfolder with body_array.npy)
S=$(cd "$(dirname "$0")" && pwd); R=$(cd "$S/../.." && pwd)   # repo root
PY=${PY:-/Users/rochan/miniconda3/envs/rbenv/bin/python}   # override: PY=python ./run_variants.sh ...
root=$1; tag=$2; shift 2
mkdir -p "$S/$tag"
for d in "$root"/*/; do
  n=$(basename "$d"); [ -f "$d/body_array.npy" ] || continue
  $PY -I "$S/bench.py" "$R" "$d" "$S/$tag/$n.pkl" "$@" > "$S/$tag/$n.json" 2> "$S/$tag/$n.err" &
  while [ $(jobs -r | wc -l) -ge ${NPAR:-5} ]; do sleep 0.5; done
done
wait
