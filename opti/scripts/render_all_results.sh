#!/usr/bin/env bash
# render_all_results.sh -- one K x K figure per run in results/
#
# Each results directory is handed to scripts/render_kxk.py, which finds the
# state inside it, infers the complex, and writes figures/<complex>_kxk_<run>_<covered>.{png,pdf}.
# Directories holding no annealer state (the *starcover* certificate dirs) are
# reported as skipped, not failed.
#
#   bash scripts/render_all_results.sh                 # everything
#   bash scripts/render_all_results.sh rp2 klein       # only matching dirs
#   PY=/usr/bin/python3 bash scripts/render_all_results.sh
#
# _ALREADY_DELIVERED are runs whose figures already exist under hand-picked
# names (figures/rp2_kxk_k5_540.*, figures/klein_kxk_k5_1186.*) -- skipped so
# the same picture does not appear twice in figures/ under two names.
set -u
PY=${PY:-python3}
cd "$(dirname "$0")/.." || exit 1
mkdir -p figures
_ALREADY_DELIVERED=" rp2_k5_push_warm klein_cold666_seed_999 "

filters=("$@")
match() {                     # match <dir> against the positional filters
    [ ${#filters[@]} -eq 0 ] && return 0
    for f in "${filters[@]}"; do case "$1" in *"$f"*) return 0 ;; esac; done
    return 1
}

ok=0; skip=0; fail=0
for d in results/*/; do
    d=${d%/}
    name=$(basename "$d")
    match "$name" || continue
    case "$_ALREADY_DELIVERED" in *" $name "*) echo "skip   $name  (figure already delivered)"; skip=$((skip + 1)); continue ;; esac
    out=$("$PY" scripts/render_kxk.py "$d" 2>&1)
    case $? in
        0) echo "ok     $name  $(echo "$out" | tail -1 | sed 's/^wrote //;s/ .*//')"; ok=$((ok + 1)) ;;
        *) if echo "$out" | grep -q "no state with best_chains"; then
               echo "skip   $name  (no state)"
           else
               echo "FAIL   $name"; echo "$out" | tail -4 | sed 's/^/         /'
           fi
           skip=$((skip + 1)) ;;
    esac
done
echo "---- $ok rendered, $skip skipped/failed"
