#!/bin/bash
# Post-training visuals + 3D test metrics, run next to (not inside) run_phases.sh so live jobs are untouched.
#  * every generator run of the fold is processed as soon as it writes DONE  -> outputs/gen/<run>_fold<k>/vis/
#  * Track P physics visuals once outputs/trackP.DONE exists               -> outputs/physics/<MOD>/vis/
#  * after run_phases.sh prints "PHASES DONE": cross-run comparison         -> outputs/vis_compare/fold<k>/,
#    outputs/METRICS3D_fold<k>.md, appended to outputs/RESULTS.md; then outputs/post_fold<k>.DONE
# GPUs: picks the one with the most free memory among POST_GPUS (only GPUs this project already uses).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/obed/miniconda3/bin/python
MOD=${MODALITY:-T2W}
FOLD=${FOLD:-0}
POST_GPUS=${POST_GPUS:-"0 2 4 7"}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8   # leave CPUs to the training dataloaders
mkdir -p logs
log() { echo "[$(date '+%F %T')] $*"; }
pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F', *' -v ok=" $POST_GPUS " 'index(ok, " " $1 " ") { print $2, $1 }' | sort -nr | head -1 | cut -d' ' -f2
}
phys_vis_done=0
sweep() {
  for d in outputs/gen/*_fold"$FOLD"; do
    [ -f "$d/DONE" ] || continue
    [ -f "$d/vis/DONE" ] || [ -f "$d/vis/FAILED" ] && continue
    local name gpu
    name=$(basename "$d"); gpu=$(pick_gpu)
    log "visualize + metrics: $name on GPU $gpu -> logs/vis_$name.log"
    if ! $PY gen/visualize.py --run_dir "$d" --gpu "$gpu" > "logs/vis_$name.log" 2>&1; then
      mkdir -p "$d/vis"; touch "$d/vis/FAILED"; log "FAILED $name (see logs/vis_$name.log)"
    fi
  done
  if [ "$phys_vis_done" = 0 ] && [ -f outputs/trackP.DONE ]; then
    log "Track P physics visuals -> logs/vis_physics_$MOD.log"
    $PY trackP/visualize_physics.py --modality "$MOD" > "logs/vis_physics_$MOD.log" 2>&1 || log "physics visuals FAILED"
    phys_vis_done=1
  fi
}

log "post-processing fold $FOLD (GPUs: $POST_GPUS)"
until grep -q "PHASES DONE" logs/run_phases.out 2>/dev/null; do
  sweep
  sleep 300
done
sweep
log "cross-run comparison"
$PY gen/visualize.py --compare --fold "$FOLD" --gpu "$(pick_gpu)" > "logs/vis_compare_fold$FOLD.log" 2>&1 || log "compare FAILED"
if [ -f "outputs/METRICS3D_fold$FOLD.md" ] && [ -f outputs/RESULTS.md ] && ! grep -q "3D test metrics — fold $FOLD" outputs/RESULTS.md; then
  { echo; sed 's/^# /## /' "outputs/METRICS3D_fold$FOLD.md"; } >> outputs/RESULTS.md
fi
touch "outputs/post_fold$FOLD.DONE"
log "POST DONE -> outputs/METRICS3D_fold$FOLD.md, outputs/vis_compare/fold$FOLD/"
