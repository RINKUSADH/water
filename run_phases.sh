#!/bin/bash
# Generator ablation ladder on one CV fold. Phases 0/1/2/4/5 start immediately; Phase-3 (physics) runs start once
# Track P has written outputs/trackP.DONE. Ends by writing outputs/RESULTS.md.
#   GPU_A/B/C/D: GPUs for the runs (two runs share a GPU; each run needs ~9-13 GB).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/obed/miniconda3/bin/python
MOD=${MODALITY:-T2W}
FOLD=${FOLD:-0}
EPOCHS=${EPOCHS:-60}
SPE=${SPE:-600}
GPU_A=${GPU_A:-2}; GPU_B=${GPU_B:-4}; GPU_C=${GPU_C:-7}; GPU_D=${GPU_D:-0}
COMMON="--modality $MOD --fold $FOLD --epochs $EPOCHS --samples_per_epoch $SPE"
GLOBAL="--attn_blocks 4 --volctx 1 --head lut_bias --adain_dec 1"   # full global model (Phases 1+2+4+5)
mkdir -p logs
log() { echo "[$(date '+%F %T')] $*"; }
launch() {
  local name=$1 gpu=$2; shift 2
  log "launch $name on GPU $gpu -> logs/gen_${name}_fold${FOLD}.log"
  $PY gen/train.py --run "$name" --gpu "$gpu" $COMMON "$@" > "logs/gen_${name}_fold${FOLD}.log" 2>&1 &
}

[ -f "outputs/cache/$MOD/DONE" ] || { log "building cache"; $PY gen/prepare_cache.py --modality "$MOD" > "logs/prepare_cache_$MOD.log" 2>&1; }

launch ph0_base   "$GPU_A"
launch ph1_attn   "$GPU_A" --attn_blocks 4
launch ph2_volctx "$GPU_B" --attn_blocks 4 --volctx 1
launch ph5_head   "$GPU_B" --attn_blocks 4 --volctx 1 --head lut_bias
launch ph4_adain  "$GPU_C" $GLOBAL

log "waiting for Track P"
until [ -f outputs/trackP.DONE ] || [ -f outputs/trackP.FAILED ]; do sleep 120; done
if [ -f outputs/trackP.DONE ]; then
  launch ph3_concat  "$GPU_C" $GLOBAL --phys concat
  launch ph3_spade   "$GPU_D" $GLOBAL --phys spade
  launch ph3_full    "$GPU_D" $GLOBAL --phys spade_xattn --proto 1
  launch ph3_shuffle "$GPU_A" $GLOBAL --phys spade_xattn --proto 1 --phys_shuffle 1
else
  log "Track P failed -> skipping Phase-3 physics runs"
fi
wait
log "all runs finished; summarizing"
$PY gen/summarize.py > logs/summary.log 2>&1
log "PHASES DONE -> outputs/RESULTS.md"
