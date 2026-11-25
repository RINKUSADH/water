#!/bin/bash
# Track P: MRF preprocessing -> ParamNet pretraining (GPU) + field-transfer fit (CPU, in parallel)
#          -> physics maps for every MRIxFields volume (needed by the Phase-3 generator runs).
# Writes outputs/trackP.DONE on success, outputs/trackP.FAILED on any error.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/obed/miniconda3/bin/python
GPU=${GPU_P:-0}
MOD=${MODALITY:-T2W}
ITERS=${P1_ITERS:-30000}
mkdir -p logs outputs
rm -f outputs/trackP.DONE outputs/trackP.FAILED
trap 'echo "[$(date "+%F %T")] TRACK P FAILED (line $LINENO)"; touch outputs/trackP.FAILED' ERR
log() { echo "[$(date '+%F %T')] $*"; }

if [ ! -f outputs/mrf/index.json ]; then
  log "P0 preprocess MRF"; $PY trackP/p0_prepare_mrf.py --workers 24 > logs/p0_prepare.log 2>&1
fi
log "P1 ParamNet ($ITERS iters) on GPU $GPU -> logs/p1_paramnet.log"
$PY trackP/p1_train_paramnet.py --gpu "$GPU" --iters "$ITERS" > logs/p1_paramnet.log 2>&1 &
P1=$!
log "P2 field transfer (CPU) -> logs/p2_field_transfer.log"
$PY trackP/p2_field_transfer.py --workers 12 > logs/p2_field_transfer.log 2>&1 &
P2=$!
wait $P2; log "P2 finished"
wait $P1; log "P1 finished: $(grep -m1 'PASS' logs/p1_paramnet.log || true)"

until [ -f "outputs/cache/$MOD/DONE" ]; do sleep 60; done
log "precompute physics maps ($MOD) -> logs/precompute_physics_$MOD.log"
$PY gen/precompute_physics.py --modality "$MOD" --gpu "$GPU" > "logs/precompute_physics_$MOD.log" 2>&1
touch outputs/trackP.DONE
log "TRACK P DONE"
