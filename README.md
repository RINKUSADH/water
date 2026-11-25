# PhysGlobal3D — physics-grounded, globally-aware 1.5T→3T MRI synthesis

Successor to `/mnt/raid/obed/Rinku/Task_2_Volumetric3D_TeacherStudent`. Plan and reasoning:
`/DATA1/obed/Rinku/l2h/conversation_log_2026-10-31.md`.

## Layout
```
common.py                  constants, normalization, Bloch equations, FieldTransfer, helpers (T1/T2 in seconds)
trackP/
  p0_prepare_mrf.py        Cambridge (4 sites, 1.5T+3T) + Dryad SAFE (3T, B1+) -> 1 mm maps, masks, split, QC
  paramnet.py              ParamNet 3D U-Net: image + modality + field -> log T1, log T2, M0, uncertainty
  p1_train_paramnet.py     physics-simulated training (random Bloch protocols, bias/B1+, Rician noise, blur);
                           held-out test vs constant baseline -> outputs/paramnet/test_report.md
  p2_field_transfer.py     two-stage rigid registration of 3T->1.5T maps; fit log T_3T = a + b log T_1.5T;
                           leave-one-subject-out vs fixed 2^0.35 rule -> outputs/field_transfer/
gen/
  data.py                  20 paired subjects x 5 fields, 5-fold CV by subject, memory-mapped slab dataset
  prepare_cache.py         normalized float16 volume cache + 2 mm context volumes
  precompute_physics.py    frozen ParamNet over every cached volume -> outputs/physics/
  physics_stack.py         physics stack / tissue-prototype inputs (field transfer + Bloch prior at target field)
  model.py                 PhysGlobalUNet3D with switchable phases (see below)
  losses.py                L1 + 3D SSIM + VGG perceptual; LUT/bias regularizers; challenge-style slice SSIM/PSNR
  train.py                 one ablation run on one fold (+ full-volume eval at the end)
  evaluate.py              sliding-window full-volume eval, identity baseline, low/high-freq error, z-jitter
  summarize.py             -> outputs/RESULTS.md
  metrics3d.py             3D test metrics: PSNR/SSIM (3D, brain voxels), LPIPS/DISTS/FID (2.5D, 3 planes)
  visualize.py             per run: NIfTI predictions, metrics (best + latest, mean ± std), 3-plane / error-frequency /
                           z-profile / head (gate, LUT, bias, residual) / attention / prototype / physics-input /
                           training-curve figures; --compare: cross-run figures + outputs/METRICS3D_fold<k>.md
trackP/visualize_physics.py ParamNet maps per MRIxFields subject x field + T1-rises-with-field check
scripts/run_trackP.sh      Track P end-to-end (writes outputs/trackP.DONE / .FAILED)
scripts/run_phases.sh      generator ablation ladder; physics runs wait for Track P
scripts/run_post.sh        watcher: visualize + metrics for each finished run, physics visuals, final comparison
tests/smoke_test.py        build/forward/backward every variant, zero-init identity check, memory/speed
```

## Model flags (gen/train.py)
| flag | phase | meaning |
|---|---|---|
| `--attn_blocks 4` | 1 | Transformer blocks in the bottleneck (2x22x27 = 1,188 tokens), conv PE, AdaLN-Zero |
| `--volctx 1` | 2 | whole volume at 2 mm -> 64 Perceiver tokens, gated cross-attention, slab z-position |
| `--phys concat/spade/spade_xattn` | 3 | physics as input channels (old) / SPADE decoder / + aligned physics cross-attention with uncertainty bias |
| `--proto 1` | 3b | tissue-prototype tokens (K=12) from physics maps at source and target field |
| `--adain_dec 1` | 4 | decoder modulation from the global vector |
| `--head gate/direct/lut_bias` | 5 | original gate / direct / monotone LUT x smooth bias field + residual |
| `--phys_shuffle 1` | control | physics maps taken from another subject |
All new paths are zero-initialized, so every variant starts as its simpler parent.

## Run
```bash
cd /DATA1/obed/Rinku/l2h/PhysGlobal3D
nohup setsid bash scripts/run_trackP.sh  > logs/run_trackP.out  2>&1 &
nohup setsid bash scripts/run_phases.sh  > logs/run_phases.out  2>&1 &
# env overrides: GPU_P, GPU_A..GPU_D, FOLD, EPOCHS, SPE, P1_ITERS, MODALITY
nohup setsid bash scripts/run_post.sh    > logs/run_post_fold0.out 2>&1 &   # visuals + 3D metrics
tail -f logs/run_phases.out logs/gen_*_fold0.log
```
One run by hand: `python gen/visualize.py --run_dir outputs/gen/ph4_adain_fold0 --gpu 0`; comparison:
`python gen/visualize.py --compare --fold 0`.

## Test metrics
Test set = the fold's held-out subjects (4 per fold); tables give mean ± sample std over subjects. `best.pth` is
selected on validation slabs of the same subjects, so `latest.pth` (final epoch) is the unbiased number; both are
reported. LPIPS / DISTS / FID use 2D ImageNet networks and are averaged over axial, coronal and sagittal slices
(2.5D); FID is per subject (slice sets) plus one pooled FID. Physics runs also get a test-time physics swap (another
subject's maps) to measure how much the model depends on subject-specific T1/T2/M0. With n = 4 the std is noisy —
run all 5 folds for n = 20.
Other folds: `FOLD=1 bash scripts/run_phases.sh` (Track P outputs are reused).

## Known limitations
* Brain masks for MRF data are heuristic (no skull-stripping tool installed) — check `outputs/mrf/qc/`.
* T1W simulation uses a simplified MPRAGE-like IR-GRE equation.
* ParamNet is trained on 1.5T/3T only; its field input is clamped to [1.5, 3] T for 0.1T/5T/7T sources.
* Field transfer is measured for 1.5T->3T only; other field pairs use a log-field extrapolation.
* Unpaired pretraining on Training_retrospective (masked-volume modeling) is not implemented yet.
* Check the MRIxFields rules before training on Validating_prospective subjects for a challenge submission.
