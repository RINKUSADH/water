"""
precompute_physics.py — run the frozen Track-P ParamNet over every cached MRIxFields volume (all subjects, all fields).

outputs/physics/<MOD>/<sid>_<field>_phys.npy     float16 (4, X, Y, Z): log T1, log T2, M0, mean log-scale
outputs/physics/<MOD>/<sid>_<field>_physctx.npy  float16 (4, Z/4, X/4, Y/4)
ParamNet was trained on 1.5T and 3T MRF data only, so its field input is clamped to [1.5, 3] T; maps for 0.1T/5T/7T
sources are extrapolations.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trackP"))
from common import FIELD_TESLA, MOD_IDX, OUT_DIR  # noqa: E402
from data import CACHE_DIR, CTX_POOL, PHYS_DIR, VolumeBank, cache_path  # noqa: E402
from paramnet import load_paramnet  # noqa: E402

SLAB_P, STRIDE_P = 32, 16


@torch.no_grad()
def infer_volume(net, vol_xyz, modality, field_t, device):
    X, Y, Z = vol_xyz.shape
    vol = torch.from_numpy(np.moveaxis(np.asarray(vol_xyz, dtype=np.float32), -1, 0).copy()).to(device)  # (Z,X,Y)
    acc = torch.zeros((4, Z, X, Y), device=device)
    wacc = torch.zeros((1, Z, 1, 1), device=device)
    win = torch.hann_window(SLAB_P + 2, periodic=False, device=device)[1:-1].clamp_min(0.05).view(1, SLAB_P, 1, 1)
    starts = list(range(0, Z - SLAB_P + 1, STRIDE_P))
    if starts[-1] != Z - SLAB_P:
        starts.append(Z - SLAB_P)
    mod = torch.tensor([MOD_IDX[modality]], device=device)
    fld = torch.tensor([min(max(field_t, 1.5), 3.0)], device=device)
    for z0 in starts:
        x = vol[z0:z0 + SLAB_P][None, None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x, mod, fld)
        o = torch.cat([out["log_t1"], out["log_t2"], out["m0"], out["log_b"].mean(1, keepdim=True)], 1)[0].float()
        acc[:, z0:z0 + SLAB_P] += o * win
        wacc[:, z0:z0 + SLAB_P] += win
    maps = acc / wacc.clamp_min(1e-6)
    brain = (vol > 0.005)[None]
    maps = torch.where(brain, maps, torch.zeros_like(maps))
    ctx = F.avg_pool3d(maps[None], CTX_POOL)[0]
    return maps.permute(0, 2, 3, 1).cpu().numpy(), ctx.cpu().numpy()  # (4,X,Y,Z), (4,Z/4,X/4,Y/4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="T2W")
    ap.add_argument("--ckpt", default=str(OUT_DIR / "paramnet" / "best.pth"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    net, ck = load_paramnet(args.ckpt, device)
    print(f"ParamNet step {ck['step']} val {ck['val']}", flush=True)
    (PHYS_DIR / args.modality).mkdir(parents=True, exist_ok=True)
    meta = json.load(open(CACHE_DIR / args.modality / "meta.json"))
    bank = VolumeBank(args.modality)
    jobs = [(s, f) for s in meta["subjects"] for f in FIELD_TESLA]
    if args.limit:
        jobs = jobs[: args.limit]
    for i, (sid, field) in enumerate(jobs):
        maps, ctx = infer_volume(net, bank.vol(sid, field), args.modality, FIELD_TESLA[field], device)
        np.save(cache_path(args.modality, sid, field, "phys", PHYS_DIR), maps.astype(np.float16))
        np.save(cache_path(args.modality, sid, field, "physctx", PHYS_DIR), ctx.astype(np.float16))
        br = maps[2] != 0
        print(f"[{i + 1}/{len(jobs)}] {sid} {field}: brain T1 med {np.exp(np.median(maps[0][br])):.3f}s "
              f"T2 med {np.exp(np.median(maps[1][br])) * 1000:.0f}ms", flush=True)
    if not args.limit:
        (PHYS_DIR / args.modality / "DONE").write_text("ok\n")
    print("PHYSICS DONE", flush=True)


if __name__ == "__main__":
    main()
