"""
prepare_cache.py — normalize every paired MRIxFields volume once and store float16 .npy files for memory-mapping.

outputs/cache/<MOD>/<sid>_<field>_vol.npy  (X, Y, Z)      globally percentile-normalized (as the original pipeline)
outputs/cache/<MOD>/<sid>_<field>_ctx.npy  (Z/4, X/4, Y/4) 2 mm average-pooled whole-volume context
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIELDS, norm_volume  # noqa: E402
from data import CACHE_DIR, CTX_POOL, cache_path, make_folds, paired_index  # noqa: E402


def job(args):
    modality, sid, field, path = args
    out_v = cache_path(modality, sid, field, "vol")
    out_c = cache_path(modality, sid, field, "ctx")
    if out_v.exists() and out_c.exists():
        return f"{sid} {field}: cached"
    vol = norm_volume(nib.load(str(path)).get_fdata(dtype=np.float32))
    np.save(out_v, vol.astype(np.float16))
    t = torch.from_numpy(np.moveaxis(vol, -1, 0)[None, None].copy())  # (1,1,Z,X,Y)
    ctx = F.avg_pool3d(t, CTX_POOL)[0, 0].numpy()
    np.save(out_c, ctx.astype(np.float16))
    return f"{sid} {field}: {vol.shape} -> ctx {ctx.shape}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="T2W")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    (CACHE_DIR / args.modality).mkdir(parents=True, exist_ok=True)
    idx = paired_index(args.modality)
    print(f"{len(idx)} paired subjects: {list(idx)}", flush=True)
    jobs = [(args.modality, s, f, idx[s][f]) for s in idx for f in FIELDS]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for msg in ex.map(job, jobs):
            print("  " + msg, flush=True)
    folds = make_folds(list(idx), 5, 0)
    meta = {"subjects": list(idx), "paths": {s: {f: str(p) for f, p in v.items()} for s, v in idx.items()},
            "folds": [{"train": tr, "val": va} for tr, va in folds]}
    json.dump(meta, open(CACHE_DIR / args.modality / "meta.json", "w"), indent=1)
    (CACHE_DIR / args.modality / "DONE").write_text("ok\n")
    print("CACHE DONE", flush=True)


if __name__ == "__main__":
    main()
