"""
evaluate.py — full-volume 1.5T->3T evaluation of a trained run on its fold's validation subjects.

Sliding-window inference along Z (slab 16, stride 8, Hann blending), then on the official slices z in [150, 180):
SSIM / PSNR / NRMSE (skimage, per slice, data_range = GT slice range) and LPIPS; the same for the identity baseline.
Extra diagnostics over the whole brain: low- vs high-frequency error (Gaussian sigma = 8 voxels = 4 mm) and
inter-slab jitter of the mean-intensity profile along Z. Writes eval_full.json and eval_<sid>.png.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import normalized_root_mse, peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIELD_IDX, FIELD_TESLA, FieldTransfer, gaussian_blur3d  # noqa: E402
from data import CACHE_DIR, SLAB, VolumeBank, slab_dhw  # noqa: E402
from model import build_model  # noqa: E402

Z_EVAL = (150, 180)
_LPIPS = {}


def lpips_fn(device):
    if device not in _LPIPS:
        import lpips
        _LPIPS[device] = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    return _LPIPS[device]


@torch.no_grad()
def predict_volume(model, cfg, bank, sid, src, tgt, ft, device, batch=4):
    from train import forward
    vol = bank.vol(sid, src)
    X, Y, Z = vol.shape
    acc = np.zeros((Z, X, Y), np.float32)
    wacc = np.zeros((Z, 1, 1), np.float32)
    win = np.clip(np.hanning(SLAB), 0.1, 1.0).astype(np.float32)[:, None, None]
    starts = list(range(0, Z - SLAB + 1, SLAB // 2))
    if starts[-1] != Z - SLAB:
        starts.append(Z - SLAB)
    use_phys = cfg["phys"] != "none" or cfg["proto"]
    ctx = torch.from_numpy(np.asarray(bank.ctx(sid, src), np.float32))[None, None].to(device)
    pctx = torch.from_numpy(np.asarray(bank.phys_ctx(sid, src), np.float32))[None].to(device) if use_phys else None
    for i in range(0, len(starts), batch):
        zs = starts[i:i + batch]
        b = len(zs)
        x = torch.from_numpy(np.stack([slab_dhw(vol, z) for z in zs])[:, None]).to(device)
        bt = {"src": x, "src_idx": [FIELD_IDX[src]] * b, "tgt_idx": [FIELD_IDX[tgt]] * b,
              "src_t": [FIELD_TESLA[src]] * b, "tgt_t": [FIELD_TESLA[tgt]] * b,
              "zpos": [(z + SLAB / 2) / Z for z in zs], "ctx": ctx.expand(b, -1, -1, -1, -1)}
        if use_phys:
            pv = bank.phys_vol(sid, src)
            bt["phys"] = torch.from_numpy(np.stack([slab_dhw(pv, z) for z in zs])).to(device)
            bt["phys_ctx"] = pctx.expand(b, -1, -1, -1, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, _ = forward(model, bt, cfg, ft, device)
        pred = pred.float().cpu().numpy()[:, 0]
        for z, p in zip(zs, pred):
            acc[z:z + SLAB] += p * win
            wacc[z:z + SLAB] += win
    return np.moveaxis(np.clip(acc / np.maximum(wacc, 1e-5), 0, 1), 0, -1)  # (X, Y, Z)


def slice_metrics(pred, gt, device):
    lp = lpips_fn(device)
    out = {"ssim": [], "psnr": [], "nrmse": [], "lpips": []}
    for z in range(*Z_EVAL):
        p, g = pred[:, :, z], gt[:, :, z]
        dr = max(float(g.max() - g.min()), 1e-4)
        out["ssim"].append(structural_similarity(g, p, data_range=dr))
        out["psnr"].append(peak_signal_noise_ratio(g, p, data_range=dr))
        out["nrmse"].append(normalized_root_mse(g, p))
        pt = torch.from_numpy(p).float()[None, None].repeat(1, 3, 1, 1).to(device) * 2 - 1
        gt_t = torch.from_numpy(g).float()[None, None].repeat(1, 3, 1, 1).to(device) * 2 - 1
        with torch.no_grad():
            out["lpips"].append(float(lp(pt, gt_t)))
    return {k: float(np.mean(v)) for k, v in out.items()}


def global_metrics(pred, gt, src, device):
    brain = torch.from_numpy(src > 0.005).to(device)
    err = (torch.from_numpy(pred).to(device) - torch.from_numpy(gt).to(device)) * brain
    low = gaussian_blur3d(err[None, None], (8.0, 8.0, 8.0))[0, 0]
    n = brain.sum().clamp_min(1)
    res = {"err_low_freq": float((low.abs() * brain).sum() / n),
           "err_high_freq": float(((err - low).abs() * brain).sum() / n)}
    bz = brain.float().sum((0, 1)).clamp_min(1)
    pp = (torch.from_numpy(pred).to(device) * brain).sum((0, 1)) / bz
    gp = (torch.from_numpy(gt).to(device) * brain).sum((0, 1)) / bz
    d = (pp - gp)[60:300]
    res["z_jitter"] = float(d.diff().abs().mean())
    res["z_profile_bias"] = float(d.abs().mean())
    return res


def save_png(src, pred, gt, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z, y = 165, src.shape[1] // 2
    rows = [("axial z=165", lambda v: np.rot90(v[:, :, z])), ("coronal", lambda v: np.rot90(v[:, y, :]))]
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    for r, (name, sl) in enumerate(rows):
        for c, (t, v) in enumerate([("1.5T input", src), ("prediction", pred), ("3T ground truth", gt)]):
            ax[r, c].imshow(sl(v), cmap="gray", vmin=0, vmax=1)
            ax[r, c].set_title(f"{t} ({name})", fontsize=9)
        ax[r, 3].imshow(sl(np.abs(pred - gt)), cmap="inferno", vmin=0, vmax=0.3)
        ax[r, 3].set_title("|pred - gt| (0..0.3)", fontsize=9)
        for a in ax[r]:
            a.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=70)
    plt.close(fig)


def evaluate_run(run_dir, device, ckpt_name="best.pth", src="1.5T", tgt="3T"):
    run_dir = Path(run_dir)
    ck = torch.load(run_dir / ckpt_name, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    use_phys = cfg["phys"] != "none" or cfg["proto"]
    ft = FieldTransfer.from_json(cfg["field_transfer"]) if use_phys else FieldTransfer()
    bank = VolumeBank(cfg["modality"], use_phys)
    meta = json.load(open(CACHE_DIR / cfg["modality"] / "meta.json"))
    subjects = meta["folds"][cfg["fold"]]["val"]
    per = []
    for i, sid in enumerate(subjects):
        pred = predict_volume(model, cfg, bank, sid, src, tgt, ft, device)
        s, g = np.asarray(bank.vol(sid, src), np.float32), np.asarray(bank.vol(sid, tgt), np.float32)
        row = {"subject": sid, "model": {**slice_metrics(pred, g, device), **global_metrics(pred, g, s, device)},
               "identity": {**slice_metrics(s, g, device), **global_metrics(s, g, s, device)}}
        per.append(row)
        print(f"  eval {sid}: model SSIM {row['model']['ssim']:.4f} PSNR {row['model']['psnr']:.2f} | "
              f"identity SSIM {row['identity']['ssim']:.4f} PSNR {row['identity']['psnr']:.2f}", flush=True)
        if i == 0:
            save_png(s, pred, g, run_dir / f"eval_{sid}.png", f"{run_dir.name} — subject {sid}")
    mean = {w: {k: float(np.mean([r[w][k] for r in per])) for k in per[0][w]} for w in ("model", "identity")}
    res = {"run": run_dir.name, "ckpt_epoch": ck["epoch"], "direction": f"{src}->{tgt}", "subjects": subjects,
           "mean": mean, "per_subject": per}
    json.dump(res, open(run_dir / "eval_full.json", "w"), indent=1)
    print(f"FULL EVAL {run_dir.name}: " + json.dumps(mean), flush=True)
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    evaluate_run(a.run_dir, torch.device(f"cuda:{a.gpu}"))
