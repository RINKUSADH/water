"""
p1_train_paramnet.py — Track P1: physics-simulated pretraining of ParamNet on real MRF maps.

Each step: pick MRF scans -> simulate a weighted image with the Bloch equations and randomized sequence
parameters -> add bias field (incl. measured B1+ for Dryad), Rician noise, blur, gamma -> normalize like
MRIxFields -> crop a 1 mm slab and upsample x2 to the 0.5 mm MRIxFields grid -> predict (log T1, log T2, M0)
with a Laplace uncertainty. Labels are the true MRF maps, so supervision is exact and aligned.

Outputs (outputs/paramnet/): best.pth, latest.pth, history.json, test_report.json/.md, qc_test.png
Pass criteria on held-out subjects (T2W input): MAE below the constant-mean baseline for T1, T2, M0, and
Pearson r > 0.7 for T2 (T1 from long-TR T2W is weakly identifiable; reported, not required).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DEFAULT_SEQ, MOD_IDX, MODALITIES, OUT_DIR, bloch_signal, gaussian_blur3d,  # noqa: E402
                    pearson, sample_seq)
from paramnet import ParamUNet3D  # noqa: E402

CROP_1MM = (16, 96, 96)        # (Z, X, Y) at 1 mm -> (32, 192, 192) at 0.5 mm for training
EVAL_CROP_1MM = (16, 176, 208)  # -> (32, 352, 416) at 0.5 mm for evaluation (covers the brain in-plane)


class MRFStore:
    """Holds the preprocessed MRF scans of one split on the GPU (float16)."""

    def __init__(self, mrf_dir: Path, split: str, device, limit: int = 0):
        index = {d["id"]: d for d in json.load(open(mrf_dir / "index.json"))}
        ids = json.load(open(mrf_dir / "split.json"))[split]
        if limit:
            ids = ids[:limit]
        self.items = []
        for sid in ids:
            maps = torch.from_numpy(np.load(mrf_dir / f"{sid}_maps.npy")).to(device)
            masks = torch.from_numpy(np.load(mrf_dir / f"{sid}_masks.npy")).to(device)
            self.items.append({"id": sid, "maps": maps, "masks": masks, **index[sid]})
        print(f"  [{split}] {len(self.items)} scans on {device}", flush=True)

    def __len__(self):
        return len(self.items)


def smooth_bias(shape, sigma, device, gen):
    """Multiplicative smooth bias field exp(N(0, sigma)) from a 4^3 control grid."""
    ctrl = torch.randn((1, 1, 4, 4, 4), generator=gen, device=device) * sigma
    return torch.exp(F.interpolate(ctrl, size=shape, mode="trilinear", align_corners=True))[0, 0]


def simulate(item, modality, rng, gen, deterministic=False):
    """Returns the normalized simulated image (X, Y, Z) float32 and the tissue mask used."""
    maps = item["maps"].float()
    t1, t2, m0, b1 = maps[0], maps[1], maps[2], maps[3]
    head, brain = item["masks"][0].bool(), item["masks"][1].bool()
    if deterministic:
        mask, seq, snr, bias = brain, DEFAULT_SEQ[modality], 40.0, None
        sig_xy, sig_z, q, gamma = 0.0, 0.0, 0.98, 1.0
    else:
        mask = brain if rng.random() < 0.8 else head  # MRIxFields inputs are skull-stripped
        seq = sample_seq(modality, rng)
        snr = rng.uniform(15.0, 80.0)
        bias = smooth_bias(t1.shape, rng.uniform(0.0, 0.3), t1.device, gen)
        if item["has_b1"] and rng.random() < 0.7:
            bias = bias * b1.clamp(0.5, 1.5) ** rng.uniform(0.0, 1.0)
        sig_xy, sig_z = rng.uniform(0.0, 0.8), rng.uniform(0.0, 1.5)
        q, gamma = rng.uniform(0.97, 0.99), math.exp(rng.normal(0.0, 0.15))

    s = bloch_signal(m0, t1, t2, modality, seq)
    if bias is not None:
        s = s * bias
    s = s * mask
    ref = torch.quantile(s[mask][::7], 0.99)
    sigma = float(ref) / snr
    n1 = torch.randn(s.shape, generator=gen, device=s.device) * sigma
    n2 = torch.randn(s.shape, generator=gen, device=s.device) * sigma
    s = torch.sqrt((s + n1) ** 2 + n2 ** 2) * mask
    if sig_xy > 0 or sig_z > 0:
        s = gaussian_blur3d(s[None, None], (sig_xy, sig_xy, sig_z))[0, 0]
    hi = torch.quantile(s[mask][::7], q)
    s = (s / hi.clamp_min(1e-6)).clamp(0.0, 1.0) ** gamma
    return s, mask


def crop_upsample(vols, center, size_1mm):
    """vols: list of (X, Y, Z) tensors. Crops size_1mm=(cz, cx, cy) around center (x, y, z) with zero padding,
    returns tensors (1, 1, 2cz, 2cx, 2cy) in (D=Z, H=X, W=Y) order upsampled x2."""
    cz, cx, cy = size_1mm
    X, Y, Z = vols[0][0].shape
    x0, y0, z0 = center[0] - cx // 2, center[1] - cy // 2, center[2] - cz // 2
    out = []
    for v, mode in vols:
        pad = (max(0, -z0), max(0, z0 + cz - Z), max(0, -y0), max(0, y0 + cy - Y), max(0, -x0), max(0, x0 + cx - X))
        vp = F.pad(v[None, None].float(), pad)
        xs, ys, zs = x0 + pad[4], y0 + pad[2], z0 + pad[0]
        c = vp[:, :, xs:xs + cx, ys:ys + cy, zs:zs + cz].permute(0, 1, 4, 2, 3)  # -> (1,1,Z,X,Y)
        if mode == "nearest":
            c = F.interpolate(c, scale_factor=2, mode="nearest")
        else:
            c = F.interpolate(c, scale_factor=2, mode="trilinear", align_corners=False)
        out.append(c)
    return out


def make_sample(item, modality, rng, gen, deterministic=False, center=None, size=CROP_1MM):
    img, mask = simulate(item, modality, rng, gen, deterministic)
    maps = item["maps"].float()
    if center is None:
        nz = torch.nonzero(item["masks"][1][::4, ::4, ::4])
        c = nz[int(rng.integers(len(nz)))] * 4
        center = [int(c[0]), int(c[1]), int(c[2])]
    log_t1, log_t2, m0 = torch.log(maps[0]), torch.log(maps[1]), maps[2]
    x, lt1, lt2, m, msk = crop_upsample(
        [(img, "linear"), (log_t1, "linear"), (log_t2, "linear"), (m0, "linear"), (mask.float(), "nearest")],
        center, size)
    if not deterministic:
        if rng.random() < 0.5:
            x, lt1, lt2, m, msk = (t.flip(3) for t in (x, lt1, lt2, m, msk))
        if rng.random() < 0.5:
            x, lt1, lt2, m, msk = (t.flip(4) for t in (x, lt1, lt2, m, msk))
    return x, torch.cat([lt1, lt2, m], 1), msk


def laplace_nll(pred, target, mask):
    mu = torch.cat([pred["log_t1"], pred["log_t2"], pred["m0"]], 1)
    log_b = pred["log_b"]
    nll = (target - mu).abs() * torch.exp(-log_b) + log_b
    l1 = (target - mu).abs()
    w = mask.expand_as(nll)
    denom = w.sum().clamp_min(1.0)
    return (nll * w).sum() / denom, (l1 * w).sum() / denom


@torch.no_grad()
def constant_baseline(store):
    """Field-specific mean of each map over training brain voxels (the trivial predictor to beat)."""
    acc = {}
    for it in store.items:
        br = it["masks"][1].bool()
        v = it["maps"].float()
        key = it["field"]
        a = acc.setdefault(key, [0.0, 0.0, 0.0, 0])
        a[0] += float(v[0][br].sum())
        a[1] += float(v[1][br].sum())
        a[2] += float(v[2][br].sum())
        a[3] += int(br.sum())
    return {k: {"t1": a[0] / a[3], "t2": a[1] / a[3], "m0": a[2] / a[3]} for k, a in acc.items()}


def eval_centers(item, n_z=4):
    br = item["masks"][1].bool()
    nz = torch.nonzero(br)
    lo, hi = nz.min(0).values, nz.max(0).values
    cx, cy = int((lo[0] + hi[0]) // 2), int((lo[1] + hi[1]) // 2)
    zs = np.linspace(int(lo[2]) + 12, int(hi[2]) - 12, n_z).astype(int)
    return [[cx, cy, int(z)] for z in zs]


@torch.no_grad()
def evaluate(net, store, const, modalities, device, max_scans=0):
    net.eval()
    rng, gen = np.random.default_rng(123), torch.Generator(device=device).manual_seed(123)
    rows = []
    for it in store.items[: max_scans or None]:
        for mod in modalities:
            preds, tgts = [], []
            for center in eval_centers(it):
                x, tgt, msk = make_sample(it, mod, rng, gen, deterministic=True, center=center, size=EVAL_CROP_1MM)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = net(x, torch.tensor([MOD_IDX[mod]], device=device), torch.tensor([it["field"]], device=device))
                m = msk[0, 0] > 0.5
                p = torch.stack([out["log_t1"][0, 0][m].exp(), out["log_t2"][0, 0][m].exp(), out["m0"][0, 0][m]])
                g = torch.stack([tgt[0, 0][m].exp(), tgt[0, 1][m].exp(), tgt[0, 2][m]])
                u = out["log_b"][0, :, m].exp().mean(0)
                preds.append(torch.cat([p, u[None]]).cpu().numpy())
                tgts.append(g.cpu().numpy())
            p, g = np.concatenate(preds, 1), np.concatenate(tgts, 1)
            c = const.get(it["field"], next(iter(const.values())))
            err = np.abs(p[:3] - g)
            row = {"id": it["id"], "field": it["field"], "modality": mod}
            for i, k in enumerate(["t1", "t2", "m0"]):
                row[f"mae_{k}"] = float(err[i].mean())
                row[f"mae_const_{k}"] = float(np.abs(c[k] - g[i]).mean())
                row[f"r_{k}"] = pearson(p[i], g[i])
            # Uncertainty calibration: correlation between predicted scale and actual |log T2 error|
            row["r_unc_vs_err_t2"] = pearson(p[3], np.abs(np.log(p[1]) - np.log(g[1])))
            rows.append(row)
    net.train()
    return rows


def summarize(rows):
    keys = [k for k in rows[0] if k.startswith(("mae", "r_"))]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def score(summ):
    """Mean MAE ratio to the constant baseline (lower is better, < 1 beats the baseline)."""
    return float(np.mean([summ[f"mae_{k}"] / max(summ[f"mae_const_{k}"], 1e-8) for k in ("t1", "t2", "m0")]))


def qc_figure(net, item, device, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng, gen = np.random.default_rng(0), torch.Generator(device=device).manual_seed(0)
    center = eval_centers(item, 3)[1]
    x, tgt, msk = make_sample(item, "T2W", rng, gen, deterministic=True, center=center, size=EVAL_CROP_1MM)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = net(x, torch.tensor([MOD_IDX["T2W"]], device=device), torch.tensor([item["field"]], device=device))
    s = x.shape[2] // 2
    m = msk[0, 0, s].cpu().numpy()
    panels = [("input T2W", x[0, 0, s], "gray", 0, 1),
              ("T1 true (s)", tgt[0, 0, s].exp(), "magma", 0, 3), ("T1 pred", out["log_t1"][0, 0, s].exp(), "magma", 0, 3),
              ("T2 true (s)", tgt[0, 1, s].exp(), "viridis", 0, 0.2), ("T2 pred", out["log_t2"][0, 0, s].exp(), "viridis", 0, 0.2),
              ("M0 true", tgt[0, 2, s], "gray", 0, 1.2), ("M0 pred", out["m0"][0, 0, s], "gray", 0, 1.2),
              ("uncertainty", out["log_b"][0, 1, s].exp(), "inferno", None, None)]
    fig, ax = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.4))
    for a, (t, v, cm, lo, hi) in zip(ax, panels):
        v = v.float().cpu().numpy() * (m if t != "input T2W" else 1)
        a.imshow(np.rot90(v), cmap=cm, vmin=lo, vmax=hi)
        a.set_title(t, fontsize=9)
        a.axis("off")
    fig.suptitle(f"{item['id']} ({item['field']}T) — T2W input, deterministic protocol")
    plt.tight_layout()
    plt.savefig(path, dpi=70)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrf_dir", default=str(OUT_DIR / "mrf"))
    ap.add_argument("--out_dir", default=str(OUT_DIR / "paramnet"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--mod_probs", default="T1W:0.25,T2W:0.5,T2FLAIR:0.25")
    ap.add_argument("--limit_scans", type=int, default=0, help="debug: scans per split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mrf_dir = Path(args.mrf_dir)

    train = MRFStore(mrf_dir, "train", device, args.limit_scans)
    val = MRFStore(mrf_dir, "val", device, args.limit_scans)
    const = constant_baseline(train)
    print("constant baseline:", json.dumps(const), flush=True)

    mods, probs = zip(*[(m.split(":")[0], float(m.split(":")[1])) for m in args.mod_probs.split(",")])
    probs = np.array(probs) / np.sum(probs)
    model_kwargs = {"base": args.base}
    net = ParamUNet3D(**model_kwargs).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = min(500, args.iters // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * min(s, args.iters) / args.iters)))
    print(f"ParamNet params: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M", flush=True)

    history, best, t0 = [], float("inf"), time.time()
    run_nll = run_l1 = 0.0
    for step in range(1, args.iters + 1):
        xs, ys, ms, mod_i, fld = [], [], [], [], []
        for _ in range(args.batch):
            it = train.items[int(rng.integers(len(train)))]
            mod = str(rng.choice(mods, p=probs))
            x, y, m = make_sample(it, mod, rng, gen)
            xs.append(x), ys.append(y), ms.append(m), mod_i.append(MOD_IDX[mod]), fld.append(it["field"])
        x, y, m = torch.cat(xs), torch.cat(ys), torch.cat(ms)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(x, torch.tensor(mod_i, device=device), torch.tensor(fld, device=device))
        nll, l1 = laplace_nll(out, y, m)
        loss = nll + l1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run_nll += nll.item()
        run_l1 += l1.item()
        if step % 100 == 0:
            print(f"step {step:6d} | nll {run_nll / 100:.4f} | l1 {run_l1 / 100:.4f} | lr {sched.get_last_lr()[0]:.2e} | "
                  f"{(time.time() - t0) / step:.2f}s/it", flush=True)
            run_nll = run_l1 = 0.0
        if step % args.eval_every == 0 or step == args.iters:
            summ = summarize(evaluate(net, val, const, ["T2W"], device))
            sc = score(summ)
            history.append({"step": step, "val_score": sc, **summ})
            print(f"  VAL step {step}: score {sc:.3f} (MAE/const; <1 beats constant) | "
                  f"T1 MAE {summ['mae_t1']:.3f}s (const {summ['mae_const_t1']:.3f}) r {summ['r_t1']:.3f} | "
                  f"T2 MAE {summ['mae_t2'] * 1000:.1f}ms (const {summ['mae_const_t2'] * 1000:.1f}) r {summ['r_t2']:.3f} | "
                  f"M0 MAE {summ['mae_m0']:.3f} (const {summ['mae_const_m0']:.3f}) r {summ['r_m0']:.3f}", flush=True)
            ck = {"model": net.state_dict(), "model_kwargs": model_kwargs, "step": step, "val": summ,
                  "const": const, "args": vars(args)}
            torch.save(ck, out_dir / "latest.pth")
            if sc < best:
                best = sc
                torch.save(ck, out_dir / "best.pth")
                print(f"  -> best (score {sc:.3f})", flush=True)
            json.dump(history, open(out_dir / "history.json", "w"), indent=1)

    # ------------------------------------------------------------------ final held-out test
    del train
    torch.cuda.empty_cache()
    test = MRFStore(mrf_dir, "test", device, args.limit_scans)
    net.load_state_dict(torch.load(out_dir / "best.pth", map_location=device, weights_only=False)["model"])
    rows = evaluate(net, test, const, list(MODALITIES), device)
    report = {"per_scan": rows, "by_modality": {}}
    for mod in MODALITIES:
        report["by_modality"][mod] = summarize([r for r in rows if r["modality"] == mod])
    t2w = report["by_modality"]["T2W"]
    report["pass"] = {
        "t1_beats_constant": t2w["mae_t1"] < t2w["mae_const_t1"],
        "t2_beats_constant": t2w["mae_t2"] < t2w["mae_const_t2"],
        "m0_beats_constant": t2w["mae_m0"] < t2w["mae_const_m0"],
        "t2_r_above_0.7": t2w["r_t2"] > 0.7,
    }
    report["pass"]["ALL"] = all(report["pass"].values())
    json.dump(report, open(out_dir / "test_report.json", "w"), indent=1)
    lines = ["# ParamNet held-out test report", "", f"PASS: **{report['pass']['ALL']}**  {report['pass']}", "",
             "| input | T1 MAE (s) | const | r | T2 MAE (ms) | const | r | M0 MAE | const | r |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for mod, s in report["by_modality"].items():
        lines.append(f"| {mod} | {s['mae_t1']:.3f} | {s['mae_const_t1']:.3f} | {s['r_t1']:.2f} | {s['mae_t2'] * 1000:.1f} | "
                     f"{s['mae_const_t2'] * 1000:.1f} | {s['r_t2']:.2f} | {s['mae_m0']:.3f} | {s['mae_const_m0']:.3f} | {s['r_m0']:.2f} |")
    (out_dir / "test_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    qc_figure(net, test.items[0], device, out_dir / "qc_test.png")
    print("P1 DONE", flush=True)


if __name__ == "__main__":
    main()
