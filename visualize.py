"""
visualize.py — post-training visual results and 3D test metrics. Reads saved checkpoints only, so it never
touches a running job.

Per run (`--run_dir outputs/gen/<run>_fold<k>`), on the fold's held-out test subjects:
  metrics3d_<ckpt>.json     PSNR / SSIM (3D) and LPIPS / DISTS / FID (2.5D, 3 planes) per subject, mean ± std,
                            pooled FID; for `best` and `latest` checkpoints (see gen/metrics3d.py)
  metrics3d_best_physswap.json   physics runs only: test-time physics maps from another subject (dependency test)
  vis/nifti/pred_<sid>.nii.gz    full predicted 3T volume (normalized intensities, affine of the 3T scan)
  vis/three_plane_<sid>.png      axial / coronal / sagittal: input, prediction, GT, |error|, signed error
  vis/err_freq_<sid>.png         error split into low (Gaussian sigma 4 mm) and high frequency
  vis/zprofile_<sid>.png         mean brain intensity per axial slice: prediction vs GT vs identity
  vis/head_<sid>.png             gate map (gate head) or LUT curve + bias field + residual (lut_bias head)
  vis/attention.png              self-attn maps + attention distance per block, context / physics x-attn,
                                 Perceiver latents (Phases 1-3)
  vis/prototypes.png             tissue-prototype map and centres (Phase 3b)
  vis/physics_input.png          the 5-channel physics stack the generator receives (Phase 3)
  vis/training_curves.png        losses, val SSIM / PSNR vs identity, x-attn gates per epoch
Cross-run (`--compare --fold k`) -> outputs/vis_compare/fold<k>/ and outputs/METRICS3D_fold<k>.md.

Test set = the held-out subjects of the fold. `best.pth` was selected on the same subjects' validation slabs, so
`latest.pth` (final epoch, no selection) is the unbiased number; both are reported.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIELD_IDX, FIELD_TESLA, LOG_T1_RANGE, LOG_T2_RANGE, OUT_DIR, FieldTransfer, gaussian_blur3d  # noqa: E402
from data import CACHE_DIR, SLAB, VolumeBank, slab_dhw  # noqa: E402
from metrics3d import Metrics3D, brain_mask, fid_from_feats, fmt, mean_std  # noqa: E402
from model import build_model  # noqa: E402
from physics_stack import build_physics  # noqa: E402

SRC, TGT = "1.5T", "3T"
Z_SHOW = 165
METRIC_KEYS = ["psnr", "ssim", "lpips", "dists", "fid"]
METRIC_INFO = {"psnr": ("PSNR (dB) ↑", 2), "ssim": ("SSIM ↑", 4), "lpips": ("LPIPS ↓", 4),
               "dists": ("DISTS ↓", 4), "fid": ("FID ↓", 2)}
RUN_ORDER = ["ph0_base", "ph1_attn", "ph2_volctx", "ph5_head", "ph4_adain", "ph3_concat", "ph3_spade", "ph3_full",
             "ph3_shuffle"]
# fixed categorical slots (reference palette, light mode); colour follows the run, never its rank
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
RUN_COLOR = dict(zip(RUN_ORDER[:8], SERIES))
RUN_COLOR["ph3_shuffle"] = SERIES[7]          # same entity family as ph3_full, drawn dashed / hatched
IDENTITY_COLOR, INK, INK2, MUTED, SURFACE = "#898781", "#0b0b0b", "#52514e", "#898781", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6, "font.size": 9,
    "legend.frameon": False,
})


# ----------------------------------------------------------------------------- loading / inference
def load_run(run_dir, ckpt, device):
    ck = torch.load(Path(run_dir) / f"{ckpt}.pth", map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    use_phys = cfg["phys"] != "none" or bool(cfg["proto"])
    ft = FieldTransfer.from_json(cfg["field_transfer"]) if use_phys else FieldTransfer()
    return model, cfg, ft, use_phys, ck["epoch"]


def make_batch(bank, cfg, sid, zs, ctx, use_phys, psid, pctx, device):
    vol = bank.vol(sid, SRC)
    b = len(zs)
    bt = {"src": torch.from_numpy(np.stack([slab_dhw(vol, z) for z in zs])[:, None]).to(device),
          "src_idx": [FIELD_IDX[SRC]] * b, "tgt_idx": [FIELD_IDX[TGT]] * b,
          "src_t": [FIELD_TESLA[SRC]] * b, "tgt_t": [FIELD_TESLA[TGT]] * b,
          "zpos": [(z + SLAB / 2) / vol.shape[2] for z in zs], "ctx": ctx.expand(b, -1, -1, -1, -1)}
    if use_phys:
        pv = bank.phys_vol(psid, SRC)
        bt["phys"] = torch.from_numpy(np.stack([slab_dhw(pv, z) for z in zs])).to(device)
        bt["phys_ctx"] = pctx.expand(b, -1, -1, -1, -1)
    return bt


def context_tensors(bank, sid, psid, use_phys, device):
    ctx = torch.from_numpy(np.asarray(bank.ctx(sid, SRC), np.float32))[None, None].to(device)
    pctx = torch.from_numpy(np.asarray(bank.phys_ctx(psid, SRC), np.float32))[None].to(device) if use_phys else None
    return ctx, pctx


@torch.no_grad()
def predict_full(model, cfg, ft, use_phys, bank, sid, device, psid=None, capture=False, batch=4):
    """Same sliding window as evaluate.predict_volume (slab 16, stride 8, Hann), optionally also blending the
    head's internal maps (gate / residual / bias field) and collecting the per-slab LUT knots."""
    from train import forward
    psid = psid or sid
    X, Y, Z = bank.vol(sid, SRC).shape
    acc, wacc, extra = np.zeros((Z, X, Y), np.float32), np.zeros((Z, 1, 1), np.float32), {}
    win = np.clip(np.hanning(SLAB), 0.1, 1.0).astype(np.float32)[:, None, None]
    starts = list(range(0, Z - SLAB + 1, SLAB // 2))
    if starts[-1] != Z - SLAB:
        starts.append(Z - SLAB)
    ctx, pctx = context_tensors(bank, sid, psid, use_phys, device)
    store, knots = {}, []
    hook = model.out_gate.register_forward_hook(lambda m, i, o: store.__setitem__("gate", o)) \
        if capture and cfg["head"] == "gate" else None
    for i in range(0, len(starts), batch):
        zs = starts[i:i + batch]
        bt = make_batch(bank, cfg, sid, zs, ctx, use_phys, psid, pctx, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, aux = forward(model, bt, cfg, ft, device)
        p = pred.float().cpu().numpy()[:, 0]
        maps = {}
        if capture:
            if "gate" in store:
                maps["gate"] = store["gate"].float().cpu().numpy()[:, 0]
            if "resid" in aux:
                maps["resid"] = aux["resid"].float().cpu().numpy()[:, 0]
                maps["bias"] = torch.exp(F.interpolate(aux["log_bias_lr"].float(), size=bt["src"].shape[2:],
                                                       mode="trilinear", align_corners=False)).cpu().numpy()[:, 0]
                knots += [(z + SLAB / 2, k) for z, k in zip(zs, aux["knots"].float().cpu().numpy())]
        for j, z in enumerate(zs):
            acc[z:z + SLAB] += p[j] * win
            wacc[z:z + SLAB] += win
            for name, m in maps.items():
                extra.setdefault(name, np.zeros((Z, X, Y), np.float32))[z:z + SLAB] += m[j] * win
    if hook is not None:
        hook.remove()
    norm = np.maximum(wacc, 1e-5)
    pred = np.moveaxis(np.clip(acc / norm, 0, 1), 0, -1)
    return pred, {k: np.moveaxis(v / norm, 0, -1) for k, v in extra.items()}, knots


# ----------------------------------------------------------------------------- attention / prototype capture
def _attn_probs(mod, x, ctx=None, bias=None):
    """Head-averaged attention probabilities (Nq, Nk) of model.Attention for batch item 0."""
    with torch.autocast("cuda", enabled=False):
        x = x.float()
        ctx = x if ctx is None else ctx.float()
        B, N, C = x.shape
        dh = C // mod.h
        q = F.linear(x, mod.q.weight.float(), mod.q.bias.float()).view(B, N, mod.h, dh).transpose(1, 2)
        k = F.linear(ctx, mod.kv.weight.float(), mod.kv.bias.float()).view(B, ctx.shape[1], 2, mod.h, dh)[:, :, 0]
        a = (q @ k.permute(0, 2, 3, 1)) / math.sqrt(dh)
        if bias is not None:
            a = a + bias.float()[:, None, None, :]
        return a.softmax(-1).mean(1)[0].cpu()


@torch.no_grad()
def attention_snapshot(model, cfg, ft, use_phys, bank, sid, device, z_center=Z_SHOW):
    """One forward pass on the slab centred at z_center with hooks on every attention module."""
    from train import forward
    z0 = z_center - SLAB // 2
    ctx, pctx = context_tensors(bank, sid, sid, use_phys, device)
    bt = make_batch(bank, cfg, sid, [z0], ctx, use_phys, sid, pctx, device)
    snap, hooks = {"z0": z0}, []

    def attn_hook(name):
        def fn(mod, args, kwargs, out):
            c = args[1] if len(args) > 1 else kwargs.get("ctx")
            b = args[2] if len(args) > 2 else kwargs.get("bias")
            snap[name] = _attn_probs(mod, args[0], c, b)
        return fn

    for bi, blk in enumerate(getattr(model, "tblocks", [])):
        hooks.append(blk.attn.register_forward_hook(attn_hook(f"self{bi}"), with_kwargs=True))
        if blk.use_ctx:
            hooks.append(blk.xattn_c.register_forward_hook(attn_hook(f"ctx{bi}"), with_kwargs=True))
        if blk.use_phys:
            hooks.append(blk.xattn_p.register_forward_hook(attn_hook(f"phys{bi}"), with_kwargs=True))
    if len(model.tblocks):
        hooks.append(model.tblocks[0].register_forward_hook(lambda m, a, o: snap.__setitem__("grid", tuple(a[0].shape[2:]))))
    if model.volctx:
        hooks.append(model.ctx_enc.cross.register_forward_hook(attn_hook("perceiver"), with_kwargs=True))
        hooks.append(model.ctx_enc.cnn.register_forward_hook(lambda m, a, o: snap.__setitem__("pgrid", tuple(o.shape[2:]))))
    if model.proto:
        hooks.append(model.protos.register_forward_hook(lambda m, a, o: snap.__setitem__("proto_w", o[1][0].float().cpu())))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        forward(model, bt, cfg, ft, device)
    for h in hooks:
        h.remove()
    if cfg["phys"] != "none":
        src_t = torch.tensor([FIELD_TESLA[SRC]], device=device)
        tgt_t = torch.tensor([FIELD_TESLA[TGT]], device=device)
        snap["phys_stack"] = build_physics(bt["phys"].float(), bt["src"], src_t, tgt_t, ft, cfg["modality"])[0].cpu().numpy()
    snap["gates"] = {n: float(torch.tanh(p).item()) for n, p in model.named_parameters() if n.endswith(("alpha_c", "alpha_p"))}
    return snap


# ----------------------------------------------------------------------------- plotting helpers
def planes(v, z=Z_SHOW):
    X, Y, _ = v.shape[-3:]
    return {"axial": np.rot90(v[..., :, :, z]), "coronal": np.rot90(v[..., :, Y // 2, :]),
            "sagittal": np.rot90(v[..., X // 2, :, :])}


def show(ax, img, title="", **kw):
    ax.imshow(img, **kw)
    ax.set_title(title, fontsize=8, color=INK2)
    ax.axis("off")


def savefig(fig, path, dpi=90):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fig_three_plane(src, pred, gt, path, title):
    P = [planes(v) for v in (src, pred, gt)]
    err = planes(pred - gt)
    fig, ax = plt.subplots(3, 5, figsize=(17, 11))
    for r, pl in enumerate(("axial", "coronal", "sagittal")):
        for c, (name, p) in enumerate(zip(("1.5T input", "prediction", "3T ground truth"), P)):
            show(ax[r, c], p[pl], f"{name} — {pl}", cmap="gray", vmin=0, vmax=1)
        show(ax[r, 3], np.abs(err[pl]), "|pred − GT|  (0 … 0.3)", cmap="inferno", vmin=0, vmax=0.3)
        show(ax[r, 4], err[pl], "pred − GT  (−0.2 … 0.2)", cmap="RdBu_r", vmin=-0.2, vmax=0.2)
    fig.suptitle(title)
    savefig(fig, path, 70)


def fig_err_freq(pred, gt, mask, path, title, device):
    with torch.no_grad():
        m = torch.from_numpy(mask).to(device)
        e = (torch.from_numpy(pred).to(device) - torch.from_numpy(gt).to(device)) * m
        low = gaussian_blur3d(e[None, None], (8.0, 8.0, 8.0))[0, 0] * m
        e, low = e.cpu().numpy(), low.cpu().numpy()
    hi = e - low
    n = max(mask.sum(), 1)
    fig, ax = plt.subplots(3, 3, figsize=(11, 11))
    for r, pl in enumerate(("axial", "coronal", "sagittal")):
        for c, (name, v) in enumerate((("total error", e), ("low-frequency (σ = 4 mm)", low), ("high-frequency", hi))):
            show(ax[r, c], planes(v)[pl], f"{name} — {pl}", cmap="RdBu_r", vmin=-0.15, vmax=0.15)
    fig.suptitle(f"{title}\nMAE in brain: total {np.abs(e).sum() / n:.4f} | low {np.abs(low).sum() / n:.4f} | "
                 f"high {np.abs(hi).sum() / n:.4f}  (pred − GT, −0.15 … 0.15)")
    savefig(fig, path, 70)


def z_profiles(src, pred, gt, mask):
    n = np.maximum(mask.sum((0, 1)), 1)
    return {k: (v * mask).sum((0, 1)) / n for k, v in (("src", src), ("pred", pred), ("gt", gt))}, mask.sum((0, 1)) > 2000


def fig_zprofile(prof, valid, path, title):
    z = np.arange(len(prof["gt"]))
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for k, lab, col, ls in (("gt", "3T ground truth", INK, "-"), ("pred", "prediction", SERIES[0], "-"),
                            ("src", "1.5T input (identity)", IDENTITY_COLOR, "--")):
        ax[0].plot(z[valid], prof[k][valid], ls, color=col, label=lab)
    ax[0].set_ylabel("mean brain intensity")
    ax[0].legend(loc="upper left")
    ax[1].axhline(0, color=MUTED, lw=0.8)
    ax[1].plot(z[valid], (prof["pred"] - prof["gt"])[valid], color=SERIES[0], label="prediction − GT")
    ax[1].plot(z[valid], (prof["src"] - prof["gt"])[valid], "--", color=IDENTITY_COLOR, label="identity − GT")
    ax[1].axvspan(150, 180, color="#e6e5e1", alpha=0.6, lw=0, label="challenge slices 150–180")
    ax[1].set_xlabel("axial slice z")
    ax[1].set_ylabel("difference")
    ax[1].legend(loc="upper left")
    fig.suptitle(title)
    savefig(fig, path)


def cond_median(x, y, bins):
    idx = np.digitize(x, bins) - 1
    out = np.full(len(bins) - 1, np.nan)
    for b in range(len(bins) - 1):
        s = y[idx == b]
        if s.size > 50:
            out[b] = np.median(s)
    return out


def fig_head(cfg, extras, knots, src, pred, gt, mask, path, title):
    if cfg["head"] == "gate" and "gate" in extras:
        g = extras["gate"]
        fig, ax = plt.subplots(1, 4, figsize=(17, 4.6))
        for c, (pl, v) in enumerate(planes(g * mask).items()):
            show(ax[c], v, f"gate G — {pl} (1 = copy input)", cmap="viridis", vmin=0, vmax=1)
        ax[3].hist(g[mask].ravel()[::7], bins=60, color=SERIES[0])
        ax[3].set_title("gate values in brain", fontsize=8)
        ax[3].set_xlabel("G")
        fig.suptitle(f"{title} — output = G·input + (1−G)·synthesis")
        savefig(fig, path, 75)
        return
    if cfg["head"] != "lut_bias" or "bias" not in extras:
        return
    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 4)
    a = fig.add_subplot(gs[:, 0:2])
    xs, ys, ps = src[mask][::5], gt[mask][::5], pred[mask][::5]
    a.hist2d(xs, ys, bins=128, range=[[0, 1], [0, 1]], cmap="Greys", norm=matplotlib.colors.LogNorm())
    bins = np.linspace(0, 1, 41)
    bc = 0.5 * (bins[1:] + bins[:-1])
    a.plot([0, 1], [0, 1], "--", color=MUTED, lw=1, label="identity")
    cmap = plt.get_cmap("Blues")
    zs = np.array([k[0] for k in knots])
    zn = (zs - zs.min()) / max(np.ptp(zs), 1)
    kx = np.linspace(0, 1, len(knots[0][1]))
    for (z, k), t in zip(knots[::3], zn[::3]):
        a.plot(kx, k, color=cmap(0.35 + 0.6 * t), lw=0.7, alpha=0.8)
    a.plot([], [], color=cmap(0.7), lw=0.9, label="learned LUT per slab (darker = higher z)")
    a.plot(bc, cond_median(xs, ys, bins), color=INK, lw=2, label="GT: median 3T | 1.5T")
    a.plot(bc, cond_median(xs, ps, bins), color=SERIES[1], lw=2, label="prediction: median | 1.5T")
    a.set_xlim(0, 1)
    a.set_ylim(0, 1.05)
    a.set_xlabel("1.5T input intensity")
    a.set_ylabel("3T intensity")
    a.set_title("joint histogram (brain voxels, log counts) + intensity mapping", fontsize=9)
    a.legend(loc="upper left", fontsize=8)
    lb = np.log2(np.where(mask, extras["bias"], 1.0))
    for c, (pl, v) in enumerate(planes(lb).items()):
        if c < 2:
            ax = fig.add_subplot(gs[0, 2 + c])
            show(ax, v, f"bias field log2 — {pl} (±0.3)", cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    for c, (pl, v) in enumerate(planes(extras["resid"] * mask).items()):
        if c < 2:
            ax = fig.add_subplot(gs[1, 2 + c])
            show(ax, v, f"residual — {pl} (±0.1)", cmap="RdBu_r", vmin=-0.1, vmax=0.1)
    fig.suptitle(f"{title} — output = LUT(input) × bias + residual  |  bias range in brain "
                 f"{extras['bias'][mask].min():.2f}–{extras['bias'][mask].max():.2f}, "
                 f"|residual| mean {np.abs(extras['resid'][mask]).mean():.4f}")
    savefig(fig, path, 75)


def _overlay(ax, base, amap, title, point=None):
    ax.imshow(base, cmap="gray", vmin=0, vmax=1)
    ax.imshow(amap, cmap="magma", alpha=0.55, vmin=0, vmax=max(float(amap.max()), 1e-6))
    if point is not None:
        ax.plot(*point, "o", ms=9, mfc="none", mec="#1baf7a", mew=2)
    ax.set_title(title, fontsize=8, color=INK2)
    ax.axis("off")


def _upsample2d(a, shape):
    t = torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)[None, None]
    return F.interpolate(t, size=shape, mode="bilinear", align_corners=False)[0, 0].numpy()


def fig_attention(snap, src, path, title):
    if "grid" not in snap:
        return None
    D, H, W = snap["grid"]
    X, Y, _ = src.shape
    base = np.rot90(src[:, :, Z_SHOW])
    brain_tok = torch.from_numpy(np.ascontiguousarray(src[:, :, snap["z0"]:snap["z0"] + SLAB] > 0.005)).float()
    brain_tok = F.adaptive_avg_pool3d(brain_tok.permute(2, 0, 1)[None], (D, H, W))[0].flatten() > 0.3
    queries = [(D - 1, int(H * fh), int(W * fw)) for fh, fw in ((0.5, 0.5), (0.32, 0.3), (0.62, 0.72))]
    qidx = [d * H * W + h * W + w for d, h, w in queries]
    blocks = sorted(int(k[4:]) for k in snap if k.startswith("self"))
    last = blocks[-1]
    # token centres in mm: D pooled 8 slices (4 mm), H/W pooled 16 voxels (8 mm)
    g = torch.stack(torch.meshgrid(torch.arange(D) * 4.0, torch.arange(H) * 8.0, torch.arange(W) * 8.0,
                                   indexing="ij"), -1).view(-1, 3)
    dist = torch.cdist(g, g)
    rows = [("self", "self-attention")] + [(k, n) for k, n in (("ctx", "context x-attn"), ("phys", "physics x-attn"))
                                          if f"{k}{last}" in snap]
    extra_rows = int("perceiver" in snap)
    fig, ax = plt.subplots(len(rows) + 1 + extra_rows, 4, figsize=(17, 4.3 * (len(rows) + 1 + extra_rows)))
    report = {}
    for r, (key, name) in enumerate(rows):
        A = snap[f"{key}{last}"]
        if key in ("self", "phys"):
            for c, (q, (d, h, w)) in enumerate(zip(qidx, queries)):
                m = A[q].view(D, H, W).sum(0).numpy()
                pt = ((h + 0.5) * X / H, Y - (w + 0.5) * Y / W)  # rot90 display: col = x, row = Y-1-y
                _overlay(ax[r, c], base, np.rot90(_upsample2d(m, (X, Y))),
                         f"{name}, block {last}: query {c + 1}", point=pt)
            if key == "phys":
                diag = float(A.diagonal()[brain_tok].mean())
                report["phys_diag_mass"] = diag
                ax[r, 3].bar(["same position", "uniform"], [diag, 1.0 / A.shape[1]], color=[SERIES[7], IDENTITY_COLOR])
                ax[r, 3].set_title("physics x-attn: mean weight on the aligned token", fontsize=8)
            else:
                md = [float((snap[f"self{b}"] * dist).sum(1)[brain_tok].mean()) for b in blocks]
                uni = float(dist[brain_tok][:, brain_tok].mean())
                report["self_attn_mean_distance_mm"] = md
                report["uniform_distance_mm"] = uni
                ax[r, 3].bar([f"block {b}" for b in blocks], md, color=SERIES[0])
                ax[r, 3].axhline(uni, ls="--", color=IDENTITY_COLOR, label=f"uniform over brain ({uni:.0f} mm)")
                ax[r, 3].set_ylabel("mean attention distance (mm)")
                ax[r, 3].legend(fontsize=7)
                ax[r, 3].set_title("how global is self-attention?", fontsize=8)
        else:  # context cross-attention: keys = [64 volume latents][12 prototypes]
            nk = A.shape[1]
            n_lat = 64 if snap.get("pgrid") else 0
            ent = -(A * (A + 1e-9).log()).sum(1) / math.log(nk)
            show(ax[r, 0], np.rot90(_upsample2d(ent.view(D, H, W).mean(0).numpy(), (X, Y))),
                 f"{name}: normalized entropy (block {last})", cmap="viridis", vmin=0, vmax=1)
            if nk > n_lat:
                pm = A[:, n_lat:].sum(1).view(D, H, W).mean(0).numpy()
                show(ax[r, 1], np.rot90(_upsample2d(pm, (X, Y))), "attention mass on tissue prototypes",
                     cmap="viridis", vmin=0, vmax=1)
                pk = A[brain_tok][:, n_lat:].mean(0).numpy()
                ax[r, 2].bar(np.arange(len(pk)), pk, color=SERIES[6])
                ax[r, 2].set_title("mean weight per prototype token", fontsize=8)
            else:
                ax[r, 1].axis("off")
                ax[r, 2].axis("off")
            top = A[brain_tok].mean(0)
            ax[r, 3].bar(np.arange(nk), top.numpy(), color=SERIES[2])
            ax[r, 3].set_title("mean weight per context token (brain queries)", fontsize=8)
    # gates row
    r = len(rows)
    gates = snap["gates"]
    for c in range(4):
        ax[r, c].axis("off")
    if gates:
        ax[r, 0].axis("on")
        ax[r, 0].barh(list(gates), list(gates.values()), color=SERIES[4])
        ax[r, 0].set_title("cross-attention gates tanh(α) (0 = path unused)", fontsize=8)
        ax[r, 0].tick_params(labelsize=6)
    report["gates"] = gates
    if "perceiver" in snap:
        r += 1
        A = snap["perceiver"]                              # (64 latents, d*h*w volume tokens)
        d, h, w = snap["pgrid"]
        ent = -(A * (A + 1e-9).log()).sum(1)
        cbase = None
        for c, li in enumerate(ent.argsort()[:4].tolist()):
            m = A[li].view(d, h, w).sum(0).numpy()        # summed over z
            cbase = np.rot90(src[::4, ::4, Z_SHOW]) if cbase is None else cbase
            _overlay(ax[r, c], cbase, np.rot90(_upsample2d(m, src[::4, ::4, 0].shape)),
                     f"volume latent {li} (most focused #{c + 1}), summed over z")
    fig.suptitle(f"{title} — slab z {snap['z0']}–{snap['z0'] + SLAB - 1}, base image = 1.5T axial z={Z_SHOW}")
    savefig(fig, path, 70)
    return report


def fig_prototypes(snap, model, src, ctx, path, title):
    if "proto_w" not in snap:
        return
    w = snap["proto_w"]                                    # (N, K) over the 2 mm grid (Z/4, X/4, Y/4)
    K = w.shape[1]
    Zc, Xc, Yc = ctx.shape
    c = model.protos.centers.detach().float().cpu().numpy()
    t1 = np.exp(c[:, 0] * (LOG_T1_RANGE[1] - LOG_T1_RANGE[0]) + LOG_T1_RANGE[0]) * 1000
    t2 = np.exp(c[:, 1] * (LOG_T2_RANGE[1] - LOG_T2_RANGE[0]) + LOG_T2_RANGE[0]) * 1000
    order = np.argsort(t1)
    rank = np.empty(K, int)
    rank[order] = np.arange(K)
    valid = (w.sum(1) > 1e-6).view(Zc, Xc, Yc).numpy()
    lab = rank[w.argmax(1).view(Zc, Xc, Yc).numpy()].astype(float)
    lab[~valid] = np.nan
    lab_xyz = np.moveaxis(lab, 0, -1)                      # (X/4, Y/4, Z/4)
    mass = w.sum(0).numpy()
    mass = mass / max(mass.sum(), 1e-9)
    fig, ax = plt.subplots(1, 5, figsize=(21, 4.6))
    cm = plt.get_cmap("viridis", K).copy()
    cm.set_bad(SURFACE)
    zc = Z_SHOW // 4
    show(ax[0], np.rot90(np.moveaxis(ctx, 0, -1)[:, :, zc]), "1.5T context (2 mm), axial", cmap="gray", vmin=0, vmax=1)
    for i, (pl, v) in enumerate(planes(lab_xyz, zc).items()):
        show(ax[1 + i], v, f"hard prototype label — {pl}", cmap=cm, vmin=-0.5, vmax=K - 0.5)
    sc = ax[4].scatter(t1, t2, s=40 + 1500 * mass, c=rank, cmap=cm, vmin=-0.5, vmax=K - 0.5, edgecolor=INK2, lw=0.5)
    for i in range(K):
        ax[4].annotate(str(rank[i]), (t1[i], t2[i]), fontsize=7, ha="center", va="center", color=INK)
    ax[4].set_xscale("log")
    ax[4].set_yscale("log")
    ax[4].set_xlabel("centre T1 at 1.5T (ms)")
    ax[4].set_ylabel("centre T2 (ms)")
    ax[4].set_title("learned prototype centres (size = voxel share; colour = T1 rank)", fontsize=8)
    fig.colorbar(sc, ax=ax[4], label="prototype rank by T1")
    fig.suptitle(title)
    savefig(fig, path, 75)


def fig_physics_input(snap, src, path, title):
    if "phys_stack" not in snap:
        return
    ps = snap["phys_stack"][:, SLAB // 2]                  # (5, X, Y) at z0 + 8 = Z_SHOW
    names = [("M0 / 1.5", "gray", 0, 1), ("norm log T1 @3T", "viridis", 0, 1), ("norm log T2 @3T", "viridis", 0, 1),
             ("Bloch prior @3T", "gray", 0, 1), ("ParamNet uncertainty", "magma", 0, 1)]
    fig, ax = plt.subplots(1, 6, figsize=(22, 4.4))
    show(ax[0], np.rot90(src[:, :, Z_SHOW]), "1.5T input", cmap="gray", vmin=0, vmax=1)
    for i, (n, cm, lo, hi) in enumerate(names):
        show(ax[i + 1], np.rot90(ps[i]), n, cmap=cm, vmin=lo, vmax=hi)
    fig.suptitle(f"{title} — physics stack fed to the generator (axial z={Z_SHOW})")
    savefig(fig, path, 75)


def fig_training_curves(hist, path, title):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3))
    for k, col in zip(("total", "l1", "ssim", "perc"), SERIES):
        if k in hist[0]["train"]:
            ax[0].plot(ep, [h["train"][k] for h in hist], color=col, label=k)
    ax[0].set_yscale("log")
    ax[0].set_title("training loss terms", fontsize=9)
    ax[0].legend()
    for i, (k, lab) in enumerate((("ssim", "val SSIM (slabs, slice-wise)"), ("psnr", "val PSNR (dB)"))):
        ax[1 + i].plot(ep, [h["val"][k] for h in hist], color=SERIES[0], label="model")
        ax[1 + i].axhline(hist[0]["val"][f"{k}_id"], ls="--", color=IDENTITY_COLOR, label="identity (1.5T input)")
        b = int(np.argmax([h["val"]["ssim"] for h in hist]))
        ax[1 + i].plot(ep[b], hist[b]["val"][k], "o", ms=8, mfc="none", mec=SERIES[1], mew=2, label=f"best.pth (ep {ep[b]})")
        ax[1 + i].set_title(lab, fontsize=9)
        ax[1 + i].legend()
    gates = sorted({g for h in hist for g in h.get("gates", {})})
    if gates:
        for gi, gname in enumerate(gates):
            col = SERIES[0] if gname.endswith("alpha_c") else SERIES[7]
            ax[3].plot(ep, [h["gates"].get(gname, np.nan) for h in hist], color=col, alpha=0.4 + 0.6 * gi / len(gates), lw=1)
        for suf, col, lab in (("alpha_c", SERIES[0], "context x-attn gates"), ("alpha_p", SERIES[7], "physics x-attn gates")):
            if any(g.endswith(suf) for g in gates):
                ax[3].plot([], [], color=col, label=lab)
        ax[3].set_title("tanh(α) per block", fontsize=9)
        ax[3].legend()
    elif "gate_mean" in hist[0]["train"]:
        ax[3].plot(ep, [h["train"]["gate_mean"] for h in hist], color=SERIES[0])
        ax[3].set_title("mean output gate G (1 = copy input)", fontsize=9)
    else:
        ax[3].axis("off")
    for a in ax:
        a.set_xlabel("epoch")
    fig.suptitle(title)
    savefig(fig, path)


# ----------------------------------------------------------------------------- per-run processing
def test_subjects(modality, fold):
    meta = json.load(open(CACHE_DIR / modality / "meta.json"))
    return meta["folds"][fold]["val"], meta


def summarize_rows(rows, fp, fg, device):
    out = {"per_subject": rows, "summary": mean_std(rows, METRIC_KEYS)}
    out["fid_pooled"] = fid_from_feats(np.concatenate(fp), np.concatenate(fg), device)
    return out


def ensure_identity(modality, fold, subjects, bank, M, device):
    path = OUT_DIR / "metrics3d" / f"fold{fold}" / "identity.json"
    if path.exists():
        res = json.load(open(path))
        if [r["subject"] for r in res["per_subject"]] == list(subjects):
            return res
    if len(subjects) < len(test_subjects(bank.modality, fold)[0]):
        return None  # debug subset: never write the shared identity file
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, fp, fg = [], [], []
    for sid in subjects:
        s = np.asarray(bank.vol(sid, SRC), np.float32)
        g = np.asarray(bank.vol(sid, TGT), np.float32)
        m, a, b = M.compute(s, g, s)
        rows.append({"subject": sid, **m})
        fp.append(a)
        fg.append(b)
        print(f"  identity {sid}: " + " ".join(f"{k} {m[k]:.4f}" for k in METRIC_KEYS), flush=True)
    res = summarize_rows(rows, fp, fg, device)
    json.dump(res, open(path, "w"), indent=1)
    return res


def process_run(run_dir, args, device):
    run_dir = Path(run_dir)
    out = Path(args.out) if args.out else run_dir / "vis"
    (out / "nifti").mkdir(parents=True, exist_ok=True)
    M = Metrics3D(device)
    ckpts = [c for c in args.ckpts.split(",") if (run_dir / f"{c}.pth").exists()]
    fig_ckpt = args.fig_ckpt if args.fig_ckpt in ckpts else ckpts[0]
    model, cfg, ft, use_phys, _ = load_run(run_dir, fig_ckpt, device)
    subjects, meta = test_subjects(cfg["modality"], cfg["fold"])
    subjects = subjects[: args.max_subjects] if args.max_subjects else subjects
    bank = VolumeBank(cfg["modality"], use_phys)
    name = run_dir.name
    if not args.no_metrics:
        ensure_identity(cfg["modality"], cfg["fold"], subjects, bank, M, device)
    # the shuffle control was trained with another subject's physics, so it is also tested that way
    other = {s: subjects[(i + 1) % len(subjects)] for i, s in enumerate(subjects)}
    mode_list = [(c, "own") for c in ckpts]
    if cfg.get("phys_shuffle"):
        mode_list = [(c, "other") for c in ckpts]
    elif use_phys and "best" in ckpts:
        mode_list.append(("best", "swap"))
    for ckpt, pmode in mode_list:
        model, cfg, ft, use_phys, epoch = load_run(run_dir, ckpt, device)
        rows, fp, fg = [], [], []
        for si, sid in enumerate(subjects):
            psid = other[sid] if pmode in ("other", "swap") else sid
            is_fig = ckpt == fig_ckpt and pmode != "swap"
            pred, extras, knots = predict_full(model, cfg, ft, use_phys, bank, sid, device, psid=psid, capture=is_fig)
            src = np.asarray(bank.vol(sid, SRC), np.float32)
            gt = np.asarray(bank.vol(sid, TGT), np.float32)
            mask = brain_mask(gt, src)
            if not args.no_metrics:
                m, a, b = M.compute(pred, gt, src)
                rows.append({"subject": sid, "phys_from": psid if use_phys else None, **m})
                fp.append(a)
                fg.append(b)
                print(f"  {name} [{ckpt}/{pmode}] {sid}: " + " ".join(f"{k} {m[k]:.4f}" for k in METRIC_KEYS), flush=True)
            if not is_fig:
                continue
            ref = nib.load(meta["paths"][sid][TGT])
            nib.save(nib.Nifti1Image(pred.astype(np.float32), ref.affine), str(out / "nifti" / f"pred_{sid}.nii.gz"))
            ttl = f"{name} [{ckpt}, epoch {epoch}] — subject {sid}"
            fig_three_plane(src, pred, gt, out / f"three_plane_{sid}.png", ttl)
            fig_err_freq(pred, gt, mask, out / f"err_freq_{sid}.png", ttl, device)
            prof, valid = z_profiles(src, pred, gt, mask)
            fig_zprofile(prof, valid, out / f"zprofile_{sid}.png", ttl)
            fig_head(cfg, extras, knots, src, pred, gt, mask, out / f"head_{sid}.png", ttl)
            np.savez_compressed(out / f"slices_{sid}.npz", **{f"pred_{k}": v.astype(np.float16) for k, v in planes(pred).items()},
                                **{f"prof_{k}": v for k, v in prof.items()}, prof_valid=valid)
            if si == 0:
                snap = attention_snapshot(model, cfg, ft, use_phys, bank, sid, device)
                rep = fig_attention(snap, src, out / "attention.png", f"{name} [{ckpt}] — subject {sid}")
                if rep:
                    json.dump(rep, open(out / "attention_stats.json", "w"), indent=1)
                fig_prototypes(snap, model, src, np.asarray(bank.ctx(sid, SRC), np.float32), out / "prototypes.png",
                               f"{name} [{ckpt}] — tissue prototypes, subject {sid}")
                fig_physics_input(snap, src, out / "physics_input.png", f"{name} — subject {sid}")
        if rows:
            res = summarize_rows(rows, fp, fg, device)
            res.update(run=name, ckpt=ckpt, epoch=epoch, phys_mode=pmode, subjects=subjects)
            suffix = "_physswap" if pmode == "swap" else ""
            json.dump(res, open((out.parent if not args.out else out) / f"metrics3d_{ckpt}{suffix}.json", "w"), indent=1)
            s = res["summary"]
            print(f"METRICS3D {name} [{ckpt}/{pmode}] " + " | ".join(f"{k} {fmt(s, k, METRIC_INFO[k][1])}" for k in METRIC_KEYS)
                  + f" | pooled FID {res['fid_pooled']:.2f}", flush=True)
    hp = run_dir / "history.json"
    if hp.exists():
        fig_training_curves(json.load(open(hp)), out / "training_curves.png", name)
    if not args.out:
        (out / "DONE").write_text("ok\n")
    print(f"VIS DONE {name} -> {out}", flush=True)


# ----------------------------------------------------------------------------- cross-run comparison
def run_dirs(fold):
    found = {p.name.rsplit("_fold", 1)[0]: p for p in (OUT_DIR / "gen").glob(f"*_fold{fold}") if (p / "vis" / "DONE").exists()}
    return [(n, found[n]) for n in sorted(found, key=lambda n: RUN_ORDER.index(n) if n in RUN_ORDER else 99)]


def metrics_table(fold, runs, ckpt):
    idp = OUT_DIR / "metrics3d" / f"fold{fold}" / "identity.json"
    entries = []
    if idp.exists():
        entries.append(("identity (1.5T input)", json.load(open(idp))))
    for name, d in runs:
        for suffix, lab in (("", name), ("_physswap", f"{name} — test-time physics swap")):
            p = d / f"metrics3d_{ckpt}{suffix}.json"
            if p.exists():
                entries.append((lab, json.load(open(p))))
    return entries


def fig_metric_bars(entries, path, title):
    fig, ax = plt.subplots(1, 5, figsize=(22, 5.2))
    labels = [e[0].replace(" — test-time physics swap", " (swap)").replace(" (1.5T input)", "") for e in entries]
    for a, k in zip(ax, METRIC_KEYS):
        for i, (lab, e) in enumerate(entries):
            run = lab.split(" ")[0]
            col = IDENTITY_COLOR if lab.startswith("identity") else RUN_COLOR.get(run, MUTED)
            s = e["summary"][k]
            hatch = "///" if ("swap" in lab or run == "ph3_shuffle") else None
            a.bar(i, s["mean"], yerr=s["std"], color=col, edgecolor=SURFACE, lw=2, hatch=hatch, capsize=3,
                  error_kw={"ecolor": INK2, "lw": 1})
            vals = [r[k] for r in e["per_subject"]]
            a.plot(np.full(len(vals), i) + np.linspace(-0.15, 0.15, len(vals)), vals, "o", ms=3.5, color=INK, alpha=0.7)
        lo = min(min(r[k] for r in e["per_subject"]) for _, e in entries)
        hi = max(max(r[k] for r in e["per_subject"]) for _, e in entries)
        a.set_ylim(lo - 0.15 * (hi - lo + 1e-9), hi + 0.15 * (hi - lo + 1e-9))
        a.set_xticks(range(len(entries)))
        a.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        a.set_title(METRIC_INFO[k][0], fontsize=10)
    fig.suptitle(f"{title}\nbars = mean ± sample std over test subjects, dots = individual subjects; hatched = mismatched physics")
    savefig(fig, path)


def fig_compare_subject(fold, runs, sid, bank, path):
    src = np.asarray(bank.vol(sid, SRC), np.float32)
    gt = np.asarray(bank.vol(sid, TGT), np.float32)
    ps, pg = planes(src)["axial"], planes(gt)["axial"]
    cols = [("1.5T input", ps)] + [(n, np.load(d / "vis" / f"slices_{sid}.npz")["pred_axial"].astype(np.float32))
                                    for n, d in runs if (d / "vis" / f"slices_{sid}.npz").exists()] + [("3T GT", pg)]
    h, w = pg.shape
    crop = (slice(h // 2 - 80, h // 2 + 80), slice(w // 2 - 80, w // 2 + 80))
    fig, ax = plt.subplots(3, len(cols), figsize=(2.6 * len(cols), 8.6))
    for c, (n, img) in enumerate(cols):
        show(ax[0, c], img, n, cmap="gray", vmin=0, vmax=1)
        show(ax[1, c], img[crop], "zoom", cmap="gray", vmin=0, vmax=1)
        if n == "3T GT":
            ax[2, c].axis("off")
        else:
            mae = np.abs(img - pg)[pg > 0.005].mean()
            show(ax[2, c], np.abs(img - pg), f"|err|  MAE {mae:.4f}", cmap="inferno", vmin=0, vmax=0.3)
    fig.suptitle(f"fold {fold} — subject {sid}, axial z={Z_SHOW} (fig checkpoint of each run)")
    savefig(fig, path, 70)


def fig_zprofiles_all(runs, sid, path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axhline(0, color=MUTED, lw=0.8)
    first = True
    for n, d in runs:
        p = d / "vis" / f"slices_{sid}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        v = z["prof_valid"]
        zz = np.arange(len(v))[v]
        if first:
            ax.plot(zz, (z["prof_src"] - z["prof_gt"])[v], "--", color=IDENTITY_COLOR, label="identity")
            first = False
        ax.plot(zz, (z["prof_pred"] - z["prof_gt"])[v], ":" if n == "ph3_shuffle" else "-",
                color=RUN_COLOR.get(n, MUTED), lw=1.2, label=n)
    ax.set_xlabel("axial slice z")
    ax.set_ylabel("mean brain intensity: model − GT")
    ax.legend(ncol=5, fontsize=7, loc="upper left")
    ax.set_title(f"z-profile bias, subject {sid}")
    savefig(fig, path)


def fig_val_curves_all(runs, path):
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    idv = None
    for n, d in runs:
        hp = d / "history.json"
        if not hp.exists():
            continue
        h = json.load(open(hp))
        idv = idv or h[0]["val"]
        for a, k in zip(ax, ("ssim", "psnr")):
            a.plot([r["epoch"] for r in h], [r["val"][k] for r in h], ":" if n == "ph3_shuffle" else "-",
                   color=RUN_COLOR.get(n, MUTED), lw=1.2, label=n)
    for a, k, t in zip(ax, ("ssim", "psnr"), ("val SSIM", "val PSNR (dB)")):
        if idv:
            a.axhline(idv[f"{k}_id"], ls="--", color=IDENTITY_COLOR, label="identity")
        a.set_xlabel("epoch")
        a.set_title(t)
    ax[1].legend(fontsize=7, ncol=2)
    savefig(fig, path)


def write_markdown(fold, runs, path):
    lines = [f"# 3D test metrics — fold {fold} (1.5T → 3T, T2W)", "",
             "Test set = the fold's held-out subjects; values are **mean ± sample std over subjects**.",
             "PSNR / SSIM are 3D over brain voxels (SSIM 11³ Gaussian window). LPIPS (AlexNet), DISTS and FID "
             "(Inception-v3, per-subject slice sets) are 2.5D: averaged over all brain slices of the axial, coronal and "
             "sagittal planes. *FID pooled* = one FID over the slices of all test subjects.",
             "`best.pth` was chosen on validation slabs of these same subjects; `latest.pth` (final epoch) has no "
             "selection and is the unbiased estimate.",
             "*test-time physics swap*: the trained physics model fed another subject's ParamNet maps — a large drop "
             "means the model really uses subject-specific T1/T2/M0. `ph3_shuffle` was trained and tested with "
             "another subject's maps.", ""]
    for ckpt in ("latest", "best"):
        entries = metrics_table(fold, runs, ckpt)
        if len(entries) <= 1:
            continue
        n = entries[0][1]["summary"]["psnr"]["n"]
        lines += [f"## checkpoint: {ckpt}.pth  (n = {n} subjects)", "",
                  "| run | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | DISTS ↓ | FID ↓ (per subject) | FID pooled ↓ |",
                  "|---|---|---|---|---|---|---|"]
        for lab, e in entries:
            s = e["summary"]
            lines.append(f"| {lab} | " + " | ".join(fmt(s, k, METRIC_INFO[k][1]) for k in METRIC_KEYS)
                         + f" | {e['fid_pooled']:.2f} |")
        lines.append("")
    lines += ["Figures: `outputs/vis_compare/fold%d/`, per run `outputs/gen/<run>_fold%d/vis/`." % (fold, fold), ""]
    path.write_text("\n".join(lines))
    print("\n".join(lines))


def compare(fold, device):
    runs = run_dirs(fold)
    out = OUT_DIR / "vis_compare" / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    if not runs:
        print("no processed runs")
        return
    cfg = json.load(open(runs[0][1] / "config.json"))
    subjects, _ = test_subjects(cfg["modality"], fold)
    bank = VolumeBank(cfg["modality"])
    for ckpt in ("latest", "best"):
        entries = metrics_table(fold, runs, ckpt)
        if len(entries) > 1:
            fig_metric_bars(entries, out / f"metrics_bars_{ckpt}.png", f"fold {fold}, {ckpt}.pth — 3D test metrics")
    for sid in subjects:
        fig_compare_subject(fold, runs, sid, bank, out / f"compare_{sid}.png")
        fig_zprofiles_all(runs, sid, out / f"zprofile_all_{sid}.png")
    fig_val_curves_all(runs, out / "val_curves_all.png")
    write_markdown(fold, runs, OUT_DIR / f"METRICS3D_fold{fold}.md")
    print(f"COMPARE DONE -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpts", default="best,latest")
    ap.add_argument("--fig_ckpt", default="best")
    ap.add_argument("--out", default="", help="debug: write everything here instead of <run_dir>/vis")
    ap.add_argument("--max_subjects", type=int, default=0, help="debug")
    ap.add_argument("--no_metrics", action="store_true")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    torch.set_num_threads(8)  # training dataloaders share these CPUs
    if args.run_dir:
        process_run(args.run_dir, args, device)
    if args.compare:
        compare(args.fold, device)


if __name__ == "__main__":
    main()
