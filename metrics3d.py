"""
metrics3d.py — whole-volume image-quality metrics for one predicted volume vs its ground truth.

Volumes are (X, Y, Z) float arrays in [0, 1] (the cache normalization). The brain mask is the union of
GT > 0.005 and source > 0.005; everything is computed inside the brain bounding box (+8 voxel margin).

  PSNR   3D  MSE over all brain voxels, data_range = GT max - GT min (whole volume)
  SSIM   3D  Gaussian window 11^3 (sigma 1.5), data_range as PSNR, SSIM map averaged over brain voxels
  LPIPS  2.5D AlexNet-LPIPS over every brain slice of the axial, coronal and sagittal planes
             (slice kept if >= 2% brain); mean over slices per plane, then over the 3 planes
  DISTS  2.5D same slices, DISTS (VGG16, Ding et al. 2020)
  FID    2.5D Inception-v3 pool3 (2048-d) features of the same slices; Frechet distance between the predicted
             and the GT slice sets of one subject (per-subject FID). Features are returned so the caller can also
             compute one pooled FID over all test subjects.
LPIPS / DISTS / FID use 2D ImageNet networks, so "3D" means the three-plane (2.5D) average used in 3D MRI
synthesis papers; PSNR and SSIM are true 3D.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

MIN_BRAIN_FRAC = 0.02
MARGIN = 8
BATCH = 16


def brain_mask(gt, src):
    return (gt > 0.005) | (src > 0.005)


def bbox(mask, margin=MARGIN):
    sl = []
    for ax in range(3):
        idx = np.where(mask.any(axis=tuple(a for a in range(3) if a != ax)))[0]
        sl.append(slice(max(int(idx[0]) - margin, 0), min(int(idx[-1]) + 1 + margin, mask.shape[ax])))
    return tuple(sl)


def _gauss_sep(x, size=11, sigma=1.5):
    k = torch.arange(size, device=x.device, dtype=torch.float32) - size // 2
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    for dim in (2, 3, 4):
        shape = [1, 1, 1, 1, 1]
        shape[dim] = size
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (4 - dim)] = pad[2 * (4 - dim) + 1] = size // 2
        x = F.conv3d(F.pad(x, pad), k.view(shape))
    return x


@torch.no_grad()
def fid_from_feats(fa, fb, device="cpu"):
    """Frechet distance between two feature sets (N, 2048). tr sqrt(S1 S2) is computed as
    sum sqrt(eig(S1^1/2 S2 S1^1/2)) (symmetric PSD, float64) — same value as scipy.linalg.sqrtm, ~100x faster."""
    a = torch.as_tensor(fa, dtype=torch.float64, device=device)
    b = torch.as_tensor(fb, dtype=torch.float64, device=device)
    s1, s2 = torch.cov(a.T), torch.cov(b.T)
    e, v = torch.linalg.eigh(s1)
    r1 = (v * e.clamp_min(0).sqrt()) @ v.T
    m = r1 @ s2 @ r1
    tr = torch.linalg.eigvalsh(0.5 * (m + m.T)).clamp_min(0).sqrt().sum()
    return float(((a.mean(0) - b.mean(0)) ** 2).sum() + s1.trace() + s2.trace() - 2 * tr)


class Metrics3D:
    def __init__(self, device):
        self.device = device
        self._lpips = self._dists = self._incep = None

    # ------------------------------------------------------------------ lazy networks
    def lpips(self):
        if self._lpips is None:
            import lpips
            self._lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
        return self._lpips

    def dists(self):
        if self._dists is None:
            from DISTS_pytorch import DISTS
            self._dists = DISTS().to(self.device).eval()
        return self._dists

    def inception(self):
        if self._incep is None:
            from pytorch_fid.inception import InceptionV3
            self._incep = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(self.device).eval()
        return self._incep

    # ------------------------------------------------------------------ pieces
    @torch.no_grad()
    def psnr_ssim(self, pred, gt, mask, dr):
        p = torch.from_numpy(pred).to(self.device)[None, None]
        g = torch.from_numpy(gt).to(self.device)[None, None]
        m = torch.from_numpy(mask).to(self.device)[None, None]
        mse = ((p - g) ** 2)[m].mean().clamp_min(1e-12)
        psnr = float(10 * torch.log10(dr ** 2 / mse))
        c1, c2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
        mx, my = _gauss_sep(p), _gauss_sep(g)
        sxx = _gauss_sep(p * p) - mx ** 2
        syy = _gauss_sep(g * g) - my ** 2
        sxy = _gauss_sep(p * g) - mx * my
        s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
        return psnr, float(s[m].mean())

    @staticmethod
    def plane_slices(vol, mask, plane):
        """(N, 1, H, W) tensor of the brain slices along one axis (0 = sagittal, 1 = coronal, 2 = axial)."""
        v, m = np.moveaxis(vol, plane, 0), np.moveaxis(mask, plane, 0)
        keep = m.reshape(m.shape[0], -1).mean(1) >= MIN_BRAIN_FRAC
        return torch.from_numpy(np.ascontiguousarray(v[keep]))[:, None], keep

    @torch.no_grad()
    def perceptual(self, pred, gt, mask):
        lp_net, di_net, inc = self.lpips(), self.dists(), self.inception()
        out = {"lpips": [], "dists": []}
        fp, fg = [], []
        for plane in (0, 1, 2):
            ps, keep = self.plane_slices(pred, mask, plane)
            gs, _ = self.plane_slices(gt, mask, plane)
            lp, di = [], []
            for i in range(0, len(ps), BATCH):
                p = ps[i:i + BATCH].to(self.device).float().repeat(1, 3, 1, 1)
                g = gs[i:i + BATCH].to(self.device).float().repeat(1, 3, 1, 1)
                lp.append(lp_net(p * 2 - 1, g * 2 - 1).flatten().cpu())
                di.append(di_net(p, g, batch_average=False).flatten().cpu())
                fp.append(inc(p)[0].flatten(1).cpu())
                fg.append(inc(g)[0].flatten(1).cpu())
            out["lpips"].append(float(torch.cat(lp).mean()))
            out["dists"].append(float(torch.cat(di).mean()))
        res = {k: float(np.mean(v)) for k, v in out.items()}
        res.update({f"lpips_{n}": v for n, v in zip(("sag", "cor", "ax"), out["lpips"])})
        res.update({f"dists_{n}": v for n, v in zip(("sag", "cor", "ax"), out["dists"])})
        fp, fg = torch.cat(fp).numpy(), torch.cat(fg).numpy()
        res["fid"] = fid_from_feats(fp, fg, self.device)
        return res, fp, fg

    # ------------------------------------------------------------------ main entry
    def compute(self, pred, gt, src):
        """Returns (metrics dict, inception feats of pred slices, inception feats of GT slices)."""
        mask_full = brain_mask(gt, src)
        dr = max(float(gt.max() - gt.min()), 1e-4)
        bb = bbox(mask_full)
        p = np.ascontiguousarray(pred[bb], dtype=np.float32)
        g = np.ascontiguousarray(gt[bb], dtype=np.float32)
        m = np.ascontiguousarray(mask_full[bb])
        psnr, ssim = self.psnr_ssim(p, g, m, dr)
        perc, fp, fg = self.perceptual(p, g, m)
        return {"psnr": psnr, "ssim": ssim, **perc}, fp, fg


def mean_std(rows, keys):
    """rows: list of dicts -> {key: (mean, sample std, n)}."""
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": int(len(v))}
    return out


def fmt(ms, k, digits):
    if k not in ms or ms[k]["n"] == 0 or math.isnan(ms[k]["mean"]):
        return "–"
    return f"{ms[k]['mean']:.{digits}f} ± {ms[k]['std']:.{digits}f}"
