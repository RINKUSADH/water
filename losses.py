"""
losses.py — focused loss suite: L1 + 3D SSIM + VGG perceptual (2D slices), plus regularizers for the LUT/bias head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _gauss1d(size=7, sigma=1.5, device="cpu"):
    k = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    return k / k.sum()


def _sep_conv3d(x, k):
    n = k.numel()
    for dim in (2, 3, 4):
        shape = [1, 1, 1, 1, 1]
        shape[dim] = n
        x = F.conv3d(x, k.view(shape), padding=tuple(n // 2 if d == dim else 0 for d in (2, 3, 4)))
    return x


def ssim3d(x, y, data_range=1.0):
    k = _gauss1d(device=x.device)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mx, my = _sep_conv3d(x, k), _sep_conv3d(y, k)
    sxx = _sep_conv3d(x * x, k) - mx ** 2
    syy = _sep_conv3d(y * y, k) - my ** 2
    sxy = _sep_conv3d(x * y, k) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return s.mean()


def ssim2d_slices(pred, gt):
    """Challenge-style SSIM: per axial slice of (B,1,D,H,W), data_range = GT slice range. Returns mean."""
    B, _, D, H, W = pred.shape
    p = pred.permute(0, 2, 1, 3, 4).reshape(B * D, 1, H, W).float()
    g = gt.permute(0, 2, 1, 3, 4).reshape(B * D, 1, H, W).float()
    dr = (g.amax((1, 2, 3)) - g.amin((1, 2, 3))).clamp_min(1e-4).view(-1, 1, 1, 1)
    k = _gauss1d(device=p.device)
    kx, ky = k.view(1, 1, 1, -1), k.view(1, 1, -1, 1)
    f = lambda t: F.conv2d(F.conv2d(t, kx, padding=(0, 3)), ky, padding=(3, 0))  # noqa: E731
    c1, c2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
    mx, my = f(p), f(g)
    sxx, syy, sxy = f(p * p) - mx ** 2, f(g * g) - my ** 2, f(p * g) - mx * my
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
    return s.mean((1, 2, 3)).view(B, D).mean(1)


def psnr_slices(pred, gt):
    B, _, D, H, W = pred.shape
    p = pred.permute(0, 2, 1, 3, 4).reshape(B * D, -1).float()
    g = gt.permute(0, 2, 1, 3, 4).reshape(B * D, -1).float()
    dr = (g.amax(1) - g.amin(1)).clamp_min(1e-4)
    mse = ((p - g) ** 2).mean(1).clamp_min(1e-10)
    return (10 * torch.log10(dr ** 2 / mse)).view(B, D).mean(1)


class VGGPerceptual(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg, self.taps = vgg, {3, 8, 15}  # relu1_2, relu2_2, relu3_3
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, gt, n_slices=4):
        B, _, D, H, W = pred.shape
        idx = torch.randint(0, D, (n_slices,), device=pred.device)
        p = pred[:, :, idx].permute(0, 2, 1, 3, 4).reshape(-1, 1, H, W).repeat(1, 3, 1, 1)
        g = gt[:, :, idx].permute(0, 2, 1, 3, 4).reshape(-1, 1, H, W).repeat(1, 3, 1, 1)
        p, g = (p - self.mean) / self.std, (g - self.mean) / self.std
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            p, g = layer(p), layer(g)
            if i in self.taps:
                loss = loss + F.l1_loss(p, g)
        return loss


class SynthesisLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_ssim=1.0, w_perc=0.1, w_resid=0.01, w_bias_tv=0.1):
        super().__init__()
        self.w = dict(l1=w_l1, ssim=w_ssim, perc=w_perc, resid=w_resid, bias_tv=w_bias_tv)
        self.perc = VGGPerceptual() if w_perc > 0 else None

    def forward(self, pred, gt, aux):
        terms = {"l1": F.l1_loss(pred, gt), "ssim": 1.0 - ssim3d(pred.float(), gt.float())}
        if self.perc is not None:
            terms["perc"] = self.perc(pred, gt)
        if "resid" in aux:
            terms["resid"] = aux["resid"].abs().mean()
            lb = aux["log_bias_lr"]
            terms["bias_tv"] = sum((lb.diff(dim=d).abs().mean() for d in (2, 3, 4) if lb.shape[d] > 1), torch.zeros((), device=lb.device))
        total = sum(self.w[k] * v for k, v in terms.items())
        return total, {k: float(v.detach()) for k, v in terms.items()}
