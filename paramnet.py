"""
paramnet.py — ParamNet: 3D U-Net that estimates (log T1, log T2, M0) and a per-voxel uncertainty from one
weighted image, conditioned on modality and field strength.

Input : image (B, 1, D, H, W) in [0, 1], modality index (B,), field in tesla (B,)
Output: dict(log_t1, log_t2, m0, log_b) each (B, 1, D, H, W); log_b is the log Laplace scale (3 channels).
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import LOG_T1_RANGE, LOG_T2_RANGE, MODALITIES  # noqa: E402


def _gn(ch):
    return nn.GroupNorm(min(8, ch), ch)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False), _gn(cout), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False), _gn(cout), nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ParamUNet3D(nn.Module):
    def __init__(self, base: int = 24, levels: int = 4):
        super().__init__()
        self.levels = levels
        in_ch = 1 + len(MODALITIES) + 1  # image + modality one-hot + field (tesla / 3)
        chs = [base * 2 ** i for i in range(levels)]
        self.enc = nn.ModuleList()
        c = in_ch
        for ch in chs:
            self.enc.append(ConvBlock(c, ch))
            c = ch
        self.mid = ConvBlock(chs[-1], chs[-1])
        self.dec = nn.ModuleList()
        for i in range(levels - 1, 0, -1):
            self.dec.append(ConvBlock(chs[i] + chs[i - 1], chs[i - 1]))
        self.head = nn.Conv3d(chs[0], 6, 1)

    def forward(self, img, modality_idx, field_t):
        B, _, D, H, W = img.shape
        mult = 2 ** (self.levels - 1)
        pd_, ph, pw = (-D) % mult, (-H) % mult, (-W) % mult
        x = F.pad(img, (0, pw, 0, ph, 0, pd_), mode="replicate")
        onehot = F.one_hot(modality_idx.long(), len(MODALITIES)).float().to(img.dtype)
        cond = torch.cat([onehot, (field_t.float() / 3.0).unsqueeze(1).to(img.dtype)], 1)
        cond = cond.view(B, -1, 1, 1, 1).expand(-1, -1, *x.shape[2:])
        x = torch.cat([x, cond], 1)

        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < self.levels - 1:
                skips.append(x)
                x = F.max_pool3d(x, 2)
        x = self.mid(x)
        for blk in self.dec:
            s = skips.pop()
            x = F.interpolate(x, size=s.shape[2:], mode="trilinear", align_corners=False)
            x = blk(torch.cat([x, s], 1))
        out = self.head(x)[:, :, :D, :H, :W].float()
        # Smooth range mapping (a hard clamp has zero gradient outside the range and can freeze training).
        lt1 = LOG_T1_RANGE[0] + (LOG_T1_RANGE[1] - LOG_T1_RANGE[0]) * torch.sigmoid(out[:, 0:1])
        lt2 = LOG_T2_RANGE[0] + (LOG_T2_RANGE[1] - LOG_T2_RANGE[0]) * torch.sigmoid(out[:, 1:2])
        return {
            "log_t1": lt1,
            "log_t2": lt2,
            "m0": out[:, 2:3],
            "log_b": out[:, 3:6].clamp(-7.0, 3.0),
        }


def load_paramnet(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = ParamUNet3D(**ck.get("model_kwargs", {})).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, ck
