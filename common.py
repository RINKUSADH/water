"""
common.py — shared constants, normalization and MR physics for PhysGlobal3D.

Array conventions
-----------------
* Volumes on disk / in numpy are (X, Y, Z): X = left-right, Y = anterior-posterior, Z = axial (slab axis).
  This holds for MRIxFields (LAS, 364x436x364 @ 0.5 mm), Cambridge MRF (RAS) and Dryad SAFE MRF.
* Network tensors are (B, C, D, H, W) with D = Z, H = X, W = Y  (i.e. vol[:, :, z0:z1].transpose(2, 0, 1)).
* Relaxation times are in SECONDS everywhere inside this project.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "outputs"

# ----------------------------------------------------------------------------- fields / modalities
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
FIELD_TESLA = {"0.1T": 0.1, "1.5T": 1.5, "3T": 3.0, "5T": 5.0, "7T": 7.0}
FIELD_IDX = {f: i for i, f in enumerate(FIELDS)}
MODALITIES = ["T1W", "T2W", "T2FLAIR"]
MOD_IDX = {m: i for i, m in enumerate(MODALITIES)}

# Physical clipping ranges (seconds). T2 is capped at 0.5 s because the Cambridge MRF dictionary saturates there.
T1_RANGE = (0.05, 5.0)
T2_RANGE = (0.005, 0.5)
LOG_T1_RANGE = (math.log(T1_RANGE[0]), math.log(T1_RANGE[1]))
LOG_T2_RANGE = (math.log(T2_RANGE[0]), math.log(T2_RANGE[1]))


def norm_volume(vol: np.ndarray, pct_lo: float = 0.5, pct_hi: float = 99.5) -> np.ndarray:
    """Global volume percentile normalization (identical to the original dataset3d._norm_volume_3d)."""
    lo = np.percentile(vol, pct_lo)
    hi = np.percentile(vol, pct_hi)
    if hi - lo < 1e-6:
        return np.zeros_like(vol, dtype=np.float32)
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def to_dhw(vol_xyz):
    """(X, Y, Z) -> (Z, X, Y) for numpy arrays or torch tensors (last three dims)."""
    if isinstance(vol_xyz, np.ndarray):
        return np.moveaxis(vol_xyz, -1, -3)
    return vol_xyz.movedim(-1, -3)


def to_xyz(vol_dhw):
    """(Z, X, Y) -> (X, Y, Z)."""
    if isinstance(vol_dhw, np.ndarray):
        return np.moveaxis(vol_dhw, -3, -1)
    return vol_dhw.movedim(-3, -1)


# ----------------------------------------------------------------------------- Bloch signal models (torch)
# Default protocols (seconds / degrees) used for deterministic evaluation and for the physics prior image.
DEFAULT_SEQ = {
    "T2W": dict(TR=4.0, TE=0.090),
    "T2FLAIR": dict(TR=9.0, TE=0.120, TI=2.5),
    "T1W": dict(TR=2.0, TE=0.003, TI=0.9),
}
# Randomization ranges for simulation-based ParamNet training.
SEQ_RANGES = {
    "T2W": dict(TR=(2.5, 6.0), TE=(0.070, 0.130)),
    "T2FLAIR": dict(TR=(7.0, 11.0), TE=(0.080, 0.140), TI=(2.0, 2.9)),
    "T1W": dict(TR=(1.5, 2.5), TE=(0.002, 0.005), TI=(0.7, 1.2)),
}


def sample_seq(modality: str, rng: np.random.Generator) -> dict:
    return {k: float(rng.uniform(*v)) for k, v in SEQ_RANGES[modality].items()}


def bloch_signal(m0, t1, t2, modality: str, seq: dict):
    """Closed-form steady-state signal. t1/t2 in seconds; tensors broadcast.

    T2W     : spin echo           M0 (1 - e^{-TR/T1}) e^{-TE/T2}
    T2FLAIR : inversion recovery  M0 |1 - 2 e^{-TI/T1} + e^{-TR/T1}| e^{-TE/T2}
    T1W     : MPRAGE-like IR-GRE  M0 max(1 - 2 e^{-TI/T1} + e^{-TR/T1}, 0) e^{-TE/T2}
              (phase-sensitive, so CSF stays dark; a simplification of the full MPRAGE equation)
    """
    t1 = t1.clamp(*T1_RANGE)
    t2 = t2.clamp(*T2_RANGE)
    e2 = torch.exp(-seq["TE"] / t2)
    if modality == "T2W":
        return m0 * (1.0 - torch.exp(-seq["TR"] / t1)) * e2
    mz = 1.0 - 2.0 * torch.exp(-seq["TI"] / t1) + torch.exp(-seq["TR"] / t1)
    if modality == "T2FLAIR":
        return m0 * mz.abs() * e2
    if modality == "T1W":
        return m0 * mz.clamp_min(0.0) * e2
    raise ValueError(modality)


# ----------------------------------------------------------------------------- field transfer
class FieldTransfer:
    """Maps source-field relaxation times to a target field.

    Fitted model (Track P2) at the reference ratio B_t/B_s = 2 (1.5T -> 3T):
        log T_tgt = a + b * log T_src
    Generalized to any field pair with r = log2(B_t / B_s) (r = 1 is the fitted pair, r = 0 identity):
        log T_tgt = r * a + (1 + r * (b - 1)) * log T_src
    The default (no fit available) reproduces the original fixed rule T1 * (B_t/B_s)^0.35 and leaves T2 unchanged.
    Extrapolation to pairs other than 1.5T->3T is an assumption, not a measurement.
    """

    def __init__(self, params: dict = None):
        default = {"t1": {"a": 0.35 * math.log(2.0), "b": 1.0}, "t2": {"a": 0.0, "b": 1.0}}
        self.params = params or default

    @classmethod
    def from_json(cls, path):
        if path is None or not Path(path).exists():
            return cls()
        with open(path) as f:
            d = json.load(f)
        return cls({"t1": d["fit"]["t1"], "t2": d["fit"]["t2"]})

    def apply_log(self, log_t, which: str, src_tesla, tgt_tesla):
        """log_t: tensor (B, 1, ...); src/tgt_tesla: tensors (B,)."""
        r = torch.log2(tgt_tesla / src_tesla).view(-1, *([1] * (log_t.dim() - 1)))
        a, b = self.params[which]["a"], self.params[which]["b"]
        return r * a + (1.0 + r * (b - 1.0)) * log_t


def norm_log_t1(log_t1):
    return (log_t1 - LOG_T1_RANGE[0]) / (LOG_T1_RANGE[1] - LOG_T1_RANGE[0])


def norm_log_t2(log_t2):
    return (log_t2 - LOG_T2_RANGE[0]) / (LOG_T2_RANGE[1] - LOG_T2_RANGE[0])


# ----------------------------------------------------------------------------- misc torch helpers
def gaussian_blur3d(x, sigmas):
    """Separable Gaussian blur of a (B, C, D, H, W) tensor; sigmas = (sd, sh, sw) in voxels (0 = skip)."""
    for dim, s in zip((2, 3, 4), sigmas):
        if s <= 1e-3:
            continue
        r = max(1, int(math.ceil(3 * s)))
        k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
        k = torch.exp(-0.5 * (k / s) ** 2)
        k = k / k.sum()
        shape = [1, 1, 1, 1, 1]
        shape[dim] = k.numel()
        k = k.view(shape).repeat(x.shape[1], 1, 1, 1, 1)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (4 - dim)] = r
        pad[2 * (4 - dim) + 1] = r
        x = F.conv3d(F.pad(x, pad, mode="replicate"), k, groups=x.shape[1])
    return x


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
