"""
data.py — MRIxFields paired data for the generator.

* 20 paired subjects (Training_prospective 006/007/009 + Validating_prospective 001..020 named files), each with
  all five fields. The duplicate `P_*_0010..0012` / unmatched `P_*_0004/5/8` files in Validating_prospective are ignored.
* Subject-level K-fold cross-validation.
* A disk cache of globally-normalized float16 volumes (X, Y, Z) and 2 mm context volumes (Z/4, X/4, Y/4) that all
  parallel runs share through memory-mapping (built once by prepare_cache.py).
* Optional physics cache from ParamNet (precompute_physics.py): (4, X, Y, Z) = [log T1, log T2, M0, log-scale].
"""

import itertools
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FIELD_IDX, FIELD_TESLA, FIELDS, OUT_DIR  # noqa: E402

DATA_ROOT = Path("/DATA1/obed/Rinku")
TRAIN_P = DATA_ROOT / "release_20260414" / "Training_prospective"
VAL_P = DATA_ROOT / "Validating_prospective"
CACHE_DIR = OUT_DIR / "cache"
PHYS_DIR = OUT_DIR / "physics"
SLAB = 16
Z_MIN, Z_MARGIN = 40, 20  # slab sampling range [Z_MIN, Z - SLAB - Z_MARGIN], as in the original dataset
CTX_POOL = 4


def paired_index(modality: str) -> dict:
    subjects = {}
    for f in FIELDS:
        for p in sorted((TRAIN_P / modality / f).glob(f"P_{modality}_{f}_*.nii.gz")):
            sid = p.name.split("_")[-1].replace(".nii.gz", "")[-3:]
            subjects.setdefault(sid, {})[f] = p
        for p in sorted((VAL_P / modality / f).glob("[0-9][0-9][0-9]_*.nii.gz")):
            subjects.setdefault(p.name[:3], {})[f] = p
    return {s: v for s, v in sorted(subjects.items()) if all(f in v for f in FIELDS)}


def make_folds(subjects, k=5, seed=0):
    s = sorted(subjects)
    random.Random(seed).shuffle(s)
    folds = [sorted(s[i::k]) for i in range(k)]
    return [(sorted(set(s) - set(f)), f) for f in folds]


def parse_directions(spec: str):
    if spec == "all":
        return [(a, b) for a, b in itertools.permutations(FIELDS, 2)]
    if spec == "up":
        return [(a, b) for a, b in itertools.permutations(FIELDS, 2) if FIELD_TESLA[b] > FIELD_TESLA[a]]
    return [tuple(d.split("-")) for d in spec.split(",")]


def cache_path(modality, sid, field, kind="vol", root=CACHE_DIR):
    return root / modality / f"{sid}_{field}_{kind}.npy"


class VolumeBank:
    """Memory-mapped access to cached volumes (shared page cache across processes and runs)."""

    def __init__(self, modality, phys=False):
        self.modality, self.phys = modality, phys
        self._mm = {}

    def _get(self, path):
        if path not in self._mm:
            self._mm[path] = np.load(path, mmap_mode="r")
        return self._mm[path]

    def vol(self, sid, field):
        return self._get(cache_path(self.modality, sid, field, "vol"))

    def ctx(self, sid, field):
        return self._get(cache_path(self.modality, sid, field, "ctx"))

    def phys_vol(self, sid, field):
        return self._get(cache_path(self.modality, sid, field, "phys", PHYS_DIR))

    def phys_ctx(self, sid, field):
        return self._get(cache_path(self.modality, sid, field, "physctx", PHYS_DIR))


def slab_dhw(vol_xyz, z0):
    """(X, Y, Z) -> slab (SLAB, X, Y) float32; works for (C, X, Y, Z) too."""
    s = np.asarray(vol_xyz[..., z0:z0 + SLAB], dtype=np.float32)
    return np.moveaxis(s, -1, -3)


class PairedSlabDataset(Dataset):
    """Training: random (subject, direction, z) slabs. Validation: fixed 1.5T->3T slabs."""

    def __init__(self, modality, subjects, directions, train=True, samples_per_epoch=400,
                 main_dir=("1.5T", "3T"), main_dir_prob=0.5, use_phys=False, phys_shuffle=False, seed=0):
        self.bank = VolumeBank(modality, use_phys)
        self.subjects, self.directions, self.train = list(subjects), list(directions), train
        self.samples_per_epoch, self.main_dir, self.main_dir_prob = samples_per_epoch, main_dir, main_dir_prob
        self.use_phys, self.phys_shuffle = use_phys, phys_shuffle
        Z = self.bank.vol(self.subjects[0], "1.5T").shape[2]
        self.Z = Z
        self.z_hi = Z - SLAB - Z_MARGIN
        if not train:
            self.items = [(s, main_dir[0], main_dir[1], z) for s in self.subjects for z in range(100, 260, SLAB)]
        self.rng = random.Random(seed)

    def __len__(self):
        return self.samples_per_epoch if self.train else len(self.items)

    def _sample(self):
        sid = self.rng.choice(self.subjects)
        if self.main_dir in self.directions and self.rng.random() < self.main_dir_prob:
            src, tgt = self.main_dir
        else:
            src, tgt = self.rng.choice(self.directions)
        return sid, src, tgt, self.rng.randint(Z_MIN, self.z_hi)

    def __getitem__(self, idx):
        sid, src, tgt, z0 = self._sample() if self.train else self.items[idx]
        b = self.bank
        out = {
            "src": slab_dhw(b.vol(sid, src), z0)[None],
            "tgt": slab_dhw(b.vol(sid, tgt), z0)[None],
            "ctx": np.asarray(b.ctx(sid, src), dtype=np.float32)[None],
        }
        if self.use_phys:
            psid = sid
            if self.phys_shuffle:  # control: physics maps of a different subject
                others = [s for s in self.subjects if s != sid] or [sid]
                psid = random.Random(hash((sid, z0, src))).choice(others)
            out["phys"] = slab_dhw(b.phys_vol(psid, src), z0)
            out["phys_ctx"] = np.asarray(b.phys_ctx(psid, src), dtype=np.float32)
        if self.train and self.rng.random() < 0.5:  # left-right flip: H axis of slabs (= X), axis 1 of ctx (Z,X,Y)
            out["src"], out["tgt"] = out["src"][:, :, ::-1], out["tgt"][:, :, ::-1]
            out["ctx"] = out["ctx"][:, :, ::-1]
            if self.use_phys:
                out["phys"], out["phys_ctx"] = out["phys"][:, :, ::-1], out["phys_ctx"][:, :, ::-1]
        out = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in out.items()}
        out.update(src_idx=FIELD_IDX[src], tgt_idx=FIELD_IDX[tgt],
                   src_t=FIELD_TESLA[src], tgt_t=FIELD_TESLA[tgt],
                   zpos=(z0 + SLAB / 2) / self.Z, sid=sid, z0=z0)
        return out
