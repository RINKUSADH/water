"""
smoke_test.py — build every phase variant, run forward/backward on real slabs, check zero-init properties,
report step time and peak memory. Physics variants need outputs/physics (use --phys to include them).
"""

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gen"))
from common import FieldTransfer  # noqa: E402
from data import PairedSlabDataset, parse_directions  # noqa: E402
from losses import SynthesisLoss  # noqa: E402
from model import build_model  # noqa: E402
from train import forward, to_device  # noqa: E402

BASE = dict(modality="T2W", base_ch=32, n_res=2, heads=8, attn_blocks=0, volctx=0, phys="none", proto=0,
            adain_dec=0, head="gate")
VARIANTS = {
    "ph0_base": {},
    "ph1_attn": dict(attn_blocks=4),
    "ph2_volctx": dict(attn_blocks=4, volctx=1),
    "ph5_head": dict(attn_blocks=4, volctx=1, head="lut_bias"),
    "ph4_adain": dict(attn_blocks=4, volctx=1, head="lut_bias", adain_dec=1),
}
PHYS_VARIANTS = {
    "ph3_concat": dict(attn_blocks=4, volctx=1, head="lut_bias", adain_dec=1, phys="concat"),
    "ph3_spade": dict(attn_blocks=4, volctx=1, head="lut_bias", adain_dec=1, phys="spade"),
    "ph3_full": dict(attn_blocks=4, volctx=1, head="lut_bias", adain_dec=1, phys="spade_xattn", proto=1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--phys", action="store_true")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    variants = {**VARIANTS, **(PHYS_VARIANTS if args.phys else {})}
    loss_fn = SynthesisLoss().to(device)
    ft = FieldTransfer()
    ds = PairedSlabDataset("T2W", ["001", "002"], parse_directions("all"), True, 4, use_phys=args.phys)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    batch = to_device(batch, device)
    for name, over in variants.items():
        cfg = {**BASE, **over}
        torch.manual_seed(0)
        model = build_model(cfg).to(device)
        torch.cuda.reset_peak_memory_stats(device)
        # zero-init check: the LUT/bias head must start as the identity mapping (inside the brain mask)
        if cfg["head"] == "lut_bias":
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred, _ = forward(model, batch, cfg, ft, device)
            x = batch["src"]
            ident_err = float((pred.float() - x * (x > 0.005)).abs().max())
            assert ident_err < 2e-2, f"{name}: LUT/bias head is not identity at init ({ident_err})"
        t0 = time.time()
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred, aux = forward(model, batch, cfg, ft, device)
            loss, terms = loss_fn(pred.float(), batch["tgt"], aux)
            loss.backward()
            bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            assert not bad, f"{name}: non-finite grads in {bad[:3]}"
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        dt = (time.time() - t0) / 3
        n = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{name:<12} params {n:6.2f}M | loss {loss.item():.4f} {terms} | {dt:.2f}s/step (B=2) | "
              f"peak {torch.cuda.max_memory_allocated(device) / 1e9:.1f}GB | pred range [{pred.min().item():.2f},{pred.max().item():.2f}]",
              flush=True)
        del model
        torch.cuda.empty_cache()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
