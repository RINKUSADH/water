"""
train.py — train one generator variant (one ablation row) on one cross-validation fold, then run full-volume eval.

Example:
  python gen/train.py --run ph1_attn --fold 0 --gpu 2 --attn_blocks 4
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_DIR, FieldTransfer  # noqa: E402
from data import CACHE_DIR, PairedSlabDataset, parse_directions  # noqa: E402
from losses import SynthesisLoss, psnr_slices, ssim2d_slices  # noqa: E402
from model import build_model  # noqa: E402
from physics_stack import build_physics, build_proto_input  # noqa: E402

SLOW_MODULES = ("tblocks", "ctx_enc", "protos", "phys_enc", "type_emb", "unc_lambda")


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--modality", default="T2W")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--samples_per_epoch", type=int, default=400)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_attn", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    # architecture / phase flags
    ap.add_argument("--base_ch", type=int, default=32)
    ap.add_argument("--n_res", type=int, default=2)
    ap.add_argument("--attn_blocks", type=int, default=0)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--volctx", type=int, default=0)
    ap.add_argument("--phys", default="none", choices=["none", "concat", "spade", "spade_xattn"])
    ap.add_argument("--proto", type=int, default=0)
    ap.add_argument("--adain_dec", type=int, default=0)
    ap.add_argument("--head", default="gate", choices=["gate", "direct", "lut_bias"])
    ap.add_argument("--phys_shuffle", type=int, default=0, help="control: physics maps from another subject")
    # data / losses
    ap.add_argument("--directions", default="all")
    ap.add_argument("--main_dir_prob", type=float, default=0.5)
    ap.add_argument("--w_l1", type=float, default=1.0)
    ap.add_argument("--w_ssim", type=float, default=1.0)
    ap.add_argument("--w_perc", type=float, default=0.1)
    ap.add_argument("--field_transfer", default=str(OUT_DIR / "field_transfer" / "field_transfer.json"))
    ap.add_argument("--eval_full", type=int, default=1)
    ap.add_argument("--max_val_slabs", type=int, default=0, help="debug")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def to_device(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def forward(model, batch, cfg, ft, device):
    """Shared forward used by training, validation and full-volume evaluation."""
    x = batch["src"]
    src_idx = torch.as_tensor(batch["src_idx"], device=device)
    tgt_idx = torch.as_tensor(batch["tgt_idx"], device=device)
    src_t = torch.as_tensor(batch["src_t"], device=device, dtype=torch.float32)
    tgt_t = torch.as_tensor(batch["tgt_t"], device=device, dtype=torch.float32)
    zpos = torch.as_tensor(batch["zpos"], device=device, dtype=torch.float32)
    phys = proto_in = None
    if cfg["phys"] != "none":
        phys = build_physics(batch["phys"], x, src_t, tgt_t, ft, cfg["modality"])
    if cfg["proto"]:
        proto_in = build_proto_input(batch["phys_ctx"], batch["ctx"], src_t, tgt_t, ft)
    return model(x, src_idx, tgt_idx, zpos=zpos, ctx_vol=batch.get("ctx"), phys=phys, proto_in=proto_in)


@torch.no_grad()
def validate(model, loader, loss_fn, cfg, ft, device):
    model.eval()
    agg = {"loss": [], "ssim": [], "psnr": [], "ssim_id": [], "psnr_id": []}
    for i, batch in enumerate(loader):
        batch = to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, aux = forward(model, batch, cfg, ft, device)
            loss, _ = loss_fn(pred.float(), batch["tgt"], aux)
        agg["loss"].append(loss.item())
        agg["ssim"] += ssim2d_slices(pred.float(), batch["tgt"]).tolist()
        agg["psnr"] += psnr_slices(pred.float(), batch["tgt"]).tolist()
        agg["ssim_id"] += ssim2d_slices(batch["src"], batch["tgt"]).tolist()
        agg["psnr_id"] += psnr_slices(batch["src"], batch["tgt"]).tolist()
        if cfg["max_val_slabs"] and (i + 1) * loader.batch_size >= cfg["max_val_slabs"]:
            break
    model.train()
    return {k: float(np.mean(v)) for k, v in agg.items()}


def main(argv=None):
    args = get_args(argv)
    cfg = vars(args)
    device = torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = OUT_DIR / "gen" / f"{args.run}_fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(out_dir / "config.json", "w"), indent=1)

    meta = json.load(open(CACHE_DIR / args.modality / "meta.json"))
    fold = meta["folds"][args.fold]
    use_phys = args.phys != "none" or bool(args.proto)
    directions = parse_directions(args.directions)
    train_ds = PairedSlabDataset(args.modality, fold["train"], directions, True, args.samples_per_epoch,
                                 main_dir_prob=args.main_dir_prob, use_phys=use_phys, phys_shuffle=bool(args.phys_shuffle),
                                 seed=args.seed)
    val_ds = PairedSlabDataset(args.modality, fold["val"], directions, False, use_phys=use_phys)
    # Each worker gets its own RNG stream.
    wi = lambda wid: setattr(torch.utils.data.get_worker_info().dataset, "rng", __import__("random").Random(args.seed * 1000 + wid + int(time.time())))  # noqa: E731
    train_dl = DataLoader(train_ds, batch_size=args.batch, num_workers=args.workers, pin_memory=True,
                          drop_last=True, persistent_workers=args.workers > 0, worker_init_fn=wi if args.workers else None)
    val_dl = DataLoader(val_ds, batch_size=args.batch, num_workers=min(args.workers, 4), pin_memory=True)
    print(f"[{args.run} fold {args.fold}] train subjects {fold['train']} | val {fold['val']} | "
          f"{len(directions)} directions | val slabs {len(val_ds)}", flush=True)

    ft = FieldTransfer.from_json(args.field_transfer) if use_phys else FieldTransfer()
    model = build_model(cfg).to(device)
    slow = [p for n, p in model.named_parameters() if n.startswith(SLOW_MODULES)]
    fast = [p for n, p in model.named_parameters() if not n.startswith(SLOW_MODULES)]
    opt = torch.optim.AdamW([{"params": fast, "lr": args.lr}, {"params": slow, "lr": args.lr_attn}], weight_decay=args.wd)
    total = args.epochs * (args.samples_per_epoch // args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    loss_fn = SynthesisLoss(args.w_l1, args.w_ssim, args.w_perc).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params {n_params:.2f}M (slow-lr {sum(p.numel() for p in slow) / 1e6:.2f}M) | {total} steps", flush=True)

    history, best, step = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        t0, sums, n = time.time(), {}, 0
        for batch in train_dl:
            batch = to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred, aux = forward(model, batch, cfg, ft, device)
            loss, terms = loss_fn(pred.float(), batch["tgt"], aux)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            n += 1
            for k, v in {**terms, "total": loss.item(), "gradnorm": float(gn)}.items():
                sums[k] = sums.get(k, 0.0) + v
            for k in ("gate_mean",):
                if k in aux:
                    sums[k] = sums.get(k, 0.0) + float(aux[k])
        tr = {k: v / max(n, 1) for k, v in sums.items()}
        val = validate(model, val_dl, loss_fn, cfg, ft, device)
        gates = {nm: float(torch.tanh(p).item()) for nm, p in model.named_parameters() if nm.endswith(("alpha_c", "alpha_p"))}
        rec = {"epoch": epoch, "step": step, "train": tr, "val": val, "gates": gates, "sec": time.time() - t0,
               "mem_gb": torch.cuda.max_memory_allocated(device) / 1e9}
        history.append(rec)
        print(f"ep {epoch:3d} | train {tr['total']:.4f} (l1 {tr['l1']:.4f}) | val SSIM {val['ssim']:.4f} PSNR {val['psnr']:.2f} "
              f"(identity {val['ssim_id']:.4f}/{val['psnr_id']:.2f}) | {rec['sec']:.0f}s | {rec['mem_gb']:.1f}GB"
              + (f" | gate {tr['gate_mean']:.3f}" if "gate_mean" in tr else "")
              + (f" | xattn {gates}" if gates else ""), flush=True)
        ck = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val": val}
        torch.save(ck, out_dir / "latest.pth")
        if val["ssim"] > best:
            best = val["ssim"]
            torch.save(ck, out_dir / "best.pth")
        json.dump(history, open(out_dir / "history.json", "w"), indent=1)

    print(f"best val SSIM {best:.4f}", flush=True)
    if args.eval_full:
        from evaluate import evaluate_run
        evaluate_run(out_dir, device)
    (out_dir / "DONE").write_text("ok\n")
    print("TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
