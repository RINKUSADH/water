"""
summarize.py — collect Track P reports and every generator run into outputs/RESULTS.md.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT_DIR  # noqa: E402

ORDER = ["ph0_base", "ph1_attn", "ph2_volctx", "ph5_head", "ph4_adain", "ph3_concat", "ph3_spade", "ph3_full", "ph3_shuffle"]
DESC = {
    "ph0_base": "CNN baseline (original architecture, gate head) + new data/losses",
    "ph1_attn": "+ bottleneck self-attention (Phase 1)",
    "ph2_volctx": "+ volume context tokens & slab z (Phase 2)",
    "ph5_head": "+ LUT x bias + residual head (Phase 5)",
    "ph4_adain": "+ decoder modulation from global vector (Phase 4) = full global model",
    "ph3_concat": "full global + physics as input channels (old method)",
    "ph3_spade": "full global + SPADE physics (3c)",
    "ph3_full": "full global + SPADE + physics x-attn + tissue prototypes (3a+3b+3c)",
    "ph3_shuffle": "control H: ph3_full with physics maps from another subject",
}


def main():
    lines = ["# PhysGlobal3D results", ""]
    pr = OUT_DIR / "paramnet" / "test_report.md"
    if pr.exists():
        lines += ["## Track P1 — ParamNet (held-out MRF subjects)", "", *pr.read_text().splitlines()[2:], ""]
    ftp = OUT_DIR / "field_transfer" / "field_transfer.json"
    if ftp.exists():
        ft = json.load(open(ftp))
        lines += ["## Track P2 — measured 1.5T -> 3T field transfer", "",
                  f"Fit: T1 a={ft['fit']['t1']['a']:.3f} b={ft['fit']['t1']['b']:.3f} | T2 a={ft['fit']['t2']['a']:.3f} "
                  f"b={ft['fit']['t2']['b']:.3f} | effective exponent at median: {ft['effective_field_exponent_at_median']} | "
                  f"pairs used {ft['n_pairs_used']}", "",
                  "| method | T1 MAE (ms) | T2 MAE (ms) |", "|---|---|---|"]
        for m in ("identity", "fixed_0.35", "power_law", "lut"):
            lines.append(f"| {m} | {ft['loso_mean']['t1'][m]['mae_s'] * 1000:.1f} | {ft['loso_mean']['t2'][m]['mae_s'] * 1000:.1f} |")
        lines.append("")

    runs = {}
    for f in sorted((OUT_DIR / "gen").glob("*/eval_full.json")):
        r = json.load(open(f))
        name, fold = r["run"].rsplit("_fold", 1)
        runs.setdefault(name, {})[fold] = r
    if runs:
        lines += ["## Generator ablation (1.5T -> 3T, full volumes, slices 150-180, fold validation subjects)", "",
                  "| run | description | folds | SSIM | PSNR | NRMSE | LPIPS | low-freq err | high-freq err | z-jitter |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        first = next(iter(runs.values()))
        idm = {k: sum(r["mean"]["identity"][k] for r in first.values()) / len(first) for k in first[next(iter(first))]["mean"]["identity"]}
        lines.append(f"| identity | output = 1.5T input | {','.join(first)} | {idm['ssim']:.4f} | {idm['psnr']:.2f} | {idm['nrmse']:.4f} | "
                     f"{idm['lpips']:.4f} | {idm['err_low_freq']:.4f} | {idm['err_high_freq']:.4f} | {idm['z_jitter']:.4f} |")
        for name in sorted(runs, key=lambda n: ORDER.index(n) if n in ORDER else 99):
            rs = runs[name]
            m = {k: sum(r["mean"]["model"][k] for r in rs.values()) / len(rs) for k in idm}
            lines.append(f"| {name} | {DESC.get(name, '')} | {','.join(rs)} | {m['ssim']:.4f} | {m['psnr']:.2f} | {m['nrmse']:.4f} | "
                         f"{m['lpips']:.4f} | {m['err_low_freq']:.4f} | {m['err_high_freq']:.4f} | {m['z_jitter']:.4f} |")
        lines += ["", "Identity row = same subjects/fold as the runs (all runs in this table share the fold)."]
    text = "\n".join(lines) + "\n"
    (OUT_DIR / "RESULTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
