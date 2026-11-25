"""
visualize_physics.py — what the frozen ParamNet produced on MRIxFields (outputs/physics/<MOD>/), and whether it is
physically plausible across field strengths.

outputs/physics/<MOD>/vis/subject_<sid>.png   rows = 5 fields; image, T1, T2, M0, uncertainty (axial z=165)
outputs/physics/<MOD>/vis/field_medians.png   brain-median T1 / T2 per field (one line per subject) and
                                              3T-measured vs 1.5T-transferred T1 (Track P2 fit)
outputs/physics/<MOD>/vis/physics_check.json/.md   numbers + PASS/FAIL: T1 must rise from 1.5T to 3T
ParamNet was trained on 1.5T / 3T only (field input clamped to [1.5, 3] T), so 0.1T / 5T / 7T maps are extrapolations.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gen"))
from common import FIELD_TESLA, FIELDS, OUT_DIR, FieldTransfer  # noqa: E402
from data import CACHE_DIR, PHYS_DIR, VolumeBank  # noqa: E402

Z_SHOW = 165
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "legend.frameon": False,
})


def show(ax, img, title, **kw):
    im = ax.imshow(np.rot90(img), **kw)
    ax.set_title(title, fontsize=8, color=INK2)
    ax.axis("off")
    return im


def subject_figure(bank, sid, path):
    fig, ax = plt.subplots(len(FIELDS), 5, figsize=(16, 3.3 * len(FIELDS)))
    ims = {}
    for r, f in enumerate(FIELDS):
        img = np.asarray(bank.vol(sid, f)[:, :, Z_SHOW], np.float32)
        p = np.asarray(bank.phys_vol(sid, f)[:, :, :, Z_SHOW], np.float32)
        br = img > 0.005
        t1 = np.where(br, np.exp(p[0]) * 1000, np.nan)
        t2 = np.where(br, np.exp(p[1]) * 1000, np.nan)
        m0 = np.where(br, p[2], np.nan)
        un = np.where(br, np.exp(p[3]), np.nan)
        ext = "" if f in ("1.5T", "3T") else "  (extrapolated)"
        show(ax[r, 0], img, f"{f} T2W input{ext}", cmap="gray", vmin=0, vmax=1)
        ims["t1"] = show(ax[r, 1], t1, f"T1 (ms) — median {np.nanmedian(t1):.0f}", cmap="viridis", vmin=300, vmax=3000)
        ims["t2"] = show(ax[r, 2], t2, f"T2 (ms) — median {np.nanmedian(t2):.0f}", cmap="viridis", vmin=40, vmax=250)
        ims["m0"] = show(ax[r, 3], m0, "M0 (a.u.)", cmap="gray", vmin=0, vmax=1.3)
        ims["un"] = show(ax[r, 4], un, "uncertainty (Laplace scale)", cmap="magma", vmin=0, vmax=0.3)
    for c, k in zip(range(1, 5), ("t1", "t2", "m0", "un")):
        fig.colorbar(ims[k], ax=ax[:, c], shrink=0.3, pad=0.01)
    fig.suptitle(f"ParamNet maps on MRIxFields subject {sid}, axial z={Z_SHOW}")
    fig.savefig(path, dpi=60, bbox_inches="tight")
    plt.close(fig)


def brain_stats(bank, sid, f):
    img = np.asarray(bank.vol(sid, f)[::2, ::2, ::2], np.float32)
    p = np.asarray(bank.phys_vol(sid, f)[:, ::2, ::2, ::2], np.float32)
    br = img > 0.005
    lt1, lt2 = p[0][br], p[1][br]
    return {"t1_med_ms": float(np.exp(np.median(lt1)) * 1000), "t2_med_ms": float(np.exp(np.median(lt2)) * 1000),
            "t1_p25_ms": float(np.exp(np.percentile(lt1, 25)) * 1000), "t1_p75_ms": float(np.exp(np.percentile(lt1, 75)) * 1000),
            "unc_med": float(np.exp(np.median(p[3][br])))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="T2W")
    args = ap.parse_args()
    if not (PHYS_DIR / args.modality / "DONE").exists():
        sys.exit("physics maps not computed yet (outputs/physics/<MOD>/DONE missing)")
    out = PHYS_DIR / args.modality / "vis"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(CACHE_DIR / args.modality / "meta.json"))
    bank = VolumeBank(args.modality, phys=True)
    ft = FieldTransfer.from_json(OUT_DIR / "field_transfer" / "field_transfer.json")
    stats = {}
    for sid in meta["subjects"]:
        subject_figure(bank, sid, out / f"subject_{sid}.png")
        stats[sid] = {f: brain_stats(bank, sid, f) for f in FIELDS}
        print(f"{sid}: " + " ".join(f"{f} T1 {stats[sid][f]['t1_med_ms']:.0f}" for f in FIELDS), flush=True)

    # field-transfer prediction of the 3T median from the 1.5T median
    def transfer(t1_ms):
        lt = torch.tensor([[np.log(t1_ms / 1000.0)]])
        return float(torch.exp(ft.apply_log(lt, "t1", torch.tensor([1.5]), torch.tensor([3.0])))[0, 0]) * 1000

    sids = meta["subjects"]
    t1_15 = np.array([stats[s]["1.5T"]["t1_med_ms"] for s in sids])
    t1_3 = np.array([stats[s]["3T"]["t1_med_ms"] for s in sids])
    t1_ft = np.array([transfer(v) for v in t1_15])
    ratio = t1_3 / t1_15
    rises = int((ratio > 1.0).sum())
    expected = float(np.median(t1_ft / t1_15))
    check = {
        "n_subjects": len(sids), "t1_rises_1p5_to_3T": rises, "median_ratio_t1_3T_over_1p5T": float(np.median(ratio)),
        "expected_ratio_from_P2_fit": expected,
        "median_abs_err_3T_vs_transfer_ms": float(np.median(np.abs(t1_3 - t1_ft))),
        "pass": bool(rises >= 0.8 * len(sids) and 1.05 <= np.median(ratio) <= 1.7), "per_subject": stats,
    }
    json.dump(check, open(out / "physics_check.json", "w"), indent=1)

    x = np.arange(len(FIELDS))
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for k, a, lab in (("t1_med_ms", ax[0], "brain median T1 (ms)"), ("t2_med_ms", ax[1], "brain median T2 (ms)")):
        m = np.array([[stats[s][f][k] for f in FIELDS] for s in sids])
        for row in m:
            a.plot(x, row, color=MUTED, lw=0.6, alpha=0.6)
        a.plot(x, np.median(m, 0), "o-", color=SERIES[0], lw=2, label="median over subjects")
        a.axvspan(-0.4, 0.5, color="#e6e5e1", lw=0)
        a.axvspan(2.5, 4.4, color="#e6e5e1", lw=0, label="outside ParamNet training fields")
        a.set_xticks(x)
        a.set_xticklabels(FIELDS)
        a.set_xlim(-0.4, 4.4)
        a.set_ylabel(lab)
        a.legend(fontsize=7)
    ax[0].plot([1, 2], [np.median(t1_15), np.median(t1_ft)], "s--", color=SERIES[1], ms=7, label="1.5T → 3T by P2 fit")
    ax[0].legend(fontsize=7)
    ax[0].set_title("T1 should rise with field strength", fontsize=9)
    ax[1].set_title("T2 should fall slightly with field strength", fontsize=9)
    lo, hi = min(t1_3.min(), t1_ft.min()) * 0.9, max(t1_3.max(), t1_ft.max()) * 1.1
    ax[2].plot([lo, hi], [lo, hi], "--", color=MUTED, lw=1, label="y = x")
    ax[2].plot(t1_ft, t1_3, "o", color=SERIES[0], ms=6, label="subjects")
    ax[2].set_xlabel("T1 median predicted from the 1.5T scan + field transfer (ms)")
    ax[2].set_ylabel("T1 median predicted from the 3T scan (ms)")
    ax[2].set_title("ParamNet consistency across fields", fontsize=9)
    ax[2].legend(fontsize=7)
    fig.suptitle(f"ParamNet on MRIxFields {args.modality}: T1 rises 1.5T→3T in {rises}/{len(sids)} subjects, "
                 f"median ratio {np.median(ratio):.2f} (P2 fit expects {expected:.2f}) — "
                 f"{'PASS' if check['pass'] else 'FAIL'}")
    fig.savefig(out / "field_medians.png", dpi=90, bbox_inches="tight")
    plt.close(fig)

    md = [f"# ParamNet physics check on MRIxFields {args.modality}", "",
          f"* T1 rises from 1.5T to 3T in **{rises}/{len(sids)}** subjects; median ratio "
          f"**{np.median(ratio):.2f}** (P2 field-transfer fit expects {expected:.2f}).",
          f"* Median |T1(3T scan) − transfer(T1(1.5T scan))| = {check['median_abs_err_3T_vs_transfer_ms']:.0f} ms.",
          f"* Verdict: **{'PASS' if check['pass'] else 'FAIL'}** (needs ≥80% rising and median ratio in [1.05, 1.7]).", "",
          "| subject | " + " | ".join(f"T1 {f}" for f in FIELDS) + " | " + " | ".join(f"T2 {f}" for f in FIELDS) + " |",
          "|---|" + "---|" * (2 * len(FIELDS))]
    for s in sids:
        md.append(f"| {s} | " + " | ".join(f"{stats[s][f]['t1_med_ms']:.0f}" for f in FIELDS) + " | "
                  + " | ".join(f"{stats[s][f]['t2_med_ms']:.0f}" for f in FIELDS) + " |")
    md += ["", "ms, brain medians. 0.1T / 5T / 7T are outside the ParamNet training range (extrapolated)."]
    (out / "physics_check.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:5]), flush=True)
    print(f"PHYSICS VIS DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
