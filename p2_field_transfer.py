"""
p2_field_transfer.py — Track P2: measure how T1 and T2 change from 1.5T to 3T on the same brains.

Pairs (Cambridge): FSM_1_5 <-> AOUP_3_0 (10 subjects) and CNR_1_5 <-> CNR_3_0 (7 subjects), test and retest.
For each pair the 3T maps are rigidly registered to the 1.5T maps (SimpleITK, Mattes MI on brain-masked T1,
multi-resolution). Voxels inside both brain masks are binned by the 1.5T value and the bin medians are fitted
in log space:  log T_3T = a + b * log T_1.5T   (separately for T1 and T2), plus a non-parametric lookup table.

Leave-one-subject-out evaluation compares: identity, the original fixed rule T1 * 2^0.35, the power law and the LUT.
Writes outputs/field_transfer/field_transfer.json and fit plots.
"""

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT_DIR, T2_RANGE  # noqa: E402

PAIRS = [("FSM_1_5", "AOUP_3_0"), ("CNR_1_5", "CNR_3_0")]
BINS = {"t1": (math.log(0.3), math.log(4.5), 40), "t2": (math.log(0.02), math.log(0.45), 40)}
MIN_BIN_COUNT = 500
SUBSAMPLE = 300_000


def to_sitk(arr_xyz):
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr_xyz.transpose(2, 1, 0)).astype(np.float32))
    img.SetSpacing((1.0, 1.0, 1.0))
    return img


def from_sitk(img):
    return sitk.GetArrayFromImage(img).transpose(2, 1, 0)


def register_pair(job):
    mrf_dir, subj, s15, s3, sess = job
    a = np.load(mrf_dir / f"cam_{s15}_{subj}_{sess}_maps.npy").astype(np.float32)
    am = np.load(mrf_dir / f"cam_{s15}_{subj}_{sess}_masks.npy")
    b = np.load(mrf_dir / f"cam_{s3}_{subj}_{sess}_maps.npy").astype(np.float32)
    bm = np.load(mrf_dir / f"cam_{s3}_{subj}_{sess}_masks.npy")

    # Stage 1: align the shapes of the smoothed brain masks (robust global alignment, moments initialization).
    shape_a = to_sitk(ndi.gaussian_filter(am[1].astype(np.float32), 2))
    shape_b = to_sitk(ndi.gaussian_filter(bm[1].astype(np.float32), 2))
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.1, seed=1)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-3, numberOfIterations=400, relaxationFactor=0.6)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2])
    R.SetSmoothingSigmasPerLevel([2.0, 1.0])
    R.SetInitialTransform(sitk.CenteredTransformInitializer(
        shape_a, shape_b, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS), inPlace=False)
    tx1 = R.Execute(shape_a, shape_b)

    # Stage 2: mutual-information refinement on brain-masked T2 (strong GM/WM/CSF contrast at both fields).
    fixed = to_sitk(a[1] * am[1])
    moving = to_sitk(b[1] * bm[1])
    R2 = sitk.ImageRegistrationMethod()
    R2.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    R2.SetMetricSamplingStrategy(R2.RANDOM)
    R2.SetMetricSamplingPercentage(0.3, seed=2)
    R2.SetInterpolator(sitk.sitkLinear)
    R2.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=500, relaxationFactor=0.6)
    R2.SetOptimizerScalesFromPhysicalShift()
    R2.SetShrinkFactorsPerLevel([2, 1])
    R2.SetSmoothingSigmasPerLevel([1.0, 0.0])
    R2.SetInitialTransform(sitk.Euler3DTransform(tx1.GetNthTransform(0)), inPlace=False)
    tx = R2.Execute(fixed, moving)
    R = R2

    def warp(arr, interp):
        return from_sitk(sitk.Resample(to_sitk(arr), fixed, tx, interp, 0.0, sitk.sitkFloat32))

    lt1_b = warp(np.log(b[0]), sitk.sitkLinear)
    lt2_b = warp(np.log(b[1]), sitk.sitkLinear)
    bmask_w = warp(bm[1].astype(np.float32), sitk.sitkNearestNeighbor) > 0.5

    # Erode the joint mask to reduce partial-volume and residual misregistration at tissue edges.
    both = ndi.binary_erosion((am[1] > 0) & bmask_w, iterations=1)
    x1, y1 = np.log(a[0])[both], lt1_b[both]
    x2, y2 = np.log(a[1])[both], lt2_b[both]
    r_after = float(np.corrcoef(x1, y1)[0, 1])
    rng = np.random.default_rng(0)
    sel = rng.choice(x1.size, size=min(SUBSAMPLE, x1.size), replace=False)
    return {
        "subject": subj, "pair": f"{s15}->{s3}", "session": sess,
        "metric": float(R.GetMetricValue()), "r_logT1_after": r_after, "n_vox": int(both.sum()),
        "t1": np.stack([x1[sel], y1[sel]]).astype(np.float32),
        "t2": np.stack([x2[sel], y2[sel]]).astype(np.float32),
    }


def bin_medians(x, y, which):
    lo, hi, n = BINS[which]
    edges = np.linspace(lo, hi, n + 1)
    idx = np.digitize(x, edges) - 1
    cx, my, cnt = [], [], []
    for i in range(n):
        sel = idx == i
        if sel.sum() >= MIN_BIN_COUNT:
            cx.append(0.5 * (edges[i] + edges[i + 1]))
            my.append(float(np.median(y[sel])))
            cnt.append(int(sel.sum()))
    return np.array(cx), np.array(my), np.array(cnt)


def valid(xy, which):
    x, y = xy
    if which == "t2":  # drop dictionary-saturated values
        cap = math.log(T2_RANGE[1]) - 0.02
        keep = (x < cap) & (y < cap)
        return x[keep], y[keep]
    return x, y


def fit(records, which):
    x = np.concatenate([valid(r[which], which)[0] for r in records])
    y = np.concatenate([valid(r[which], which)[1] for r in records])
    cx, my, cnt = bin_medians(x, y, which)
    w = np.sqrt(cnt)
    A = np.stack([np.ones_like(cx), cx], 1) * w[:, None]
    a, b = np.linalg.lstsq(A, my * w, rcond=None)[0]
    return {"a": float(a), "b": float(b), "lut_x": cx.tolist(), "lut_y": my.tolist(), "lut_count": cnt.tolist()}


def predict(model, x, which, method):
    if method == "identity":
        return x
    if method == "fixed_0.35":
        return x + (0.35 * math.log(2.0) if which == "t1" else 0.0)
    if method == "power_law":
        return model["a"] + model["b"] * x
    if method == "lut":
        return np.interp(x, model["lut_x"], model["lut_y"])
    raise ValueError(method)


def evaluate(model, records, which):
    out = {}
    for method in ("identity", "fixed_0.35", "power_law", "lut"):
        errs_s, errs_log = [], []
        for r in records:
            x, y = valid(r[which], which)
            p = predict(model, x, which, method)
            errs_s.append(np.abs(np.exp(p) - np.exp(y)))
            errs_log.append(np.abs(p - y))
        out[method] = {"mae_s": float(np.mean(np.concatenate(errs_s))),
                       "median_abs_log_err": float(np.median(np.concatenate(errs_log)))}
    return out


def plot(records, models, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for a, which, unit in [(ax[0], "t1", "s"), (ax[1], "t2", "s")]:
        x = np.concatenate([valid(r[which], which)[0] for r in records])
        y = np.concatenate([valid(r[which], which)[1] for r in records])
        lo, hi, _ = BINS[which]
        a.hist2d(x, y, bins=200, range=[[lo, hi], [lo, hi]], cmap="Greys", norm=matplotlib.colors.LogNorm())
        xs = np.linspace(lo, hi, 50)
        m = models[which]
        a.plot(xs, xs, "b--", label="identity")
        if which == "t1":
            a.plot(xs, xs + 0.35 * math.log(2), "g--", label="fixed 2^0.35")
        a.plot(xs, m["a"] + m["b"] * xs, "r-", label=f"fit a={m['a']:.3f} b={m['b']:.3f}")
        a.plot(m["lut_x"], m["lut_y"], "o", color="orange", ms=3, label="bin medians (LUT)")
        a.set_xlabel(f"log {which.upper()} at 1.5T (log {unit})")
        a.set_ylabel(f"log {which.upper()} at 3T")
        a.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrf_dir", default=str(OUT_DIR / "mrf"))
    ap.add_argument("--out_dir", default=str(OUT_DIR / "field_transfer"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--min_reg_r", type=float, default=0.85, help="reject pairs whose post-registration r(logT1) is lower")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    mrf_dir, out_dir = Path(args.mrf_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for s15, s3 in PAIRS:
        for f in sorted(mrf_dir.glob(f"cam_{s15}_MRF3D*_maps.npy")):
            subj, sess = f.name.split("_")[-3], f.name.split("_")[-2]
            if (mrf_dir / f"cam_{s3}_{subj}_{sess}_maps.npy").exists():
                jobs.append((mrf_dir, subj, s15, s3, sess))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"registering {len(jobs)} cross-field pairs", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        records = list(ex.map(register_pair, jobs))
    for r in records:
        print(f"  {r['subject']} {r['pair']} {r['session']}: r(logT1) after reg = {r['r_logT1_after']:.3f}  n={r['n_vox']}", flush=True)
    good = [r for r in records if r["r_logT1_after"] >= args.min_reg_r]
    print(f"kept {len(good)}/{len(records)} pairs (r >= {args.min_reg_r})", flush=True)

    # Leave-one-subject-out evaluation
    subjects = sorted({r["subject"] for r in good})
    loso = {"t1": [], "t2": []}
    for s in subjects:
        tr = [r for r in good if r["subject"] != s]
        te = [r for r in good if r["subject"] == s]
        for which in ("t1", "t2"):
            loso[which].append({"subject": s, **evaluate(fit(tr, which), te, which)})
    loso_mean = {}
    for which in ("t1", "t2"):
        loso_mean[which] = {m: {k: float(np.mean([row[m][k] for row in loso[which]])) for k in ("mae_s", "median_abs_log_err")}
                            for m in ("identity", "fixed_0.35", "power_law", "lut")}

    models = {w: fit(good, w) for w in ("t1", "t2")}
    eff_exp = {w: None for w in ("t1", "t2")}
    for w in ("t1", "t2"):  # effective exponent gamma at the median 1.5T value: T3 = T1.5 * 2^gamma
        xm = float(np.median(np.concatenate([valid(r[w], w)[0] for r in good])))
        eff_exp[w] = ((models[w]["a"] + models[w]["b"] * xm) - xm) / math.log(2.0)
    result = {
        "fit": {w: {"a": models[w]["a"], "b": models[w]["b"]} for w in ("t1", "t2")},
        "lut": {w: {k: models[w][k] for k in ("lut_x", "lut_y", "lut_count")} for w in ("t1", "t2")},
        "effective_field_exponent_at_median": eff_exp,
        "loso_mean": loso_mean, "loso_per_subject": loso,
        "registration": [{k: v for k, v in r.items() if k not in ("t1", "t2")} for r in records],
        "n_pairs_used": len(good), "subjects": subjects,
    }
    json.dump(result, open(out_dir / "field_transfer.json", "w"), indent=1)
    plot(good, models, out_dir / "field_transfer_fit.png")

    print(json.dumps({"fit": result["fit"], "effective_exponent": eff_exp}, indent=1))
    for w in ("t1", "t2"):
        print(f"LOSO {w.upper()}:")
        for m, v in loso_mean[w].items():
            print(f"   {m:<11} MAE {v['mae_s'] * 1000:8.1f} ms   median|log err| {v['median_abs_log_err']:.4f}")
    print("P2 DONE", flush=True)


if __name__ == "__main__":
    main()
