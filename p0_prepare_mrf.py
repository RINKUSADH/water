"""
p0_prepare_mrf.py — Track P0: convert Cambridge multi-site MRF and Dryad SAFE 3D-MRF into one format.

For every scan writes outputs/mrf/<scan_id>_maps.npy  float16 (4, X, Y, Z) = [T1 (s), T2 (s), M0 (rel), B1+ (rel)]
                     outputs/mrf/<scan_id>_masks.npy uint8   (2, X, Y, Z) = [head, brain]
on a 1 mm isotropic grid, plus outputs/mrf/index.json, outputs/mrf/split.json and QC PNGs in outputs/mrf/qc/.

Masks are heuristic (no skull-stripping tool is installed): head = Otsu on smoothed M0 + largest component + hole
filling; brain = head voxels above a fat-T1 threshold, eroded ~7 mm, largest component, dilated back. Check the QC.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.io import loadmat
from skimage.filters import threshold_otsu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT_DIR, T1_RANGE, T2_RANGE  # noqa: E402

CAM_ROOT = Path("/DATA1/obed/Rinku/l2h/MRF_Buonincontri/extracted/4Pub")
DRY_ROOT = Path("/DATA1/obed/Rinku/l2h/MRF_SAFE_Dryad")
CAM_SITES = {"FSM_1_5": 1.5, "CNR_1_5": 1.5, "AOUP_3_0": 3.0, "CNR_3_0": 3.0}  # mag_* folders excluded (undocumented)

# Subject-level split (every scan of a subject goes to the same split).
CAM_SPLIT = {"test": ["MRF3D009", "MRF3D010"], "val": ["MRF3D008"]}
DRY_SPLIT = {"test": ["10", "20"], "val": ["9", "19"]}  # one Siemens + one GE subject each


def largest_cc(m):
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def fill_holes(m):
    m = ndi.binary_fill_holes(m)
    for z in range(m.shape[2]):  # axial slice-wise filling closes ventricles open to the border
        m[:, :, z] = ndi.binary_fill_holes(m[:, :, z])
    return m


def make_masks(t1, m0, field):
    sm = ndi.gaussian_filter(m0, 1.0)
    thr = threshold_otsu(sm[np.isfinite(sm)])
    head = largest_cc(sm > 0.6 * thr)
    head = ndi.binary_closing(head, iterations=3)
    head = fill_holes(largest_cc(head))
    fat_thr = 0.40 if field < 2.0 else 0.50  # scalp fat has T1 below this; WM is above
    # Skull (cortical bone) has almost no proton density: the dark M0 ring separates brain from scalp/muscle.
    m0_rel = m0 / max(np.percentile(m0[head], 90), 1e-8)
    cand = head & (t1 > fat_thr) & (m0_rel > 0.30)
    cand = ndi.binary_opening(cand, iterations=2)
    core = largest_cc(ndi.binary_erosion(cand, iterations=7))
    # Geodesic dilation: regrow only through candidate tissue, so the mask cannot jump the skull gap.
    brain = ndi.binary_dilation(core, iterations=8, mask=cand)
    brain = fill_holes(largest_cc(brain))
    return head.astype(np.uint8), brain.astype(np.uint8)


def clean_maps(t1, t2, m0, head):
    t1 = np.nan_to_num(t1, nan=0.0).clip(*T1_RANGE)
    t2 = np.nan_to_num(t2, nan=0.0).clip(*T2_RANGE)
    m0 = np.nan_to_num(np.abs(m0), nan=0.0)
    p99 = np.percentile(m0[head > 0], 99.0) if head.any() else m0.max()
    m0 = (m0 / max(p99, 1e-8)).clip(0.0, 1.5)
    return t1, t2, m0


def qc_png(maps, masks, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t1, t2, m0 = maps[0], maps[1], maps[2]
    fig, ax = plt.subplots(3, 4, figsize=(14, 10))
    for r, axis in enumerate([2, 1, 0]):
        idx = t1.shape[axis] // 2
        sl = lambda v: np.rot90(np.take(v, idx, axis=axis))  # noqa: E731
        ax[r, 0].imshow(sl(t1), cmap="magma", vmin=0, vmax=3.0)
        ax[r, 1].imshow(sl(t2), cmap="viridis", vmin=0, vmax=0.2)
        ax[r, 2].imshow(sl(m0), cmap="gray", vmin=0, vmax=1.2)
        ax[r, 3].imshow(sl(m0), cmap="gray", vmin=0, vmax=1.2)
        ax[r, 3].contour(sl(masks[0]), levels=[0.5], colors="y", linewidths=0.7)
        ax[r, 3].contour(sl(masks[1]), levels=[0.5], colors="r", linewidths=0.7)
        for c, name in enumerate(["T1 (s)", "T2 (s)", "M0", "head(y)/brain(r)"]):
            ax[r, c].set_title(f"{name} axis{axis}", fontsize=8)
            ax[r, c].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=60)
    plt.close(fig)


def save_scan(scan_id, t1, t2, m0, b1, field, out_dir):
    head0 = (m0 > 0).astype(np.uint8)
    t1, t2, m0_n = clean_maps(t1, t2, m0, np.ones_like(head0))
    head, brain = make_masks(t1, m0_n, field)
    t1, t2, m0_n = clean_maps(t1, t2, m0, head)  # renormalize M0 within the head mask
    maps = np.stack([t1, t2, m0_n, b1]).astype(np.float16)
    masks = np.stack([head, brain]).astype(np.uint8)
    np.save(out_dir / f"{scan_id}_maps.npy", maps)
    np.save(out_dir / f"{scan_id}_masks.npy", masks)
    qc_png(maps.astype(np.float32), masks, f"{scan_id}  ({field}T)", out_dir / "qc" / f"{scan_id}.png")
    br = brain > 0
    return {
        "shape": list(t1.shape),
        "brain_voxels": int(br.sum()),
        "head_voxels": int(head.sum()),
        "brain_T1_median_s": float(np.median(t1[br])) if br.any() else None,
        "brain_T2_median_s": float(np.median(t2[br])) if br.any() else None,
    }


def process_cambridge(job):
    site, subj, sess, out_dir = job
    path = CAM_ROOT / site / subj / f"{subj}_{sess}.nii"
    img = nib.as_closest_canonical(nib.load(str(path)))
    vol = img.get_fdata(dtype=np.float32)  # (X, Y, Z, 3): T1 s, T2 s, M0
    zoom = float(img.header.get_zooms()[0]) / 1.0  # 1.12 mm -> 1 mm
    t1, t2, m0 = (ndi.zoom(vol[..., c], zoom, order=1) for c in range(3))
    scan_id = f"cam_{site}_{subj}_{sess}"
    info = save_scan(scan_id, t1, t2, m0, np.ones_like(t1), CAM_SITES[site], out_dir)
    info.update(id=scan_id, dataset="cambridge", site=site, subject=subj, session=sess,
                field=CAM_SITES[site], vendor="cambridge_single_vendor", has_b1=False)
    return info


def process_dryad(job):
    num, out_dir = job
    d = loadmat(str(DRY_ROOT / f"subject_{num}.mat"), variable_names=["T1_map", "T2_map", "PD_map", "B1_map"])
    t1 = np.abs(d["T1_map"]).astype(np.float32) / 1000.0  # ms -> s
    t2 = np.abs(d["T2_map"]).astype(np.float32) / 1000.0
    m0 = np.abs(d["PD_map"]).astype(np.float32)
    b1 = np.nan_to_num(np.abs(d["B1_map"]).astype(np.float32), nan=1.0).clip(0.3, 2.0)
    scan_id = f"dry_{num}"
    info = save_scan(scan_id, t1, t2, m0, b1, 3.0, out_dir)
    info.update(id=scan_id, dataset="dryad", site="dryad", subject=str(num), session="single",
                field=3.0, vendor="siemens_vida" if int(num) <= 10 else "ge_premier", has_b1=True)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(OUT_DIR / "mrf"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="debug: process only N scans per dataset")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "qc").mkdir(parents=True, exist_ok=True)

    cam_jobs = []
    for site in CAM_SITES:
        for sdir in sorted((CAM_ROOT / site).glob("MRF3D*")):
            for sess in ("test", "retest"):
                if (sdir / f"{sdir.name}_{sess}.nii").exists():
                    cam_jobs.append((site, sdir.name, sess, out_dir))
    dry_jobs = [(n, out_dir) for n in range(1, 21) if (DRY_ROOT / f"subject_{n}.mat").exists()]
    if args.limit:
        cam_jobs, dry_jobs = cam_jobs[: args.limit], dry_jobs[: args.limit]
    print(f"Cambridge scans: {len(cam_jobs)} | Dryad scans: {len(dry_jobs)}", flush=True)

    index = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for info in ex.map(process_cambridge, cam_jobs):
            index.append(info)
            print(f"  [ok] {info['id']}  brain T1 med {info['brain_T1_median_s']:.3f}s  T2 med {info['brain_T2_median_s']:.3f}s", flush=True)
        for info in ex.map(process_dryad, dry_jobs):
            index.append(info)
            print(f"  [ok] {info['id']}  brain T1 med {info['brain_T1_median_s']:.3f}s  T2 med {info['brain_T2_median_s']:.3f}s", flush=True)

    split = {"train": [], "val": [], "test": []}
    for info in index:
        rule = CAM_SPLIT if info["dataset"] == "cambridge" else DRY_SPLIT
        part = "test" if info["subject"] in rule["test"] else "val" if info["subject"] in rule["val"] else "train"
        info["split"] = part
        split[part].append(info["id"])
    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f, indent=1)
    with open(out_dir / "split.json", "w") as f:
        json.dump(split, f, indent=1)
    print({k: len(v) for k, v in split.items()}, flush=True)
    print("P0 DONE", flush=True)


if __name__ == "__main__":
    main()
