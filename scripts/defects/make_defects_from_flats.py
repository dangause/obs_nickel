#!/usr/bin/env python
"""
Build rectangular defect masks from cpFlat outputs, optionally ingest as a Butler
`defects` calib, and (optionally) save QA plots.

Designed for lsst-scipipe-9.x and obs_nickel (single detector).

Usage (most common):
  python make_defects_from_flats.py \
    --repo ~/Desktop/lick/lsst/data/nickel/062424 \
    --collection Nickel/run/cp_flat/20250730T135912Z \
    --register --ingest --defects-run Nickel/calib/defects/<TS> \
    --plot --qa-dir qa_defects
"""
from __future__ import annotations
import argparse, os, sys
from datetime import datetime
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # safe in headless
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.ndimage import gaussian_filter, label, find_objects, binary_opening

from lsst.daf.butler import Butler, DatasetType
from lsst.geom import Box2I, Point2I, Extent2I
try:
    from lsst.ip.isr import Defects          # newer stacks
except ImportError:
    from lsst.afw.image import Defects       # older stacks

# ------------------------------ helpers --------------------------------

def ts_utc() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def _query_flat_refs(b: Butler, instrument: str):
    """Return all flat refs for this instrument (no detector needed)."""
    return list(b.registry.queryDatasets(
        datasetType="flat",
        where=f"instrument='{instrument}'",
    ))

def median_flat(b: Butler, instrument: str) -> Tuple[np.ndarray, list]:
    refs = _query_flat_refs(b, instrument)
    if not refs:
        raise RuntimeError("No flats found in the given collection(s).")
    arrs = [b.get(r).image.array.astype(np.float32) for r in refs]
    return np.median(np.stack(arrs, axis=0), axis=0), refs

def detect_rectangles_from_flat(
    img: np.ndarray,
    sigma_pix: int = 7,
    ratio_hi: float = 1.10,
    ratio_lo: float = 0.90,
    min_area_px: int = 8,
    open_kernel: int = 2,
) -> List[Tuple[int,int,int,int]]:
    """Return rectangles (x0, y0, w, h) indicating likely sensor defects."""
    img = img.astype(np.float32)
    smooth = gaussian_filter(img, sigma=sigma_pix)
    smooth = np.maximum(smooth, 1e-6)
    ratio  = img / smooth

    mask = (ratio > ratio_hi) | (ratio < ratio_lo)
    if open_kernel > 0:
        mask = binary_opening(mask, structure=np.ones((open_kernel, open_kernel), dtype=bool))

    labels, _ = label(mask)
    rects: List[Tuple[int,int,int,int]] = []
    for sl in find_objects(labels):
        if sl is None:
            continue
        ys, xs = sl
        h = int(ys.stop - ys.start)
        w = int(xs.stop - xs.start)
        if w * h < min_area_px:
            continue
        rects.append((int(xs.start), int(ys.start), w, h))
    return rects

def rectangles_to_boxes(rects: Iterable[Tuple[int,int,int,int]]) -> List[Box2I]:
    boxes: List[Box2I] = []
    for x0,y0,w,h in rects:
        boxes.append(Box2I(Point2I(int(x0), int(y0)), Extent2I(int(w), int(h))))
    return boxes

def masked_fraction(shape: Tuple[int,int], rects: Iterable[Tuple[int,int,int,int]]) -> float:
    mask = np.zeros(shape, dtype=bool)
    for x0,y0,w,h in rects:
        mask[y0:y0+h, x0:x0+w] = True
    return float(mask.mean())

def ensure_defects_dataset_type(b):
    from lsst.daf.butler import DatasetType
    dims = b.registry.dimensions.conform({"instrument", "detector"})
    try:
        dt = b.registry.getDatasetType("defects")
        if not getattr(dt, "isCalibration", False):
            raise RuntimeError(
                "Existing dataset type 'defects' is not marked as calibration. "
                "Please create a fresh repo (or a new path) and re-run."
            )
    except Exception:
        b.registry.registerDatasetType(DatasetType("defects", dims, "Defects", isCalibration=True))


def save_overlay_png(img: np.ndarray, rects: List[Tuple[int,int,int,int]], title: str, out_png: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img, origin="lower")
    for x0,y0,w,h in rects:
        ax.add_patch(patches.Rectangle((x0, y0), w, h, fill=False, linewidth=0.8))
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

def save_mask_png(img: np.ndarray, rects: List[Tuple[int,int,int,int]], out_png: str):
    mask = np.zeros_like(img, dtype=bool)
    for x0,y0,w,h in rects:
        mask[y0:y0+h, x0:x0+w] = True
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(mask, origin="lower")
    ax.set_title("Defect mask (True=masked)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

# ------------------------------- CLI -----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Create and (optionally) ingest Nickel defects from flats.")
    ap.add_argument("--repo", required=True, help="Butler repo path")
    ap.add_argument("--collection", required=True, help="Collection(s) with flats (e.g. Nickel/run/cp_flat/...)")
    ap.add_argument("--instrument", default="Nickel")
    ap.add_argument("--sigma", type=int, default=7, help="Gaussian sigma (pixels) for smoothing")
    ap.add_argument("--ratio-hi", type=float, default=1.10, dest="ratio_hi", help="Upper ratio threshold")
    ap.add_argument("--ratio-lo", type=float, default=0.90, dest="ratio_lo", help="Lower ratio threshold")
    ap.add_argument("--min-area", type=int, default=8, dest="min_area", help="Minimum rectangle area (pixels)")
    ap.add_argument("--open", type=int, default=2, dest="open_kernel", help="Morphological opening kernel (square side)")
    ap.add_argument("--csv-out", default=None, help="Path to write rectangles CSV (default: nickel_defects_rects_<TS>.csv)")
    ap.add_argument("--ingest", action="store_true", help="Ingest defects to Butler after creating CSV")
    ap.add_argument("--register", action="store_true", help="Register dataset type 'defects' if missing")
    ap.add_argument("--defects-run", default=None, help="Run name for ingest (default: Nickel/calib/defects/<TS>)")
    ap.add_argument("--plot", action="store_true", help="Write PNG overlays/masks to --qa-dir")
    ap.add_argument("--qa-dir", default="qa_defects", help="Directory for QA PNGs (default: qa_defects)")
    args = ap.parse_args()

    repo = os.path.expanduser(args.repo)
    b_flat = Butler(repo, collections=args.collection)

    # Build median flat (no detector arg used)
    print(f"Building median flat from {args.collection} ...")
    med, refs = median_flat(b_flat, args.instrument)
    print(f"Using {len(refs)} flat(s)")

    # Detect rectangles
    rects = detect_rectangles_from_flat(
        med,
        sigma_pix=args.sigma,
        ratio_hi=args.ratio_hi,
        ratio_lo=args.ratio_lo,
        min_area_px=args.min_area,
        open_kernel=args.open_kernel,
    )
    frac = masked_fraction(med.shape, rects)
    print(f"Found {len(rects)} rectangles; masked fraction ≈ {frac:.3%}")

    # Save CSV (no detector column)
    ts = ts_utc()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(script_dir, f"nickel_defects_rects_{ts}.csv")
    csv_out = os.path.expanduser(args.csv_out) if args.csv_out else default_csv

    pd.DataFrame(
        [dict(x0=x0, y0=y0, width=w, height=h, label="auto-flat") for (x0,y0,w,h) in rects],
        columns=["x0","y0","width","height","label"],
    ).to_csv(csv_out, index=False)
    print(f"Wrote rectangles -> {csv_out}")


    # Optional QA
    if args.plot:
        os.makedirs(args.qa_dir, exist_ok=True)
        overlay_png = os.path.join(args.qa_dir, "overlay.png")
        mask_png    = os.path.join(args.qa_dir, "mask.png")
        save_overlay_png(med, rects, "Median flat + defects", overlay_png)
        save_mask_png(med, rects, mask_png)
        print(f"QA images: {overlay_png}, {mask_png}")

    # Optional ingest
    if args.ingest:
        defects_run = args.defects_run or f"Nickel/calib/defects/{ts}"
        print(f"Ingesting defects to run: {defects_run}")
        b_write = Butler(repo, run=defects_run)
        if args.register:
            ensure_defects_dataset_type(b_write)

        # Derive the (single) detector id from any flat ref (no CLI 'det' anywhere)
        try:
            det_id = int(refs[0].dataId["detector"])
        except Exception as e:
            raise RuntimeError("Could not determine detector ID from flats; "
                               "ensure flats exist in the given collection.") from e

        boxes = rectangles_to_boxes(rects)
        defects_obj = Defects(boxes)
        dataId = dict(instrument=args.instrument, detector=det_id)
        b_write.put(defects_obj, "defects", dataId=dataId)
        print("Done.")
        print("DEFECTS_RUN =", defects_run)

    return 0

if __name__ == "__main__":
    sys.exit(main())
