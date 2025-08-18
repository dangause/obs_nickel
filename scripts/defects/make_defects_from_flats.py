#!/usr/bin/env python3
"""
nickel_defects.py — derive rectangular defect masks from Nickel flats (or darks),
visualize them, and ingest into Butler as a `defects` calib.

Works with lsst-scipipe-9.x. Uses only matplotlib for plots.

USAGE (CLI examples):
  # Build defects from cpFlat run, save CSV, and ingest into a new calib run
  python nickel_defects.py \
    --repo /Users/you/Desktop/lick/lsst/data/nickel/062424 \
    --collections Nickel/run/cp_flat/20250730T135912Z \
    --detectors 0 \
    --save-csv nickel_defects_rects.csv \
    --register --ingest --verbose

  # Visualize overlay only (from CSV, no ingest)
  python nickel_defects.py \
    --repo /Users/you/Desktop/lick/lsst/data/nickel/062424 \
    --collections Nickel/run/cp_flat/20250730T135912Z \
    --detectors 0 \
    --csv nickel_defects_rects.csv \
    --show

In notebooks, `from nickel_defects import *` and call helpers directly.
"""
from __future__ import annotations

import os
import sys
import math
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter, label, find_objects, binary_opening

from lsst.daf.butler import Butler, DatasetType
from lsst.geom import Box2I, Point2I, Extent2I
try:
    from lsst.ip.isr import Defects  # modern stacks
except Exception:  # pragma: no cover
    from lsst.afw.image import Defects  # older stacks

# ----------------------------- Utilities -----------------------------------

def now_ts_utc() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

@dataclass
class DetectConfig:
    sigma_pix: int = 7
    ratio_hi: float = 1.10
    ratio_lo: float = 0.90
    min_area_px: int = 8
    open_struct_xy: Tuple[int, int] = (2, 2)
    merge_columns: bool = False
    x_tol: int = 1       # for merging
    gap_tol: int = 4     # for merging

# ----------------------------- Loading -------------------------------------

def list_flat_refs(b: Butler, det: int) -> list:
    return list(b.registry.queryDatasets(
        datasetType="flat",
        where="instrument='Nickel' AND detector=@det",
        bind={"det": det},
    ))


def load_all_flats_median(b: Butler, det: int) -> Tuple[np.ndarray, list]:
    refs = list_flat_refs(b, det)
    if not refs:
        raise RuntimeError("No flats found. Check --collections and detector.")
    arrs = [b.get(r).image.array.astype(np.float32) for r in refs]
    return np.median(np.stack(arrs, axis=0), axis=0), refs

# ----------------------------- Detection -----------------------------------

def detect_rectangles_from_flat(img: np.ndarray, cfg: DetectConfig) -> List[Tuple[int,int,int,int]]:
    img = img.astype(np.float32)
    smooth = gaussian_filter(img, sigma=cfg.sigma_pix)
    smooth = np.maximum(smooth, 1e-6)
    ratio = img / smooth

    mask = (ratio > cfg.ratio_hi) | (ratio < cfg.ratio_lo)
    mask = binary_opening(mask, structure=np.ones(cfg.open_struct_xy, dtype=bool))

    labels, _ = label(mask)
    rects: List[Tuple[int,int,int,int]] = []
    for sl in find_objects(labels):
        if sl is None: continue
        ysl, xsl = sl
        h = int(ysl.stop - ysl.start)
        w = int(xsl.stop - xsl.start)
        if w * h < cfg.min_area_px:
            continue
        rects.append((int(xsl.start), int(ysl.start), w, h))
    if cfg.merge_columns and rects:
        rects = merge_vertical_segments(rects, x_tol=cfg.x_tol, gap_tol=cfg.gap_tol)
    return rects


def merge_vertical_segments(rects: Sequence[Tuple[int,int,int,int]], x_tol: int = 1, gap_tol: int = 4) -> List[Tuple[int,int,int,int]]:
    """Merge vertically adjacent rectangles that share nearly the same x-span.
    rect = (x0, y0, w, h)
    """
    rects = sorted(rects, key=lambda r: (r[0], r[1]))
    merged: List[Tuple[int,int,int,int]] = []
    for x0, y0, w, h in rects:
        if merged:
            px, py, pw, ph = merged[-1]
            same_left  = abs(x0 - px) <= x_tol
            same_right = abs((x0 + w) - (px + pw)) <= x_tol
            touching   = y0 <= py + ph + gap_tol
            if same_left and same_right and touching:
                # extend previous
                new_y0 = min(py, y0)
                new_h  = max(py + ph, y0 + h) - new_y0
                merged[-1] = (px, new_y0, pw, new_h)
                continue
        merged.append((x0, y0, w, h))
    return merged

# ----------------------------- Converters ----------------------------------

def rects_to_boxes(rects: Iterable[Tuple[int,int,int,int]]) -> List[Box2I]:
    out = []
    for x0, y0, w, h in rects:
        out.append(Box2I(Point2I(int(x0), int(y0)), Extent2I(int(w), int(h))))
    return out


def defects_to_boxes(defs_obj) -> List[Box2I]:
    """Normalize various Defects flavors to a list of Box2I."""
    if hasattr(defs_obj, "getBBoxList"):
        return list(defs_obj.getBBoxList())
    boxes = []
    try:
        for d in defs_obj:
            if hasattr(d, "getBBox"):
                boxes.append(d.getBBox())
            elif hasattr(d, "bbox"):
                boxes.append(d.bbox)
    except TypeError:
        pass
    if not boxes and hasattr(defs_obj, "getDefects"):
        for d in defs_obj.getDefects():
            if hasattr(d, "getBBox"):
                boxes.append(d.getBBox())
            elif hasattr(d, "bbox"):
                boxes.append(d.bbox)
    if not boxes:
        raise RuntimeError("Could not extract bboxes from Defects object.")
    return boxes

# ----------------------------- Butler ops ----------------------------------

def ensure_defects_dataset_type(b: Butler, verbose: bool = True) -> None:
    dims = b.registry.dimensions.conform({"instrument", "detector"})
    try:
        b.registry.getDatasetType("defects")
        if verbose:
            print("Dataset type 'defects' already exists.")
    except Exception:
        dt = DatasetType("defects", dims, "Defects")
        b.registry.registerDatasetType(dt)
        if verbose:
            print("Registered dataset type 'defects'.")


def ingest_defects(b_repo_path: str, run: str, instrument: str, detector: int, boxes: List[Box2I]) -> None:
    b_write = Butler(b_repo_path, run=run)
    ensure_defects_dataset_type(Butler(b_repo_path, writeable=True), verbose=False)
    defects_obj = Defects(boxes)
    b_write.put(defects_obj, "defects", dataId=dict(instrument=instrument, detector=int(detector)))

# ----------------------------- Plotting ------------------------------------

def plot_overlay(image: np.ndarray, boxes: Sequence[Box2I], title: str = "Flat + defects") -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image, origin="lower")
    for box in boxes:
        rect = patches.Rectangle((box.getMinX(), box.getMinY()), box.getWidth(), box.getHeight(),
                                 fill=False, linewidth=0.9)
        ax.add_patch(rect)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    plt.show()

# ----------------------------- CLI -----------------------------------------

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build and ingest Nickel defects from flats.")
    p.add_argument("--repo", required=True, help="Butler repo path")
    p.add_argument("--collections", required=True, help="Comma-separated collections to read flats from (e.g. Nickel/run/cp_flat/...) ")
    p.add_argument("--detectors", required=True, nargs="+", type=int, help="Detector id(s), e.g. 0")
    p.add_argument("--save-csv", dest="save_csv", default=None, help="Write rectangles CSV here")
    p.add_argument("--csv", dest="csv", default=None, help="Load rectangles from CSV instead of detecting")
    p.add_argument("--defects-run", dest="defects_run", default=None, help="Calib run to write, default Nickel/calib/defects/<UTC>")
    p.add_argument("--register", action="store_true", help="Ensure dataset type 'defects' exists")
    p.add_argument("--ingest", action="store_true", help="Ingest defects into Butler")
    p.add_argument("--show", action="store_true", help="Show overlay for first detector")
    p.add_argument("--merge", action="store_true", help="Merge vertical segments of columns")
    p.add_argument("--sigma", type=int, default=7)
    p.add_argument("--ratio-hi", type=float, default=1.10)
    p.add_argument("--ratio-lo", type=float, default=0.90)
    p.add_argument("--min-area", type=int, default=8)
    p.add_argument("--open-xy", type=int, nargs=2, default=(2,2))
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    ns = parse_args(sys.argv[1:] if argv is None else argv)
    repo = os.path.expanduser(ns.repo)
    collections = [c.strip() for c in ns.collections.split(",") if c.strip()]
    dets = list(ns.detectors)
    defects_run = ns.defects_run or f"Nickel/calib/defects/{now_ts_utc()}"

    cfg = DetectConfig(
        sigma_pix=ns.sigma,
        ratio_hi=ns.ratio_hi,
        ratio_lo=ns.ratio_lo,
        min_area_px=ns.min_area,
        open_struct_xy=tuple(ns.open_xy),
        merge_columns=ns.merge,
    )

    if ns.register:
        ensure_defects_dataset_type(Butler(repo, writeable=True))

    # One Butler for reading flats (can handle multiple collections via chaining semantics)
    b_read = Butler(repo, collections=collections if len(collections) > 1 else collections[0])

    all_rects = {}
    for det in dets:
        if ns.csv:
            df = pd.read_csv(ns.csv)
            rows = df[df["detector"] == det]
            rects = [(int(r.x0), int(r.y0), int(r.width), int(r.height)) for _, r in rows.iterrows()]
            if ns.verbose:
                print(f"[det {det}] loaded {len(rects)} rectangles from {ns.csv}")
        else:
            medflat, refs = load_all_flats_median(b_read, det)
            if ns.verbose:
                print(f"[det {det}] median of {len(refs)} flats (collections={collections})")
            rects = detect_rectangles_from_flat(medflat, cfg)
            if ns.verbose:
                print(f"[det {det}] detected {len(rects)} rectangles (sigma={cfg.sigma_pix}, ratio=[{cfg.ratio_lo},{cfg.ratio_hi}], min_area={cfg.min_area_px})")
            if ns.show:
                mask = np.zeros_like(medflat, dtype=bool)
                for (x0,y0,w,h) in rects:
                    mask[y0:y0+h, x0:x0+w] = True
                print(f"[det {det}] masked fraction ≈ {mask.mean():.3%}")
                boxes = rects_to_boxes(rects)
                try:
                    pf = next(iter(refs)).dataId.get("physical_filter", "?")
                except StopIteration:
                    pf = "?"
                plot_overlay(medflat, boxes, title=f"Median flat + defects (det={det}, filter={pf})")

        all_rects[det] = rects

    # Save CSV if requested
    if ns.save_csv:
        rows = []
        for det, rects in all_rects.items():
            for (x0,y0,w,h) in rects:
                rows.append(dict(detector=det, x0=x0, y0=y0, width=w, height=h, label="auto-flat"))
        pd.DataFrame(rows, columns=["detector","x0","y0","width","height","label"]).to_csv(ns.save_csv, index=False)
        if ns.verbose:
            print(f"Wrote CSV: {ns.save_csv}")

    # Ingest if requested (one dataset per detector)
    if ns.ingest:
        for det, rects in all_rects.items():
            boxes = rects_to_boxes(rects)
            ingest_defects(repo, defects_run, "Nickel", det, boxes)
            if ns.verbose:
                print(f"[det {det}] put defects into run {defects_run}")
        print(f"DEFECTS_RUN={defects_run}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())