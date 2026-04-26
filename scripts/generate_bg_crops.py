"""
One-time hard negative mining script.
Runs MSER proposal pipeline on all training images (train/ + extra/ from train.txt)
and test/ images, saves valid background crops as JPEGs, writes manifests.

Outputs:
    data/bg_crops/train/       — background crops for training
    data/bg_crops/test/        — background crops for test loader
    data/bg_crops/train_manifest.json
    data/bg_crops/test_manifest.json

Usage:
    python scripts/generate_bg_crops.py
    python scripts/generate_bg_crops.py --config my_config.json
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json
import argparse
import random
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.data.dataset import parse_digit_struct, _gt_boxes, _filter_encompassing
from src.pipeline.preprocessing import preprocess_image, get_proposals
from src.pipeline.nms import iou as _iou


def _touches_gt(box, gt_boxes):
    """True if box has any pixel overlap with any GT digit box."""
    bx, by, bw, bh = box
    for gx, gy, gw, gh in gt_boxes:
        if (bx < gx + gw and bx + bw > gx and
                by < gy + gh and by + bh > gy):
            return True
    return False


def _dedup_bg_proposals(proposals, iou_threshold=0.5):
    """Remove near-duplicate proposals per image before saving."""
    kept = []
    for box in proposals:
        if not any(_iou(box, k) > iou_threshold for k in kept):
            kept.append(box)
    return kept


def _spatially_spread(proposals, n):
    """Greedy farthest-point selection — picks n proposals maximally spread across the image."""
    if len(proposals) <= n:
        return proposals
    centers = [(x + w / 2, y + h / 2) for x, y, w, h in proposals]
    picked  = [0]
    while len(picked) < n:
        best_idx, best_dist = -1, -1
        for i, c in enumerate(centers):
            if i in picked:
                continue
            min_dist = min((c[0] - centers[p][0])**2 + (c[1] - centers[p][1])**2 for p in picked)
            if min_dist > best_dist:
                best_dist, best_idx = min_dist, i
        picked.append(best_idx)
    return [proposals[i] for i in picked]


def _process_one(args):
    """Worker: process a single image and save its background crops."""
    split_dir, rec, out_dir, inf_cfg, size, max_per_image, start_idx = args
    img_path = os.path.join(split_dir, rec["filename"])
    img_bgr  = cv2.imread(img_path)
    if img_bgr is None:
        return []

    preprocessed = preprocess_image(img_bgr)
    proposals    = get_proposals(preprocessed, cfg=inf_cfg)
    proposals    = _filter_encompassing(proposals)
    proposals    = _dedup_bg_proposals(proposals, iou_threshold=0.5)
    gt           = [(x, y, w, h) for x, y, w, h in _gt_boxes(rec)]
    proposals    = [b for b in proposals if not _touches_gt(b, gt)]
    proposals    = _spatially_spread(proposals, max_per_image)

    stem   = os.path.splitext(rec["filename"])[0].replace("/", "_")
    saved  = []
    for idx, box in enumerate(proposals, start=start_idx):
        x, y, w, h = box
        x, y = max(0, x), max(0, y)
        crop = img_bgr[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        crop  = cv2.resize(crop, (size, size))
        fpath = os.path.join(out_dir, f"{stem}_{idx}.jpg")
        Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(fpath, quality=85)
        saved.append(fpath)
    return saved


def mine_and_save(images, data_dir, out_dir, inf_cfg, size, seed, max_per_image, num_workers=8):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)

    # stem -> max existing index already on disk
    existing_indices = {}
    for fname in os.listdir(out_dir):
        if not fname.endswith('.jpg'):
            continue
        parts = os.path.splitext(fname)[0].rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            existing_indices[parts[0]] = max(existing_indices.get(parts[0], -1), int(parts[1]))

    tasks = [
        (split_dir, rec, out_dir, inf_cfg, size, max_per_image,
         existing_indices.get(os.path.splitext(rec["filename"])[0].replace("/", "_"), -1) + 1)
        for split_dir, rec in images
    ]

    all_paths = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_process_one, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"  HNM → {os.path.basename(out_dir)}"):
            all_paths.extend(fut.result())

    rng.shuffle(all_paths)
    return all_paths


def main(cfg):
    ds_cfg           = cfg["dataset"]
    inf_cfg          = cfg["inference"]
    data_dir         = cfg["data_dir"]
    splits_dir       = cfg["splits_dir"]
    bg_crops_dir     = cfg["bg_crops_dir"]
    iou_threshold    = ds_cfg["bg_iou_threshold"]
    img_size         = ds_cfg["image_size"]
    seed             = ds_cfg["seed"]
    max_per_image    = ds_cfg.get("bg_max_per_image", 10)
    num_workers      = ds_cfg.get("num_workers", 8)

    # ── Training images from train.txt ────────────────────────────────────────
    train_txt = os.path.join(splits_dir, "train.txt")
    with open(train_txt) as f:
        rel_paths = [l.strip() for l in f if l.strip()]

    # group by split so we parse each digitStruct.mat once
    by_split = {}
    for rel in rel_paths:
        split_name, filename = rel.split("/", 1)
        by_split.setdefault(split_name, set()).add(filename)

    train_records = []
    n_digit_crops = 0
    for split_name, filenames in by_split.items():
        split_dir = os.path.join(data_dir, split_name)
        records   = parse_digit_struct(os.path.join(split_dir, "digitStruct.mat"))
        for rec in records:
            if rec["filename"] not in filenames:
                continue
            if not os.path.exists(os.path.join(split_dir, rec["filename"])):
                continue
            train_records.append((split_dir, rec))
            n_digit_crops += len(rec["labels"])

    n_bg_train = int(n_digit_crops * ds_cfg["bg_ratio"])
    print(f"Training: {len(train_records)} images, {n_digit_crops} digit crops → target {n_bg_train} bg crops (bg_ratio={ds_cfg['bg_ratio']})")

    train_manifest    = os.path.join(bg_crops_dir, "train_manifest.json")
    train_out         = os.path.join(bg_crops_dir, "train")
    new_train_paths   = mine_and_save(train_records, data_dir, train_out, inf_cfg, img_size, seed, max_per_image, num_workers)
    if os.path.exists(train_manifest):
        with open(train_manifest) as f:
            existing_train = json.load(f)
    else:
        existing_train = []
    train_paths = existing_train + new_train_paths
    print(f"  {len(new_train_paths)} new + {len(existing_train)} existing = {len(train_paths)} total train bg crops")

    # ── Test images ───────────────────────────────────────────────────────────
    test_dir     = os.path.join(data_dir, "test")
    test_records = parse_digit_struct(os.path.join(test_dir, "digitStruct.mat"))
    test_records = [(test_dir, r) for r in test_records
                    if os.path.exists(os.path.join(test_dir, r["filename"]))]

    n_test_digits = sum(len(r["labels"]) for _, r in test_records)
    print(f"Test: {len(test_records)} images, {n_test_digits} digit crops")

    test_manifest     = os.path.join(bg_crops_dir, "test_manifest.json")
    test_out          = os.path.join(bg_crops_dir, "test")
    new_test_paths    = mine_and_save(test_records, data_dir, test_out, inf_cfg, img_size, seed, max_per_image, num_workers)
    if os.path.exists(test_manifest):
        with open(test_manifest) as f:
            existing_test = json.load(f)
    else:
        existing_test = []
    test_paths = existing_test + new_test_paths
    print(f"  {len(new_test_paths)} new + {len(existing_test)} existing = {len(test_paths)} total test bg crops")

    # ── Write manifests ───────────────────────────────────────────────────────
    with open(train_manifest, "w") as f:
        json.dump(train_paths, f)
    with open(test_manifest, "w") as f:
        json.dump(test_paths, f)

    print(f"\nManifests written:")
    print(f"  {train_manifest}  ({len(train_paths)} crops)")
    print(f"  {test_manifest}   ({len(test_paths)} crops)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    main(cfg)
