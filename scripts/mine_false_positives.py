"""
Mine false positive detections from eval_sequence results.

For --split test (default):
    Reads wrong.json from seq_eval_<model>/
    Saves FP crops to bg_crops/test/, appends to test_manifest.json
    Used to make eval metrics harder and more honest.

For --split train:
    Reads wrong_train.json from seq_eval_<model>_train/
    Saves FP crops to bg_crops/train/, appends to train_manifest.json
    Used for iterative hard negative mining on training data.

Usage:
    python scripts/mine_false_positives.py --model custom
    python scripts/mine_false_positives.py --model custom --split train
    python scripts/mine_false_positives.py --model both --split train
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json
import argparse
import cv2
from PIL import Image
from tqdm import tqdm

from src.pipeline.predict import load_model
from src.evaluation.eval_utils import run_image, build_gt_boxes_map
from src.pipeline.nms import iou as _iou


def _is_false_positive(box, gt_boxes, iou_threshold=0.3):
    """True if box has low IoU with all GT digit boxes."""
    return all(_iou(box, (gx, gy, gw, gh)) < iou_threshold
               for gx, gy, gw, gh, _ in gt_boxes)


def mine_for_model(model_name, cfg, split="test"):
    data_dir     = cfg["data_dir"]
    inf_cfg      = cfg["inference"]
    results_dir  = cfg["results_dir"]
    bg_crops_dir = cfg["bg_crops_dir"]
    img_size     = cfg["dataset"]["image_size"]

    if split == "train":
        wrong_path = os.path.join(results_dir, f"seq_eval_{model_name}_train", "wrong_train.json")
        out_dir    = os.path.join(bg_crops_dir, "train")
        manifest   = os.path.join(bg_crops_dir, "train_manifest.json")
        split_key  = "split"   # wrong_train entries have a "split" field
    else:
        wrong_path = os.path.join(results_dir, f"seq_eval_{model_name}", "wrong.json")
        out_dir    = os.path.join(bg_crops_dir, "test")
        manifest   = os.path.join(bg_crops_dir, "test_manifest.json")
        split_key  = None

    if not os.path.exists(wrong_path):
        print(f"  No wrong json found at {wrong_path}, skipping")
        return

    with open(wrong_path) as f:
        wrong = json.load(f)
    print(f"  [{model_name}] {len(wrong)} wrong images to process ({split} split)")

    model, device = load_model(
        model_name,
        ckpt_dir=cfg["ckpt_dir"],
        num_classes=cfg["num_classes"],
        custom_cfg=cfg.get("custom"),
    )

    # build gt_boxes_map — for train split entries carry their own split name
    if split == "train":
        # group by split dir to avoid re-parsing mat files
        by_split = {}
        for entry in wrong:
            by_split.setdefault(entry["split"], []).append(entry["filename"])
        gt_boxes_map = {}
        for sname, fnames in by_split.items():
            smap = build_gt_boxes_map(os.path.join(data_dir, sname, "digitStruct.mat"))
            for fname in fnames:
                gt_boxes_map[f"{sname}/{fname}"] = smap.get(fname, [])
    else:
        raw_map = build_gt_boxes_map(os.path.join(data_dir, "test", "digitStruct.mat"))
        gt_boxes_map = {fname: boxes for fname, boxes in raw_map.items()}

    os.makedirs(out_dir, exist_ok=True)

    # existing indices on disk to avoid overwriting
    existing_indices = {}
    for fname in os.listdir(out_dir):
        if not fname.endswith('.jpg'):
            continue
        parts = os.path.splitext(fname)[0].rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            existing_indices[parts[0]] = max(existing_indices.get(parts[0], -1), int(parts[1]))

    new_paths = []
    for entry in tqdm(wrong, desc=f"  mining FP [{model_name}] {split}"):
        filename  = entry["filename"]
        split_dir = entry["split"] if split == "train" else "test"
        img_path  = os.path.join(data_dir, split_dir, filename)
        img_bgr   = cv2.imread(img_path)
        if img_bgr is None:
            continue

        _, det_list, _ = run_image(img_bgr, model, device, inf_cfg)
        if not det_list:
            continue

        map_key  = f"{split_dir}/{filename}" if split == "train" else filename
        gt_boxes = gt_boxes_map.get(map_key, [])
        stem     = os.path.splitext(filename)[0]
        start_idx = existing_indices.get(stem, -1) + 1

        for i, (x, y, w, h, digit, conf) in enumerate(det_list, start=start_idx):
            if not _is_false_positive((x, y, w, h), gt_boxes):
                continue
            x, y = max(0, x), max(0, y)
            crop = img_bgr[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            crop  = cv2.resize(crop, (img_size, img_size))
            fpath = os.path.join(out_dir, f"{stem}_{i}.jpg")
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(fpath, quality=85)
            new_paths.append(fpath)
            existing_indices[stem] = i

    print(f"  [{model_name}] {len(new_paths)} FP crops saved → {out_dir}")

    if os.path.exists(manifest):
        with open(manifest) as f:
            existing = json.load(f)
    else:
        existing = []

    with open(manifest, "w") as f:
        json.dump(existing + new_paths, f)
    print(f"  {split}_manifest updated: {len(existing)} → {len(existing) + len(new_paths)} entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["custom", "vgg16", "both"], default="custom")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    models = ["custom", "vgg16"] if args.model == "both" else [args.model]
    for m in models:
        mine_for_model(m, cfg, split=args.split)
