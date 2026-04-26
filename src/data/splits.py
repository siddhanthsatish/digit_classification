"""
Generate and save image-level train/val splits to data/splits/.

Split strategy (Route B):
  - train: train/ 100% + extra/ 95%  (~108k images)
  - val:   extra/ 5% random          (~4,100 images, zero overlap with train or test)
  - test:  test/ 100%                (loaded separately via SVHNFormat1Dataset, never touched here)

Outputs:
    data/splits/train.txt   — one relative path per line (e.g. train/1.png, extra/1.png)
    data/splits/val.txt     — one relative path per line (e.g. extra/27640.png)
    data/splits/meta.json   — seed, val_split, and counts

Usage:
    python scripts/splits.py                        # uses config.json defaults
    python scripts/splits.py --val_split 0.05 --seed 42
"""

import os
import json
import random
import argparse
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.data.dataset import parse_digit_struct


def generate_splits(data_dir="data", val_split=0.05, seed=42, splits_dir="data/splits"):
    os.makedirs(splits_dir, exist_ok=True)

    rng = random.Random(seed)

    # train/ — all images go to train
    mat_path = os.path.join(data_dir, "train", "digitStruct.mat")
    records  = parse_digit_struct(mat_path)
    train_filenames = [
        r["filename"] for r in records
        if os.path.exists(os.path.join(data_dir, "train", r["filename"]))
    ]

    # extra/ — 5% val, 95% train
    mat_path_extra  = os.path.join(data_dir, "extra", "digitStruct.mat")
    records_extra   = parse_digit_struct(mat_path_extra)
    extra_filenames = [
        r["filename"] for r in records_extra
        if os.path.exists(os.path.join(data_dir, "extra", r["filename"]))
    ]
    rng.shuffle(extra_filenames)
    n_val         = int(len(extra_filenames) * val_split)
    val_imgs      = extra_filenames[:n_val]
    extra_train   = extra_filenames[n_val:]

    # write splits
    train_txt = os.path.join(splits_dir, "train.txt")
    with open(train_txt, "w") as f:
        for name in train_filenames:
            f.write(f"train/{name}\n")
        for name in extra_train:
            f.write(f"extra/{name}\n")

    val_txt = os.path.join(splits_dir, "val.txt")
    with open(val_txt, "w") as f:
        for name in val_imgs:
            f.write(f"extra/{name}\n")

    meta = {
        "seed":      seed,
        "val_split": val_split,
        "counts": {
            "train_from_train": len(train_filenames),
            "train_from_extra": len(extra_train),
            "total_train":      len(train_filenames) + len(extra_train),
            "val_imgs":         len(val_imgs),
        },
    }
    with open(os.path.join(splits_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Splits saved to {splits_dir}/")
    print(f"  train: {meta['counts']['train_from_train']} from train/ + "
          f"{meta['counts']['train_from_extra']} from extra/ = {meta['counts']['total_train']} total")
    print(f"  val:   {meta['counts']['val_imgs']} from extra/ (5%, zero overlap)")
    print(f"  test:  loaded separately from test/ via SVHNFormat1Dataset")
    print(f"  seed:  {seed}")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--val_split",  type=float, default=0.05)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--splits_dir", default="data/splits")
    parser.add_argument("--config",     default="config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    generate_splits(
        data_dir=cfg.get("data_dir", args.data_dir),
        val_split=cfg["dataset"].get("val_split", args.val_split),
        seed=cfg["dataset"].get("seed", args.seed),
        splits_dir=cfg.get("splits_dir", args.splits_dir),
    )
