"""
Generate all report visualisations from saved training history JSONs and SVHN cache.
Run after both models have been trained:

    python scripts/generate_report_figures.py

Outputs to outputs/report_figures/
  EDA:
    eda_class_distribution.png / .json
    eda_sequence_length.png    / .json
    eda_bbox_size.png          / .json
    eda_aspect_ratio.png       / .json
    eda_image_size.png         / .json
    eda_digit_position.png     / .json
    eda_sample_crops.png       / .json
  Model comparison (requires trained models):
    proposal_hit_rate.png      / .json
    pipeline_steps.png         / .json
    comparison_loss.png        / .json
    comparison_accuracy.png    / .json
    comparison_bar.png         / .json
    model_summary.txt
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json
import pickle
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT_DIR     = "outputs/report_figures"
OUT_EDA     = "outputs/report_figures/exploratory"
OUT_COMPARE = "outputs/report_figures/comparison"
RESULTS_DIR = "outputs/results"
CACHE_DIR  = "data/cache"
MODELS     = ["custom", "vgg16"]
MODEL_LABELS = {"custom": "CustomCNN (scratch)", "vgg16": "VGG16 (fine-tuned)"}
COLORS       = {"custom": "steelblue",            "vgg16": "darkorange"}
SPLIT_COLORS = {"train": "steelblue", "test": "darkorange", "extra": "seagreen"}
EDA_SPLITS   = ("train", "test")
IMAGE_SAMPLE = 3000   # images sampled for image-size distribution (speed)
SEED         = 42


# ── helpers ───────────────────────────────────────────────────────────────────

def save_meta(stem, meta, out_dir):
    path = os.path.join(out_dir, stem + ".json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved → {path}")


def savefig(stem, out_dir):
    path = os.path.join(out_dir, stem + ".png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")
    return path


def load_records(split_name):
    path = os.path.join(CACHE_DIR, f"{split_name}_records.pkl")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping {split_name}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_history(model_name):
    path = os.path.join(RESULTS_DIR, f"{model_name}_history.json")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping {model_name}")
        return None
    with open(path) as f:
        return json.load(f)


# ── EDA ───────────────────────────────────────────────────────────────────────

def eda_class_distribution():
    stem = "eda_class_distribution"
    counts = {}
    for split in EDA_SPLITS:
        records = load_records(split)
        if records is None:
            continue
        c = np.zeros(10, dtype=int)
        for rec in records:
            for lbl in rec["labels"]:
                c[0 if lbl == 10 else lbl] += 1
        counts[split] = c

    if not counts:
        return

    x, width = np.arange(10), 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (split, c) in enumerate(counts.items()):
        ax.bar(x + i * width, c, width, label=split.capitalize(),
               color=SPLIT_COLORS[split], edgecolor="white")
    ax.set_xticks(x + width * (len(counts) - 1) / 2)
    ax.set_xticklabels([str(d) for d in range(10)])
    ax.set_xlabel("Digit class"); ax.set_ylabel("Crop count")
    ax.set_title("Class Distribution (digit crops)")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    savefig(stem, OUT_EDA)

    meta = {
        "description": "Per-class digit crop counts across splits",
        "splits": {s: c.tolist() for s, c in counts.items()},
        "classes": list(range(10)),
        "most_frequent": {s: int(np.argmax(c)) for s, c in counts.items()},
        "least_frequent": {s: int(np.argmin(c)) for s, c in counts.items()},
        "imbalance_ratio": {s: round(float(c.max() / c.min()), 3) for s, c in counts.items()},
    }
    save_meta(stem, meta, OUT_EDA)


def eda_sequence_length_distribution():
    stem = "eda_sequence_length"
    all_lengths = {}
    for split in EDA_SPLITS:
        records = load_records(split)
        if records is None:
            continue
        all_lengths[split] = [len(rec["labels"]) for rec in records]

    if not all_lengths:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for split, lengths in all_lengths.items():
        max_len = max(lengths)
        bins = np.arange(1, max_len + 2) - 0.5
        ax.hist(lengths, bins=bins, label=split.capitalize(), alpha=0.7,
                color=SPLIT_COLORS[split], edgecolor="white")
    ax.set_xlabel("Digits per image"); ax.set_ylabel("Image count")
    ax.set_title("Sequence Length Distribution")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    savefig(stem, OUT_EDA)

    meta = {
        "description": "Distribution of number of digits per image",
        "splits": {
            s: {
                "mean":   round(float(np.mean(l)), 3),
                "median": float(np.median(l)),
                "max":    int(np.max(l)),
                "counts": {str(k): int(v)
                           for k, v in zip(*np.unique(l, return_counts=True))},
            }
            for s, l in all_lengths.items()
        },
    }
    save_meta(stem, meta, OUT_EDA)


def eda_bbox_size():
    stem = "eda_bbox_size"
    data = {}
    for split in EDA_SPLITS:
        records = load_records(split)
        if records is None:
            continue
        heights, widths = [], []
        for rec in records:
            heights.extend(rec["heights"])
            widths.extend(rec["widths"])
        data[split] = {"heights": heights, "widths": widths}

    if not data:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for split, d in data.items():
        clip = 200
        axes[0].hist(np.clip(d["heights"], 0, clip), bins=60, alpha=0.6,
                     label=split.capitalize(), color=SPLIT_COLORS[split], edgecolor="none")
        axes[1].hist(np.clip(d["widths"],  0, clip), bins=60, alpha=0.6,
                     label=split.capitalize(), color=SPLIT_COLORS[split], edgecolor="none")
    for ax, dim in zip(axes, ["Height (px)", "Width (px)"]):
        ax.set_xlabel(dim); ax.set_ylabel("Count")
        ax.set_title(f"Bounding Box {dim.split()[0]} Distribution")
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.suptitle("Bounding Box Size Distribution", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(stem, OUT_EDA)

    meta = {
        "description": "Bounding box height and width distributions across splits",
        "splits": {
            s: {
                "height": {"mean": round(float(np.mean(d["heights"])), 2),
                           "median": float(np.median(d["heights"])),
                           "p5":  float(np.percentile(d["heights"], 5)),
                           "p95": float(np.percentile(d["heights"], 95))},
                "width":  {"mean": round(float(np.mean(d["widths"])), 2),
                           "median": float(np.median(d["widths"])),
                           "p5":  float(np.percentile(d["widths"], 5)),
                           "p95": float(np.percentile(d["widths"], 95))},
            }
            for s, d in data.items()
        },
    }
    save_meta(stem, meta, OUT_EDA)


def eda_aspect_ratio():
    stem = "eda_aspect_ratio"
    data = {}
    for split in EDA_SPLITS:
        records = load_records(split)
        if records is None:
            continue
        ratios = []
        for rec in records:
            for h, w in zip(rec["heights"], rec["widths"]):
                if w > 0:
                    ratios.append(h / w)
        data[split] = ratios

    if not data:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for split, ratios in data.items():
        ax.hist(np.clip(ratios, 0, 6), bins=80, alpha=0.6,
                label=split.capitalize(), color=SPLIT_COLORS[split], edgecolor="none")
    ax.axvline(0.2, color="red",  linestyle="--", linewidth=1, label="mser_min_aspect (0.2)")
    ax.axvline(5.0, color="green", linestyle="--", linewidth=1, label="mser_max_aspect (5.0)")
    ax.set_xlabel("Aspect ratio (h/w)"); ax.set_ylabel("Count")
    ax.set_title("Bounding Box Aspect Ratio Distribution")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    savefig(stem, OUT_EDA)

    meta = {
        "description": "Bounding box aspect ratio (h/w) distribution; mser thresholds overlaid",
        "mser_min_aspect": 0.2,
        "mser_max_aspect": 5.0,
        "splits": {
            s: {
                "mean":   round(float(np.mean(r)), 3),
                "median": round(float(np.median(r)), 3),
                "p5":     round(float(np.percentile(r, 5)), 3),
                "p95":    round(float(np.percentile(r, 95)), 3),
                "pct_within_mser_bounds": round(
                    float(np.mean((np.array(r) >= 0.2) & (np.array(r) <= 5.0))) * 100, 2),
            }
            for s, r in data.items()
        },
    }
    save_meta(stem, meta, OUT_EDA)


def eda_image_size():
    stem = "eda_image_size"
    rng = random.Random(SEED)
    data = {}
    for split in EDA_SPLITS:
        records = load_records(split)
        if records is None:
            continue
        sample = rng.sample(records, min(IMAGE_SAMPLE, len(records)))
        widths, heights = [], []
        for rec in sample:
            path = os.path.join("data", split, rec["filename"])
            if not os.path.exists(path):
                continue
            try:
                w, h = Image.open(path).size
                widths.append(w); heights.append(h)
            except Exception:
                continue
        data[split] = {"widths": widths, "heights": heights}

    if not data:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for split, d in data.items():
        axes[0].hist(d["widths"],  bins=50, alpha=0.6, label=split.capitalize(),
                     color=SPLIT_COLORS[split], edgecolor="none")
        axes[1].hist(d["heights"], bins=50, alpha=0.6, label=split.capitalize(),
                     color=SPLIT_COLORS[split], edgecolor="none")
    for ax, dim in zip(axes, ["Width (px)", "Height (px)"]):
        ax.set_xlabel(dim); ax.set_ylabel("Image count")
        ax.set_title(f"Image {dim.split()[0]} Distribution")
        ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.suptitle(f"Full Scene Image Size Distribution (sample n={IMAGE_SAMPLE})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(stem, OUT_EDA)

    meta = {
        "description": f"Full scene image dimensions sampled from each split (n={IMAGE_SAMPLE})",
        "sample_size": IMAGE_SAMPLE,
        "splits": {
            s: {
                "width":  {"mean": round(float(np.mean(d["widths"])), 1),
                           "median": float(np.median(d["widths"])),
                           "min": int(np.min(d["widths"])),
                           "max": int(np.max(d["widths"]))},
                "height": {"mean": round(float(np.mean(d["heights"])), 1),
                           "median": float(np.median(d["heights"])),
                           "min": int(np.min(d["heights"])),
                           "max": int(np.max(d["heights"]))},
            }
            for s, d in data.items()
        },
    }
    save_meta(stem, meta, OUT_EDA)


def eda_digit_position():
    """Digit frequency by position (0=first, 1=second, …) across classes."""
    stem = "eda_digit_position"
    records = load_records("train")
    if records is None:
        return

    max_pos = max(len(rec["labels"]) for rec in records)
    # matrix: rows=position, cols=digit 0-9
    matrix = np.zeros((max_pos, 10), dtype=int)
    for rec in records:
        for pos, lbl in enumerate(rec["labels"]):
            digit = 0 if lbl == 10 else lbl
            matrix[pos, digit] += 1

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(10)
    width = 0.8 / max_pos
    cmap = plt.get_cmap("tab10")
    for pos in range(max_pos):
        ax.bar(x + pos * width, matrix[pos], width,
               label=f"Position {pos + 1}", color=cmap(pos / max_pos), edgecolor="none")
    ax.set_xticks(x + width * (max_pos - 1) / 2)
    ax.set_xticklabels([str(d) for d in range(10)])
    ax.set_xlabel("Digit class"); ax.set_ylabel("Count")
    ax.set_title("Digit Frequency by Sequence Position (train)")
    ax.legend(fontsize=8, ncol=max_pos); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    savefig(stem, OUT_EDA)

    meta = {
        "description": "Frequency of each digit class at each sequence position (train split)",
        "max_position": max_pos,
        "positions": {
            str(pos + 1): {str(d): int(matrix[pos, d]) for d in range(10)}
            for pos in range(max_pos)
        },
    }
    save_meta(stem, meta, OUT_EDA)


def eda_sample_crops(n_per_class=8):
    """Grid of sample digit crops, n_per_class per digit 0-9."""
    stem = "eda_sample_crops"
    records = load_records("train")
    if records is None:
        return

    rng = random.Random(SEED)
    # bucket samples by digit
    buckets = {d: [] for d in range(10)}
    shuffled = list(records)
    rng.shuffle(shuffled)
    for rec in shuffled:
        for lbl, top, left, h, w in zip(
                rec["labels"], rec["tops"], rec["lefts"],
                rec["heights"], rec["widths"]):
            digit = 0 if lbl == 10 else lbl
            if len(buckets[digit]) < n_per_class:
                buckets[digit].append((
                    os.path.join("data", "train", rec["filename"]),
                    int(top), int(left), int(h), int(w)
                ))
        if all(len(v) == n_per_class for v in buckets.values()):
            break

    fig, axes = plt.subplots(10, n_per_class, figsize=(n_per_class * 1.2, 10 * 1.2))
    for digit in range(10):
        for col, (img_path, top, left, h, w) in enumerate(buckets[digit]):
            ax = axes[digit, col]
            ax.axis("off")
            try:
                img = Image.open(img_path).convert("RGB")
                iw, ih = img.size
                crop = img.crop((max(0, left), max(0, top),
                                 min(iw, left + w), min(ih, top + h)))
                crop = crop.resize((32, 32), Image.BILINEAR)
                ax.imshow(np.array(crop))
            except Exception:
                pass
            if col == 0:
                ax.set_ylabel(str(digit), fontsize=10, rotation=0,
                              labelpad=14, va="center")
    plt.suptitle("Sample Digit Crops by Class (train)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(stem, OUT_EDA)

    meta = {
        "description": f"Sample digit crops from train split, {n_per_class} per class",
        "n_per_class": n_per_class,
        "classes": list(range(10)),
        "samples_found": {str(d): len(v) for d, v in buckets.items()},
    }
    save_meta(stem, meta, OUT_EDA)


# ── proposal hit rate (pre-classification) ────────────────────────────────────

def _proposal_worker(args):
    """Process-pool worker: preprocess one image and return GT boxes + proposals."""
    filename, gt_lefts, gt_tops, gt_widths, gt_heights, inf_cfg = args
    import cv2
    from src.pipeline.preprocessing import preprocess_image, get_proposals
    img = cv2.imread(os.path.join("data", "test", filename))
    if img is None:
        return None
    preprocessed = preprocess_image(img)
    proposals = get_proposals(
        preprocessed,
        use_mser=inf_cfg.get("use_mser", True),
        use_sliding=inf_cfg.get("use_sliding", False),
        cfg=inf_cfg,
    )
    gt_boxes = list(zip(gt_lefts, gt_tops, gt_widths, gt_heights))
    return gt_boxes, proposals


def _compute_proposal_hit_rate():
    """Compute recall, precision, full-coverage from test records + proposal pipeline.

    Caches result to OUT_COMPARE/proposal_hit_rate.json so it only runs once.
    """
    cache_path = os.path.join(OUT_COMPARE, "proposal_hit_rate.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        if "recall" in data:
            return data

    from concurrent.futures import ProcessPoolExecutor
    from src.pipeline.nms import iou as box_iou

    records = load_records("test")
    if records is None:
        return None

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    with open(cfg_path) as f:
        inf_cfg = json.load(f).get("inference", {})

    tasks = [
        (r["filename"], r["lefts"], r["tops"], r["widths"], r["heights"], inf_cfg)
        for r in records
    ]

    print("  Computing proposal hit rate (one-time, ~13k images)...")
    num_workers = os.cpu_count() or 4
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for res in pool.map(_proposal_worker, tasks, chunksize=64):
            if res is not None:
                results.append(res)

    hit_iou = 0.3
    total_gt = total_gt_hit = total_props = total_prop_hit = 0
    images_all_hit = images_total = 0
    by_seq_len = {}

    for gt_boxes, proposals in results:
        seq_len = len(gt_boxes)
        sl = by_seq_len.setdefault(seq_len, dict(
            gt_hit=0, gt_total=0, prop_hit=0, prop_total=0,
            img_all_hit=0, img_total=0))
        images_total += 1
        sl["img_total"] += 1
        img_gt_hits = 0
        for gb in gt_boxes:
            total_gt += 1; sl["gt_total"] += 1
            if any(box_iou(gb, p) >= hit_iou for p in proposals):
                total_gt_hit += 1; sl["gt_hit"] += 1; img_gt_hits += 1
        if img_gt_hits == len(gt_boxes):
            images_all_hit += 1; sl["img_all_hit"] += 1
        for p in proposals:
            total_props += 1; sl["prop_total"] += 1
            if any(box_iou(gb, p) >= hit_iou for gb in gt_boxes):
                total_prop_hit += 1; sl["prop_hit"] += 1

    recall    = total_gt_hit   / total_gt    if total_gt    else 0.0
    precision = total_prop_hit / total_props if total_props else 0.0
    full_cov  = images_all_hit / images_total if images_total else 0.0

    hit_data = {
        "iou_threshold": hit_iou,
        "recall":    {"hits": total_gt_hit,   "total": total_gt,    "rate": round(recall, 4)},
        "precision": {"hits": total_prop_hit,  "total": total_props, "rate": round(precision, 4)},
        "full_coverage": {"images": images_all_hit, "total": images_total, "rate": round(full_cov, 4)},
        "by_seq_len": {
            str(k): {
                "recall":    round(v["gt_hit"]  / v["gt_total"], 4)   if v["gt_total"]   else 0.0,
                "precision": round(v["prop_hit"] / v["prop_total"], 4) if v["prop_total"] else 0.0,
                "full_coverage": round(v["img_all_hit"] / v["img_total"], 4) if v["img_total"] else 0.0,
                **v,
            }
            for k, v in sorted(by_seq_len.items())
        },
    }
    with open(cache_path, "w") as f:
        json.dump(hit_data, f, indent=2)
    print(f"  Cached → {cache_path}")
    return hit_data


def proposal_hit_rate():
    """3-panel chart: proposal recall, precision, full-coverage by seq length."""
    stem = "proposal_hit_rate"
    hit_data = _compute_proposal_hit_rate()
    if hit_data is None:
        return

    by_len  = hit_data["by_seq_len"]
    lengths = sorted(by_len.keys(), key=int)
    iou_thr = hit_data["iou_threshold"]
    metrics    = ["recall", "precision", "full_coverage"]
    titles     = ["Recall (GT digits found)", "Precision (proposals matching GT)",
                  "Full Coverage (all GT digits hit)"]
    overall    = [hit_data["recall"]["rate"], hit_data["precision"]["rate"],
                  hit_data["full_coverage"]["rate"]]
    count_keys = ["gt_total", "prop_total", "img_total"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    x = np.arange(len(lengths))
    for ax, metric, title, ov, ck in zip(axes, metrics, titles, overall, count_keys):
        rates  = [by_len[l][metric] * 100 for l in lengths]
        counts = [by_len[l][ck] for l in lengths]
        bars = ax.bar(x, rates, color="steelblue", edgecolor="white", width=0.6)
        for bar, r, c in zip(bars, rates, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{r:.1f}%\n(n={c})", ha="center", va="bottom", fontsize=7)
        ax.axhline(ov * 100, color="red", linestyle="--", linewidth=1,
                   label=f"Overall: {ov*100:.1f}%")
        ax.set_xticks(x); ax.set_xticklabels(lengths)
        ax.set_xlabel("GT Sequence Length")
        ax.set_ylabel("%"); ax.set_ylim(0, 115)
        ax.set_title(title); ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle(f"Region Proposal Quality (IoU \u2265 {iou_thr})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(stem, OUT_COMPARE)
    save_meta(stem, hit_data, OUT_COMPARE)


# ── pipeline step visualisation ───────────────────────────────────────────────

def pipeline_steps_figure():
    """Full pipeline visualisation for 1 correct and 1 wrong image.

    8 columns: Original → Polarity → Blur → CLAHE → MSER → Proposals
               → Classifier hits → Final prediction
    Rows labelled ✓ CORRECT / ✗ WRONG.
    """
    import cv2
    import matplotlib.patches as mpatches
    from src.pipeline.predict import load_model, DIGIT_LABELS, BG_CLASS
    from src.evaluation.eval_utils import run_image

    stem = "pipeline_steps"

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    if not os.path.exists(cfg_path):
        print("  config.json not found — skipping pipeline steps.")
        return
    with open(cfg_path) as f:
        cfg = json.load(f)
    inf_cfg = cfg.get("inference", {})

    # Pick 1 correct + 1 wrong (deterministic: first 3-digit match sorted by filename)
    entries = {}
    for name in MODELS:
        cpath = os.path.join(RESULTS_DIR, f"seq_eval_{name}", "correct.json")
        wpath = os.path.join(RESULTS_DIR, f"seq_eval_{name}", "wrong.json")
        if not (os.path.exists(cpath) and os.path.exists(wpath)):
            continue
        with open(cpath) as f:
            correct = json.load(f)
        with open(wpath) as f:
            wrong = json.load(f)
        good = sorted(
            [e for e in correct if 2 <= len(e["gt"]) <= 3],
            key=lambda e: e["filename"],
        )
        bad = sorted(
            [e for e in wrong
             if 2 <= len(e["gt"]) <= 3 and e["pred"] != ""
             and len(e["pred"]) != len(e["gt"])],
            key=lambda e: e["filename"],
        )
        if good and bad:
            entries["correct"] = good[0]
            entries["wrong"]   = bad[0]
            entries["_model"]  = name
            break
    if len(entries) < 2:
        print("  Could not find suitable correct+wrong entries — skipping.")
        return

    # Load model for classifier hits + final prediction
    model_name = entries.pop("_model")
    try:
        model, device = load_model(
            model_name, ckpt_dir=cfg["ckpt_dir"],
            num_classes=cfg["num_classes"], custom_cfg=cfg.get("custom"),
        )
    except FileNotFoundError as e:
        print(f"  {e} — skipping pipeline steps.")
        return

    # Load GT boxes
    records = load_records("test")
    if records is None:
        return
    gt_map = {}
    for rec in records:
        gt_map[rec["filename"]] = list(zip(
            rec["lefts"], rec["tops"], rec["widths"], rec["heights"], rec["labels"],
        ))

    def _bgr_to_rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 8, figsize=(30, 8))
    for row, key in enumerate(["correct", "wrong"]):
        entry = entries[key]
        filename = entry["filename"]
        img_path = os.path.join("data", "test", filename)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        pred_seq, det_list, im = run_image(img_bgr, model, device, inf_cfg)

        panels = [
            (_bgr_to_rgb(img_bgr),       "1. Original"),
            (_bgr_to_rgb(im["step1"]),   "2. Polarity norm"),
            (_bgr_to_rgb(im["step2"]),   "3. Gaussian blur"),
            (_bgr_to_rgb(im["step3"]),   "4. CLAHE"),
            (None, f"5. MSER ({len(im['mser_boxes'])})"),
            (None, f"6. Proposals ({len(im['all_proposals'])})"),
            (None, f"7. Hits ({len(im['digit_boxes'])})"),
            (None, f"8. Pred: \"{pred_seq}\""),
        ]

        is_correct = entry["correct"]
        icon = "\u2713" if is_correct else "\u2717"
        colour = "green" if is_correct else "red"
        row_label = f"{icon} {'CORRECT' if is_correct else 'WRONG'}\nGT=\"{entry['gt']}\"  Pred=\"{entry['pred']}\""

        step3_rgb = _bgr_to_rgb(im["step3"])
        for col, (img_rgb, title) in enumerate(panels):
            ax = axes[row][col]
            ax.imshow(img_rgb if img_rgb is not None else step3_rgb)
            # GT boxes on all panels
            for (lx, ty, bw, bh, lbl) in gt_map.get(filename, []):
                ax.add_patch(mpatches.Rectangle(
                    (lx, ty), bw, bh, linewidth=1.5,
                    edgecolor="lime", facecolor="none", linestyle="--"))
            # MSER boxes
            if col == 4:
                for (bx, by, bw, bh) in im["mser_boxes"]:
                    ax.add_patch(mpatches.Rectangle(
                        (bx, by), bw, bh, linewidth=0.7,
                        edgecolor="cyan", facecolor="none"))
            # Deduped proposals
            elif col == 5:
                for (bx, by, bw, bh) in im["all_proposals"]:
                    ax.add_patch(mpatches.Rectangle(
                        (bx, by), bw, bh, linewidth=0.7,
                        edgecolor="orange", facecolor="none"))
            # Classifier hits (pre-NMS)
            elif col == 6:
                for (bx, by, bw, bh), lbl, conf in zip(
                        im["digit_boxes"], im["digit_labels"], im["digit_scores"]):
                    ax.add_patch(mpatches.Rectangle(
                        (bx, by), bw, bh, linewidth=0.8,
                        edgecolor="red", facecolor="none"))
                    ax.text(bx, by - 1, f"{DIGIT_LABELS[lbl]}:{conf:.0%}",
                            color="red", fontsize=5)
            # Final prediction (post-NMS)
            elif col == 7:
                for (dx, dy, dw, dh, digit, conf) in det_list:
                    ax.add_patch(mpatches.Rectangle(
                        (dx, dy), dw, dh, linewidth=2,
                        edgecolor="yellow", facecolor="none"))
                    ax.text(dx, dy - 2, f"{digit}",
                            color="yellow", fontsize=9, fontweight="bold")
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        axes[row][0].set_ylabel(row_label, fontsize=10, rotation=0,
                                labelpad=120, va="center", ha="right",
                                color=colour, fontweight="bold")

    plt.suptitle(
        f"Full Pipeline: Preprocessing → Proposals → Classification → NMS "
        f"(green dashed = GT, model = {MODEL_LABELS[model_name]})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    savefig(stem, OUT_COMPARE)

    meta = {
        "description": "Full pipeline visualisation for 1 correct and 1 wrong image",
        "model": model_name,
        "correct": entries["correct"],
        "wrong":   entries["wrong"],
    }
    save_meta(stem, meta, OUT_COMPARE)


# ── model comparison ──────────────────────────────────────────────────────────

def comparison_loss_curves(histories):
    stem = "comparison_loss"
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for split, ax in zip(["train", "val"], axes):
        for name, h in histories.items():
            epochs = range(1, len(h[f"{split}_loss"]) + 1)
            ax.plot(epochs, h[f"{split}_loss"],
                    label=MODEL_LABELS[name], color=COLORS[name], marker="o", markersize=3)
        ax.set_title(f"{split.capitalize()} Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle("Loss Curves — Model Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(stem, OUT_COMPARE)

    meta = {
        "description": "Train and val loss curves for each model",
        "models": {
            name: {
                "train_loss_final": round(h["train_loss"][-1], 4),
                "val_loss_final":   round(h["val_loss"][-1], 4),
                "epochs": len(h["train_loss"]),
            }
            for name, h in histories.items()
        },
    }
    save_meta(stem, meta, OUT_COMPARE)


def comparison_accuracy_curves(histories):
    stem = "comparison_accuracy"
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for split, ax in zip(["train", "val"], axes):
        for name, h in histories.items():
            epochs = range(1, len(h[f"{split}_acc"]) + 1)
            ax.plot(epochs, [a * 100 for a in h[f"{split}_acc"]],
                    label=MODEL_LABELS[name], color=COLORS[name], marker="o", markersize=3)
        ax.set_title(f"{split.capitalize()} Accuracy")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle("Accuracy Curves — Model Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(stem, OUT_COMPARE)

    meta = {
        "description": "Train and val accuracy curves for each model",
        "models": {
            name: {
                "train_acc_final": round(h["train_acc"][-1] * 100, 2),
                "val_acc_final":   round(h["val_acc"][-1]   * 100, 2),
                "epochs": len(h["train_acc"]),
            }
            for name, h in histories.items()
        },
    }
    save_meta(stem, meta, OUT_COMPARE)


def comparison_bar_chart(histories):
    stem = "comparison_bar"
    metrics = ["train_acc", "val_acc", "test_acc"]
    labels  = ["Train Acc", "Val Acc", "Test Acc"]
    x, width = np.arange(len(labels)), 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, h) in enumerate(histories.items()):
        vals = [h.get(m, [0])[-1] * 100 if isinstance(h.get(m, 0), list)
                else h.get(m, 0) * 100
                for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=MODEL_LABELS[name],
                      color=COLORS[name], edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x + width / 2); ax.set_xticklabels(labels)
    ax.set_ylim(0, 110); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Performance Comparison")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    savefig(stem, OUT_COMPARE)

    meta = {
        "description": "Train / val / test accuracy bar chart per model",
        "models": {
            name: {
                m: round((h.get(m, [0])[-1] if isinstance(h.get(m, 0), list)
                          else h.get(m, 0)) * 100, 2)
                for m in metrics
            }
            for name, h in histories.items()
        },
    }
    save_meta(stem, meta, OUT_COMPARE)


def comparison_sequence_accuracy():
    """Bar chart comparing sequence-level accuracy + error breakdown."""
    stem = "comparison_seq_accuracy"
    summaries = {}
    wrongs = {}
    for name in MODELS:
        spath = os.path.join(RESULTS_DIR, f"seq_eval_{name}", "summary.json")
        wpath = os.path.join(RESULTS_DIR, f"seq_eval_{name}", "wrong.json")
        if not os.path.exists(spath):
            continue
        with open(spath) as f:
            summaries[name] = json.load(f)
        if os.path.exists(wpath):
            with open(wpath) as f:
                wrongs[name] = json.load(f)

    if len(summaries) < 2:
        print("  Sequence eval results not found for both models — skipping.")
        return

    # ── bar chart: overall accuracy ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: overall accuracy
    ax = axes[0]
    names = list(summaries.keys())
    accs = [summaries[n]["accuracy"] * 100 for n in names]
    bars = ax.bar([MODEL_LABELS[n] for n in names], accs,
                  color=[COLORS[n] for n in names], edgecolor="white", width=0.5)
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Sequence Accuracy (%)"); ax.set_ylim(0, 100)
    ax.set_title("Overall Sequence Accuracy")
    ax.grid(True, axis="y", alpha=0.3)

    # Right: error breakdown (stacked)
    ax = axes[1]
    categories = ["Correct", "Wrong digits", "Wrong length", "No detection"]
    cat_colors = ["#4CAF50", "#FF9800", "#F44336", "#9E9E9E"]
    x = np.arange(len(names))
    width = 0.5
    bottoms = np.zeros(len(names))
    breakdown = {}
    for i, name in enumerate(names):
        s = summaries[name]
        w = wrongs.get(name, [])
        empty = sum(1 for e in w if e["pred"] == "")
        len_err = sum(1 for e in w if len(e["pred"]) != len(e["gt"]) and e["pred"] != "")
        digit_err = sum(1 for e in w if len(e["pred"]) == len(e["gt"]))
        breakdown[name] = [s["correct"], digit_err, len_err, empty]

    for j, (cat, color) in enumerate(zip(categories, cat_colors)):
        vals = [breakdown[n][j] for n in names]
        ax.bar(x, vals, width, bottom=bottoms, label=cat, color=color, edgecolor="white")
        bottoms += np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels([MODEL_LABELS[n] for n in names])
    ax.set_ylabel("Image count"); ax.set_title("Error Breakdown")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Sequence-Level Evaluation Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(stem, OUT_COMPARE)

    meta = {
        "description": "Sequence-level accuracy comparison and error breakdown",
        "models": {
            n: {"accuracy_pct": round(summaries[n]["accuracy"] * 100, 2),
                "correct": breakdown[n][0], "wrong_digits": breakdown[n][1],
                "wrong_length": breakdown[n][2], "no_detection": breakdown[n][3],
                "total": summaries[n]["total"]}
            for n in names
        },
    }
    save_meta(stem, meta, OUT_COMPARE)


def comparison_sequence_by_length():
    """Line chart: sequence accuracy vs GT sequence length."""
    stem = "comparison_seq_by_length"
    all_logs = {}
    for name in MODELS:
        cpath = os.path.join(RESULTS_DIR, f"seq_eval_{name}", "correct.json")
        wpath = os.path.join(RESULTS_DIR, f"seq_eval_{name}", "wrong.json")
        if not (os.path.exists(cpath) and os.path.exists(wpath)):
            continue
        with open(cpath) as f:
            correct = json.load(f)
        with open(wpath) as f:
            wrong = json.load(f)
        all_logs[name] = correct + wrong

    if len(all_logs) < 2:
        print("  Sequence eval results not found for both models — skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    meta_models = {}
    for name, log in all_logs.items():
        by_len = {}
        for e in log:
            gl = len(e["gt"])
            by_len.setdefault(gl, [0, 0])
            by_len[gl][1] += 1
            if e["correct"]:
                by_len[gl][0] += 1
        lengths = sorted(by_len.keys())
        accs = [by_len[l][0] / by_len[l][1] * 100 for l in lengths]
        counts = [by_len[l][1] for l in lengths]
        ax.plot(lengths, accs, marker="o", label=MODEL_LABELS[name],
                color=COLORS[name], linewidth=2, markersize=6)
        meta_models[name] = {str(l): {"accuracy_pct": round(a, 2), "count": c}
                             for l, a, c in zip(lengths, accs, counts)}

    ax.set_xlabel("GT Sequence Length (# digits)")
    ax.set_ylabel("Sequence Accuracy (%)")
    ax.set_title("Sequence Accuracy by Number of Digits")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylim(0, 105)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig(stem, OUT_COMPARE)

    save_meta(stem, {"description": "Sequence accuracy broken down by GT length",
                     "models": meta_models}, OUT_COMPARE)


def model_summary_table(histories):
    lines = ["=" * 60, "MODEL PERFORMANCE SUMMARY", "=" * 60,
             f"{'Model':<25} {'Train Acc':>10} {'Val Acc':>10} {'Test Acc':>10} {'Epochs':>8}",
             "-" * 60]
    for name, h in histories.items():
        lines.append(
            f"{MODEL_LABELS[name]:<25} "
            f"{h['train_acc'][-1]*100:>9.2f}% "
            f"{h['val_acc'][-1]*100:>9.2f}% "
            f"{h.get('test_acc', 0)*100:>9.2f}% "
            f"{len(h['train_acc']):>8}"
        )
    lines.append("=" * 60)
    text = "\n".join(lines)
    print(text)
    path = os.path.join(OUT_COMPARE, "model_summary.txt")
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"  Saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_EDA, exist_ok=True)
    os.makedirs(OUT_COMPARE, exist_ok=True)

    print("── EDA figures ──────────────────────────────────────────")
    eda_class_distribution()
    eda_sequence_length_distribution()
    eda_bbox_size()
    eda_aspect_ratio()
    eda_image_size()
    eda_digit_position()
    eda_sample_crops()

    print("\n── Proposal hit rate ────────────────────────────────────")
    proposal_hit_rate()

    print("\n── Pipeline steps visualisation ─────────────────────────")
    pipeline_steps_figure()

    print("\n── Model comparison figures ─────────────────────────────")
    histories = {name: h for name in MODELS
                 if (h := load_history(name)) is not None}

    if not histories:
        print("  No history files found — skipping model comparison.")
    elif len(histories) >= 2:
        comparison_loss_curves(histories)
        comparison_accuracy_curves(histories)
        comparison_bar_chart(histories)
        model_summary_table(histories)
    else:
        print("  Only one model found — skipping comparison plots.")
        model_summary_table(histories)

    print("\n── Sequence evaluation comparison ───────────────────────")
    comparison_sequence_accuracy()
    comparison_sequence_by_length()

    print(f"\nDone. All figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
