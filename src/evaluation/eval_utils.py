"""
Shared evaluation utilities used by both eval_sequence.py and visualize_results.ipynb.

Public API
----------
build_gt_map(mat_path)
    Parse digitStruct.mat → {filename: gt_sequence_str}

run_image(img_bgr, model, device, inf_cfg)
    Run the full pipeline on one image.
    Returns (pred_seq, det_list, intermediates)

evaluate_dataset(model_name, cfg)
    Run eval on the full SVHN test set and write correct/wrong/summary JSON files.
    Returns seq_accuracy (float).
"""

import os
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import torch
from tqdm import tqdm

from src.data.dataset import parse_digit_struct
from src.pipeline.predict import load_model, detect_digits, classify_crops, DIGIT_LABELS, BG_CLASS
from src.pipeline.preprocessing import (
    normalize_polarity, denoise, normalize_lighting,
    mser_proposals, get_proposals, crop_proposal,
)


# ── GT map ────────────────────────────────────────────────────────────────────

def build_gt_map(mat_path):
    """Parse digitStruct.mat and return {filename: gt_sequence_str}."""
    records = parse_digit_struct(mat_path)
    gt_map = {}
    for rec in records:
        labels_left = sorted(zip(rec["lefts"], rec["labels"]))
        gt_map[rec["filename"]] = "".join(
            str(0 if l == 10 else l) for _, l in labels_left
        )
    return gt_map


def build_gt_boxes_map(mat_path):
    """Parse digitStruct.mat and return {filename: [(left,top,w,h,label), ...]}."""
    records = parse_digit_struct(mat_path)
    return {
        rec["filename"]: list(zip(
            rec["lefts"], rec["tops"], rec["widths"], rec["heights"], rec["labels"]
        ))
        for rec in records
    }


# ── Single-image inference ────────────────────────────────────────────────────

def run_image(img_bgr, model, device, inf_cfg):
    """
    Run the full detection pipeline on one BGR image.

    Returns
    -------
    pred_seq : str
    det_list : list of (x, y, w, h, digit_str, conf)
    intermediates : dict
    """
    step1 = normalize_polarity(img_bgr)
    step2 = denoise(step1)
    step3 = normalize_lighting(step2)

    mser_boxes = mser_proposals(
        step3,
        min_area=inf_cfg.get("mser_min_area", 100),
        max_area=inf_cfg.get("mser_max_area", 5000),
        min_aspect=inf_cfg.get("mser_min_aspect", 0.2),
        max_aspect=inf_cfg.get("mser_max_aspect", 5.0),
    )

    all_proposals = get_proposals(
        step3,
        use_mser=inf_cfg.get("use_mser", True),
        use_sliding=inf_cfg.get("use_sliding", False),
        cfg=inf_cfg,
    )

    target_size = inf_cfg.get("win_size", 32)
    crops, valid_boxes = [], []
    for box in all_proposals:
        crop = crop_proposal(step3, box, target_size=target_size)
        if crop is not None:
            crops.append(crop)
            valid_boxes.append(box)

    predictions = classify_crops(crops, model, device, inf_cfg.get("batch_size", 64))
    conf_threshold = inf_cfg.get("conf_threshold", 0.85)
    digit_boxes, digit_scores, digit_labels = [], [], []
    for box, (pred, conf) in zip(valid_boxes, predictions):
        if pred != BG_CLASS and conf >= conf_threshold:
            digit_boxes.append(box)
            digit_scores.append(conf)
            digit_labels.append(pred)

    # Compute DBSCAN cluster labels on the pre-NMS hits (for visualisation)
    import numpy as np
    dbscan_labels   = None
    dbscan_centers  = {}
    if len(digit_boxes) > 1:
        from sklearn.cluster import DBSCAN
        hit_centers = np.array(
            [[b[0] + b[2] / 2, b[1] + b[3] / 2] for b in digit_boxes], dtype=float
        )
        eps    = float(np.median([b[2] for b in digit_boxes])) * 1.5
        labels = DBSCAN(eps=eps, min_samples=2).fit_predict(hit_centers)
        dbscan_labels = labels
        cluster_ids, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(cluster_ids) > 0:
            max_count = counts.max()
            strong = [
                cid for cid, cnt in zip(cluster_ids, counts)
                if cnt > max_count * 0.5
            ]
            dbscan_centers = {
                cid: hit_centers[labels == cid].mean(axis=0)
                for cid in strong
            }

    pred_seq, det_list, _ = detect_digits(
        img_bgr, model, device,
        conf_threshold=conf_threshold,
        nms_iou=inf_cfg.get("nms_iou", 0.3),
        use_mser=inf_cfg.get("use_mser", True),
        use_sliding=inf_cfg.get("use_sliding", False),
        inf_cfg=inf_cfg,
    )

    intermediates = dict(
        step1=step1, step2=step2, step3=step3,
        mser_boxes=mser_boxes,
        all_proposals=all_proposals,
        digit_boxes=digit_boxes,
        digit_scores=digit_scores,
        digit_labels=digit_labels,
        dbscan_labels=dbscan_labels,
        dbscan_centers=dbscan_centers,
    )
    return pred_seq, det_list, intermediates


# ── Preprocessing worker (CPU-bound, runs in separate process) ───────────────

def _preprocess_and_extract_crops(args):
    """Worker: preprocess one image and extract all proposal crops."""
    filename, data_dir, inf_cfg = args
    img_path = os.path.join(data_dir, "test", filename)
    img_bgr  = cv2.imread(img_path)
    if img_bgr is None:
        return None
    
    from src.pipeline.preprocessing import preprocess_image, get_proposals, crop_proposal
    
    preprocessed = preprocess_image(img_bgr)
    proposals    = get_proposals(
        preprocessed,
        use_mser=inf_cfg.get("use_mser", True),
        use_sliding=inf_cfg.get("use_sliding", False),
        cfg=inf_cfg,
    )
    
    target_size = inf_cfg.get("win_size", 32)
    crops, valid_boxes = [], []
    for box in proposals:
        crop = crop_proposal(preprocessed, box, target_size=target_size)
        if crop is not None:
            crops.append(crop)
            valid_boxes.append(box)
    
    return filename, crops, valid_boxes


# ── Dataset evaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_dataset(model_name, cfg):
    data_dir    = cfg.get("data_dir", "data")
    inf_cfg     = cfg.get("inference", {})
    results_dir = cfg.get("results_dir", "outputs/results")
    num_workers = cfg.get("dataset", {}).get("num_workers", 8)

    out_dir = os.path.join(results_dir, f"seq_eval_{model_name}")
    os.makedirs(out_dir, exist_ok=True)

    model, device = load_model(
        model_name, ckpt_dir=cfg["ckpt_dir"],
        num_classes=cfg["num_classes"], custom_cfg=cfg.get("custom"),
    )

    gt_map = build_gt_map(os.path.join(data_dir, "test", "digitStruct.mat"))

    # Phase 1: parallel preprocessing + crop extraction (CPU-bound)
    tasks = [(fn, data_dir, inf_cfg) for fn in gt_map.keys()]
    image_data = {}  # filename -> (crops, valid_boxes)
    
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_preprocess_and_extract_crops, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{model_name} preprocess"):
            result = fut.result()
            if result:
                filename, crops, valid_boxes = result
                image_data[filename] = (crops, valid_boxes)

    # ── Proposal hit-rate (model-independent, cached per eval run) ─────────────
    gt_boxes_map = build_gt_boxes_map(os.path.join(data_dir, "test", "digitStruct.mat"))
    from src.pipeline.nms import iou as box_iou
    hit_iou = 0.3
    total_gt = 0
    total_gt_hit = 0
    total_proposals = 0
    total_prop_hit = 0
    images_all_hit = 0
    images_total = 0
    by_seq_len = {}  # seq_len -> {gt_hit, gt_total, prop_hit, prop_total, img_all_hit, img_total}
    for filename, gt_boxes in gt_boxes_map.items():
        if filename not in image_data:
            continue
        _, proposals = image_data[filename]
        seq_len = len(gt_boxes)
        if seq_len not in by_seq_len:
            by_seq_len[seq_len] = dict(gt_hit=0, gt_total=0, prop_hit=0,
                                       prop_total=0, img_all_hit=0, img_total=0)
        sl = by_seq_len[seq_len]
        images_total += 1
        sl["img_total"] += 1
        # recall: how many GT digits have a matching proposal
        img_gt_hits = 0
        for (gl, gt, gw, gh, _lbl) in gt_boxes:
            total_gt += 1
            sl["gt_total"] += 1
            if any(box_iou((gl, gt, gw, gh), p) >= hit_iou for p in proposals):
                total_gt_hit += 1
                sl["gt_hit"] += 1
                img_gt_hits += 1
        if img_gt_hits == len(gt_boxes):
            images_all_hit += 1
            sl["img_all_hit"] += 1
        # precision: how many proposals match at least one GT digit
        for p in proposals:
            total_proposals += 1
            sl["prop_total"] += 1
            if any(box_iou((gl, gt, gw, gh), p) >= hit_iou
                   for (gl, gt, gw, gh, _) in gt_boxes):
                total_prop_hit += 1
                sl["prop_hit"] += 1

    recall    = total_gt_hit   / total_gt       if total_gt       else 0.0
    precision = total_prop_hit / total_proposals if total_proposals else 0.0
    full_cov  = images_all_hit / images_total    if images_total   else 0.0
    hit_meta = {
        "iou_threshold": hit_iou,
        "recall":    {"hits": total_gt_hit,   "total": total_gt,       "rate": round(recall, 4)},
        "precision": {"hits": total_prop_hit,  "total": total_proposals, "rate": round(precision, 4)},
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
    with open(os.path.join(out_dir, "proposal_hit_rate.json"), "w") as f:
        json.dump(hit_meta, f, indent=2)
    print(f"  [{model_name}] Proposals — recall: {recall:.4f}  precision: {precision:.4f}  full-coverage: {full_cov:.4f}")

    # Phase 2: batch GPU inference on all crops
    from src.pipeline.predict import detect_digits
    
    log = []
    for filename, gt_seq in tqdm(gt_map.items(), desc=f"{model_name} inference"):
        if filename not in image_data:
            continue
        
        img_path = os.path.join(data_dir, "test", filename)
        img_bgr  = cv2.imread(img_path)
        if img_bgr is None:
            continue
        
        pred_seq, _, _ = detect_digits(
            img_bgr, model, device,
            conf_threshold=inf_cfg.get("conf_threshold", 0.85),
            nms_iou=inf_cfg.get("nms_iou", 0.3),
            use_mser=inf_cfg.get("use_mser", True),
            use_sliding=inf_cfg.get("use_sliding", False),
            inf_cfg=inf_cfg,
        )
        
        log.append({"filename": filename, "gt": gt_seq, "pred": pred_seq, "correct": pred_seq == gt_seq})

    correct = sum(e["correct"] for e in log)
    total   = len(log)
    seq_acc = correct / total if total else 0.0
    print(f"  [{model_name}] Sequence accuracy: {correct}/{total} = {seq_acc:.4f}")

    with open(os.path.join(out_dir, "correct.json"), "w") as f:
        json.dump([e for e in log if e["correct"]], f, indent=2)
    with open(os.path.join(out_dir, "wrong.json"), "w") as f:
        json.dump([e for e in log if not e["correct"]], f, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"model": model_name, "accuracy": seq_acc,
                   "correct": correct, "total": total}, f, indent=2)
    print(f"  Saved → {out_dir}/")
    return seq_acc


def _preprocess_and_extract_crops_train(args):
    """Worker: preprocess one image from train split and extract all proposal crops."""
    split_name, filename, data_dir, inf_cfg = args
    img_path = os.path.join(data_dir, split_name, filename)
    img_bgr  = cv2.imread(img_path)
    if img_bgr is None:
        return None
    
    from src.pipeline.preprocessing import preprocess_image, get_proposals, crop_proposal
    
    preprocessed = preprocess_image(img_bgr)
    proposals    = get_proposals(
        preprocessed,
        use_mser=inf_cfg.get("use_mser", True),
        use_sliding=inf_cfg.get("use_sliding", False),
        cfg=inf_cfg,
    )
    
    target_size = inf_cfg.get("win_size", 32)
    crops, valid_boxes = [], []
    for box in proposals:
        crop = crop_proposal(preprocessed, box, target_size=target_size)
        if crop is not None:
            crops.append(crop)
            valid_boxes.append(box)
    
    return split_name, filename, crops, valid_boxes


@torch.no_grad()
def evaluate_train_split(model_name, cfg):
    data_dir    = cfg.get("data_dir", "data")
    inf_cfg     = cfg.get("inference", {})
    results_dir = cfg.get("results_dir", "outputs/results")
    splits_dir  = cfg.get("splits_dir", "data/splits")
    num_workers = cfg.get("dataset", {}).get("num_workers", 8)

    out_dir = os.path.join(results_dir, f"seq_eval_{model_name}_train")
    os.makedirs(out_dir, exist_ok=True)

    model, device = load_model(
        model_name, ckpt_dir=cfg["ckpt_dir"],
        num_classes=cfg["num_classes"], custom_cfg=cfg.get("custom"),
    )

    train_txt = os.path.join(splits_dir, "train.txt")
    with open(train_txt) as f:
        rel_paths = [l.strip() for l in f if l.strip()]

    by_split = {}
    for rel in rel_paths:
        split_name, filename = rel.split("/", 1)
        by_split.setdefault(split_name, []).append(filename)

    gt_map = {}
    for split_name, filenames in by_split.items():
        split_gt      = build_gt_map(os.path.join(data_dir, split_name, "digitStruct.mat"))
        filenames_set = set(filenames)
        for fname, gt_seq in split_gt.items():
            if fname in filenames_set:
                gt_map[f"{split_name}/{fname}"] = (gt_seq, split_name, fname)

    # Phase 1: parallel preprocessing
    tasks = [(sn, fn, data_dir, inf_cfg) for _, sn, fn in gt_map.values()]
    image_data = {}  # (split_name, filename) -> (crops, valid_boxes)
    
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_preprocess_and_extract_crops_train, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{model_name} train preprocess"):
            result = fut.result()
            if result:
                split_name, filename, crops, valid_boxes = result
                image_data[(split_name, filename)] = (crops, valid_boxes)

    # Phase 2: GPU inference
    from src.pipeline.predict import detect_digits
    
    log = []
    for rel_key, (gt_seq, split_name, filename) in tqdm(gt_map.items(), desc=f"{model_name} train inference"):
        if (split_name, filename) not in image_data:
            continue
        
        img_path = os.path.join(data_dir, split_name, filename)
        img_bgr  = cv2.imread(img_path)
        if img_bgr is None:
            continue
        
        pred_seq, _, _ = detect_digits(
            img_bgr, model, device,
            conf_threshold=inf_cfg.get("conf_threshold", 0.85),
            nms_iou=inf_cfg.get("nms_iou", 0.3),
            use_mser=inf_cfg.get("use_mser", True),
            use_sliding=inf_cfg.get("use_sliding", False),
            inf_cfg=inf_cfg,
        )
        
        log.append({"filename": filename, "split": split_name, "gt": gt_seq,
                    "pred": pred_seq, "correct": pred_seq == gt_seq})

    correct   = sum(e["correct"] for e in log)
    total     = len(log)
    seq_acc   = correct / total if total else 0.0
    wrong_list = [e for e in log if not e["correct"]]
    print(f"  [{model_name}] Train split accuracy: {correct}/{total} = {seq_acc:.4f}")
    print(f"  Wrong: {len(wrong_list)}")

    with open(os.path.join(out_dir, "wrong_train.json"), "w") as f:
        json.dump(wrong_list, f, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"model": model_name, "accuracy": seq_acc,
                   "correct": correct, "total": total}, f, indent=2)
    print(f"  Saved → {out_dir}/")
    return seq_acc
