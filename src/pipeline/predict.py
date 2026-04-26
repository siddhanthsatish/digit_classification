"""
End-to-end digit detection and recognition pipeline.

Flow:
  image → preprocess → region proposals → CNN classifier → NMS → digit sequence
"""

import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from src.pipeline.preprocessing import preprocess_image, get_proposals, crop_proposal
from src.pipeline.nms import nms

NORMALIZE = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                         std=[0.1980, 0.2010, 0.1970]),
])

DIGIT_LABELS = [str(d) for d in range(10)]   # index 0-9 → '0'-'9', index 10 = background
BG_CLASS = 10


def load_model(model_name, ckpt_dir="outputs/checkpoints", device=None, num_classes=11,
               custom_cfg=None):
    """
    Load a trained checkpoint. For ``custom``, pass ``custom_cfg`` (the ``custom``
    dict from config.json) so architecture matches training.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps"  if torch.backends.mps.is_available() else "cpu")

    if model_name == "custom":
        from src.models.custom_cnn import CustomCNN, custom_cnn_kwargs_from_cfg
        kw = custom_cnn_kwargs_from_cfg(custom_cfg or {})
        model = CustomCNN(num_classes=num_classes, **kw)
    else:
        from src.models.vgg16_finetune import build_vgg16
        model = build_vgg16(num_classes=num_classes)

    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run: python scripts/train.py --model {model_name}"
        )
    state_dict = torch.load(ckpt_path, map_location=device)
    # Strip '_orig_mod.' prefix added by torch.compile
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, device


@torch.no_grad()
def classify_crops(crops_bgr, model, device, batch_size=64):
    """
    Classify a list of BGR crops.
    Returns (predicted_class, confidence) for each crop.
    """
    results = []
    for i in range(0, len(crops_bgr), batch_size):
        batch = crops_bgr[i:i + batch_size]
        tensors = []
        for crop in batch:
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensors.append(NORMALIZE(pil))
        tensor = torch.stack(tensors).to(device)
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        for pred, conf in zip(preds.cpu().tolist(), confs.cpu().tolist()):
            results.append((pred, conf))
    return results


def detect_digits(img_bgr, model, device,
                  conf_threshold=0.85,
                  nms_iou=0.3,
                  use_mser=True,
                  use_sliding=True,
                  inf_cfg=None):
    """
    Full detection pipeline on a single BGR image.

    Returns:
        digit_sequence (str): left-to-right ordered digit string
        detections (list):    [(x, y, w, h, digit, confidence), ...]
    """
    if inf_cfg is None:
        inf_cfg = {}

    preprocessed = preprocess_image(img_bgr)
    proposals    = get_proposals(
        preprocessed,
        use_mser=use_mser,
        use_sliding=use_sliding,
        cfg=inf_cfg,
    )

    if not proposals:
        return "", [], []

    target_size = inf_cfg.get("win_size", 32)
    crops, valid_boxes = [], []
    for box in proposals:
        crop = crop_proposal(preprocessed, box, target_size=target_size)
        if crop is not None:
            crops.append(crop)
            valid_boxes.append(box)

    if not crops:
        return "", [], []

    batch_size  = inf_cfg.get("batch_size", 64)
    predictions = classify_crops(crops, model, device, batch_size=batch_size)

    # Filter: discard background class and low-confidence predictions
    digit_boxes, digit_scores, digit_labels = [], [], []
    bg_boxes = []
    for box, (pred, conf) in zip(valid_boxes, predictions):
        if pred != BG_CLASS and conf >= conf_threshold:
            digit_boxes.append(box)
            digit_scores.append(conf)
            digit_labels.append(pred)
        else:
            bg_boxes.append(box)

    if not digit_boxes:
        return "", [], bg_boxes

    # Per-class NMS
    max_boxes = inf_cfg.get("max_boxes", 5)
    final_boxes, final_scores, final_labels = [], [], []
    for cls in range(10):
        cls_boxes  = [b for b, l in zip(digit_boxes, digit_labels) if l == cls]
        cls_scores = [s for s, l in zip(digit_scores, digit_labels) if l == cls]
        if cls_boxes:
            kept_b, kept_s = nms(cls_boxes, cls_scores, iou_threshold=nms_iou, max_boxes=max_boxes)
            final_boxes.extend(kept_b)
            final_scores.extend(kept_s)
            final_labels.extend([cls] * len(kept_b))

    # Cross-class encompassing + overlap filter
    from src.pipeline.nms import is_encompassing, iou as _iou

    def _contained(small, large):
        """True if small box is mostly inside large box (>50% of small area)."""
        sx, sy, sw, sh = small
        lx, ly, lw, lh = large
        ix1 = max(sx, lx); iy1 = max(sy, ly)
        ix2 = min(sx+sw, lx+lw); iy2 = min(sy+sh, ly+lh)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        return inter / (sw * sh + 1e-6) > 0.5

    combined = sorted(zip(final_scores, final_boxes, final_labels), reverse=True)
    kept = []
    for score, box, lbl in combined:
        if not any(
            is_encompassing(box, k_box)
            or _iou(box, k_box) > nms_iou
            or _contained(box, k_box)
            for _, k_box, _ in kept
        ):
            kept.append((score, box, lbl))
    final_scores = [s for s, _, _ in kept]
    final_boxes  = [b for _, b, _ in kept]
    final_labels = [l for _, _, l in kept]

    # Global cap
    if len(final_boxes) > max_boxes:
        top = sorted(zip(final_scores, final_boxes, final_labels), reverse=True)[:max_boxes]
        final_scores, final_boxes, final_labels = zip(*top)
        final_boxes, final_scores, final_labels = list(final_boxes), list(final_scores), list(final_labels)

    # ── OLD cluster filter (median-based, x+y axes) — kept for reference ──────
    # if len(final_boxes) > 1:
    #     cx_arr   = np.array([b[0] + b[2] / 2 for b in final_boxes], dtype=float)
    #     cy_arr   = np.array([b[1] + b[3] / 2 for b in final_boxes], dtype=float)
    #     x_thresh = float(np.median([b[2] for b in final_boxes])) * 2.0
    #     y_thresh = float(np.median([b[3] for b in final_boxes])) * 0.5
    #     median_cx = float(np.median(cx_arr))
    #     median_cy = float(np.median(cy_arr))
    #     keep_nn = [
    #         i for i, (cx, cy) in enumerate(zip(cx_arr, cy_arr))
    #         if abs(cx - median_cx) <= x_thresh and abs(cy - median_cy) <= y_thresh
    #     ]
    #     if keep_nn:
    #         final_boxes  = [final_boxes[i]  for i in keep_nn]
    #         final_scores = [final_scores[i] for i in keep_nn]
    #         final_labels = [final_labels[i] for i in keep_nn]

    # ── DBSCAN cluster filter ─────────────────────────────────────────────────
    # Cluster the pre-NMS digit hits in 2D (cx, cy) — digit-class agnostic.
    # eps = median box width * 1.5 (adapts to digit size in the image).
    # Only clusters whose hit count >= 30% of the dominant cluster survive.
    # Final boxes are kept only if their center falls within eps of a
    # surviving cluster center.
    if len(final_boxes) > 1 and len(digit_boxes) > 1:
        from sklearn.cluster import DBSCAN

        hit_centers = np.array(
            [[b[0] + b[2] / 2, b[1] + b[3] / 2] for b in digit_boxes],
            dtype=float
        )
        eps         = float(np.median([b[2] for b in digit_boxes])) * 1.5
        min_samples = 2

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(hit_centers)

        cluster_ids, counts = np.unique(labels[labels >= 0], return_counts=True)

        if len(cluster_ids) > 0:
            max_count = counts.max()
            # Keep clusters with > 50% of dominant cluster's hits, up to max_boxes
            strong = [
                cid for cid, cnt in sorted(
                    zip(cluster_ids, counts), key=lambda x: -x[1]
                )[:max_boxes]
                if cnt > max_count * 0.5
            ]
            if strong:
                cluster_centers = {
                    cid: hit_centers[labels == cid].mean(axis=0)
                    for cid in strong
                }
                keep_db = [
                    i for i, box in enumerate(final_boxes)
                    if any(
                        np.hypot(
                            box[0] + box[2] / 2 - cc[0],
                            box[1] + box[3] / 2 - cc[1]
                        ) <= eps
                        for cc in cluster_centers.values()
                    )
                ]
                if keep_db:
                    final_boxes  = [final_boxes[i]  for i in keep_db]
                    final_scores = [final_scores[i] for i in keep_db]
                    final_labels = [final_labels[i] for i in keep_db]

    # Sort left-to-right by x coordinate
    detections = sorted(
        zip(final_boxes, final_labels, final_scores),
        key=lambda d: d[0][0]
    )

    digit_sequence = "".join(DIGIT_LABELS[lbl] for _, lbl, _ in detections)
    det_list = [(x, y, w, h, DIGIT_LABELS[lbl], conf)
                for (x, y, w, h), lbl, conf in detections]

    return digit_sequence, det_list, bg_boxes


def draw_detections(img_bgr, detections, bg_boxes=None):
    """Draw bounding boxes and labels on a copy of the image."""
    out = img_bgr.copy()
    for (x, y, w, h, digit, conf) in detections:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(out, digit, (x, max(y - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out
