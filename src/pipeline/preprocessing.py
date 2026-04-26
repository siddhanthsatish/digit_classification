"""
Preprocessing pipeline:
  1. Noise reduction (Gaussian blur)
  2. Lighting normalization (CLAHE)
  3. Region proposal via MSER + sliding window over image pyramid

All tunable parameters are passed via the `cfg` dict (from config.json "inference").
"""

import cv2
import numpy as np


# ── Noise & lighting ──────────────────────────────────────────────────────────

def denoise(img_bgr):
    """Mild Gaussian blur to suppress high-frequency noise."""
    return cv2.GaussianBlur(img_bgr, (3, 3), 0)


def normalize_lighting(img_bgr):
    """CLAHE on the L channel of LAB to handle uneven illumination."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def normalize_polarity(img_bgr):
    """
    Invert image if background is light (dominant brightness > 127).
    Ensures digits are always bright on dark background for MSER.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if np.median(gray) > 127:
        img_bgr = cv2.bitwise_not(img_bgr)
    return img_bgr


def preprocess_image(img_bgr):
    """Full preprocessing: polarity normalization → denoise → lighting normalization."""
    img = normalize_polarity(img_bgr)
    img = denoise(img)
    img = normalize_lighting(img)
    return img


# ── MSER region proposals ─────────────────────────────────────────────────────

def mser_proposals(img_bgr, min_area=100, max_area=5000,
                   min_aspect=0.2, max_aspect=5.0):
    """
    Detect candidate digit regions using MSER on both the image and its inverse.
    Returns list of (x, y, w, h) bounding boxes.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create(min_area=min_area, max_area=max_area)

    boxes = []
    for g in [gray, cv2.bitwise_not(gray)]:
        regions, _ = mser.detectRegions(g)
        for pts in regions:
            x, y, w, h = cv2.boundingRect(pts)
            aspect = w / (h + 1e-6)
            if min_aspect < aspect < max_aspect:
                boxes.append((x, y, w, h))
    return boxes


# ── Image pyramid ─────────────────────────────────────────────────────────────

def build_pyramid(img_bgr, min_size=32, scale=0.75):
    """
    Yield (scale_factor, resized_image) pairs from original down to min_size.
    """
    factor  = 1.0
    current = img_bgr.copy()
    while True:
        yield factor, current
        new_w = int(current.shape[1] * scale)
        new_h = int(current.shape[0] * scale)
        if new_w < min_size or new_h < min_size:
            break
        current = cv2.resize(current, (new_w, new_h))
        factor *= scale


# ── Sliding window ────────────────────────────────────────────────────────────

def sliding_window(img_bgr, win_size=32, step=16):
    """Yield (x, y, window_crop) for every window position."""
    h, w = img_bgr.shape[:2]
    for y in range(0, h - win_size + 1, step):
        for x in range(0, w - win_size + 1, step):
            yield x, y, img_bgr[y:y + win_size, x:x + win_size]


# ── Combined region proposals ─────────────────────────────────────────────────

def _dedup_proposals(proposals, iou_threshold=0.8):
    """Remove near-duplicate proposals (IoU > threshold), keeping first occurrence."""
    if not proposals:
        return proposals
    from src.pipeline.nms import _iou_vectorised
    boxes = np.array(proposals, dtype=np.float32)
    kept  = []
    for i, box in enumerate(boxes):
        if not kept or not np.any(_iou_vectorised(box, boxes[kept]) > iou_threshold):
            kept.append(i)
    return [proposals[i] for i in kept]


def get_proposals(img_bgr, use_mser=True, use_sliding=True, cfg=None):
    """
    Combine MSER and sliding-window proposals across an image pyramid.
    Returns list of (x, y, w, h) in original image coordinates.

    Args:
        cfg: dict from config.json "inference" section. Recognised keys:
             win_size, sliding_step, pyramid_scale, mser_min_area,
             mser_max_area, mser_min_aspect, mser_max_aspect
    """
    if cfg is None:
        cfg = {}

    win_size      = cfg.get("win_size", 32)
    step          = cfg.get("sliding_step", 16)
    pyramid_scale = cfg.get("pyramid_scale", 0.75)
    mser_min_area = cfg.get("mser_min_area", 100)
    mser_max_area = cfg.get("mser_max_area", 5000)
    mser_min_asp  = cfg.get("mser_min_aspect", 0.2)
    mser_max_asp  = cfg.get("mser_max_aspect", 5.0)

    proposals = []

    for scale_factor, scaled_img in build_pyramid(img_bgr, min_size=win_size,
                                                   scale=pyramid_scale):
        if use_mser:
            for (x, y, w, h) in mser_proposals(
                scaled_img,
                min_area=mser_min_area, max_area=mser_max_area,
                min_aspect=mser_min_asp, max_aspect=mser_max_asp,
            ):
                proposals.append((
                    int(x / scale_factor), int(y / scale_factor),
                    int(w / scale_factor), int(h / scale_factor),
                ))

        if use_sliding:
            for (x, y, _) in sliding_window(scaled_img, win_size=win_size, step=step):
                proposals.append((
                    int(x / scale_factor), int(y / scale_factor),
                    int(win_size / scale_factor), int(win_size / scale_factor),
                ))

    return _dedup_proposals(proposals, iou_threshold=0.8)


def crop_proposal(img_bgr, box, target_size=32):
    """Crop and resize a proposal box to target_size x target_size."""
    x, y, w, h = box
    x, y = max(0, x), max(0, y)
    crop = img_bgr[y:y + h, x:x + w]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (target_size, target_size))
