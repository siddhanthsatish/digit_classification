"""
Non-Maximum Suppression (NMS) to remove overlapping bounding boxes.
"""

import numpy as np


def iou(box_a, box_b):
    """Compute IoU between two boxes (x, y, w, h)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    inter_x1 = max(ax, bx);  inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw); inter_y2 = min(ay + ah, by + bh)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    return inter_area / (aw * ah + bw * bh - inter_area + 1e-6)


def is_encompassing(box_a, box_b):
    """Check if box_a fully contains box_b or vice versa."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    a_contains_b = (ax <= bx and ay <= by and ax + aw >= bx + bw and ay + ah >= by + bh)
    b_contains_a = (bx <= ax and by <= ay and bx + bw >= ax + aw and by + bh >= ay + ah)
    return a_contains_b or b_contains_a


def _iou_vectorised(box, boxes):
    """IoU between one box and an array of boxes (x, y, w, h). Returns 1-D array."""
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2])
    iy2 = np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    union = box[2] * box[3] + boxes[:, 2] * boxes[:, 3] - inter
    return inter / (union + 1e-6)


def _encompassing_vectorised(box, boxes):
    """True for each row in boxes that fully contains or is contained by box."""
    ax, ay, aw, ah = box
    bx = boxes[:, 0]; by = boxes[:, 1]; bw = boxes[:, 2]; bh = boxes[:, 3]
    a_in_b = (bx <= ax) & (by <= ay) & (bx + bw >= ax + aw) & (by + bh >= ay + ah)
    b_in_a = (ax <= bx) & (ay <= by) & (ax + aw >= bx + bw) & (ay + ah >= by + bh)
    return a_in_b | b_in_a


def nms(boxes, scores, iou_threshold=0.3, max_boxes=5):
    """
    Args:
        boxes:         list of (x, y, w, h)
        scores:        list of confidence scores (same length)
        iou_threshold: boxes with IoU > threshold are suppressed
        max_boxes:     maximum number of boxes to return

    Returns:
        kept boxes and their scores
    """
    if not boxes:
        return [], []

    boxes  = np.array(boxes,  dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    order  = scores.argsort()[::-1]
    kept   = []

    while order.size > 0 and len(kept) < max_boxes:
        i = order[0]
        if not kept or not np.any(_encompassing_vectorised(boxes[i], boxes[kept])):
            kept.append(i)
        rest = order[1:]
        suppress = _iou_vectorised(boxes[i], boxes[rest]) > iou_threshold
        order = rest[~suppress]

    kept_boxes  = [tuple(boxes[i].astype(int)) for i in kept]
    kept_scores = [float(scores[i])            for i in kept]
    return kept_boxes, kept_scores
