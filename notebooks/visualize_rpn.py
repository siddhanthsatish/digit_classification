"""
Visualize RPN outputs on demo images.

Draws:
  - Blue boxes:  all MSER proposals after geometric filter + cross-scale NMS
  - Green boxes: proposals the RPN scored above threshold (digit)
  - Red boxes:   proposals the RPN scored below threshold (background)

Output saved to graded_images/debug/rpn_<filename>.png

Usage:
    python visualize_rpn.py
    python visualize_rpn.py --images_dir demo_images --scale 4
"""

import os
import argparse
import json
import cv2
import numpy as np
import torch

from pipeline.preprocessing import (
    preprocess_image, get_proposals, geometric_filter,
    cross_scale_nms, load_rpn
)
from models.rpn import build_rpn_input


def visualize_rpn(args):
    with open(args.config) as f:
        cfg = json.load(f)
    inf_cfg = cfg["inference"]

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    rpn, _ = load_rpn(ckpt_dir=cfg["ckpt_dir"], cfg=cfg, device=device)

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    image_paths = sorted([
        os.path.join(args.images_dir, f)
        for f in os.listdir(args.images_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])[:5]

    out_dir = os.path.join("graded_images", "debug")
    os.makedirs(out_dir, exist_ok=True)

    size          = cfg["dataset"].get("image_size", 32)
    context_scale = cfg["rpn"].get("context_scale", 1.5)
    threshold     = inf_cfg.get("rpn_objectness_threshold", 0.05)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w   = img.shape[:2]
        pre    = preprocess_image(img)
        scale  = args.scale
        out    = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

        # Get proposals → geometric filter → cross-scale NMS
        proposals = get_proposals(pre, use_mser=True, use_sliding=False,
                                  use_rpn=False, cfg={**inf_cfg, "use_rpn": False})
        proposals = geometric_filter(proposals, h, w, cfg=inf_cfg)
        proposals = cross_scale_nms(proposals, iou_threshold=inf_cfg.get("nms_iou", 0.3))

        # Build RPN inputs
        inputs, valid = [], []
        for box in proposals:
            x, y, bw, bh = box
            x, y = max(0, x), max(0, y)
            tight = pre[y:y + bh, x:x + bw]
            if tight.size == 0:
                continue
            tight = cv2.resize(tight, (size, size))
            cx, cy = x + bw / 2, y + bh / 2
            nw, nh = bw * context_scale, bh * context_scale
            cx1 = max(0, int(cx - nw / 2)); cy1 = max(0, int(cy - nh / 2))
            cx2 = min(w, int(cx + nw / 2)); cy2 = min(h, int(cy + nh / 2))
            ctx = pre[cy1:cy2, cx1:cx2]
            if ctx.size == 0:
                ctx = tight
            ctx = cv2.resize(ctx, (size, size))
            inputs.append(build_rpn_input(tight, ctx, size=size))
            valid.append(box)

        # Draw all proposals in blue first
        for (x, y, bw, bh) in valid:
            cv2.rectangle(out, (x*scale, y*scale), ((x+bw)*scale, (y+bh)*scale), (255, 100, 0), 1)

        if inputs:
            with torch.no_grad():
                tensor = torch.from_numpy(np.stack(inputs)).to(device)
                scores = torch.softmax(rpn(tensor), dim=1)[:, 1].cpu().numpy()

            for (x, y, bw, bh), score in zip(valid, scores):
                x1, y1 = x * scale, y * scale
                x2, y2 = (x + bw) * scale, (y + bh) * scale
                if score >= threshold:
                    color = (0, 255, 0)   # green = digit
                else:
                    color = (0, 0, 255)   # red = background
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(out, f"{score:.2f}", (x1, max(y1 - 3, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale / 4, color, 1)

        fname   = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(out_dir, f"rpn_{fname}.png")
        cv2.imwrite(out_path, out)
        n_digit = sum(1 for s in scores if s >= threshold) if inputs else 0
        print(f"  {fname}: {len(valid)} proposals, {n_digit} above threshold={threshold} → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.json")
    parser.add_argument("--images_dir", default="demo_images")
    parser.add_argument("--scale",      type=int, default=4)
    args = parser.parse_args()
    visualize_rpn(args)
