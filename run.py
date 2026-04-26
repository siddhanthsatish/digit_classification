"""
run.py — Main entry point for grading.

Runs digit detection on up to 5 images per model and writes annotated results under
graded_images/<custom|vgg16>/.
Defaults to demo_images/ if no --images_dir is provided.

By default runs both models (each into its own subfolder). Single-model runs use the
same subfolder layout for consistency.

Usage:
    python run.py
    python run.py --model custom
    python run.py --model vgg16 --images_dir path/to/images
    python run.py --model both --conf_threshold 0.80
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
import cv2

from src.pipeline.predict import load_model, detect_digits, draw_detections


def _collect_image_paths(images_dir):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])[:5]


def run(args):
    with open(args.config) as f:
        cfg = json.load(f)
    inf_cfg = cfg["inference"]

    if args.conf_threshold is not None:
        inf_cfg["conf_threshold"] = args.conf_threshold

    images_dir = args.images_dir or "demo_images"
    if not os.path.isdir(images_dir):
        print(f"ERROR: images directory not found: {images_dir}")
        return

    image_paths = _collect_image_paths(images_dir)
    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    models_to_run = ["custom", "vgg16"] if args.model == "both" else [args.model]

    ran_any = False
    for model_name in models_to_run:
        try:
            model, device = load_model(
                model_name,
                ckpt_dir=cfg["ckpt_dir"],
                num_classes=cfg["num_classes"],
                custom_cfg=cfg.get("custom"),
            )
            print(f"\n── {model_name}  |  device: {device}")
        except FileNotFoundError:
            ckpt = os.path.join(cfg["ckpt_dir"], f"{model_name}_best.pth")
            print(
                f"WARN: Skipping {model_name}: no checkpoint at {ckpt}. "
                f"Train with: python train.py --model {model_name}"
            )
            continue

        ran_any = True
        out_dir = os.path.join("outputs/graded_images", model_name)
        os.makedirs(out_dir, exist_ok=True)

        for i, img_path in enumerate(image_paths, start=1):
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [{i}] Could not read {img_path}, skipping.")
                continue

            sequence, detections, bg_boxes = detect_digits(
                img, model, device,
                conf_threshold=inf_cfg["conf_threshold"],
                nms_iou=inf_cfg["nms_iou"],
                use_mser=inf_cfg["use_mser"],
                use_sliding=inf_cfg["use_sliding"],
                inf_cfg=inf_cfg,
            )

            annotated = draw_detections(img, detections, bg_boxes=bg_boxes)
            out_path = os.path.join(out_dir, f"{i}.png")
            cv2.imwrite(out_path, annotated)
            print(f"  [{i}] {os.path.basename(img_path):30s} → '{sequence}' "
                  f"({len(detections)} detections) → {out_path}")

    if ran_any:
        print("\nDone. Results under graded_images/<model>/")
    else:
        print("\nERROR: No model could be loaded. Train at least one model, then rerun.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["custom", "vgg16", "both"],
        default="both",
        help="Which model(s) to run; 'both' writes to graded_images/custom/ and graded_images/vgg16/",
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--images_dir",
        default=None,
        help="Directory of input images (defaults to demo_images/)",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=None,
        help="Override config.json inference.conf_threshold",
    )
    args = parser.parse_args()
    run(args)
