"""
select_demo_images.py

Scans the SVHN test set and picks 5 real street-view images that best
demonstrate the 4 grading criteria:
  1. Scale invariance      — image where digits are large relative to frame
  2. Orientation           — image where the house number is at an angle
  3. Location invariance   — digit sequence in a corner / edge of the image
  4. Lighting              — dark or high-contrast image

Only images that BOTH models (custom + vgg16) predicted correctly are
considered, so the demo always shows successful detections.

Copies selected images to demo_images/ as:
  demo1_scale.png
  demo2_orientation.png
  demo3_location.png
  demo4_lighting.png
  demo5_noise.png   (noisiest / most cluttered image)
"""

import os
import shutil
import h5py
import numpy as np
from PIL import Image

DATA_DIR    = "data/test"
OUT_DIR     = "demo_images/demo_real"
MAT_PATH    = os.path.join(DATA_DIR, "digitStruct.mat")
RESULTS_DIR = "outputs/results"


def read_attr(f, obj, key):
    val = obj[key]
    if val.shape == (1, 1):
        return [int(val[0][0])]
    return [int(f[val[j][0]][0][0]) for j in range(val.shape[0])]


def parse(mat_path, max_images=500):
    records = []
    with h5py.File(mat_path, "r") as f:
        ds = f["digitStruct"]
        n  = min(len(ds["name"]), max_images)
        for i in range(n):
            name     = "".join(chr(c) for c in f[ds["name"][i][0]][:].flatten())
            bbox     = f[ds["bbox"][i][0]]
            labels   = read_attr(f, bbox, "label")
            labels   = [0 if l == 10 else l for l in labels]
            tops     = read_attr(f, bbox, "top")
            lefts    = read_attr(f, bbox, "left")
            heights  = read_attr(f, bbox, "height")
            widths   = read_attr(f, bbox, "width")
            records.append(dict(name=name, labels=labels,
                                tops=tops, lefts=lefts,
                                heights=heights, widths=widths))
    return records


def image_stats(img_path, rec):
    img  = np.array(Image.open(img_path).convert("RGB"))
    h, w = img.shape[:2]

    # digit bounding box union
    all_tops    = rec["tops"]
    all_lefts   = rec["lefts"]
    all_heights = rec["heights"]
    all_widths  = rec["widths"]

    min_y = max(0, min(all_tops))
    min_x = max(0, min(all_lefts))
    max_y = min(h, max(t + ht for t, ht in zip(all_tops, all_heights)))
    max_x = min(w, max(l + wd for l, wd in zip(all_lefts, all_widths)))

    digit_h = max_y - min_y
    digit_w = max_x - min_x

    # scale score: digit region relative to image size
    scale_score = (digit_h / h) * (digit_w / w)

    # location score: how close to a corner/edge
    cx = (min_x + max_x) / 2 / w
    cy = (min_y + max_y) / 2 / h
    edge_score = min(cx, 1 - cx, cy, 1 - cy)   # small = near edge

    # brightness score: mean luminance (low = dark image)
    brightness = img.mean() / 255.0

    # noise/clutter score: std of pixel values
    noise_score = img.std() / 255.0

    return dict(
        scale_score=scale_score,
        edge_score=edge_score,
        brightness=brightness,
        noise_score=noise_score,
        img_h=h, img_w=w,
        digit_h=digit_h, digit_w=digit_w,
    )


def _load_correct_filenames():
    """Return set of filenames that BOTH models predicted correctly."""
    import json
    sets = []
    for model in ("custom", "vgg16"):
        path = os.path.join(RESULTS_DIR, f"seq_eval_{model}", "correct.json")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping filter for {model}")
            continue
        with open(path) as f:
            sets.append({e["filename"] for e in json.load(f)})
    if not sets:
        return None
    return set.intersection(*sets)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Parsing {MAT_PATH} ...")
    records = parse(MAT_PATH, max_images=1000)

    both_correct = _load_correct_filenames()
    if both_correct is not None:
        print(f"Images correct in both models: {len(both_correct)}")
    else:
        print("  No seq eval results found — selecting without model filter.")

    candidates = []
    for rec in records:
        if both_correct is not None and rec["name"] not in both_correct:
            continue
        img_path = os.path.join(DATA_DIR, rec["name"])
        if not os.path.exists(img_path):
            continue
        try:
            stats = image_stats(img_path, rec)
        except Exception:
            continue
        candidates.append((img_path, rec, stats))

    print(f"Scored {len(candidates)} candidates.")

    # ── 1. Scale: largest digit region relative to image ─────────────────────
    scale_img = max(candidates, key=lambda x: x[2]["scale_score"])
    print(f"Scale:       {scale_img[0]}  score={scale_img[2]['scale_score']:.3f}  "
          f"digits={scale_img[1]['labels']}")

    # ── 2. Orientation: use an image where digits are tall and narrow ─────────
    # proxy: digit_h >> digit_w suggests a vertical/angled layout
    orient_img = max(
        [c for c in candidates if c[0] != scale_img[0]],
        key=lambda x: x[2]["digit_h"] / max(x[2]["digit_w"], 1)
    )
    print(f"Orientation: {orient_img[0]}  "
          f"h/w={orient_img[2]['digit_h']}/{orient_img[2]['digit_w']}  "
          f"digits={orient_img[1]['labels']}")

    # ── 3. Location: digit sequence closest to an edge ────────────────────────
    used = {scale_img[0], orient_img[0]}
    loc_img = min(
        [c for c in candidates if c[0] not in used],
        key=lambda x: x[2]["edge_score"]
    )
    print(f"Location:    {loc_img[0]}  edge_score={loc_img[2]['edge_score']:.3f}  "
          f"digits={loc_img[1]['labels']}")

    # ── 4. Lighting: darkest image ────────────────────────────────────────────
    used.add(loc_img[0])
    light_img = min(
        [c for c in candidates if c[0] not in used],
        key=lambda x: x[2]["brightness"]
    )
    print(f"Lighting:    {light_img[0]}  brightness={light_img[2]['brightness']:.3f}  "
          f"digits={light_img[1]['labels']}")

    # ── 5. Noise: highest pixel std ───────────────────────────────────────────
    used.add(light_img[0])
    noise_img = max(
        [c for c in candidates if c[0] not in used],
        key=lambda x: x[2]["noise_score"]
    )
    print(f"Noise:       {noise_img[0]}  noise={noise_img[2]['noise_score']:.3f}  "
          f"digits={noise_img[1]['labels']}")

    # ── Copy to demo_images/ ──────────────────────────────────────────────────
    mapping = [
        (scale_img[0],  "demo1_scale.png"),
        (orient_img[0], "demo2_orientation.png"),
        (loc_img[0],    "demo3_location.png"),
        (light_img[0],  "demo4_lighting.png"),
        (noise_img[0],  "demo5_noise.png"),
    ]
    for src, dst in mapping:
        dst_path = os.path.join(OUT_DIR, dst)
        shutil.copy2(src, dst_path)
        print(f"  Copied {src} → {dst_path}")

    print("\nDone. Run: python run.py")


if __name__ == "__main__":
    main()
