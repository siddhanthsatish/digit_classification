# CNN Digit Detection & Recognition
**GT CS 6476 — Final Project**

Detects and recognizes sequences of digits in images using:
- **MSER + sliding window + image pyramid** for region proposals
- **Custom CNN** (trained from scratch, 11 classes) and **VGG16** (fine-tuned, 11 classes)
- **Non-Maximum Suppression** to merge overlapping detections
- Class 10 = background/non-digit, trained on synthetic background crops

---

## Setup

### Option A — conda (recommended, matches grading environment)
```bash
conda env create -f environment.yml
conda activate digit_classification
```

### Option B — pip
```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
```

---

## Dataset

Download SVHN Format 1 from http://ufldl.stanford.edu/housenumbers/

```bash
mkdir -p data
# Download train.tar.gz and test.tar.gz into data/, then:
cd data && tar -xzf train.tar.gz && tar -xzf test.tar.gz
```

Expected layout:
```
data/
├── train/
│   ├── 1.png … N.png
│   └── digitStruct.mat
└── test/
    ├── 1.png … N.png
    └── digitStruct.mat
```

---

## Configuration

All hyperparameters and pipeline settings live in **`config.json`** — edit this file to tune anything without touching code:

| Section | Key examples |
|---|---|
| `custom` | `epochs`, `batch_size`, `lr`, `patience`, `dropout_conv`, `filters` |
| `vgg16` | `epochs`, `batch_size`, `lr`, `backbone_lr_scale`, `freeze_features` |
| `dataset` | `val_split`, `bg_ratio`, `augment_degrees`, `color_jitter_*` |
| `inference` | `conf_threshold`, `nms_iou`, `win_size`, `sliding_step`, `pyramid_scale`, `mser_*` |

---

## Training

```bash
# Train custom CNN from scratch
python scripts/train.py --model custom

# Fine-tune VGG16 with pretrained weights
python scripts/train.py --model vgg16

# Use a different config file
python scripts/train.py --model custom --config my_config.json
```

Each run saves to `outputs/results/`:
- `{model}_curves.png` — loss & accuracy over epochs
- `{model}_confusion_matrix.png` — per-class confusion matrix
- `{model}_per_class_accuracy.png` — per-digit accuracy bar chart
- `{model}_lr_schedule.png` — learning rate decay
- `{model}_generalisation_gap.png` — train−val accuracy gap
- `{model}_classification_report.txt` — precision / recall / F1
- `{model}_history.json` — all epoch metrics + test results
- `{model}_config_used.json` — exact config snapshot for reproducibility
- `{model}_train.log` — full console output

---

## Sequence Evaluation

Evaluate sequence-level accuracy on the SVHN test set:

```bash
python scripts/eval_sequence.py --model both    # custom + vgg16
python scripts/eval_sequence.py --model custom
python scripts/eval_sequence.py --model vgg16
```

Results are saved to `outputs/results/seq_eval_{model}/` as `correct.json` and `wrong.json`.

---

## Report Figures

After training both models, generate cross-model comparison plots:

```bash
python scripts/generate_report_figures.py
```

Outputs to `outputs/report_figures/comparison/`:
- `proposal_hit_rate.png` — region proposal recall, precision, and full-coverage by sequence length
- `pipeline_steps.png` — preprocessing pipeline steps for 1 correct and 1 wrong image
- `comparison_loss.png` — train/val loss side-by-side
- `comparison_accuracy.png` — train/val accuracy side-by-side
- `comparison_bar.png` — train/val/test accuracy bar chart
- `comparison_seq_accuracy.png` — sequence accuracy + error breakdown (correct / wrong digits / wrong length / no detection)
- `comparison_seq_by_length.png` — sequence accuracy vs GT sequence length
- `model_summary.txt` — summary table

---

## Run (Grading)

```bash
python run.py
```

Reads from `demo_images/` by default and writes annotated images to `outputs/graded_images/custom/` and `outputs/graded_images/vgg16/`.

### Options
| Flag | Default | Description |
|---|---|---|
| `--model` | `both` | `custom`, `vgg16`, or `both` |
| `--config` | `config.json` | Path to config file |
| `--images_dir` | `demo_images/` | Directory of input images |
| `--conf_threshold` | *(from config)* | Override detection confidence threshold |

### Using your own images
```bash
python run.py --model vgg16 --images_dir path/to/your/images
```
Place up to 5 images in the directory. The first 5 (sorted) will be used.

---

## Notebooks

```
notebooks/
├── visualize_results.ipynb   # interactive GT vs. predicted viewer + pipeline step visualizer
└── visualize_rpn.py          # region proposal visualization
```

---

## Project Structure

```
digit_classification/
├── run.py                          # ← grader runs this
├── config.json                     # all hyperparameters and pipeline settings
├── requirements.txt
├── environment.yml
├── scripts/
│   ├── train.py                    # training script
│   ├── eval_sequence.py            # sequence-level accuracy evaluation
│   ├── generate_report_figures.py  # cross-model comparison plots
│   ├── select_demo_images.py       # demo image selection utility
│   └── orchestrate.sh              # end-to-end run script
├── src/
│   ├── data/
│   │   ├── dataset.py              # SVHN loading + synthetic background class
│   │   └── splits.py               # train/val/test split logic
│   ├── models/
│   │   ├── custom_cnn.py           # custom CNN from scratch (11 classes)
│   │   └── vgg16_finetune.py       # VGG16 fine-tuning (11 classes)
│   ├── pipeline/
│   │   ├── preprocessing.py        # denoise, CLAHE, MSER, sliding window, pyramid
│   │   ├── nms.py                  # non-maximum suppression
│   │   └── predict.py              # end-to-end inference
│   └── evaluation/
│       └── eval_utils.py           # sequence-level evaluation helpers
├── notebooks/
│   ├── visualize_results.ipynb     # interactive results viewer
│   └── visualize_rpn.py            # region proposal visualization
├── demo_images/
│   ├── demo_real/                  # real-world demo images
│   └── demo_synthetic/             # synthetic demo images
├── outputs/
│   ├── checkpoints/                # saved model weights (after training)
│   ├── results/                    # training curves, metrics, logs
│   └── graded_images/              # output images written by run.py
└── docs/
    └── LEARNING.md
```

---

## Pipeline Details

### Preprocessing
- **Gaussian blur** — suppresses high-frequency noise
- **CLAHE** — adaptive histogram equalization on the L channel (LAB space) for lighting invariance

### Region Proposals
- **MSER** — detects maximally stable extremal regions (text-like blobs)
- **Sliding window + image pyramid** — ensures scale and location invariance

### Classification (11 classes)
| Model | Weights | Output classes |
|---|---|---|
| CustomCNN | Random init | 11 (digits 0–9 + background) |
| VGG16 | ImageNet pretrained | 11 (digits 0–9 + background) |

Background class is trained on synthetic crops: between-digit patches from SVHN images, noise patches, and gradient patches (controlled by `dataset.bg_ratio` in config.json).

### Post-processing
- Background class (index 10) detections are discarded
- Per-class **NMS** with configurable IoU threshold
- Detections sorted left-to-right to assemble digit sequence
