"""
Evaluate sequence-level accuracy on the SVHN test or train split.

Usage:
    python scripts/eval_sequence.py --model custom
    python scripts/eval_sequence.py --model vgg16
    python scripts/eval_sequence.py --model both
    python scripts/eval_sequence.py --model custom --split train
"""

import argparse
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.eval_utils import evaluate_dataset, evaluate_train_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  choices=["custom", "vgg16", "both"], default="both")
    parser.add_argument("--split",  choices=["test", "train"], default="test")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    models = ["custom", "vgg16"] if args.model == "both" else [args.model]
    for m in models:
        try:
            if args.split == "train":
                evaluate_train_split(m, cfg)
            else:
                evaluate_dataset(m, cfg)
        except FileNotFoundError as e:
            print(f"  [{m}] Skipping: {e}")

