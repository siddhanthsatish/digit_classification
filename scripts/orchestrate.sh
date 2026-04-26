#!/bin/bash
# train_all.sh — Train both models and generate all report figures.
# Usage: bash train_all.sh

set -e  # exit immediately on any error

echo "=== Training CustomCNN ==="
python train.py --model custom

echo ""
echo "=== Training VGG16 ==="
python train.py --model vgg16

echo ""
echo "=== Generating report figures ==="
python generate_report_figures.py

echo ""
echo "=== All done. Results in results/ ==="
