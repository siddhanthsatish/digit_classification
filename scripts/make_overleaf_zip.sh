#!/usr/bin/env bash
# Build overleaf_upload.zip for the final report.
# Overleaf expects report.tex + references.bib at the zip root,
# with figures reachable via \graphicspath{{./}}.
#
# Usage:  bash scripts/make_overleaf_zip.sh

set -euo pipefail

if ! command -v zip &>/dev/null; then
  sudo apt-get install -y zip
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/overleaf_upload"
ZIP="$ROOT/overleaf_upload.zip"

rm -rf "$OUT" "$ZIP"
mkdir -p "$OUT"

# ── LaTeX sources ─────────────────────────────────────────────────────────────
cp "$ROOT/docs/final_report/report.tex"      "$OUT/"
cp "$ROOT/docs/final_report/references.bib"  "$OUT/"

# ── Figures (preserve subdirectory structure matching \includegraphics paths) ─
FIGURES=(
  report_figures/comparison/pipeline_steps.png
  report_figures/comparison/comparison_loss.png
  report_figures/comparison/comparison_accuracy.png
  report_figures/comparison/comparison_bar.png
  report_figures/comparison/proposal_hit_rate.png
  report_figures/comparison/comparison_seq_accuracy.png
  report_figures/comparison/comparison_seq_by_length.png
  report_figures/exploratory/eda_class_distribution.png
  report_figures/exploratory/eda_sequence_length.png
  results/vgg16_confusion_matrix.png
  results/custom_confusion_matrix.png
  results/custom_per_class_accuracy.png
  results/vgg16_per_class_accuracy.png
  graded_images/custom/1.png
  graded_images/custom/2.png
  graded_images/custom/3.png
  graded_images/custom/4.png
  graded_images/custom/5.png
  graded_images/vgg16/1.png
  graded_images/vgg16/2.png
  graded_images/vgg16/3.png
  graded_images/vgg16/4.png
  graded_images/vgg16/5.png
)

for fig in "${FIGURES[@]}"; do
  src="$ROOT/outputs/$fig"
  dest="$OUT/$fig"
  if [ ! -f "$src" ]; then
    echo "MISSING: $src" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$dest")"
  cp "$src" "$dest"
done

# ── Zip ───────────────────────────────────────────────────────────────────────
cd "$OUT"
zip -r "$ZIP" .
cd "$ROOT"
rm -rf "$OUT"

echo "Created $ZIP ($(du -h "$ZIP" | cut -f1))"
