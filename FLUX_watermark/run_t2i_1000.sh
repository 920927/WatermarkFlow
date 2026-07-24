#!/usr/bin/env bash
# FLUX 文生图 1000 张：生成干净图+水印图 → 鲁棒性 → 图像质量
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 请先激活已安装依赖的 Python 环境，例如:
# source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate watermarkflow

CLEAN_DIR="./FLUX_watermark/output_flux_t2i_1000_clean"
WM_DIR="./FLUX_watermark/output_flux_t2i_1000"
LOG_DIR="./FLUX_watermark/logs_t2i_1000"
mkdir -p "$LOG_DIR"

echo "[$(date)] ===== Phase 1: parallel generation (clean@GPU0, wm@GPU1) ====="

python FLUX_watermark/generate_t2i_batch.py \
  --task clean --img_num 1000 --gpu_ids 0 \
  --clean_dir "$CLEAN_DIR" \
  2>&1 | tee "$LOG_DIR/generate_clean.log" &
PID_CLEAN=$!

python FLUX_watermark/generate_t2i_batch.py \
  --task wm --img_num 1000 --gpu_ids 1 \
  --wm_dir "$WM_DIR" \
  2>&1 | tee "$LOG_DIR/generate_wm.log" &
PID_WM=$!

wait $PID_CLEAN
echo "[$(date)] Clean generation finished."
wait $PID_WM
echo "[$(date)] WM generation finished."

echo "[$(date)] ===== Phase 2: robustness evaluation ====="
python FLUX_watermark/text_to_image.py \
  --eval_only --img_num 1000 --gpu_ids 1 \
  --output_dir "$WM_DIR" --clean_dir "$CLEAN_DIR" \
  --metric both --attacks traditional \
  2>&1 | tee "$LOG_DIR/robustness.log"

echo "[$(date)] ===== Phase 3: image quality (CLIP/FID/IS) ====="
python t2i_quality.py \
  --wm_dir "$WM_DIR" --real_dir "$CLEAN_DIR" --metrics all \
  2>&1 | tee "$LOG_DIR/quality.log"

echo "[$(date)] ===== ALL DONE ====="
echo "Clean: $CLEAN_DIR"
echo "WM:    $WM_DIR"
echo "Logs:  $LOG_DIR"
