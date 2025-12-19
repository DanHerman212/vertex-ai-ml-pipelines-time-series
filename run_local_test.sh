#!/bin/bash
set -e

# Set environment variables to prevent macOS crashes
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "========================================================"
echo "STARTING FAST FEEDBACK LOOP TEST"
echo "========================================================"

# 1. Create Mini Dataset
echo "[1/3] Creating mini dataset (first 2000 rows)..."
mkdir -p local_test_data
# Use preproc.csv as source since full_clean_ml_datset.csv might not be available or correct
head -n 2000 training_and_preprocessing_workflows/preproc.csv > local_test_data/mini_data.csv
echo "Mini dataset created at local_test_data/mini_data.csv"

# 2. Run Training (Fast Mode)
echo "[2/3] Running Training (max_steps=10)..."
python src/train_nhits.py \
    --input_csv local_test_data/mini_data.csv \
    --model_dir local_test_data/mini_model \
    --test_output_csv local_test_data/mini_test.csv \
    --max_steps 10

echo "Training completed. Model saved to local_test_data/mini_model"

# 3. Run Evaluation
echo "[3/3] Running Evaluation..."
python src/evaluate_nhits.py \
    --test_dataset_path local_test_data/mini_data.csv \
    --model_dir local_test_data/mini_model \
    --metrics_output_path local_test_data/mini_metrics.json \
    --plot_output_path local_test_data/mini_loss_plot.html \
    --prediction_plot_path local_test_data/mini_prediction_plot.html

echo "========================================================"
echo "TEST COMPLETED SUCCESSFULLY"
echo "Metrics: $(cat local_test_data/mini_metrics.json)"
echo "Prediction Plot: local_test_data/mini_prediction_plot.html"
echo "========================================================"
