#!/bin/bash
set -e

echo "========================================================"
echo "STARTING FAST FEEDBACK LOOP TEST"
echo "========================================================"

# 1. Create Mini Dataset
echo "[1/3] Creating mini dataset (first 2000 rows)..."
mkdir -p local_test_data
head -n 2000 training_and_preprocessing_workflows/full_clean_ml_datset.csv > local_test_data/mini_data.csv
echo "Mini dataset created at local_test_data/mini_data.csv"

# 2. Run Training (Fast Mode)
echo "[2/3] Running Training (max_steps=10)..."
python src/train_nhits.py \
    --input_csv local_test_data/mini_data.csv \
    --model_dir local_test_data/mini_model \
    --test_output_csv local_test_data/mini_test.csv \
    --full_dataset_output_csv local_test_data/mini_full.csv \
    --max_steps 10 \
    --val_check_steps 5

echo "Training completed. Model saved to local_test_data/mini_model"

# 3. Run Evaluation
echo "[3/3] Running Evaluation..."
python src/evaluate_nhits.py \
    --full_dataset_path local_test_data/mini_full.csv \
    --model_dir local_test_data/mini_model \
    --metrics_output_path local_test_data/mini_metrics.json \
    --plot_output_path local_test_data/mini_loss_plot.html \
    --prediction_plot_path local_test_data/mini_prediction_plot.html

echo "========================================================"
echo "TEST COMPLETED SUCCESSFULLY"
echo "Metrics: $(cat local_test_data/mini_metrics.json)"
echo "Prediction Plot: local_test_data/mini_prediction_plot.html"
echo "========================================================"
