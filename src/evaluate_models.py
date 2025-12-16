import argparse
import pandas as pd
import numpy as np
import os
import json
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from google.cloud import storage

def load_data_manual(input_path):
    print(f"Loading data from {input_path}...")
    with open(input_path) as f:
        data = f.read()
    lines = [line for line in data.split("\n") if line.strip()]
    header = lines[0].split(",")
    lines = lines[1:]
    
    mbt = np.zeros((len(lines),))
    # raw_data excludes the first column (date)
    raw_data = np.zeros((len(lines), len(header) - 1))
    
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(",")[1:]]
        mbt[i] = values[1] 
        raw_data[i, :] = values[:]
        
    return mbt, raw_data

def split_data_indices(n):
    num_train_samples = int(0.6 * n)
    num_val_samples = int(0.2 * n)
    num_test_samples = n - num_train_samples - num_val_samples
    
    test_start_idx = num_train_samples + num_val_samples
    return test_start_idx

def evaluate_gru(model_dir, raw_data, mbt, test_start_idx, sequence_length=150):
    print("Evaluating GRU Model...")
    model_path = os.path.join(model_dir, "gru_model.keras")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    # Load Model
    model = tf.keras.models.load_model(model_path)
    
    # Load Scaler
    scaler = joblib.load(scaler_path)
    train_mean = scaler['mean']
    train_std = scaler['std']
    
    # Scale Data
    raw_data_scaled = (raw_data - train_mean) / train_std
    
    # Prepare Test Data (Sliding Window)
    # We need to start 'sequence_length' steps before the test_start_idx to predict the first test point
    start_idx = test_start_idx - sequence_length
    
    if start_idx < 0:
        raise ValueError("Not enough data history for the first test point.")

    # Create dataset
    test_ds = tf.keras.utils.timeseries_dataset_from_array(
        data=raw_data_scaled[start_idx:-1], # Input up to the last point
        targets=mbt[test_start_idx:],       # Targets starting from test_start_idx
        sequence_length=sequence_length,
        sampling_rate=1,
        batch_size=128,
        shuffle=False
    )
    
    # Predict
    print("Generating predictions...")
    predictions_scaled = model.predict(test_ds, verbose=1)
    
    # Inverse Transform
    # mbt was at index 1 of the features passed to scale_data in train_gru.py
    target_col_idx = 1 
    pred_mean = train_mean[target_col_idx]
    pred_std = train_std[target_col_idx]
    
    predictions = (predictions_scaled * pred_std) + pred_mean
    
    # Get Actuals
    actuals = np.concatenate([y for x, y in test_ds], axis=0)
    
    mae = float(mean_absolute_error(actuals, predictions))
    print(f"GRU Test MAE: {mae}")
    return mae

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--metrics_output_path', type=str, required=True)
    args = parser.parse_args()

    # 1. Load Data
    mbt, raw_data = load_data_manual(args.input_csv)
    
    # 2. Determine Split
    n = len(raw_data)
    test_start_idx = split_data_indices(n)
    print(f"Total samples: {n}. Test starts at index: {test_start_idx}")
    
    # 3. Evaluate
    mae = evaluate_gru(args.model_dir, raw_data, mbt, test_start_idx)
    
    # 4. Save Metrics for Vertex AI
    metrics = {
        "metrics": [
            {
                "name": "mae",
                "numberValue": mae,
                "format": "RAW"
            }
        ]
    }
    
    with open(args.metrics_output_path, 'w') as f:
        json.dump(metrics, f)
    
    print(f"Metrics saved to {args.metrics_output_path}")
