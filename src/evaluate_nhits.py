import argparse
import pandas as pd
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from neuralforecast import NeuralForecast
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, rmse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_nhits(model_dir, test_csv_path, metrics_output_path, plot_output_path, prediction_plot_path):
    logger.info("Starting evaluation script...")
    
    # 1. Load Test Data
    logger.info(f"Loading test data from {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path)
    
    if 'ds' in test_df.columns:
        test_df['ds'] = pd.to_datetime(test_df['ds'])
        if test_df['ds'].dt.tz is not None:
            test_df['ds'] = test_df['ds'].dt.tz_localize(None)
            
    if 'unique_id' not in test_df.columns:
        test_df['unique_id'] = 'E'

    # 2. Load Model
    logger.info(f"Loading model from {model_dir}...")
    nf = NeuralForecast.load(path=model_dir)
    
    # 3. Configure Model for Inference (Disable Early Stopping & Validation)
    logger.info("Configuring model for inference (disabling early stopping)...")
    for model in nf.models:
        # Disable Early Stopping
        model.early_stop_patience_steps = None
        
        # Disable Validation Checks
        model.val_check_steps = 1_000_000 
        model.limit_val_batches = 0
        
        # Clear callbacks
        if hasattr(model, 'callbacks'):
            model.callbacks = []

    # 4. Determine Window Size
    input_size = nf.models[0].input_size
    logger.info(f"Model input_size: {input_size}")
    
    dataset_len = len(test_df)
    n_windows = dataset_len - input_size
    
    logger.info(f"Dataset length: {dataset_len}")
    logger.info(f"Calculated n_windows: {n_windows} (len - input_size)")
    
    if n_windows <= 0:
        raise ValueError(f"Test dataset is too small ({dataset_len}) for input_size ({input_size}).")

    # 5. Generate Predictions
    logger.info(f"Generating predictions using cross_validation (n_windows={n_windows}, refit=False)...")
    
    forecasts = nf.cross_validation(
        df=test_df,
        n_windows=n_windows,
        step_size=1,
        refit=False
    )
    
    logger.info(f"Forecasts generated. Shape: {forecasts.shape}")
    
    # 6. Calculate Metrics
    logger.info("Calculating metrics...")
    
    pred_cols = [c for c in forecasts.columns if c.startswith('NHITS') and 'lo' not in c and 'hi' not in c]
    if not pred_cols:
        raise ValueError("Could not find prediction column (NHITS...)")
    pred_col = pred_cols[0]
    logger.info(f"Using prediction column: {pred_col}")
    
    eval_df = forecasts.copy()
    if pred_col != 'NHITS':
        eval_df = eval_df.rename(columns={pred_col: 'NHITS'})
        
    evaluation_df = evaluate(
        eval_df,
        metrics=[mae, rmse],
        models=['NHITS'],
        target_col='y',
        id_col='unique_id',
        time_col='ds'
    )
    
    logger.info(f"Evaluation results:\n{evaluation_df}")
    
    if 'mae' in evaluation_df.columns:
        mae_val = evaluation_df['mae'].iloc[0]
        rmse_val = evaluation_df['rmse'].iloc[0]
    else:
        mae_val = evaluation_df[evaluation_df['metric']=='mae']['NHITS'].values[0]
        rmse_val = evaluation_df[evaluation_df['metric']=='rmse']['NHITS'].values[0]

    # Save Metrics JSON
    metrics_data = {
        "metrics": [
            {"name": "mae", "numberValue": float(mae_val), "format": "RAW"},
            {"name": "rmse", "numberValue": float(rmse_val), "format": "RAW"}
        ]
    }
    
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics_data, f)
    logger.info(f"Metrics saved to {metrics_output_path}")

    # 7. Plotting
    if prediction_plot_path:
        logger.info(f"Generating plot to {prediction_plot_path}...")
        
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('seaborn-whitegrid')
            
        plt.figure(figsize=(16, 6))
        
        plot_data = forecasts
        
        plt.plot(plot_data['ds'], plot_data['y'], label='Actual MBT', color='black', linewidth=1.5, alpha=0.7)
        plt.plot(plot_data['ds'], plot_data[pred_col], label='Predicted Median MBT', color='blue', linewidth=2.0)
        
        lo_cols = [c for c in forecasts.columns if 'lo' in c]
        hi_cols = [c for c in forecasts.columns if 'hi' in c]
        
        if lo_cols and hi_cols:
            lo_cols.sort()
            hi_cols.sort()
            lo_col = lo_cols[0]
            hi_col = hi_cols[0]
            
            lo_80 = [c for c in lo_cols if '80' in c]
            hi_80 = [c for c in hi_cols if '80' in c]
            if lo_80 and hi_80:
                lo_col = lo_80[0]
                hi_col = hi_80[0]
            
            plt.fill_between(plot_data['ds'], plot_data[lo_col], plot_data[hi_col], color='blue', alpha=0.2, label='Confidence Interval')
            
        plt.title('Subway Headway Prediction: Actual vs Predicted (with Uncertainty)', fontsize=14, pad=15)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Minutes Between Trains (MBT)', fontsize=12)
        plt.legend(loc='upper right', frameon=True, framealpha=0.9)
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        
        import base64
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        html_content = f"""
        <html>
        <body>
            <div style="text-align: center;">
                <img src="data:image/png;base64,{img_base64}" style="max-width: 100%; height: auto;">
            </div>
        </body>
        </html>
        """
        
        os.makedirs(os.path.dirname(prediction_plot_path), exist_ok=True)
        with open(prediction_plot_path, 'w') as f:
            f.write(html_content)
            
    if plot_output_path:
        with open(plot_output_path, 'w') as f:
            f.write("<html><body><h3>Training loss not available during evaluation</h3></body></html>")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--metrics_output_path', type=str, required=True)
    parser.add_argument('--plot_output_path', type=str, default=None)
    parser.add_argument('--prediction_plot_path', type=str, default=None)
    
    args = parser.parse_args()
    
    evaluate_nhits(
        args.model_dir,
        args.test_dataset_path,
        args.metrics_output_path,
        args.plot_output_path,
        args.prediction_plot_path
    )
