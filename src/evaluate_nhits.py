import sys
print("Starting evaluate_nhits.py script...", flush=True)

try:
    import argparse
    import pandas as pd
    import numpy as np
    import os
    import json
    import base64
    from io import BytesIO
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.numpy import mae, rmse
    print("All imports successful.", flush=True)
except ImportError as e:
    print(f"CRITICAL: Import failed: {e}", flush=True)
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL: Unexpected error during imports: {e}", flush=True)
    sys.exit(1)

def get_plot_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_loss(logs_dir):
    print(f"Plotting loss from logs in {logs_dir}...", flush=True)
    # Expected path: logs_dir/training_logs/version_0/metrics.csv
    # Note: The structure depends on how it was copied. 
    # In train_nhits.py: shutil.copytree(temp_log_dir, logs_dir)
    # temp_log_dir contained 'training_logs/version_0/metrics.csv' because name="training_logs"
    
    # Let's try to find metrics.csv recursively
    metrics_path = None
    for root, dirs, files in os.walk(logs_dir):
        if "metrics.csv" in files:
            metrics_path = os.path.join(root, "metrics.csv")
            break
            
    if not metrics_path:
        print("Warning: metrics.csv not found in logs_dir. Skipping loss plot.", flush=True)
        return None

    try:
        df = pd.read_csv(metrics_path)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by epoch to get one value per epoch if multiple steps
        # Or just plot by step
        if 'step' in df.columns:
            x_axis = 'step'
        else:
            x_axis = df.index
            
        if 'train_loss' in df.columns:
            # Filter out NaNs which might happen if validation is on different steps
            train_df = df.dropna(subset=['train_loss'])
            ax.plot(train_df[x_axis], train_df['train_loss'], label='Train Loss')
            
        if 'valid_loss' in df.columns:
            val_df = df.dropna(subset=['valid_loss'])
            ax.plot(val_df[x_axis], val_df['valid_loss'], label='Validation Loss')
            
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        img_base64 = get_plot_base64(fig)
        plt.close(fig)
        return img_base64
        
    except Exception as e:
        print(f"Error plotting loss: {e}", flush=True)
        return None

def plot_predictions(forecasts_df):
    # Plot a segment of the forecasts
    plot_df = forecasts_df.iloc[:200] # First 200 predictions

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(plot_df['ds'], plot_df['y'], label='Actual MBT', color='black', alpha=0.7)
    ax.plot(plot_df['ds'], plot_df['NHITS-median'], label='Predicted Median MBT', color='blue', linewidth=2)
    
    # Check for confidence intervals
    if 'NHITS-lo-90.0' in plot_df.columns and 'NHITS-hi-90.0' in plot_df.columns:
         ax.fill_between(plot_df['ds'], plot_df['NHITS-lo-90.0'], plot_df['NHITS-hi-90.0'], color='blue', alpha=0.2, label='90% Confidence Interval')
    elif 'NHITS-lo-80.0' in plot_df.columns and 'NHITS-hi-80.0' in plot_df.columns:
        ax.fill_between(plot_df['ds'], plot_df['NHITS-lo-80.0'], plot_df['NHITS-hi-80.0'], color='blue', alpha=0.2, label='80% Confidence Interval')
        
    ax.set_title('Subway Headway Prediction: Actual vs Predicted')
    ax.set_xlabel('Time')
    ax.set_ylabel('Minutes Between Trains (MBT)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    img_base64 = get_plot_base64(fig)
    plt.close(fig)
    return img_base64

def generate_html_report(output_path, metrics_dict, pred_plot_b64, loss_plot_b64):
    metrics_html = ""
    if metrics_dict:
        rows = ""
        for k, v in metrics_dict.items():
            rows += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
        metrics_html = f"""
        <div style="margin-bottom: 20px;">
            <h3>Metrics</h3>
            <table border="1" style="border-collapse: collapse; width: 300px;">
                <tr style="background-color: #f2f2f2;">
                    <th style="padding: 8px; text-align: left;">Metric</th>
                    <th style="padding: 8px; text-align: left;">Value</th>
                </tr>
                {rows}
            </table>
        </div>
        """
    
    loss_html = ""
    if loss_plot_b64:
        loss_html = f"""
        <div style="margin-bottom: 40px;">
            <h3>Training & Validation Loss</h3>
            <img src="data:image/png;base64,{loss_plot_b64}" alt="Loss Plot" style="max-width: 100%; border: 1px solid #ddd;">
        </div>
        """
        
    pred_html = ""
    if pred_plot_b64:
        pred_html = f"""
        <div style="margin-bottom: 40px;">
            <h3>Forecast Visualization</h3>
            <img src="data:image/png;base64,{pred_plot_b64}" alt="Prediction Plot" style="max-width: 100%; border: 1px solid #ddd;">
        </div>
        """

    html_content = f"""
    <html>
    <head>
        <title>NHITS Model Evaluation</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border: 1px solid #ddd; }}
            th, td {{ text-align: left; padding: 8px; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            h1, h3 {{ color: #333; }}
        </style>
    </head>
    <body>
        <h1>NHITS Model Evaluation Report</h1>
        {metrics_html}
        {loss_html}
        {pred_html}
    </body>
    </html>
    """
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    print(f"HTML report saved to {output_path}", flush=True)

def evaluate_nhits(model_dir, df_csv_path, metrics_output_path, html_output_path, logs_dir=None):
    print(f"Starting evaluation script...", flush=True)
    print(f"Loading full data from {df_csv_path}...", flush=True)
    df = pd.read_csv(df_csv_path)
    
    print(f"data columns: {df.columns.tolist()}", flush=True)
    print(f"data shape: {df.shape}", flush=True)
    
    # Ensure datetime
    if 'ds' in df.columns:
        df['ds'] = pd.to_datetime(df['ds'])
    
    # Ensure unique_id exists
    if 'unique_id' not in df.columns:
        df['unique_id'] = 'E'
        
    # Ensure data is sorted by time
    df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        

    print(f"Loading model from {model_dir}...", flush=True)
    try:
        nf = NeuralForecast.load(path=model_dir)
        print("Model loaded successfully.", flush=True)
            
    except Exception as e:
        print(f"Failed to load model: {e}", flush=True)
        raise e
    try:
        # Use user requested setup
        print(f"Running Cross Validation on df with horion = 1", flush=True)
        
        horizon = 1

        forecasts = nf.cross_validation(
            df=df,
            step_size=horizon,
            val_size=horizon,
            test_size=int(len(df)*0.2),
            n_windows=None
        )
        print(f"Forecasts generated. Shape: {forecasts.shape}", flush=True)

    except Exception as e:
        print(f"CRITICAL ERROR during cross_validation: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise e
    
    print("Forecasts generated. Columns:", forecasts.columns, flush=True)
    
    # Calculate Metrics manually as requested
    print("Calculating metrics...", flush=True)
    
    # Full Set Metrics
    y_true = forecasts['y']
    y_pred = forecasts['NHITS-median']
    
    # Use neuralforecast losses
    mae_val = mae(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    
    print(f"MAE: {mae_val:.4f}")
    print(f"RMSE: {rmse_val:.4f}")

    # Save Metrics (using Full Set results)
    metrics_list = [
        {
            "name": "mae",
            "numberValue": float(mae_val),
            "format": "RAW"
        },
        {
            "name": "rmse",
            "numberValue": float(rmse_val),
            "format": "RAW"
        }
    ]
    
    metrics = {"metrics": metrics_list}
    
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f)
    print(f"Metrics saved to {metrics_output_path}")

    # Generate Plots and HTML Report
    print("Generating plots...", flush=True)
    
    # 1. Prediction Plot
    pred_plot_b64 = plot_predictions(forecasts)
    
    # 2. Loss Plot (if logs available)
    loss_plot_b64 = None
    if logs_dir:
        loss_plot_b64 = plot_loss(logs_dir)
        
    # 3. Generate HTML
    metrics_dict = {"MAE": mae_val, "RMSE": rmse_val}
    generate_html_report(html_output_path, metrics_dict, pred_plot_b64, loss_plot_b64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_csv_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--metrics_output_path', type=str, required=True)
    parser.add_argument('--html_output_path', type=str, required=True)
    parser.add_argument('--logs_dir', type=str, required=False)
    
    args = parser.parse_args()
    
    evaluate_nhits(
        args.model_dir, 
        args.df_csv_path, 
        args.metrics_output_path, 
        args.html_output_path,
        args.logs_dir
    )