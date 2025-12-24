import os
import uvicorn
import logging
import torch
from fastapi import FastAPI, Request
import pandas as pd
from neuralforecast import NeuralForecast
# Explicitly import NHITS to ensure it's available for unpickling
from neuralforecast.models import NHITS
from neuralforecast.core import MODEL_FILENAME_DICT

# Patch MODEL_FILENAME_DICT to handle uppercase 'NHITS'
# The saved model seems to reference 'NHITS' but the registry has 'nhits'
if 'NHITS' not in MODEL_FILENAME_DICT:
    MODEL_FILENAME_DICT['NHITS'] = NHITS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()
model = None

@app.on_event("startup")
def load_model():
    global model
    # Vertex AI sets AIP_STORAGE_URI to the path where model artifacts are downloaded
    base_path = os.environ.get("AIP_STORAGE_URI", "nhits_model")
    
    logger.info(f"Base model path from env: {base_path}")
    
    # 1. Use the base path directly
    # We rely on NeuralForecast to handle the GCS path or local path
    actual_model_path = base_path
    
    # Debug logging for local paths only (GCS listing can be flaky with permissions)
    if not base_path.startswith("gs://") and os.path.exists(base_path):
        logger.info(f"Listing contents of {base_path}:")
        for root, dirs, files in os.walk(base_path):
            for file in files:
                full_path = os.path.join(root, file)
                logger.info(f"Found file: {full_path}")
    elif not base_path.startswith("gs://"):
        logger.warning(f"Local base path {base_path} does not exist!")

    # 2. Load the model
    logger.info(f"Attempting to load model from: {actual_model_path}")
    try:
        # NeuralForecast.load expects the directory containing the saved model
        model = NeuralForecast.load(path=actual_model_path)
        
        # PATCH: Fix broken logger paths from training environment
        # The model artifacts might contain absolute paths to temporary training directories
        # which don't exist in the serving environment. We redirect them to a new temp dir.
        import tempfile
        safe_log_dir = tempfile.mkdtemp()
        logger.info(f"Patching model loggers to use {safe_log_dir}")
        
        for i, m in enumerate(model.models):
            # Check for PyTorch Lightning logger in the model itself
            if hasattr(m, 'logger') and m.logger is not None:
                logger.info(f"Disabling logger for model {i}")
                m.logger = None
                    
            # Also check trainer if it exists
            try:
                # Accessing .trainer raises RuntimeError if not attached
                if m.trainer is not None:
                     if m.trainer.logger is not None:
                         logger.info(f"Disabling trainer logger for model {i}")
                         m.trainer.logger = None
            except RuntimeError:
                pass
            
            # Check trainer_kwargs (common in NeuralForecast)
            if hasattr(m, 'trainer_kwargs') and isinstance(m.trainer_kwargs, dict):
                if 'logger' in m.trainer_kwargs:
                    logger.info(f"Disabling logger in trainer_kwargs for model {i}")
                    m.trainer_kwargs['logger'] = None
            
            # Force CPU execution if no GPU is present
            if not torch.cuda.is_available():
                logger.info(f"No GPU detected. Forcing model {i} to use CPU.")
                
                # 1. Update direct attribute
                if hasattr(m, 'accelerator'):
                    m.accelerator = "cpu"
                
                # 2. Update trainer_kwargs (used for re-creating Trainer)
                if hasattr(m, 'trainer_kwargs') and isinstance(m.trainer_kwargs, dict):
                    logger.info(f"Overriding accelerator in trainer_kwargs to 'cpu' for model {i}")
                    m.trainer_kwargs['accelerator'] = "cpu"
                    m.trainer_kwargs['devices'] = 1
                    
                # 3. Update hparams (used for saving/loading)
                if hasattr(m, 'hparams') and hasattr(m.hparams, 'accelerator'):
                    m.hparams.accelerator = "cpu"
                elif hasattr(m, 'hparams') and isinstance(m.hparams, dict) and 'accelerator' in m.hparams:
                    m.hparams['accelerator'] = "cpu"
                    
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model from {actual_model_path}: {e}")
        # Crash the app if model fails to load so Vertex AI knows deployment failed
        raise e

@app.get("/health")
def health_check():
    # Simple health check for Vertex AI
    return {"status": "healthy"}

@app.post("/predict")
async def predict(request: Request):
    global model
    if not model:
        logger.error("Predict called but model is not loaded.")
        return {"error": "Model not loaded"}
    
    try:
        body = await request.json()
        
        # Vertex AI sends data in {"instances": [...]} format
        instances = body.get("instances")
        if not instances:
            # Fallback if raw list is sent
            instances = body
            
        if not isinstance(instances, list):
             return {"error": "Input must be a list of records or {'instances': [...]}"}

        # Convert to DataFrame
        df = pd.DataFrame(instances)
        
        # Ensure 'ds' is datetime
        if 'ds' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])
            # Strip timezone to avoid mismatches (NeuralForecast can be picky)
            if df['ds'].dt.tz is not None:
                df['ds'] = df['ds'].dt.tz_localize(None)
            
        # NeuralForecast predict requires a dataframe with history.
        # It will predict 'h' steps into the future for each unique_id found in df.
        
        # Handle Future Exogenous Variables
        # We assume the client sends History + Future rows.
        # We split the dataframe based on the model's horizon.
        
        # Get the first model (we assume only one model is loaded for serving)
        inner_model = model.models[0]
        horizon = inner_model.h
        
        # Check if model uses future exogenous variables
        uses_future_exog = hasattr(inner_model, 'futr_exog_list') and inner_model.futr_exog_list and len(inner_model.futr_exog_list) > 0
        
        if uses_future_exog:
            # Split into history and future
            # The last 'horizon' rows are treated as future
            if len(df) <= horizon:
                 return {"error": f"Input length ({len(df)}) must be greater than horizon ({horizon}) when using future exogenous variables."}
            
            hist_df = df.iloc[:-horizon].reset_index(drop=True)
            futr_df = df.tail(horizon).reset_index(drop=True)
            
            # Fix for irregular timestamps:
            # The model expects specific future timestamps based on its frequency.
            # We generate them and overwrite the 'ds' in futr_df to avoid "missing combinations" error.
            try:
                expected_futr_df = model.make_future_dataframe(df=hist_df)
                if len(futr_df) == len(expected_futr_df):
                    logger.info("Aligning future timestamps to model frequency.")
                    # Ensure alignment by unique_id if possible, but for single series direct assignment works
                    futr_df['ds'] = expected_futr_df['ds'].values
                else:
                    logger.warning(f"Mismatch in future rows. Provided: {len(futr_df)}, Expected: {len(expected_futr_df)}")
            except Exception as align_error:
                logger.warning(f"Failed to align future timestamps: {align_error}")

            logger.info(f"Predicting with Future Exog. History: {len(hist_df)}, Future: {len(futr_df)}")
            forecast = model.predict(df=hist_df, futr_df=futr_df)
        else:
            # Standard prediction
            forecast = model.predict(df=df)
        
        # Return results
        return {"predictions": forecast.to_dict(orient="records")}
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Vertex AI sets AIP_HTTP_PORT
    port = int(os.environ.get("AIP_HTTP_PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
