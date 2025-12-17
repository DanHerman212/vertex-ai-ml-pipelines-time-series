# Model Training Pipeline: Hard Blockers & Resolutions Summary

This document summarizes the critical technical blockers encountered during the development and deployment of the GRU training pipeline on Vertex AI, along with their resolutions.

## 1. Evaluation Component (`src/evaluate_models.py`)

### Blocker 1: Data Type Mismatch (`float64` vs `float32`)
*   **Symptom:** `ValueError: Tensor conversion requested dtype float32 for Tensor with dtype float64`.
*   **Root Cause:** The input data loaded from CSV was in `float64` (Pandas default), but the TensorFlow model expected `float32`.
*   **Resolution:** Implemented explicit type casting in the inference wrapper: `input_tensor = tf.cast(input_data, dtype=tf.float32)`.

### Blocker 2: Model Signature Mismatch (Positional vs. Keyword Args)
*   **Symptom:** `TypeError` indicating unexpected keyword arguments or signature mismatches during inference.
*   **Root Cause:** The `SavedModel` format exported by Keras often requires specific input keys (e.g., `input_1`) rather than positional arguments, depending on how the signature was defined.
*   **Resolution:** Added dynamic inspection of `inference_func.structured_input_signature` to detect whether the model expects positional arguments or keyword arguments and format the input accordingly.

### Blocker 3: Missing CuDNN Kernels on CPU
*   **Symptom:** `No OpKernel was registered to support Op 'CudnnRNNV3'`.
*   **Root Cause:** The GRU model was trained on a GPU using optimized CuDNN kernels (`CudnnRNNV3`). These kernels are hardware-specific and cannot run on a standard CPU environment.
*   **Resolution:** Updated `pipeline.py` to assign an `NVIDIA_TESLA_T4` GPU to the `evaluate_model_component`, ensuring the hardware matched the model's requirements.

### Blocker 4: Incorrect MAE Calculation (Inverse Transform Logic)
*   **Symptom:** Test MAE was ~42, whereas the expected range was ~2.5.
*   **Root Cause:** The evaluation script applied an "inverse scaling" transformation `(pred * std) + mean` to the predictions. However, the model was trained to predict the **raw** target value directly (only inputs were scaled), so the output was already in the correct unit. The transformation inflated the error.
*   **Resolution:** Removed the inverse transformation logic for the target variable in `evaluate_models.py`.

### Blocker 5: Plotting Failure in Headless Environment
*   **Symptom:** The evaluation component failed during the plotting step.
*   **Root Cause:** `matplotlib` defaults to an interactive backend (like TkAgg or X11) which requires a display. Docker containers are headless.
*   **Resolution:** Explicitly set the backend to non-interactive at the top of the script: `matplotlib.use('Agg')`.

## 2. Infrastructure & Serving (`pipeline.py`, `Dockerfile`)

### Blocker 1: Serving Image Version Compatibility & EOL
*   **Symptom:** `403 Forbidden` or `Permission Denied` when attempting to use `tf2-cpu.2-17` or `tf2-cpu.2-14`.
*   **Root Cause:** 
    1. `tf2-cpu.2-17` does not exist in the public Vertex AI registry yet.
    2. `tf2-cpu.2-14` is End-of-Life (EOL).
    3. **Critical:** TF 2.16 introduced Keras 3.0, which broke backward compatibility with TF 2.15 (Keras 2) serving containers.
*   **Resolution:** Standardized the entire stack (Training and Serving) on **TensorFlow 2.15**.
    *   Training: `us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-15.py310:latest`
    *   Serving: `us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest`

### Blocker 2: `gcloud` CLI Crash
*   **Symptom:** `TypeError: string indices must be integers` during log streaming.
*   **Root Cause:** A bug in the `google-auth` library interacting with the Cloud Shell environment's credentials.
*   **Resolution:** 
    1. Workaround: Used `--skip-build` to bypass the crashing log stream when the image was already pushed.
    2. Fix: Refreshed local credentials using `gcloud auth application-default login`.

## 3. Deployment Workflow

### Blocker 1: Code Syncing & Image Rebuilds
*   **Symptom:** Deployed pipeline runs exhibited behavior from older code versions (e.g., old MAE logic, old image tags).
*   **Root Cause:** Modifying Python scripts (`src/*.py`) locally does not automatically update the pipeline unless the Docker image is **rebuilt** and **pushed**. The `src/` directory is copied into the image at build time.
*   **Resolution:** Established a strict workflow: Make changes -> Rebuild Image (`./deploy_pipeline.sh`) -> Submit Pipeline.
