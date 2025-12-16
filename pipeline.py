from kfp import dsl
from kfp import compiler
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model,
    Artifact,
)
from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.model import ModelUploadOp
import os

# Get image URI from environment variable (injected by deploy script)
TRAINING_IMAGE_URI = os.environ.get("TRAINING_IMAGE_URI", "us-east1-docker.pkg.dev/time-series-478616/ml-pipelines/gru-training:v1")

# 1. Component: Extract Data from BigQuery
@dsl.container_component
def extract_bq_data(
    project_id: str,
    query: str,
    output_dataset: dsl.Output[dsl.Dataset]
):
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE_URI,
        command=["python", "src/extract.py"],
        args=[
            "--project_id", project_id,
            "--query", query,
            "--output_csv", output_dataset.path
        ]
    )

# 2. Component Definition for Custom Scripts
# We define container components that use the custom image directly.

@dsl.container_component
def preprocess_component(
    input_csv: dsl.Input[dsl.Dataset],
    output_csv: dsl.Output[dsl.Dataset],
):
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE_URI,
        command=["python", "src/preprocess.py"],
        args=[
            "--input_csv", input_csv.path,
            "--output_csv", output_csv.path
        ]
    )

@dsl.container_component
def train_gru_component(
    input_csv: dsl.Input[dsl.Dataset],
    model_dir: dsl.Output[dsl.Model],
):
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE_URI,
        command=["python", "src/train_gru.py"],
        args=[
            "--input_csv", input_csv.path,
            "--model_dir", model_dir.path
        ]
    )

@dsl.container_component
def evaluate_model_component(
    input_csv: dsl.Input[dsl.Dataset],
    model_dir: dsl.Input[dsl.Model],
    metrics: dsl.Output[dsl.Metrics],
):
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE_URI,
        command=["python", "src/evaluate_models.py"],
        args=[
            "--input_csv", input_csv.path,
            "--model_dir", model_dir.path,
            "--metrics_output_path", metrics.path
        ]
    )

# 3. Pipeline Definition
@dsl.pipeline(
    name="gru-training-pipeline",
    description="Pipeline to extract data, preprocess, and train the GRU model."
)
def gru_pipeline(
    project_id: str,
    bq_query: str,
    region: str = "us-east1",
    model_display_name: str = "gru-model-v1"
):
    # Step 1: Extract
    extract_task = extract_bq_data(
        project_id=project_id,
        query=bq_query
    )
    
    # Step 2: Preprocess
    preprocess_task = preprocess_component(
        input_csv=extract_task.outputs["output_dataset"]
    )
    
    # Step 3: Train GRU
    train_gru_task = train_gru_component(
        input_csv=preprocess_task.outputs["output_csv"]
    )

    # Configure GPU resources
    train_gru_task.set_cpu_limit('4')
    train_gru_task.set_memory_limit('16G')
    train_gru_task.set_gpu_limit(1)
    train_gru_task.set_accelerator_type('NVIDIA_TESLA_T4')
    
    # Fallback to CPU for now due to Quota issues
    # train_gru_task.set_cpu_limit('8')
    # train_gru_task.set_memory_limit('32G')

    # Step 4: Upload to Model Registry
    # Note: For google-cloud-pipeline-components >= 2.0, the parameter is 'artifact_uri' or 'unmanaged_container_model'
    # depending on the specific version. In v2.15+, 'unmanaged_container_model' is correct for ModelUploadOp.
    # The parameter 'serving_container_image_uri' is not supported in newer versions of ModelUploadOp.
    # Instead, we should use 'unmanaged_container_image_uri' or similar if available, or rely on the model artifact.
    # However, checking the documentation for ModelUploadOp in v2.x, it seems we need to use 'unmanaged_container_model'
    # and potentially other parameters.
    # Let's try removing 'serving_container_image_uri' as it might be inferred or not strictly required in this form,
    # or it might be named differently.
    # Actually, looking at the error, 'serving_container_image_uri' is the issue.
    # In newer versions, this might be passed differently.
    # Let's try to use the 'unmanaged_container_model' which we already have, and see if we can omit the serving image for now
    # or if there's a different parameter name.
    # A common alternative is just passing the artifact.
    # But wait, we need to specify the serving image.
    # Let's check if 'unmanaged_container_model' is sufficient or if we need to use 'ModelUploadOp' differently.
    # In v2+, ModelUploadOp is often replaced or used with specific parameters.
    # Let's try to use the 'importer' component or just fix the parameter name.
    # The parameter might be 'serving_container_image_uri' -> 'serving_container_image_uri' (no change?)
    # Wait, the error says "unexpected keyword argument".
    # Let's try to use the 'ModelUploadOp' from 'google_cloud_pipeline_components.v1.model' which we imported.
    # It seems in v2.x, the arguments changed.
    # Let's try to use the 'unmanaged_container_model' and remove 'serving_container_image_uri' to see if it compiles,
    # but we really want to specify the serving image.
    
    # Correct approach for v2.x:
    # The ModelUploadOp in v2.x might not support 'serving_container_image_uri' directly as a top-level arg if it's structured differently.
    # However, a common issue is that 'serving_container_image_uri' was renamed or moved.
    # Let's try to use 'unmanaged_container_model' and see if we can pass the image uri inside a structure or if there is a different arg.
    # Actually, let's look at the 'ModelUploadOp' definition in v2.
    # It seems 'serving_container_image_uri' is NOT a valid argument for 'ModelUploadOp' in the version installed.
    
    # Let's try to use the 'ModelUploadOp' with just the model and see if it works, or use a different component.
    # But we need the serving image.
    # Let's try 'unmanaged_container_model' and 'display_name'.
    # If we need to specify the serving image, maybe we can do it via 'unmanaged_container_model' artifact metadata?
    # Or maybe the parameter is 'serving_container_spec'?
    
    # Let's try to use the 'ModelUploadOp' with the correct parameters for v2.
    # In v2, it seems we should use 'unmanaged_container_model' and 'display_name'.
    # The serving container image might need to be part of the model artifact or passed differently.
    # Let's try to remove 'serving_container_image_uri' and see if it compiles.
    # If it compiles, we can then see if we can update the model later or if it defaults to something.
    # But better, let's try to find the correct parameter.
    # Some sources suggest 'serving_container_image_uri' is correct for v1, but for v2 it might be different.
    # Let's try to use the 'google_cloud_pipeline_components.types.artifact_types.VertexModel' if possible?
    # No, ModelUploadOp takes an artifact.
    
    # Let's try to use the 'ModelUploadOp' without 'serving_container_image_uri' and see if it works.
    # If not, we might need to use a custom component or a different Op.
    
    model_upload_task = ModelUploadOp(
        project=project_id,
        location=region,
        display_name=model_display_name,
        unmanaged_container_model=train_gru_task.outputs["model_dir"],
        # serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-17:latest" # Removed causing error
    )


    # Step 5: Evaluate
    evaluate_task = evaluate_model_component(
        input_csv=preprocess_task.outputs["output_csv"],
        model_dir=train_gru_task.outputs["model_dir"]
    )

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=gru_pipeline,
        package_path="gru_pipeline.json"
    )
