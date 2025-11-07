# src/vertex_integration.py
import os
from google.cloud import aiplatform
from src.config import MODEL_SAVE_DIR
from dotenv import load_dotenv

load_dotenv()

GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")
VERTEX_BUCKET = os.getenv("VERTEX_BUCKET")

def upload_model_to_vertex():
    """Upload trained model to Vertex AI Model Registry."""
    model_path = MODEL_SAVE_DIR / "aqi_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Train first.")
    
    aiplatform.init(project=GCP_PROJECT, location=GCP_REGION, staging_bucket=VERTEX_BUCKET)

    model = aiplatform.Model.upload(
        display_name="aqi-predictor-model",
        artifact_uri=str(MODEL_SAVE_DIR),
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
    )

    print(f"✅ Model uploaded to Vertex: {model.resource_name}")

if __name__ == "__main__":
    upload_model_to_vertex()
