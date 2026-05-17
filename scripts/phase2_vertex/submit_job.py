from google.cloud import aiplatform

# ==============================================================================
# Configuration
# ==============================================================================
PROJECT_ID = "your-gcp-project-id"
REGION = "us-central1"
BUCKET_NAME = "your-gcs-bucket-name"  # e.g., gs://my-rlhf-bucket

# Artifact Registry URI
IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/rlhf-repo/rlhf-trainer:latest"

# Machine Configuration
MACHINE_TYPE = "a2-highgpu-1g"      # 1x A100 VM
ACCELERATOR_TYPE = "NVIDIA_TESLA_A100"
ACCELERATOR_COUNT = 1

def submit_custom_job():
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET_NAME}/staging")

    job = aiplatform.CustomContainerTrainingJob(
        display_name="rlhf-training-job",
        container_uri=IMAGE_URI,
        # You can pass arguments to your python script here
        command=["python", "src/train_rlhf.py"],
        model_serving_container_image_uri=None,
    )

    print(f"Submitting job using image {IMAGE_URI} on {MACHINE_TYPE} with {ACCELERATOR_COUNT}x {ACCELERATOR_TYPE}...")

    model = job.run(
        machine_type=MACHINE_TYPE,
        accelerator_type=ACCELERATOR_TYPE,
        accelerator_count=ACCELERATOR_COUNT,
        replica_count=1,
        # Optional: Set environment variables or args
        # environment_variables={"WANDB_API_KEY": "your-key"}
        sync=False # Run asynchronously
    )
    
    print(f"Job submitted successfully. You can monitor it in the Vertex AI Console.")

if __name__ == "__main__":
    submit_custom_job()
