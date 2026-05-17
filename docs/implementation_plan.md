# LLM RLHF on Google Cloud: Architecture Design Document

This document outlines the architecture and implementation plan for running Large Language Model (LLM) Reinforcement Learning from Human Feedback (RLHF) training on Google Cloud Platform (GCP). 

The strategy is divided into three phases, starting from local sanity checks to massive cloud scaling.

## 💻 Phase 0: Local Prototyping & Sanity Checks (Local Machine)

Before spending any cloud credits, the core logic should be verified on a local machine using whatever CPU/GPU resources are available.

### Architecture Overview
- **Compute:** Local workstation or laptop. If local VRAM is limited, techniques like QLoRA (4-bit quantization) or testing on a tiny proxy model (e.g., 1B or 0.5B parameters) will be used.
- **Storage:** Local filesystem.
- **Workflow:**
  1. **Data Loading Check:** Verify datasets load, tokenize, and pad correctly.
  2. **Model Compilation Check:** Ensure the RLHF (PPO/DPO) loop runs forward and backward passes without crashing or causing Out-Of-Memory (OOM) errors.
  3. **Logic Validation:** Confirm loss curves decrease on tiny dummy datasets.

### Cost Management
- **Free.** No cloud resources are provisioned.

---

## 🚀 Phase 1: Agile Development & Debugging (Compute Engine)

In the initial phase, code needs frequent adjustments, debugging is common, and interactive access to the GPU is critical. We will use a **GCP Compute Engine GPU Virtual Machine** with an on-demand start/stop strategy.

### Architecture Overview
- **Compute:** A dedicated Compute Engine VM (e.g., equipped with NVIDIA L4 or A100 GPUs) using the Deep Learning VM Image.
- **Storage:** Persistent Disk (PD) attached to the VM to store datasets, code, and checkpoints.
- **Workflow:** 
  1. **Start:** Developer runs a local script to start the VM.
  2. **Sync:** Code is synced to the VM (via `rsync` or `git pull`).
  3. **Execute:** Developer runs the training script (interactively via `tmux` or automated via SSH).
  4. **Auto-Stop:** Once the training script finishes, it triggers a `sudo shutdown -h now` command to stop the VM and halt GPU billing.

### Cost Management
- While the VM is **Stopped**, you only pay for the Persistent Disk (storage costs), which is negligible compared to GPU costs.
- You are billed for the GPU/CPU only when the instance is explicitly in the **Running** state.

---

## 🏭 Phase 2: Large-Scale Production (Vertex AI)

Once the RLHF code is stable and requires large-scale, automated, or parallel hyperparameter tuning runs, we will migrate to **Vertex AI Custom Training Jobs**.

### Architecture Overview
- **Compute:** Vertex AI managed infrastructure. GPUs are provisioned strictly for the lifespan of the job.
- **Storage:** Google Cloud Storage (GCS) buckets are used to store training data and write model checkpoints.
- **Packaging:** Code is packaged into a Docker Container and hosted on Google Artifact Registry (GAR).
- **Workflow:**
  1. **Build:** Developer builds a Docker image containing the code and dependencies.
  2. **Push:** Image is pushed to Artifact Registry.
  3. **Submit:** A training job is submitted to Vertex AI specifying the image, hardware requirements (e.g., 8x A100), and GCS paths.
  4. **Release:** Vertex AI runs the container and automatically releases all resources upon completion or failure.

### Cost Management
- True Serverless billing: You are charged down to the second only for the time the Vertex AI job is executing. No idle storage or compute costs are incurred.

---

## 📁 Proposed Directory Structure

To support all three phases seamlessly, the project repository will be structured as follows:

```text
gcloud_llm/
├── src/                      # Core RLHF training code
│   ├── train_rlhf.py         # Main training entrypoint
│   ├── data_loader.py        
│   └── models/               
├── scripts/                  # Management scripts
│   ├── phase0_local/         
│   │   └── run_local_test.sh # Script for small-scale local dry-run
│   ├── phase1_ce/            
│   │   ├── start_vm.sh       # Script to start the VM
│   │   ├── run_training.sh   # Script run inside VM (ends with shutdown)
│   │   └── sync_code.sh      # Sync local code to VM
│   └── phase2_vertex/        
│   │   ├── build_docker.sh   # Build and push container to GAR
│   │   └── submit_job.py     # Vertex AI Python SDK submission script
├── Dockerfile                # For Phase 2 Vertex AI container
└── requirements.txt          # Python dependencies
```

---

## ❓ User Review Required

> [!IMPORTANT]  
> Please review the proposed architecture and workflow. 
> 1. Do you already have a GCP Project set up with Billing enabled?
> 2. Have you requested GPU quotas in your GCP region (e.g., `us-central1`)? This is often the biggest hurdle for new GCP projects.
> 3. Should we begin by creating the skeleton directory structure and the Phase 1 VM management bash scripts?
