# Project Skills & Knowledge Base: GCP RLHF

> **🤖 AI Agent Directive**: Whenever you start a new session or work on this repository, you MUST read this file first. As the project evolves, if you encounter new bugs, hardware limitations, or make architectural decisions, you are required to **continuously update and append** to this skill file so that the project's context is never lost.

## 1. Architectural Philosophy
*   **Three-Phase Deployment Strategy**:
    *   `Phase 0 (Local)`: Prototyping and logic debugging on local limited hardware.
    *   `Phase 1 (GCP Compute Engine)`: High-freedom agile development on an on-demand GPU VM (Start -> Sync -> Train -> Auto-Shutdown).
    *   `Phase 2 (GCP Vertex AI)`: Serverless, containerized Custom Training Jobs for large-scale unattended runs.
*   **Modular Reward System**: All environment reward models MUST be modularized. We use a registry pattern (`src/rewards/registry.py`) and a base class (`BaseRewardModel`). Never hardcode reward logic into the main training loop.

## 2. Hardware Constraints & Workarounds (Local RTX 4080 16GB)
*   **Constraint**: Standard FP16 RLHF training for modern models (e.g., 4B+ parameters like Gemma 4) will instantly cause Out-Of-Memory (OOM) errors on a 16GB VRAM GPU due to the need for multiple model copies (Active, Reference, Reward) and optimizer states.
*   **Solution implemented**:
    *   **QLoRA (4-bit)**: Must load models using `bitsandbytes` in 4-bit `nf4` precision.
    *   **TRL Native PEFT**: Rely on `trl`'s native handling of `peft_config`. By doing this, `trl` will use the base model weights with adapters turned off for the reference model pass, saving massive amounts of VRAM (eliminating the need to load a separate reference model).
    *   **Tiny Batch Sizes**: Keep `batch_size` and `mini_batch_size` down to 1 or 2 for Phase 0 testing.

## 3. Security & Git Hygiene
*   **NEVER** commit GCP service account keys (`*key.json`, `*credentials.json`), local environment variables (`.env`), or massive model checkpoints (`*.safetensors`, `*.pt`) to Git. The `.gitignore` has been specifically tailored to prevent this.
*   **COMMIT CONFIRMATION RULE**: When the project reaches a certain milestone and passes necessary tests, the AI MUST explicitly ask the USER if they want to create a Git commit. **The AI MUST NOT run `git commit` or any auto-commit scripts without first receiving explicit confirmation from the USER.**

## 4. Current Setup Status (May 2026)
- **Framework Replacement**: Abandoned `trl`'s `PPOTrainer`/`GRPOTrainer` due to their inability to handle interactive multi-turn POMDPs. Implemented a custom decoupled architecture (`ReplayBuffer` -> Custom Advantage Trainer).
- **Environment Decoupling**: All Mahjong physics and LangGraph orchestration logic moved to `src/tasks/mahjong`.
- **Secrets Management**: Implemented `python-dotenv` for securely loading `HF_TOKEN` from `.env`.
- **Local Testing Constraint**: The `peft` library upcasts embeddings to fp32. For Gemma models with 256k vocabs, this allocates >10GB VRAM instantly, causing OOM on 16GB GPUs even for 2B models. Use `Qwen2.5-0.5B` for local Phase 0 verifications.

---
*(End of SKILLS.md. Append new learnings below this line in the future.)*
