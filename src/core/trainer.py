import os
import re
import json
import random
from dotenv import load_dotenv
load_dotenv() # Load variables from .env before doing anything else
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from src.core.registry import get_task
from src.core.rollout import ReplayBuffer
import src.tasks.mahjong.task 

def parse_args():
    parser = argparse.ArgumentParser(description="Custom Multi-Turn RLHF Trainer")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file to load arguments from.")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--peft_model_path", type=str, default=None, help="Path to a pre-trained PEFT adapter to load.")
    parser.add_argument("--task", type=str, default="mahjong")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_episodes", type=int, default=2, help="Number of games to rollout per epoch")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--training_phase", type=int, default=1, choices=[1, 2],
        help="Phase 1: format-only reward (teach legal output). Phase 2: full game reward (optimize strategy)."
    )
    parser.add_argument(
        "--sft_epochs", type=int, default=3,
        help="Number of SFT warm-up epochs using random legal actions as ground truth before RL starts."
    )
    parser.add_argument(
        "--sft_data", type=str, default="data/sft_mahjong.jsonl",
        help="Path to pre-generated SFT JSONL file."
    )
    parser.add_argument(
        "--sft_batch_size", type=int, default=8,
        help="Batch size for SFT warm-up (can be larger than RL batch since no rollout memory needed)."
    )
    parser.add_argument(
        "--sft_max_samples", type=int, default=500,
        help="Max number of JSONL samples to use per SFT run (subset for speed)."
    )
    parser.add_argument(
        "--sft_learning_rate", type=float, default=1e-4,
        help="Learning rate for SFT warm-up (typically much higher than RL, e.g. 1e-4 for QLoRA)."
    )
    parser.add_argument(
        "--exp_name", type=str, default=None,
        help="Experiment name (used for logging directory). Defaults to timestamp if not provided."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="If set, resumes the experiment in the same directory (no new timestamp appended)."
    )
    parser.add_argument(
        "--min_format_rate", type=float, default=0.3,
        help="Phase 2 only: abort if rollout format compliance stays below this for 3 consecutive epochs."
    )
    args = parser.parse_args()
    
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            for k, v in config_data.items():
                if hasattr(args, k) and k != "config":
                    setattr(args, k, v)
                    
    return args

def save_trajectory_log(buffer: ReplayBuffer, epoch: int, task_name: str, exp_dir: str):
    """Saves readable game rollouts to a log file."""
    os.makedirs(exp_dir, exist_ok=True)
    log_path = os.path.join(exp_dir, f"{task_name}_epoch_{epoch}_rollouts.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"=== EPOCH {epoch} ROLLOUT LOGS ===\n\n")
        for ep_idx, episode in enumerate(buffer.episodes):
            f.write(f"--- Episode {ep_idx} (Total Steps: {len(episode)}) ---\n")
            for step_idx, step in enumerate(episode):
                f.write(f"[Step {step_idx}] Reward: {step.reward:.2f} | Terminal: {step.is_terminal}\n")
                f.write(f"PROMPT:\n{step.prompt_text.strip()}\n")
                f.write(f"ACTION: {step.action_text}\n")
                f.write("-" * 40 + "\n")
            f.write("\n\n")
    print(f"📄 Saved rollout log to {log_path}")

def plot_metrics(metrics: dict, task_name: str, exp_dir: str):
    """Generates a visualization of training metrics."""
    os.makedirs(exp_dir, exist_ok=True)
    epochs = range(len(metrics["loss"]))
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss Plot
    axs[0].plot(epochs, metrics["loss"], marker='o', color='red')
    axs[0].set_title("Training Loss (Advantage-Weighted NLL)")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].grid(True)
    
    # Reward/Advantage Plot
    axs[1].plot(epochs, metrics["avg_reward"], marker='x', color='blue', label="Avg Raw Reward")
    axs[1].set_title("Environment Performance")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Reward")
    axs[1].legend()
    axs[1].grid(True)
    
    plot_path = os.path.join(exp_dir, f"{task_name}_training_metrics.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"📊 Saved metrics visualization to {plot_path}")

def main():
    import datetime
    args = parse_args()
    
    if args.exp_name is None:
        args.exp_name = f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    elif not args.resume:
        # Append timestamp to the specified exp_name to ensure a fresh directory
        args.exp_name = f"{args.exp_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    exp_dir = os.path.join("experiments", args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)
    print(f"📁 Experiment Directory: {exp_dir}")
    print(f"📝 Saved config to {config_path}")

    print(f"🚀 Starting Custom RLHF Training for Task: {args.task}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model and Tokenizer
    quantization_config = None
    if args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto" if args.use_qlora else None
    )

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)
        if args.peft_model_path and os.path.exists(args.peft_model_path):
            from peft import PeftModel
            print(f"Loading PEFT adapter from {args.peft_model_path}...")
            model = PeftModel.from_pretrained(model, args.peft_model_path, is_trainable=True)
        else:
            peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
            model = get_peft_model(model, peft_config)
    else:
        model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    task = get_task(args.task, device=device)

    # TensorBoard writer
    tb_log_dir = os.path.join(exp_dir, "tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"📈 TensorBoard logging to: {tb_log_dir}")
    print(f"   Run: tensorboard --logdir {tb_log_dir}")

    # --- Phase 1 format reward helper ---
    # Whitelist of action types the table engine actually emits — hallucinated
    # types (e.g. "reveal", "hold") must NOT count as format-compliant.
    _action_re = re.compile(
        r'<action\s+type="(discard|chi|pon|kan|riichi|ron|tsumo|skip)"(?:\s+tile="([^"]+)")?\s*/>'
    )
    _hand_re = re.compile(r'手牌: ((?:[1-9][mpsz] )*[1-9][mpsz])')

    def phase1_format_reward(prompt: str, action: str) -> float:
        """Phase 1: reward a well-formed action tag; discard/riichi tiles must be in hand."""
        m = _action_re.search(action)
        if not m:
            return -5.0      # No valid action tag
        action_type, tile = m.group(1), m.group(2)
        if action_type in ("discard", "riichi"):
            hand_match = _hand_re.search(prompt)
            if not hand_match or tile not in hand_match.group(1).split():
                return -5.0  # Discarding a tile not in hand
        return 5.0

    # Metrics storage
    history = {"loss": [], "avg_reward": [], "avg_advantage": []}

    # =========================================================
    # SFT WARM-UP: load pre-generated data and train with NLL
    # =========================================================
    sft_data_path = args.sft_data
    if args.sft_epochs > 0 and sft_data_path and os.path.exists(sft_data_path):
        print(f"\n{'='*50}")
        print(f"SFT Warm-up: {args.sft_epochs} epoch(s) from {sft_data_path}")
        print(f"{'='*50}")

        # Load JSONL — each line has "text" (full ChatML) or "messages"
        sft_records = []
        with open(sft_data_path, encoding="utf-8") as fj:
            for line in fj:
                line = line.strip()
                if line:
                    sft_records.append(json.loads(line))
        # Optionally cap the number of samples for speed
        if args.sft_max_samples and len(sft_records) > args.sft_max_samples:
            sft_records = sft_records[:args.sft_max_samples]
            print(f"  Using first {args.sft_max_samples} samples (sft_max_samples cap)")

        sft_bs = args.sft_batch_size
        sft_optimizer = torch.optim.AdamW(model.parameters(), lr=args.sft_learning_rate)

        sft_log_path = os.path.join(exp_dir, "sft_warmup.txt")
        model.train()
        with open(sft_log_path, "w", encoding="utf-8") as sft_log:
            sft_log.write(f"=== SFT WARM-UP LOG ===\n")
            sft_log.write(f"Data: {sft_data_path} ({len(sft_records)} samples used)\n")
            sft_log.write(f"Epochs: {args.sft_epochs}  Batch size: {sft_bs}\n")
            sft_log.write("=" * 60 + "\n\n")

        for sft_epoch in range(args.sft_epochs):
            random.shuffle(sft_records)
            sft_losses = []
            LOG_SAMPLES = 3  # number of samples to log per epoch

            for i in range(0, len(sft_records), sft_bs):
                batch = sft_records[i : i + sft_bs]

                # Build full text and prompt-only text for each sample
                full_texts, prompt_texts, response_texts = [], [], []
                for rec in batch:
                    if "text" in rec:
                        full_texts.append(rec["text"])
                        prompt_part = rec["text"].split("<|im_start|>assistant")[0] + "<|im_start|>assistant\n"
                        prompt_texts.append(prompt_part)
                        # Extract assistant response for logging
                        parts = rec["text"].split("<|im_start|>assistant\n", 1)
                        response_texts.append(parts[1].replace("<|im_end|>", "").strip() if len(parts) > 1 else "")
                    else:
                        msgs = rec["messages"]
                        full = tokenizer.apply_chat_template(msgs, tokenize=False)
                        prompt_msgs = [m for m in msgs if m["role"] != "assistant"]
                        prompt = tokenizer.apply_chat_template(
                            prompt_msgs, tokenize=False, add_generation_prompt=True
                        )
                        full_texts.append(full)
                        prompt_texts.append(prompt)
                        asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                        response_texts.append(asst)

                # Log first LOG_SAMPLES of epoch 0 for inspection
                if i == 0 and len(sft_losses) == 0:
                    with open(sft_log_path, "a", encoding="utf-8") as sft_log:
                        sft_log.write(f"--- Epoch {sft_epoch+1} sample log ---\n")
                        for si in range(min(LOG_SAMPLES, len(batch))):
                            # Only log the user-facing state part (skip long system prompt)
                            user_part = prompt_texts[si].split("<|im_start|>user\n", 1)
                            user_display = user_part[1].split("<|im_end|>")[0].strip() if len(user_part) > 1 else prompt_texts[si][-300:]
                            sft_log.write(f"[Sample {si+1}]\n")
                            sft_log.write(f"STATE:\n{user_display}\n")
                            sft_log.write(f"TARGET RESPONSE:\n{response_texts[si]}\n")
                            sft_log.write("-" * 40 + "\n")
                        sft_log.write("\n")

                inputs = tokenizer(
                    full_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=1024
                ).to(device)
                prompt_inputs = tokenizer(
                    prompt_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=1024
                ).to(device)

                outputs = model(**inputs)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs.input_ids[..., 1:].contiguous()

                # Mask out the prompt tokens — only compute loss on assistant response
                prompt_lengths = prompt_inputs.attention_mask.sum(dim=1)
                loss_mask = inputs.attention_mask[..., 1:].contiguous().bool()
                for b_idx, p_len in enumerate(prompt_lengths):
                    loss_mask[b_idx, : p_len - 1] = False

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nll = nll.view(shift_labels.size())
                sft_loss = (nll * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1)
                sft_loss = sft_loss.mean()

                sft_optimizer.zero_grad()
                sft_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                sft_optimizer.step()
                sft_losses.append(sft_loss.item())

                # TensorBoard: per-step SFT loss
                global_sft_step = sft_epoch * len(sft_records) + i
                writer.add_scalar("sft/step_loss", sft_loss.item(), global_sft_step)

            avg_sft_loss = sum(sft_losses) / max(len(sft_losses), 1)
            print(f"  [SFT Epoch {sft_epoch+1}/{args.sft_epochs}] Loss: {avg_sft_loss:.4f}")
            writer.add_scalar("sft/epoch_loss", avg_sft_loss, sft_epoch)
            with open(sft_log_path, "a", encoding="utf-8") as sft_log:
                sft_log.write(f"[SFT Epoch {sft_epoch+1}/{args.sft_epochs}] Avg Loss: {avg_sft_loss:.4f}\n")
        with open(sft_log_path, "a", encoding="utf-8") as sft_log:
            sft_log.write("\n=== SFT WARM-UP COMPLETE ===\n")
        print(f"  📄 SFT log saved to {sft_log_path}")
        
        sft_checkpoint_path = os.path.join(exp_dir, f"checkpoints_sft_warmup_{args.task}")
        print(f"  💾 Saving SFT checkpoint to {sft_checkpoint_path}")
        model.save_pretrained(sft_checkpoint_path)
        tokenizer.save_pretrained(sft_checkpoint_path)
    elif args.sft_epochs > 0:
        print(f"\n⚠️  sft_data not found at '{sft_data_path}', skipping SFT warm-up.")

    # =========================================================
    # MAIN RL LOOP
    # =========================================================

    # Re-initialize optimizer for RL with the lower RL learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 2. Main RL Loop (Rollout -> Train)
    low_format_streak = 0
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        # --- PHASE A: ROLLOUT ---
        if not args.debug:
            model.eval()
        rollout_model = None if args.debug else model
        rollout_tok = None if args.debug else tokenizer
        
        buffer = task.collect_rollouts(num_episodes=args.num_episodes, model=rollout_model, tokenizer=rollout_tok, exp_dir=exp_dir)
        
        # Log the raw text trajectories for inspection
        save_trajectory_log(buffer, epoch+1, args.task, exp_dir)

        # Format compliance BEFORE the update: fraction of steps with a valid action tag.
        # Checked early so a format collapse aborts instead of burning epochs on garbage.
        valid_steps = sum(
            1 for ep in buffer.episodes for step in ep
            if _action_re.search(step.action_text)
        )
        total_steps = sum(len(ep) for ep in buffer.episodes)
        format_rate = valid_steps / max(total_steps, 1)
        writer.add_scalar("rl/format_compliance", format_rate, epoch)
        print(f"Format compliance: {format_rate:.1%} ({valid_steps}/{total_steps})")

        if args.training_phase == 2 and format_rate < args.min_format_rate:
            low_format_streak += 1
            print(f"⚠️  Format rate below {args.min_format_rate:.0%} ({low_format_streak}/3 consecutive) — Phase 2 would train on mostly unparseable outputs.")
            if low_format_streak >= 3:
                print("🛑 Aborting: format collapsed for 3 consecutive epochs. Strengthen SFT warm-up (or lower --min_format_rate).")
                break
        else:
            low_format_streak = 0

        # --- Phase 1: Override rewards with format-only signal ---
        if args.training_phase == 1:
            for episode in buffer.episodes:
                for step in episode:
                    step.reward = phase1_format_reward(step.prompt_text, step.action_text)
            phase_label = "[Phase 1: Format]"
        else:
            phase_label = "[Phase 2: Strategy]"
        
        # Calculate advantages
        samples = buffer.calculate_advantages_and_flatten()
        
        if not samples:
            print("No valid samples collected. Skipping epoch.")
            continue
            
        print(f"Collected {len(samples)} state-action transitions.")

        # Compute average raw reward for tracking
        avg_raw_reward = sum([sum([step.reward for step in ep]) for ep in buffer.episodes]) / max(len(buffer.episodes), 1)

        # --- PHASE B: UPDATE ---
        model.train()
        epoch_losses = []
        epoch_advs = []
        
        for i in range(0, len(samples), args.batch_size):
            batch = samples[i:i+args.batch_size]
            
            prompts = [s["prompt"] for s in batch]
            actions = [s["action"] for s in batch]
            raw_advantages = torch.tensor([s["advantage"] for s in batch], dtype=torch.float32)
            # Clip advantages to prevent runaway gradients from catastrophically bad episodes
            advantages = raw_advantages.clamp(-5.0, 5.0).to(device)

            texts = [p + a for p, a in zip(prompts, actions)]
            
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
            prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

            outputs = model(**inputs)
            logits = outputs.logits 
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs.input_ids[..., 1:].contiguous()
            
            prompt_lengths = prompt_inputs.attention_mask.sum(dim=1)
            loss_mask = inputs.attention_mask[..., 1:].contiguous().bool()
            for b_idx, p_len in enumerate(prompt_lengths):
                loss_mask[b_idx, :p_len-1] = False 
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            nll = nll.view(shift_labels.size())
            
            action_nll = (nll * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1)
            
            pg_loss = (action_nll * advantages).mean()
            
            optimizer.zero_grad()
            pg_loss.backward()
            # Gradient clipping as a last-resort safeguard against weight explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(pg_loss.item())
            epoch_advs.append(advantages.mean().item())

        # Record metrics for the epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_adv = sum(epoch_advs) / len(epoch_advs)

        print(f"Batch completed {phase_label} - Loss: {avg_loss:.4f} | Adv: {avg_adv:.4f} | Reward: {avg_raw_reward:.4f} | Format: {format_rate:.1%}")

        # TensorBoard: RL metrics
        writer.add_scalar("rl/loss", avg_loss, epoch)
        writer.add_scalar("rl/avg_advantage", avg_adv, epoch)
        writer.add_scalar("rl/avg_episode_reward", avg_raw_reward, epoch)

        history["loss"].append(avg_loss)
        history["avg_advantage"].append(avg_adv)
        history["avg_reward"].append(avg_raw_reward)

        # Checkpoint management: Keep top 3 and latest
        epoch_checkpoint_path = os.path.join(exp_dir, f"checkpoint_epoch_{epoch+1}")
        model.save_pretrained(epoch_checkpoint_path)
        tokenizer.save_pretrained(epoch_checkpoint_path)
        
        if "checkpoints" not in history:
            history["checkpoints"] = []
        history["checkpoints"].append({
            "epoch": epoch + 1,
            "reward": avg_raw_reward,
            "path": epoch_checkpoint_path
        })
        
        # Delete checkpoints that are neither latest nor top 3
        sorted_ckpts = sorted(history["checkpoints"], key=lambda x: x["reward"], reverse=True)
        top_3_paths = [c["path"] for c in sorted_ckpts[:3]]
        latest_path = history["checkpoints"][-1]["path"]
        
        import shutil
        for c in list(history["checkpoints"]):
            if c["path"] not in top_3_paths and c["path"] != latest_path:
                if os.path.exists(c["path"]):
                    shutil.rmtree(c["path"])
                history["checkpoints"].remove(c)

    writer.flush()
    writer.close()

    # --- PHASE C: VISUALIZATION ---
    if len(history["loss"]) > 0:
        plot_metrics(history, args.task, exp_dir)

    print("✅ Custom Multi-Turn Training complete.")
    final_checkpoint_path = os.path.join(exp_dir, "checkpoints_final_mahjong")
    model.save_pretrained(final_checkpoint_path)
    tokenizer.save_pretrained(final_checkpoint_path)

if __name__ == "__main__":
    main()
