import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import matplotlib.pyplot as plt

from src.core.registry import get_task
from src.core.rollout import ReplayBuffer
import src.tasks.mahjong.task 

def parse_args():
    parser = argparse.ArgumentParser(description="Custom Multi-Turn RLHF Trainer")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--task", type=str, default="mahjong")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_episodes", type=int, default=2, help="Number of games to rollout per epoch")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

def save_trajectory_log(buffer: ReplayBuffer, epoch: int, task_name: str):
    """Saves readable game rollouts to a log file."""
    os.makedirs("./logs", exist_ok=True)
    log_path = f"./logs/{task_name}_epoch_{epoch}_rollouts.txt"
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

def plot_metrics(metrics: dict, task_name: str):
    """Generates a visualization of training metrics."""
    os.makedirs("./logs", exist_ok=True)
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
    
    plot_path = f"./logs/{task_name}_training_metrics.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"📊 Saved metrics visualization to {plot_path}")

def main():
    args = parse_args()
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
        peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, peft_config)
    else:
        model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    task = get_task(args.task, device=device)

    # Metrics storage
    history = {"loss": [], "avg_reward": [], "avg_advantage": []}

    # 2. Main RL Loop (Rollout -> Train)
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        # --- PHASE A: ROLLOUT ---
        rollout_model = None if args.debug else model
        rollout_tok = None if args.debug else tokenizer
        
        buffer = task.collect_rollouts(num_episodes=args.num_episodes, model=rollout_model, tokenizer=rollout_tok)
        
        # Log the raw text trajectories for inspection
        save_trajectory_log(buffer, epoch+1, args.task)
        
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
            advantages = torch.tensor([s["advantage"] for s in batch], dtype=torch.float32).to(device)

            texts = [p + a for p, a in zip(prompts, actions)]
            
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
            prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

            outputs = model(**inputs)
            logits = outputs.logits 
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs.input_ids[..., 1:].contiguous()
            
            prompt_lengths = prompt_inputs.attention_mask.sum(dim=1)
            loss_mask = torch.ones_like(shift_labels, dtype=torch.bool)
            for b_idx, p_len in enumerate(prompt_lengths):
                loss_mask[b_idx, :p_len-1] = False 
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            nll = nll.view(shift_labels.size())
            
            action_nll = (nll * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1)
            
            pg_loss = (action_nll * advantages).mean()
            
            optimizer.zero_grad()
            pg_loss.backward()
            optimizer.step()
            
            epoch_losses.append(pg_loss.item())
            epoch_advs.append(advantages.mean().item())

        # Record metrics for the epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_adv = sum(epoch_advs) / len(epoch_advs)
        print(f"Batch completed - Avg Loss: {avg_loss:.4f} | Avg Advantage: {avg_adv:.4f} | Avg Ep Reward: {avg_raw_reward:.4f}")
        
        history["loss"].append(avg_loss)
        history["avg_advantage"].append(avg_adv)
        history["avg_reward"].append(avg_raw_reward)

    # --- PHASE C: VISUALIZATION ---
    if len(history["loss"]) > 0:
        plot_metrics(history, args.task)

    print("✅ Custom Multi-Turn Training complete.")
    model.save_pretrained("./checkpoints/custom_rl_mahjong")
    tokenizer.save_pretrained("./checkpoints/custom_rl_mahjong")

if __name__ == "__main__":
    main()
