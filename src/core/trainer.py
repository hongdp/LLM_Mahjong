import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.core.registry import get_task
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

    # 2. Main RL Loop (Rollout -> Train)
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        # --- PHASE A: ROLLOUT ---
        # Note: In Phase 0 local tests, generating with a local LLM in a tight loop is too slow.
        # We pass None for model/tokenizer to use the fast mock generator inside orchestrator for the test.
        # In actual GCP Phase 1, we pass the real model.
        rollout_model = None if args.debug else model
        rollout_tok = None if args.debug else tokenizer
        
        buffer = task.collect_rollouts(num_episodes=args.num_episodes, model=rollout_model, tokenizer=rollout_tok)
        samples = buffer.calculate_advantages_and_flatten()
        
        if not samples:
            print("No valid samples collected. Skipping epoch.")
            continue
            
        print(f"Collected {len(samples)} state-action transitions.")

        # --- PHASE B: UPDATE ---
        model.train()
        for i in range(0, len(samples), args.batch_size):
            batch = samples[i:i+args.batch_size]
            
            prompts = [s["prompt"] for s in batch]
            actions = [s["action"] for s in batch]
            advantages = torch.tensor([s["advantage"] for s in batch], dtype=torch.float32).to(device)

            # Format the full sequence: prompt + action
            texts = [p + a for p, a in zip(prompts, actions)]
            
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
            prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits # (batch, seq_len, vocab_size)
            
            # We only want to compute loss on the `action` tokens.
            # Shift logits and labels
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs.input_ids[..., 1:].contiguous()
            
            # Mask out the prompt
            prompt_lengths = prompt_inputs.attention_mask.sum(dim=1)
            loss_mask = torch.ones_like(shift_labels, dtype=torch.bool)
            for b_idx, p_len in enumerate(prompt_lengths):
                loss_mask[b_idx, :p_len-1] = False # Do not train on prompt tokens
            
            # Calculate negative log likelihood for each token
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            nll = nll.view(shift_labels.size())
            
            # Average NLL over the action tokens
            action_nll = (nll * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1)
            
            # Advantage Weighted Loss (Policy Gradient)
            # Minimize Action NLL * Advantage
            pg_loss = (action_nll * advantages).mean()
            
            optimizer.zero_grad()
            pg_loss.backward()
            optimizer.step()
            
            if i == 0:
                print(f"Batch 0 - Loss: {pg_loss.item():.4f} | Avg Advantage: {advantages.mean().item():.4f}")

    print("✅ Custom Multi-Turn Training complete.")
    model.save_pretrained("./checkpoints/custom_rl_mahjong")
    tokenizer.save_pretrained("./checkpoints/custom_rl_mahjong")

if __name__ == "__main__":
    main()
