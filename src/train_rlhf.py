import argparse
import os
import torch
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from peft import LoraConfig

# Import our modular reward system
from src.rewards.registry import get_reward_model
# Ensure dummy rewards are imported so they register themselves
import src.rewards.dummy_rewards

def parse_args():
    parser = argparse.add_argument_group("Training Arguments")
    parser.add_argument("--model_name", type=str, default="gpt2", help="HuggingFace model name or path")
    parser.add_argument("--reward_model", type=str, default="length_penalty", help="Name of the registered reward model")
    parser.add_argument("--learning_rate", type=float, default=1.41e-5)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mini_batch_size", type=int, default=4)
    parser.add_argument("--use_qlora", action="store_true", help="Use 4-bit QLoRA (required for 4B+ models on 16GB VRAM)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (Phase 0)")
    return argparse.ArgumentParser(parents=[parser]).parse_args()

def build_dataset(tokenizer, debug=False):
    """
    Placeholder: returns a simple list of dummy prompts.
    In a real scenario, this would load from datasets library.
    """
    prompts = [
        "Explain the theory of relativity.",
        "How do I bake a chocolate cake?",
        "Write a python script to reverse a string.",
        "What is the capital of France?"
    ]
    # Replicate to make a dataset
    prompts = prompts * (2 if debug else 100)
    
    dataset = []
    for p in prompts:
        dataset.append({
            "query": p,
            "input_ids": tokenizer.encode(p, return_tensors="pt")[0]
        })
    return dataset

def main():
    args = parse_args()
    print("🚀 Starting RLHF PPO Training...")
    print(f"Configuration: {args}")

    # 1. Initialize Configuration
    config = PPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        steps=args.max_steps,
        optimize_cuda_cache=True,
    )

    # 2. Load Model and Tokenizer
    print(f"Loading model: {config.model_name} (QLoRA: {args.use_qlora})")
    
    quantization_config = None
    peft_config = None
    
    if args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # Using trl's wrapper for PPO. If using PEFT, it automatically manages the reference model!
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        peft_config=peft_config,
        device_map="auto" if args.use_qlora else None
    )
    
    if not args.use_qlora:
        # If not using PEFT, we need a separate reference model
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    else:
        # PEFT automatically handles the reference model natively in TRL
        ref_model = None

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Ensure pad token is set (gpt2 doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 3. Load modular reward model
    print(f"Loading reward model environment: {args.reward_model}")
    reward_model_env = get_reward_model(args.reward_model)

    # 4. Prepare dataset
    dataset = build_dataset(tokenizer, debug=args.debug)
    
    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    # 5. Initialize PPOTrainer
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator
    )

    # Generation kwargs for the LLM rollout
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": 32,
    }

    # 6. RLHF Training Loop
    print("Starting Training Loop...")
    for epoch, batch in enumerate(ppo_trainer.dataloader):
        if epoch >= config.steps:
            break
            
        query_tensors = batch["input_ids"]
        
        # Phase 1: Generate rollouts (responses)
        response_tensors = []
        for query in query_tensors:
            # Generate response from the active model
            response = ppo_trainer.generate(query, **generation_kwargs)
            # Remove the prompt part from the generated response
            response_tensors.append(response.squeeze()[-generation_kwargs["max_new_tokens"]:])
            
        batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
        
        # Phase 2: Compute Modular Rewards
        # Pass queries and responses to our modular reward system
        rewards = reward_model_env.compute_reward(prompts=batch["query"], responses=batch["response"])
        
        # Phase 3: Run PPO Step (Update Policy)
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # Logging
        mean_reward = torch.stack(rewards).mean().item()
        print(f"Step {epoch}/{config.steps} - Mean Reward: {mean_reward:.4f} - Loss: {stats['ppo/loss/total']}")

    print("✅ Training complete. Saving model...")
    ppo_trainer.save_pretrained("./checkpoints/final_model")

if __name__ == "__main__":
    main()
