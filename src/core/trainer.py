import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

from src.core.registry import get_task
# Import tasks to ensure they are registered
import src.tasks.mahjong.task 

def parse_args():
    parser = argparse.ArgumentParser(description="Training Arguments")
    parser.add_argument("--model_name", type=str, default="gpt2", help="HuggingFace model name or path")
    parser.add_argument("--task", type=str, default="mahjong", help="Name of the registered task to run")
    parser.add_argument("--learning_rate", type=float, default=1.41e-5)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=4, help="G value for GRPO")
    parser.add_argument("--use_qlora", action="store_true", help="Use 4-bit QLoRA")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (Phase 0)")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"🚀 Starting RLHF Training for Task: {args.task}...")
    print(f"Configuration: {args}")

    # 1. Initialize GRPO Configuration
    training_args = GRPOConfig(
        output_dir=f"./checkpoints/grpo_{args.task}",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        max_steps=args.max_steps,
        num_generations=args.num_generations, 
        logging_steps=1,
        use_cpu=not torch.cuda.is_available(),
    )

    # 2. Load Model and Tokenizer
    print(f"Loading model: {args.model_name} (QLoRA: {args.use_qlora})")
    
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto" if args.use_qlora else None
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 3. Load Task
    print(f"Loading task environment: {args.task}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = get_task(args.task, device=device)

    # 4. Prepare dataset and rewards
    dataset = task.get_train_dataset(num_samples=10 if args.debug else 100)
    reward_funcs = task.get_reward_funcs()
    
    # 5. Initialize GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config
    )

    # 6. RLHF Training Loop
    print("Starting GRPO Training Loop...")
    trainer.train()

    print("✅ Training complete. Saving model...")
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()
