import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from datasets import Dataset

# Import our modular reward system
from src.rewards.registry import get_reward_model
import src.rewards.mahjong_rewards  # Ensures it's registered
from src.mahjong_env.table import PyMahjongTable

def parse_args():
    parser = argparse.ArgumentParser(description="Training Arguments")
    parser.add_argument("--model_name", type=str, default="gpt2", help="HuggingFace model name or path")
    parser.add_argument("--reward_model", type=str, default="mahjong_step", help="Name of the registered reward model")
    parser.add_argument("--learning_rate", type=float, default=1.41e-5)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=4, help="G value for GRPO: number of sampled trajectories per prompt")
    parser.add_argument("--use_qlora", action="store_true", help="Use 4-bit QLoRA")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (Phase 0)")
    return parser.parse_args()

def build_mahjong_dataset(num_samples=100):
    """
    Bootstrapping dataset generator:
    Uses the mock table engine to generate initial starting states.
    """
    table = PyMahjongTable()
    prompts = []
    
    for _ in range(num_samples):
        obs = table.reset()
        # We take player 0's perspective for bootstrapping
        prompt = obs[0] + "\nAction: "
        prompts.append({"prompt": prompt})
        
    return Dataset.from_list(prompts)

def main():
    args = parse_args()
    print("🚀 Starting Mahjong GRPO Training...")
    print(f"Configuration: {args}")

    # 1. Initialize GRPO Configuration
    training_args = GRPOConfig(
        output_dir="./checkpoints/grpo_mahjong",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        max_steps=args.max_steps,
        num_generations=args.num_generations, # 'G' value in GRPO
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

    # Note: GRPOTrainer takes the standard CausalLM model, not the WithValueHead wrapper!
    # Because GRPO eliminates the Critic/Value model entirely.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto" if args.use_qlora else None
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 3. Load modular reward model
    print(f"Loading reward environment: {args.reward_model}")
    reward_model_env = get_reward_model(args.reward_model)

    # GRPOTrainer expects a function signature: (prompts: List[str], completions: List[str]) -> List[float]
    def trl_reward_wrapper(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        # Extract the prompt text if passed as a list of dicts (datasets format)
        prompt_texts = [p if isinstance(p, str) else p[-1]["content"] if isinstance(p, list) else str(p) for p in prompts]
        # Our BaseRewardModel returns torch.Tensors, we convert them to floats for TRL
        rewards_tensors = reward_model_env.compute_reward(prompts=prompt_texts, responses=completions)
        return [r.item() for r in rewards_tensors]

    # 4. Prepare dataset
    dataset = build_mahjong_dataset(num_samples=10 if args.debug else 100)
    
    # 5. Initialize GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[trl_reward_wrapper],
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
