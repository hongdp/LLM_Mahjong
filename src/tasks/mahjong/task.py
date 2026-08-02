import torch
from src.core.task import BaseTask
from src.core.registry import register_task
from src.core.rollout import ReplayBuffer
from src.tasks.mahjong.orchestrator import run_rollout
from src.tasks.mahjong.batch_rollout import run_rollout_batched
from src.tasks.mahjong.rewards import REWARD_MODELS

@register_task("mahjong")
class MahjongTask(BaseTask):
    """
    Mahjong multi-agent task.
    Runs the LangGraph orchestrator to collect 4-player trajectories.
    """
    def __init__(self, **kwargs):
        self.device = kwargs.get('device', 'cpu')
        # "step" = legacy absolute shaping; "potential" = energy-consistent
        # PBRS (see rewards.MahjongPotentialReward). Selected via config.
        name = kwargs.get('reward_model', 'step')
        if name not in REWARD_MODELS:
            raise ValueError(f"Unknown reward_model '{name}'. "
                             f"Available: {list(REWARD_MODELS)}")
        self.step_reward_model = REWARD_MODELS[name](device=self.device)
        print(f"🏅 Reward model: {name} ({type(self.step_reward_model).__name__})")
        # Prompt template variant: computed value facts (自家宝牌 line).
        # Must match the template the SFT adapter was trained on.
        self.value_facts = bool(kwargs.get('value_facts', False))
        if self.value_facts:
            print("💠 Prompt value facts: ON (自家宝牌 line in private state)")
        # >1 routes rollouts through the batched scheduler (near-linear
        # speedup on the host-launch-bound decode; semantics unchanged).
        self.parallel_games = int(kwargs.get('parallel_games', 1))
        if self.parallel_games > 1:
            print(f"⚡ Parallel rollout: {self.parallel_games} concurrent games")
        # Deal-luck control variate: subtract a fitted function of the
        # initial hand quality from episode returns before normalization.
        self.covariate_baseline = bool(kwargs.get('covariate_baseline', False))
        if self.covariate_baseline:
            print("🎯 Covariate baseline: ON (returns corrected by initial-hand quality)")

    def collect_rollouts(self, num_episodes: int, model=None, tokenizer=None,
                         exp_dir: str = None,
                         capture_logprobs: bool = False) -> ReplayBuffer:
        print(f"🎲 Rolling out {num_episodes} Mahjong games...")

        # Run the interactive graph (batched scheduler when parallel > 1)
        if self.parallel_games > 1:
            episodes = run_rollout_batched(
                num_episodes, model, tokenizer, exp_dir,
                capture_logprobs=capture_logprobs,
                value_facts=self.value_facts,
                parallel=self.parallel_games)
        else:
            episodes = run_rollout(num_episodes, model, tokenizer, exp_dir,
                                   capture_logprobs=capture_logprobs,
                                   value_facts=self.value_facts)
        
        buffer = ReplayBuffer(gamma=0.99)
        
        # We need to apply our Step-level heuristic rewards to the raw trajectory steps 
        # (since table engine only gave sparse rewards or basic XML validation)
        # We batch process all prompts and actions to calculate the step reward.
        for episode in episodes:
            prompts = [step.prompt_text for step in episode]
            actions = [step.action_text for step in episode]
            
            # Use our registered reward model
            step_rewards = self.step_reward_model.compute_reward(prompts, actions)
            
            # Combine engine reward with heuristic reward
            for step, s_reward in zip(episode, step_rewards):
                step.reward += s_reward.item()

            if self.covariate_baseline and episode:
                from src.tasks.mahjong.rewards import initial_hand_energy
                buffer.add_episode(
                    episode,
                    covariate=initial_hand_energy(episode[0].prompt_text))
            else:
                buffer.add_episode(episode)
            
        print(f"📊 Collected {len(buffer.episodes)} player trajectories.")
        return buffer
