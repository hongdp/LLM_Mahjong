import torch
from src.core.task import BaseTask
from src.core.registry import register_task
from src.core.rollout import ReplayBuffer
from src.tasks.mahjong.orchestrator import run_rollout
from src.tasks.mahjong.rewards import MahjongStepReward

@register_task("mahjong")
class MahjongTask(BaseTask):
    """
    Mahjong multi-agent task. 
    Runs the LangGraph orchestrator to collect 4-player trajectories.
    """
    def __init__(self, **kwargs):
        self.device = kwargs.get('device', 'cpu')
        self.step_reward_model = MahjongStepReward(device=self.device)

    def collect_rollouts(self, num_episodes: int, model=None, tokenizer=None) -> ReplayBuffer:
        print(f"🎲 Rolling out {num_episodes} Mahjong games...")
        
        # Run the interactive graph
        episodes = run_rollout(num_episodes, model, tokenizer)
        
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
                
            buffer.add_episode(episode)
            
        print(f"📊 Collected {len(buffer.episodes)} player trajectories.")
        return buffer
