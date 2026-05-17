from datasets import Dataset
from typing import List, Callable
import torch

from src.core.task import BaseTask
from src.core.registry import register_task
from src.tasks.mahjong.table import PyMahjongTable
from src.tasks.mahjong.rewards import MahjongStepReward

@register_task("mahjong")
class MahjongTask(BaseTask):
    """
    The implementation of the Mahjong task.
    Provides starting datasets and specific reward functions.
    """
    def __init__(self, **kwargs):
        self.device = kwargs.get('device', 'cpu')
        self.table = PyMahjongTable()
        self.step_reward_model = MahjongStepReward(device=self.device)

    def get_train_dataset(self, num_samples: int = 100) -> Dataset:
        prompts = []
        for _ in range(num_samples):
            obs = self.table.reset()
            # Taking player 0's perspective
            prompt = obs[0] + "\nAction: "
            prompts.append({"prompt": prompt})
        return Dataset.from_list(prompts)

    def get_reward_funcs(self) -> List[Callable]:
        # Wrap the BaseRewardModel to match GRPOTrainer's expected signature
        def trl_reward_wrapper(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
            # Extract content from list of dicts if needed
            prompt_texts = [p if isinstance(p, str) else p[-1]["content"] if isinstance(p, list) else str(p) for p in prompts]
            rewards_tensors = self.step_reward_model.compute_reward(prompts=prompt_texts, responses=completions)
            return [r.item() for r in rewards_tensors]

        return [trl_reward_wrapper]
