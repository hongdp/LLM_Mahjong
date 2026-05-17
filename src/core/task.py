from abc import ABC, abstractmethod
from typing import List, Callable
from datasets import Dataset

class BaseTask(ABC):
    """
    Abstract Base Class for RLHF tasks.
    A task encapsulates its own environment, trajectory sampling, and reward logic.
    The core RL trainer only needs the train dataset and the list of reward functions.
    """

    @abstractmethod
    def get_train_dataset(self, num_samples: int = 100) -> Dataset:
        """
        Returns a HuggingFace Dataset containing the initial states (prompts).
        Each item must at least contain a 'prompt' string key.
        """
        pass

    @abstractmethod
    def get_reward_funcs(self) -> List[Callable]:
        """
        Returns a list of reward functions for GRPOTrainer.
        Signature of each function: (prompts: List[str], completions: List[str], **kwargs) -> List[float]
        """
        pass
