from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch

class BaseRewardModel(ABC):
    """
    Abstract base class for all reward models.
    Different tasks should inherit from this class and implement the `compute_reward` method.
    """
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu", **kwargs):
        self.device = device
        
    @abstractmethod
    def compute_reward(self, prompts: List[str], responses: List[str], **kwargs) -> List[torch.Tensor]:
        """
        Compute the reward for a given set of prompts and responses.
        
        Args:
            prompts: List of prompt strings.
            responses: List of response strings.
            
        Returns:
            A list of 1D tensors containing the scalar reward for each (prompt, response) pair.
        """
        pass
