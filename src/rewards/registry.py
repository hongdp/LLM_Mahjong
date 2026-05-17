from typing import Type, Dict
from src.rewards.base import BaseRewardModel

_REWARD_REGISTRY: Dict[str, Type[BaseRewardModel]] = {}

def register_reward(name: str):
    """Decorator to register a reward model."""
    def decorator(cls: Type[BaseRewardModel]):
        _REWARD_REGISTRY[name] = cls
        return cls
    return decorator

def get_reward_model(name: str, **kwargs) -> BaseRewardModel:
    """Factory method to instantiate a reward model by name."""
    if name not in _REWARD_REGISTRY:
        raise ValueError(f"Reward model '{name}' not found. Available models: {list(_REWARD_REGISTRY.keys())}")
    return _REWARD_REGISTRY[name](**kwargs)
