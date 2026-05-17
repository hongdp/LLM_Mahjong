from typing import Dict, Type, Any
from src.core.task import BaseTask

_TASK_REGISTRY: Dict[str, Type[BaseTask]] = {}

def register_task(name: str):
    """Decorator to register a task."""
    def decorator(cls: Type[BaseTask]):
        _TASK_REGISTRY[name] = cls
        return cls
    return decorator

def get_task(name: str, **kwargs) -> BaseTask:
    """Factory method to instantiate a task by name."""
    if name not in _TASK_REGISTRY:
        raise ValueError(f"Task '{name}' not found. Available models: {list(_TASK_REGISTRY.keys())}")
    return _TASK_REGISTRY[name](**kwargs)
