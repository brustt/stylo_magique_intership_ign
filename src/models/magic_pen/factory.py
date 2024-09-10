from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import Any, Type

def create_module(config: DictConfig, base_class: Type[Any], **kwargs: Any) -> Any:
    """
    Generic factory function to create modules.
    
    Args:
        config (DictConfig): The configuration for the module.
        base_class (Type[Any]): The base class of the module to create.
        **kwargs: Additional keyword arguments to pass to the module constructor.
    
    Returns:
        An instance of the specified module.
    """
    if not isinstance(config, DictConfig):
        raise ValueError(f"Expected DictConfig, got {type(config)}")

    if '_target_' not in config:
        raise ValueError("Configuration must include a '_target_' field")

    module_class = instantiate(config, _recursive_=False)
    
    if not isinstance(module_class, base_class):
        raise TypeError(f"Created object is not an instance of {base_class}")

    return module_class(**kwargs)