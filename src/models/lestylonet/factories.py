from models.lestylonet.blocks import AdapterModule, TunerStrategyModule
from models.lestylonet.fusion_strategies import BitemporalEmbeddingFusion, FusionStrategyModule
from models.lestylonet.heads import IouPredHead, MaskHead
from models.lestylonet.neck import NeckModule
from models.lestylonet.prompt_encoder import PromptEncoder
from models.lestylonet.prompt_to_image import PromptToImageFusion

from omegaconf import DictConfig
import torch.nn as nn
from omegaconf import DictConfig
import hydra
from typing import Any, Optional, Type

from .stem import StemModule

def _create_module(config: DictConfig, **kwargs) -> Any:
    """
    Generic factory function to create modules.
    
    Args:
        config (DictConfig): The configuration for the module.
        kwargs : Additionnal keywords arguments or override arguments
    
    Returns:
        An instance of the specified module.
    """
    if not isinstance(config, DictConfig):
        raise ValueError(f"Expected DictConfig, got {type(config)}")

    if '_target_' not in config:
        raise ValueError("Configuration must include a '_target_' field")

    return hydra.utils.instantiate(config, _recursive_=False, **kwargs)

def create_stem_module(config: DictConfig) -> StemModule:
    _config = config.model.network.image_encoder.stem
    return _create_module(_config)

def create_bitemporal_transformer_blocks(config: DictConfig, tuner_strategy: TunerStrategyModule) -> nn.Module:
    _config = config.model.network.image_encoder.block
    blocks = nn.ModuleList()
    for i in range(_config.depth):
        block = _create_module(_config, layer_id=i, tuner_strategy=tuner_strategy)
        blocks.append(block)
    return blocks

def create_encoder_neck(config: DictConfig) -> NeckModule:
    _config=config.model.network.image_encoder.neck
    return _create_module(_config)

def create_prompt_encoder(config: DictConfig) -> PromptEncoder:
    _config=config.model.network.prompt_encoder
    return _create_module(_config)

def create_prompt_to_image_module(config: DictConfig) -> PromptToImageFusion:
    _config=config.model.network.prompt_to_image
    return _create_module(_config)

def create_bitemporal_embedding_fusion_module(config: DictConfig, fusion_strategy: FusionStrategyModule) -> BitemporalEmbeddingFusion:
    _config=config.model.network.bitemporal_embedding_fusion
    return _create_module(_config, fusion_strategy=fusion_strategy)

def create_iou_score_head(config: DictConfig) -> IouPredHead:
    _config=config.model.network.iou_head
    return _create_module(_config)

def create_mask_head(config: DictConfig) -> MaskHead:
    _config=config.model.network.mask_head
    return _create_module(_config)

def create_sam_model(config: DictConfig) -> nn.Module:
    # Implementation to create and return SAM model components
    pass

def create_fusion_strategy(config: DictConfig) -> FusionStrategyModule:
    _config = config.model.network.fusion_strategy
    return FusionStrategyModule.create(_config)

def create_tuner_strategy(config: DictConfig) -> TunerStrategyModule:
    _config = config.model.network.tuner_strategy
    return TunerStrategyModule.create(_config)

