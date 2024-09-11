from typing import Optional, Tuple, Type
from models.magic_pen.fusion_strategies import FusionStrategyModule
from omegaconf import DictConfig
import torch.nn as nn
from omegaconf import DictConfig
import hydra
from typing import Any, Type

from models.magic_pen.blocks import AdapterMLP, Attention, LoRA_Attention, MLPBlock, NeckModule
from models.magic_pen.tuner_strategies import AdapterModule, LoraModule, TunerStrategyModule
from .stem import StemModule
from .modules import (

    PromptEncoder,
    ImagePromptFusion,
    BitemporalEmbeddingFusion,
    IouScoreHead,
    MaskHead
)

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
    return _create_module(_config, tuner_strategy=tuner_strategy)

def create_encoder_neck(config: DictConfig) -> NeckModule:
    _config=config.model.network.image_encoder.neck
    return _create_module(_config)

def create_prompt_encoder(config: DictConfig) -> PromptEncoder:
    _config=config.model.network.prompt_encoder
    return _create_module(_config)

def create_image_prompt_fusion_module(config: DictConfig) -> ImagePromptFusion:
    _config=config.prompt_to_image
    return _create_module(_config)

def create_bitemporal_embedding_fusion_module(config: DictConfig, fusion_strategy: FusionStrategyModule) -> BitemporalEmbeddingFusion:
    return _create_module(config, BitemporalEmbeddingFusion, fusion_strategy=fusion_strategy)

def create_iou_score_head(config: DictConfig) -> IouScoreHead:
    return _create_module(config, IouScoreHead)

def create_mask_head(config: DictConfig) -> MaskHead:
    return _create_module(config, MaskHead)

def create_sam_model(config: DictConfig) -> nn.Module:
    # Implementation to create and return SAM model components
    pass

def create_fusion_strategy(config: DictConfig) -> FusionStrategyModule:
    return _create_module(config, FusionStrategyModule)

def create_tuner_strategy(config: DictConfig) -> TunerStrategyModule:
    return _create_module(config.tuner_strategy)


def create_attention(dim: int, num_heads: int, qkv_bias: bool, use_rel_pos: bool, 
                     rel_pos_zero_init: bool, input_size: Optional[Tuple[int, int]], 
                     tuner_strategy: Optional[TunerStrategyModule] = None) -> nn.Module:
    common_args = {
        'dim': dim,
        'num_heads': num_heads,
        'qkv_bias': qkv_bias,
        'use_rel_pos': use_rel_pos,
        'rel_pos_zero_init': rel_pos_zero_init,
        'input_size': input_size,
    }
    
    if isinstance(tuner_strategy, LoraModule):
        return LoRA_Attention(**common_args, r=tuner_strategy.r, alpha=tuner_strategy.alpha)
    else:
        return Attention(**common_args)
    
def create_mlp(dim: int, mlp_ratio: float, act_layer: Type[nn.Module], 
               tuner_strategy: Optional[TunerStrategyModule] = None) -> nn.Module:
    if isinstance(tuner_strategy, AdapterModule):
        return AdapterMLP(dim, 
                    mlp_ratio, 
                    act_layer, 
                    adapter_dim=tuner_strategy.dim, 
                    adapter_mlp_ratio=tuner_strategy.mlp_ratio
                )
    else:
        return MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
