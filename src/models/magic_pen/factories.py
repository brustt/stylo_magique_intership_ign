from typing import Optional, Tuple, Type
from omegaconf import DictConfig
import torch.nn as nn

from models.magic_pen.blocks import Attention, LoRA_Attention, MLPBlock
from models.magic_pen.tuner_strategies import AdapterModule, LoraModule
from .factory import create_module
from .stem import StemModule, BasicStemModule
from .modules import (
    BitemporalTransformerBlocks,
    EncoderNeck,
    PromptEncoder,
    ImagePromptFusion,
    BitemporalEmbeddingFusion,
    IouScoreHead,
    MaskHead
)
from .strategies import FusionStrategyModule, TunerStrategyModule

_REGISTER_STEM_MODULE = {
    'basic': BasicStemModule,
    }

def create_stem_module(config: DictConfig) -> StemModule:
    stem_type = config.get('type', 'basic')
    stem_config = config.get('params', {})
    
    if stem_type not in _REGISTER_STEM_MODULE:
        raise ValueError(f"Unknown stem type: {stem_type}")
    
    return create_module(stem_config, _REGISTER_STEM_MODULE[stem_type])

def create_bitemporal_transformer_blocks(config: DictConfig, tuner_strategy: TunerStrategyModule) -> nn.Module:
    attention_block = create_attention(
        dim=config.dim,
        num_heads=config.num_heads,
        qkv_bias=config.qkv_bias,
        use_rel_pos=config.use_rel_pos,
        rel_pos_zero_init=config.rel_pos_zero_init,
        input_size=config.input_size,
        tuner_strategy=tuner_strategy
    )
    
    mlp_block = create_mlp(
        dim=config.dim,
        mlp_ratio=config.mlp_ratio,
        act_layer=config.act_layer,
        tuner_strategy=tuner_strategy
    )
    
    return create_module(
        config, 
        BitemporalTransformerBlocks, 
        attention_block=attention_block,
        mlp_block=mlp_block,
        tuner_strategy=tuner_strategy
    )

def create_encoder_neck(config: DictConfig) -> EncoderNeck:
    return create_module(config, EncoderNeck)

def create_prompt_encoder(config: DictConfig) -> PromptEncoder:
    return create_module(config, PromptEncoder)

def create_image_prompt_fusion_module(config: DictConfig) -> ImagePromptFusion:
    return create_module(config, ImagePromptFusion)

def create_bitemporal_embedding_fusion_module(config: DictConfig, fusion_strategy: FusionStrategyModule) -> BitemporalEmbeddingFusion:
    return create_module(config, BitemporalEmbeddingFusion, fusion_strategy=fusion_strategy)

def create_iou_score_head(config: DictConfig) -> IouScoreHead:
    return create_module(config, IouScoreHead)

def create_mask_head(config: DictConfig) -> MaskHead:
    return create_module(config, MaskHead)

def create_sam_model(config: DictConfig) -> nn.Module:
    # Implementation to create and return SAM model components
    pass



def create_fusion_strategy(config: DictConfig) -> FusionStrategyModule:
    return create_module(config, FusionStrategyModule)

def create_tuner_strategy(config: DictConfig) -> TunerStrategyModule:
    return create_module(config, TunerStrategyModule)


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
        return nn.Sequential([
            
            MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer),
            AdapterMLP(dim, 
                    mlp_ratio, 
                    act_layer, 
                    adapter_dim=tuner_strategy.dim, 
                    adapter_mlp_ratio=tuner_strategy.mlp_ratio
                )
        ])
    else:
        return MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
