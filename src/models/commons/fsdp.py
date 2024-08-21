""" FSDP strategy, seze: https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html """
from torch.nn import Module
from lightning.pytorch.strategies import FSDPStrategy

from src.models.segment_anything.modeling.image_encoder_dev import ImageEncoderViT
from src.models.segment_anything.modeling.image_encoder import ImageEncoderViT as OldImageEncoderViT

def fsdp_strategy() -> FSDPStrategy:
    return FSDPStrategy(sharding_strategy="HYBRID_SHARD",
                        auto_wrap_policy={ImageEncoderViT, OldImageEncoderViT},)
