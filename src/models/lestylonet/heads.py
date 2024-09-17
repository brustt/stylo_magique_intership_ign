from models.lestylonet.blocks import LayerNorm2d
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch.nn import functional as F

"""MASKHEAD"""
class MaskHeadStrategy(nn.Module,ABC):
    @abstractmethod
    def forward(self, hs, src):
        raise NotImplementedError

class MaskHeadFactory:
    @staticmethod
    def create(strategy_type, **kwargs):
        if strategy_type == "default":
            return MaskHead(SAMMaskHead(**kwargs))
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
class SAMMaskHead(MaskHeadStrategy):
    def __init__(self, 
            num_mask_tokens,
            transformer_dim: int):
        self.num_mask_tokens = num_mask_tokens
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            nn.GELU(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

    def forward(self, tokens, mask_embedding):
        b, n, c, h, w = mask_embedding.shape
        mask_tokens_out = tokens[:, :, 1 : (1 + self.num_mask_tokens), :]

        upscaled_embedding = torch.stack(
            [self.output_upscaling(im) for im in mask_embedding], dim=0
        )

        hyper_in = torch.stack(
            [
                model(mask_tokens_out[:, :, i, :])
                for i, model in enumerate(self.output_hypernetworks_mlps)
            ],
            dim=2,
        )

        masks = (hyper_in @ upscaled_embedding.view(b, n, c, h * w)).view(
            b, n, -1, h, w
        )

        return masks


class MaskHead(nn.Module):
    def __init__(self, strategy):
        super().__init__()
        self.strategy = strategy

    def forward(self, tokens, mask_embedding):
        return self.strategy.forward(tokens, mask_embedding)

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
    

"""IOUHEAD"""
class IouPredHead(nn.Module):
    def __init__(self, 
                 transformer_dim, 
                 iou_head_hidden_dim, 
                 num_mask_tokens, 
                 iou_head_depth):
        super().__init__()
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, num_mask_tokens, iou_head_depth
        )

    def forward(self, tokens):
        iou_token_out = tokens[:, :, 0, :]
        iou_pred = self.iou_prediction_head(iou_token_out)
        return iou_pred