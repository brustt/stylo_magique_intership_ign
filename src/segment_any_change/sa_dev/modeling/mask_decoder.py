# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mas src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)k.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred
    
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        # torch.Size([num_mask+1+1, 256])
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        print(f"init tokens shape : {output_tokens.shape}")

        # torch.Size([N, 2, 256])
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(1), -1, -1
        )

        # torch.Size([B, N, 2, 256])
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1, -1
        )
        print(f"inter tokens shape : {output_tokens.shape}")

        # torch.Size([B, (N), num_mask+3, 256])
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        print(f"tokens shape : {tokens.shape}")

        print(f"img_embedding (src) : {image_embeddings.shape}")
        # Expand per-image data in batch direction to be per-mask => B*B ?
        src = torch.repeat_interleave(image_embeddings.unsqueeze(1), tokens.shape[1], dim=1)
        #dense_prompt_embeddings = torch.repeat_interleave(dense_prompt_embeddings, tokens.shape[0], dim=0)

        print(f"src first interleave shape : {src.shape}")


        print(f"src shape : {src.shape}")
        print(f"src dense_prompt_embeddings : {dense_prompt_embeddings.shape}")

        if dense_prompt_embeddings.ndim < src.ndim:
            dense_prompt_embeddings = dense_prompt_embeddings.unsqueeze(0).expand(
            src.size(0), -1, -1, -1, -1
            )
                    
            print(f"expanded")

        src = src + dense_prompt_embeddings
        # expand over number of points
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[1], dim=0)

        # expand over batch
        pos_src = pos_src.unsqueeze(0).expand(
            tokens.size(0), -1, -1, -1, -1
            )
        b, n, c, h, w = src.shape
        print("--in transformer--")
        print(f"src : {src.shape}")

        print(f"pos_src : {pos_src.shape}")
        print(f"tokens : {tokens.shape}")

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        print("out transformer")
        print(f"hs shape : {hs.shape}")
        print(f"src shape : {src.shape}")

        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        print(f"iou out shape : {iou_token_out.shape}")
        print(f"masks tokens out shape : {mask_tokens_out.shape}")
        
        # edit
        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b*n, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        print(f"upscaled src tokens out shape : {upscaled_embedding.shape}")

        hyper_in_list: List[torch.Tensor] = []
        # need paralelization
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        print(hyper_in_list[0].shape)
        hyper_in = torch.stack(hyper_in_list, dim=1)
        print(f"hyper in shape : {hyper_in.shape}")

        b, c, h, w = upscaled_embedding.shape
        """
        maybe a issue with upscaling
        """
        #masks = (hyper_in @ upscaled_embedding.view(b*n, c, h * w)).view(b, n, -1, h, w)
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w))
        print(f" mask_low : {masks.shape}")

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
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
