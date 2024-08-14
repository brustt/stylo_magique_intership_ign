"""
Scale attention
https://github.com/pytorch-labs/segment-anything-fast/blob/main/segment_anything_fast/flash_4.py#L335
"""

import os
import torch

# deactivate flash attn issue with torch.ops version => update to pytorch nightly
USE_CUSTOM_KERNEL = bool(int(os.environ.get('SEGMENT_ANYTHING_FAST_USE_FLASH_4', 0)))

def _attention_rel_h_rel_w(q_, k_, v_, rel_h_, rel_w_):
    """
    Writing this as a composite allows torch.compile to fuse
    the needed padding into previous operations and memory
    allocations.
    """

    import math
    sm_scale = 1. / math.sqrt(q_.size(-1))
    # Check if second last dimension is multiple of 256
    q_size_2_padded = (((q_.size(-2) + 256 - 1) // 256) * 256) - q_.size(-2)

    def kernel_guards(q_, k_, v_):
        return (q_.dtype == torch.bfloat16 or q_.dtype == torch.float16) and q_.dtype == k_.dtype and k_.dtype == v_.dtype and USE_CUSTOM_KERNEL
    # vit_b and vit_l
    # TODO: This kernel currently does not produce correct results for batch size 1 for this case
    if q_.size(0) > 1 and q_size_2_padded == 0 and q_.size(-1) == 64 and kernel_guards(q_, k_, v_):
        rel_h_w = torch.cat([rel_h_.squeeze(-1), rel_w_.squeeze(-2)], dim=-1)
        o = torch.ops.customflash.custom_flash_aligned(
            q_, k_, v_, rel_h_w, sm_scale)
        if o.numel() > 0:
            return o
    # vit_h
    if q_size_2_padded == 0 and q_.size(-1) == 80 and kernel_guards(q_, k_, v_):
        # Only support multiples of 64, so need to pad
        q = torch.nn.functional.pad(q_, (0, 128 - 80, 0, 0), "constant", 0)
        k = torch.nn.functional.pad(k_, (0, 128 - 80, 0, 0), "constant", 0)
        v = torch.nn.functional.pad(v_, (0, 128 - 80, 0, 0), "constant", 0)
        rel_h_w = torch.cat([rel_h_.squeeze(-1), rel_w_.squeeze(-2)], dim=-1)
        o = torch.ops.customflash.custom_flash_aligned(
            q, k, v, rel_h_w, sm_scale)
        if o.numel() > 0:
            return o[:, :, :, :80]
    attn_bias = (rel_h_ + rel_w_).view(q_.size(0), q_.size(1),
                                       rel_h_.size(2), rel_h_.size(3) * rel_w_.size(4))
    return torch.nn.functional.scaled_dot_product_attention(q_, k_, v_, attn_mask=attn_bias)