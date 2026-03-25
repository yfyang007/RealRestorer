import math

import torch
import torch.nn.functional as F

try:
    from xfuser.model_executor.layers.usp import USP
except ImportError:
    USP = None

try:
    from flash_attn.flash_attn_interface import flash_attn_func
except ImportError:
    flash_attn_func = None


MEMORY_LAYOUT = {
    "flash": (
        lambda x: x,
        lambda x: x,
    ),
    "torch": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "xdit": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}


def attention(
    q,
    k,
    v,
    mode="flash",
    drop_rate=0.0,
    attn_mask=None,
    causal=False,
):
    if mode == "flash" and flash_attn_func is None:
        mode = "torch"

    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]

    q = pre_attn_layout(q)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)

    if mode == "torch":
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        )
    elif mode == "flash":
        if attn_mask is not None:
            raise ValueError("Flash attention mode does not support attention masks in RealRestorer.")
        x = flash_attn_func(q, k, v, dropout_p=drop_rate, causal=causal, softmax_scale=None)
    elif mode == "vanilla":
        scale_factor = 1 / math.sqrt(q.size(-1))

        b, a, s, _ = q.shape
        s1 = k.size(2)
        attn_bias = torch.zeros(b, a, s, s1, dtype=q.dtype, device=q.device)

        if causal:
            if attn_mask is not None:
                raise ValueError("Causal mask and attn_mask cannot be used together.")
            temp_mask = torch.ones(b, a, s, s, dtype=torch.bool, device=q.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias = attn_bias.to(q.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        attn = (q @ k.transpose(-2, -1)) * scale_factor
        attn += attn_bias
        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=True)
        x = attn @ v
    elif mode == "xdit":
        if USP is None:
            raise ImportError("xDiT attention requires xfuser to be installed.")
        x = USP(q, k, v, dropout_p=drop_rate, is_causal=causal)
    else:
        raise NotImplementedError(f"Unsupported attention mode: {mode}")

    x = post_attn_layout(x)

    b, s, a, d = x.shape
    return x.reshape(b, s, -1)
