"""Current-ComfyUI Cosmos attention hooks used by Anima NegPiP.

Adapted from ComfyUI-ppm under the AGPL-3.0 license.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import einops
import torch

from comfy.ldm.cosmos.predict2 import Attention as CosmosAttention
from comfy.ldm.cosmos.predict2 import MiniTrainDIT
from comfy.model_patcher import ModelPatcher

PATCH_PREFIX = "cyberdelia_negpip_attn"
ATTN_V_PROJ_CA_KEY = f"{PATCH_PREFIX}_v_proj_ca"

try:
    import comfy.quant_ops

    apply_rope_split_half = comfy.quant_ops.ck.apply_rope_split_half
except (ImportError, AttributeError):

    def _apply_rotary_pos_emb(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        tensor_float = (
            tensor.reshape(*tensor.shape[:-1], 2, -1)
            .movedim(-2, -1)
            .unsqueeze(-2)
            .float()
        )
        output = freqs[..., 0] * tensor_float[..., 0] + freqs[..., 1] * tensor_float[..., 1]
        return output.movedim(-1, -2).reshape(*tensor.shape).type_as(tensor)

    def apply_rope_split_half(
        q: torch.Tensor,
        k: torch.Tensor,
        rope_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _apply_rotary_pos_emb(q, rope_emb), _apply_rotary_pos_emb(k, rope_emb)


def patch_cosmos_attention(model_patcher: ModelPatcher) -> None:
    cosmos_model: MiniTrainDIT = model_patcher.get_model_object("diffusion_model")  # type: ignore
    for block_name, block in cosmos_model.named_modules():
        if not (
            ("cross_attn" in block_name or "self_attn" in block_name)
            and isinstance(block, CosmosAttention)
        ):
            continue

        patch_name = f"diffusion_model.{block_name}.forward"
        if patch_name not in model_patcher.object_patches:
            model_patcher.add_object_patch(patch_name, partial(_forward_patched, block))


def _forward_patched(
    attention: CosmosAttention,
    x: torch.Tensor,
    context: torch.Tensor | None = None,
    rope_emb: torch.Tensor | None = None,
    transformer_options: dict | None = None,
) -> torch.Tensor:
    options = transformer_options if transformer_options is not None else {}
    is_cross_attention = context is not None

    x, context, rope_emb, options = _calculate_patched(
        f"{PATCH_PREFIX}_pre_ca" if is_cross_attention else f"{PATCH_PREFIX}_pre_sa",
        options,
        None,
        x,
        context,
        rope_emb,
        options,
    )

    q, k, v = _compute_qkv_patched(
        attention,
        x,
        context,
        rope_emb=rope_emb,
        transformer_options=options,
    )
    output = attention.compute_attention(q, k, v, transformer_options=options)
    return _calculate_patched(
        f"{PATCH_PREFIX}_output_ca" if is_cross_attention else f"{PATCH_PREFIX}_output_sa",
        options,
        None,
        output,
    )


def _compute_qkv_patched(
    attention: CosmosAttention,
    x: torch.Tensor,
    context: torch.Tensor | None = None,
    rope_emb: torch.Tensor | None = None,
    transformer_options: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    options = transformer_options if transformer_options is not None else {}
    if context is not None:
        q = _calculate_patched(f"{PATCH_PREFIX}_q_proj_ca", options, attention.q_proj, x)
        k = _calculate_patched(f"{PATCH_PREFIX}_k_proj_ca", options, attention.k_proj, context)
        v = _calculate_patched(ATTN_V_PROJ_CA_KEY, options, attention.v_proj, context)
    else:
        q = _calculate_patched(f"{PATCH_PREFIX}_q_proj_sa", options, attention.q_proj, x)
        k = _calculate_patched(f"{PATCH_PREFIX}_k_proj_sa", options, attention.k_proj, x)
        v = _calculate_patched(f"{PATCH_PREFIX}_v_proj_sa", options, attention.v_proj, x)

    q, k, v = map(
        lambda tensor: einops.rearrange(
            tensor,
            "b ... (h d) -> b ... h d",
            h=attention.n_heads,
            d=attention.head_dim,
        ),
        (q, k, v),
    )
    q = _calculate_patched(f"{PATCH_PREFIX}_q_norm", options, attention.q_norm, q)
    k = _calculate_patched(f"{PATCH_PREFIX}_k_norm", options, attention.k_norm, k)
    v = _calculate_patched(f"{PATCH_PREFIX}_v_norm", options, attention.v_norm, v)

    if attention.is_selfattn and rope_emb is not None:
        q, k = _calculate_patched(
            f"{PATCH_PREFIX}_rope",
            options,
            apply_rope_split_half,
            q,
            k,
            rope_emb,
        )
    return q, k, v


def _calculate_patched(
    patch_key: str,
    transformer_options: dict[str, Any],
    function: Callable | None,
    *args,
    **kwargs,
):
    patch = transformer_options.get(patch_key)
    if patch is not None:
        if function is not None:
            return patch(function, transformer_options, *args, **kwargs)
        return patch(transformer_options, *args, **kwargs)
    if function is not None:
        return function(*args, **kwargs)
    if not args:
        return None
    if len(args) == 1:
        return args[0]
    return args
