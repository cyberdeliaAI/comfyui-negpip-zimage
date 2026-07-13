"""NegPiP conditioning and diffusion wrapper for Anima.

Adapted from ComfyUI-ppm under the AGPL-3.0 license.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

import comfy.conds

from .cosmos_attention import ATTN_V_PROJ_CA_KEY

COND_NEGPIP_MASK_KEY = "c_cyberdelia_negpip_anima_mask"
NEGPIP_MASK_KEY = "cyberdelia_negpip_anima_mask"


def anima_extra_conds_negpip(
    extra_conds: Callable[..., dict],
    **kwargs,
):
    """Preserve the sign of Anima's T5 weights as a conditioning mask."""

    t5xxl_weights = kwargs.get("t5xxl_weights")
    negpip_mask = None
    if t5xxl_weights is not None:
        absolute_weights = torch.abs(t5xxl_weights)
        negpip_mask = torch.where(
            t5xxl_weights == absolute_weights,
            torch.ones_like(t5xxl_weights),
            -torch.ones_like(t5xxl_weights),
        ).unsqueeze(0).unsqueeze(-1)

        if negpip_mask.shape[1] < 512:
            negpip_mask = torch.nn.functional.pad(
                negpip_mask,
                (0, 0, 0, 512 - negpip_mask.shape[1]),
                value=1.0,
            )
        kwargs["t5xxl_weights"] = absolute_weights

    output = extra_conds(**kwargs)
    if negpip_mask is not None:
        output[COND_NEGPIP_MASK_KEY] = comfy.conds.CONDRegular(negpip_mask)
    return output


def anima_diffusion_negpip_wrapper(executor, *args, **kwargs):
    """Apply the Anima sign mask to cross-attention value projections."""

    negpip_mask = kwargs.pop(COND_NEGPIP_MASK_KEY, None)
    if negpip_mask is None or len(args) < 3:
        return executor(*args, **kwargs)

    context: torch.Tensor = args[2]
    transformer_options: dict[str, Any] = dict(kwargs.get("transformer_options") or {})
    transformer_options[ATTN_V_PROJ_CA_KEY] = _anima_v_projection_negpip
    transformer_options[NEGPIP_MASK_KEY] = negpip_mask.to(context)
    kwargs["transformer_options"] = transformer_options
    return executor(*args, **kwargs)


def _anima_v_projection_negpip(
    function: Callable,
    transformer_options: dict[str, Any],
    context: torch.Tensor,
) -> torch.Tensor:
    negpip_mask = transformer_options.get(NEGPIP_MASK_KEY)
    value_context = context if negpip_mask is None else context * negpip_mask
    return function(value_context)
