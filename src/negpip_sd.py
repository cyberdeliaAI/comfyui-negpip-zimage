"""NegPiP encoder and cross-attention patches for SD1 and SDXL.

Derived from ComfyUI-ppm's NegPiP implementation, which in turn credits
laksjdjf and hako-mikan. Released as part of this AGPL-3.0 project.
"""

from __future__ import annotations

from typing import Any

import torch

from comfy import model_management
from comfy.sd1_clip import SDClipModel, gen_empty_tokens


def sd_attn2_negpip(q, k, v, extra_options: dict[str, Any]):
    """Collapse paired key/value tokens produced by the patched encoder."""

    del extra_options
    return q, k[:, 0::2], v[:, 1::2]


def encode_token_weights_negpip(
    encoder: SDClipModel,
    token_weight_pairs,
):
    """Encode prompt weights as paired key/value embeddings for NegPiP."""

    to_encode = []
    max_token_len = 0
    has_weights = False
    for section in token_weight_pairs:
        tokens = [item[0] for item in section]
        max_token_len = max(len(tokens), max_token_len)
        has_weights = has_weights or any(item[1] != 1.0 for item in section)
        to_encode.append(tokens)

    sections = len(to_encode)
    if has_weights or sections == 0:
        if hasattr(encoder, "gen_empty_tokens"):
            to_encode.append(encoder.gen_empty_tokens(encoder.special_tokens, max_token_len))
        else:
            to_encode.append(gen_empty_tokens(encoder.special_tokens, max_token_len))

    encoded = encoder.encode(to_encode)
    embeddings, pooled = encoded[:2]
    first_pooled = (
        pooled[0:1].to(device=model_management.intermediate_device())
        if pooled is not None
        else None
    )

    output = []
    for section_index in range(sections):
        key_embeddings = embeddings[section_index : section_index + 1].clone()
        value_embeddings = embeddings[section_index : section_index + 1].clone()

        if has_weights:
            empty_embeddings = embeddings[-1]
            for token_index in range(key_embeddings.shape[1]):
                weight = token_weight_pairs[section_index][token_index][1]
                if weight == 1.0:
                    continue

                key_embeddings[0, token_index] = (
                    key_embeddings[0, token_index] - empty_embeddings[token_index]
                ) * abs(weight) + empty_embeddings[token_index]
                value_embeddings[0, token_index] = (
                    value_embeddings[0, token_index] - empty_embeddings[token_index]
                ) * abs(weight) + empty_embeddings[token_index]
                if weight < 0:
                    value_embeddings[0, token_index] = -value_embeddings[0, token_index]

        paired = torch.zeros_like(key_embeddings).repeat(1, 2, 1)
        paired[:, 0::2, :] = key_embeddings
        paired[:, 1::2, :] = value_embeddings
        output.append(paired)

    if output:
        result = (
            torch.cat(output, dim=-2).to(device=model_management.intermediate_device()),
            first_pooled,
        )
    else:
        result = (
            embeddings[-1:].to(device=model_management.intermediate_device()),
            first_pooled,
        )

    if len(encoded) > 2:
        extra = {}
        for key, value in encoded[2].items():
            if key == "attention_mask":
                value = value[:sections].flatten().unsqueeze(0).to(
                    device=model_management.intermediate_device()
                )
            extra[key] = value
        result += (extra,)

    return result
