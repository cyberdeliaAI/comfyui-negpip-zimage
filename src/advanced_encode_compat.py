"""Optional NegPiP compatibility for Advanced CLIP Text Encode.

Adapted from ComfyUI-ppm under the AGPL-3.0 license.
"""

from __future__ import annotations

import logging
import os
import sys
from math import copysign
from types import ModuleType

import torch

_INITIALIZED = False


def patch_advanced_encode() -> None:
    """Patch already-loaded Advanced CLIP Text Encode modules once."""

    global _INITIALIZED
    if _INITIALIZED:
        return

    try:
        for module in _find_loaded_modules("ComfyUI_ADV_CLIP_emb"):
            advanced_encode = module.adv_encode
            advanced_encode.advanced_encode_from_tokens = _negpip_wrapper(
                advanced_encode.advanced_encode_from_tokens,
                advanced_encode.from_zero,
            )
            logging.info("Advanced CLIP Text Encode was patched by comfyui-negpip-zimage")
    finally:
        _INITIALIZED = True


def _find_loaded_modules(name: str) -> list[ModuleType]:
    import folder_paths

    modules = []
    for custom_node_path in folder_paths.get_folder_paths("custom_nodes"):
        for candidate in os.listdir(os.path.realpath(custom_node_path)):
            if name.lower() not in candidate.lower() or candidate.endswith(".disabled"):
                continue

            module_path = os.path.join(custom_node_path, candidate)
            if os.path.isfile(module_path) and os.path.splitext(module_path)[1] != ".py":
                continue

            if os.path.isfile(module_path):
                module_name = os.path.splitext(module_path)[0]
            elif os.path.isdir(module_path):
                module_name = module_path.replace(".", "_x_")
            else:
                continue

            module = sys.modules.get(module_name)
            if module is not None:
                modules.append(module)
    return modules


def _negpip_wrapper(advanced_encode_from_tokens, from_zero):
    def advanced_encode_from_tokens_negpip(
        tokenized,
        token_normalization,
        weight_interpretation,
        encode_func,
        m_token=266,
        length=77,
        w_max=1.0,
        return_pooled=False,
        apply_to_pooled=False,
        **extra_args,
    ):
        absolute_tokens = [
            [(token, abs(weight), word_id) for token, weight, word_id in section]
            for section in tokenized
        ]
        weight_signs = [[copysign(1, weight) for _, weight, _ in section] for section in tokenized]

        marker_tokens = [[(m_token, 1.0) for _ in range(length)]]
        marker_embedding, _ = encode_func(marker_tokens)
        if marker_embedding.shape[1] == length:
            encoded_with_negpip = False
        elif marker_embedding.shape[1] == length * 2:
            encoded_with_negpip = True
        else:
            raise ValueError(
                "Unknown embedding shape: expected "
                f"{length} or {length * 2}, found {marker_embedding.shape[1]}. "
                "Do not apply more than one NegPiP node to the same branch."
            )

        def encode_without_pairs(tokens):
            embedding, pooled = encode_func(tokens)
            if encoded_with_negpip:
                return embedding[:, 0::2, :], pooled
            return embedding, pooled

        weighted_embedding, pooled = advanced_encode_from_tokens(
            absolute_tokens,
            token_normalization,
            weight_interpretation,
            encode_without_pairs,
            m_token,
            length,
            w_max,
            return_pooled,
            apply_to_pooled,
            **extra_args,
        )

        if encoded_with_negpip:
            signed_embedding = torch.empty_like(weighted_embedding).repeat(1, 2, 1)
            signed_embedding[:, 0::2, :] = weighted_embedding
            signed_embedding[:, 1::2, :] = from_zero(weight_signs, weighted_embedding)
            weighted_embedding = signed_embedding

        return weighted_embedding, pooled

    return advanced_encode_from_tokens_negpip
