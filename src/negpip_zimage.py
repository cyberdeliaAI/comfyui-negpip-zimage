from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import torch

import comfy.utils
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.lumina.model import JointAttention
from comfy.ldm.modules.attention import optimized_attention_masked

NEGPIP_TOKENS_KEY = "cyberdelia_negpip_zimage_tokens"
NEGPIP_STRENGTH_KEY = "cyberdelia_negpip_zimage_strength"
NEGPIP_SUFFIX_LENGTH_KEY = "cyberdelia_negpip_zimage_suffix_length"

_ORIGINAL_TOKEN_COUNT_KEY = "cyberdelia_negpip_zimage_original_token_count"
_NEGATIVE_TOKEN_COUNT_KEY = "cyberdelia_negpip_zimage_negative_token_count"
_COND_OR_UNCOND_KEY = "cyberdelia_negpip_zimage_cond_or_uncond"

ZIMAGE_CHAT_PREFIX = "<|im_start|>user\n"
ZIMAGE_CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
NEGATIVE_BLOCK = re.compile(r"\(([^()]*?):\s*(-?\d+(?:\.\d+)?)\)")


class TokenWeightList(list):
    """Token list carrying extracted negative spans to the encoder patch."""

    negpip_spans: list[tuple[str, float]]


def make_zimage_tokenize_with_weights(original: Callable):
    def tokenize_with_weights(text: str, return_word_ids: bool = False, **kwargs):
        negated_spans: list[tuple[str, float]] = []

        def remove_negative_block(match: re.Match[str]) -> str:
            weight = float(match.group(2))
            if weight < 0:
                negated_spans.append((match.group(1), weight))
                return " "
            return match.group(0)

        clean_text = NEGATIVE_BLOCK.sub(remove_negative_block, text)
        result = original(clean_text, return_word_ids=return_word_ids, **kwargs)

        if negated_spans and "qwen3_4b" in result:
            tokens = TokenWeightList(result["qwen3_4b"])
            tokens.negpip_spans = negated_spans
            result["qwen3_4b"] = tokens

        return result

    return tokenize_with_weights


def zimage_encode_token_weights_negpip(real_encoder, hf_tokenizer, token_weight_pairs):
    """Encode the positive prompt normally and negative spans separately."""

    from comfy import model_management

    sections = len(token_weight_pairs)
    token_ids = [[item[0] for item in chunk] for chunk in token_weight_pairs]
    encoded = real_encoder.encode(token_ids)
    embeddings, pooled = encoded[:2]

    first_pooled = (
        pooled[0:1].to(device=model_management.intermediate_device())
        if pooled is not None
        else None
    )
    normal_embeddings = (
        torch.cat([embeddings[index : index + 1].clone() for index in range(sections)], dim=-2)
        if sections
        else embeddings[-1:].clone()
    ).to(device=model_management.intermediate_device())

    extra: dict[str, Any] = {}
    if len(encoded) > 2 and encoded[2]:
        for key, value in encoded[2].items():
            if key == "attention_mask":
                value = value[:sections].flatten().unsqueeze(0).to(
                    device=model_management.intermediate_device()
                )
            extra[key] = value

    negated_spans = getattr(token_weight_pairs, "negpip_spans", [])
    if negated_spans:
        prefix_length = len(hf_tokenizer.encode(ZIMAGE_CHAT_PREFIX, add_special_tokens=False))
        suffix_length = len(hf_tokenizer.encode(ZIMAGE_CHAT_SUFFIX, add_special_tokens=False))
        negative_embeddings = []
        negative_strengths: list[float] = []

        for span_text, weight in negated_spans:
            templated = ZIMAGE_CHAT_PREFIX + span_text + ZIMAGE_CHAT_SUFFIX
            negative_ids = hf_tokenizer.encode(templated, add_special_tokens=False)
            span_encoded = real_encoder.encode([negative_ids])[0][0:1]

            end = span_encoded.shape[1] - suffix_length if suffix_length else span_encoded.shape[1]
            if end > prefix_length:
                span_embeddings = span_encoded[:, prefix_length:end, :]
            else:
                span_embeddings = span_encoded

            negative_embeddings.append(span_embeddings)
            negative_strengths.extend([float(weight)] * span_embeddings.shape[1])

        if negative_embeddings:
            injected = torch.cat(negative_embeddings, dim=1).to(
                device=model_management.intermediate_device()
            )
            strength = torch.tensor(
                negative_strengths,
                dtype=torch.float32,
                device=model_management.intermediate_device(),
            ).unsqueeze(0)

            extra[NEGPIP_TOKENS_KEY] = injected
            extra[NEGPIP_STRENGTH_KEY] = strength
            extra[NEGPIP_SUFFIX_LENGTH_KEY] = suffix_length

            attention_mask = extra.get("attention_mask")
            if torch.is_tensor(attention_mask) and attention_mask.ndim == 2:
                mask_tokens = torch.ones(
                    attention_mask.shape[0],
                    injected.shape[1],
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                insert_at = max(0, attention_mask.shape[1] - suffix_length)
                extra["attention_mask"] = torch.cat(
                    [attention_mask[:, :insert_at], mask_tokens, attention_mask[:, insert_at:]],
                    dim=1,
                )

    result = (normal_embeddings, first_pooled)
    if extra:
        result += (extra,)
    return result


def _is_unconditional(batch_index: int, batch_size: int, cond_or_uncond) -> bool:
    if not cond_or_uncond:
        return False
    group_size = max(1, batch_size // len(cond_or_uncond))
    group_index = min(batch_index // group_size, len(cond_or_uncond) - 1)
    return cond_or_uncond[group_index] == 1


def _extend_attention_mask(mask, original_length: int, negative_count: int, suffix_length: int):
    if not torch.is_tensor(mask) or mask.ndim != 2:
        return mask

    target_length = original_length + negative_count
    if mask.shape[1] == target_length:
        # The text encoder already inserted the negative-token mask.
        return mask
    if mask.shape[1] != original_length:
        return mask

    mask_tokens = torch.ones(
        mask.shape[0], negative_count, dtype=mask.dtype, device=mask.device
    )
    insert_at = max(0, original_length - suffix_length)
    return torch.cat([mask[:, :insert_at], mask_tokens, mask[:, insert_at:]], dim=1)


def lumina_diffusion_negpip_wrapper(executor, *args, **kwargs):
    transformer_options: dict[str, Any] = kwargs.get("transformer_options", {})
    negative_embeddings = kwargs.pop(NEGPIP_TOKENS_KEY, None)
    negative_strength = kwargs.pop(NEGPIP_STRENGTH_KEY, None)
    suffix_length = kwargs.pop(NEGPIP_SUFFIX_LENGTH_KEY, 5)

    if negative_embeddings is None:
        return executor(*args, **kwargs)
    if len(args) < 3:
        return executor(*args, **kwargs)

    context: torch.Tensor = args[2]
    original_length = context.shape[1]
    suffix_length = int(suffix_length)
    if original_length <= suffix_length:
        return executor(*args, **kwargs)

    batch_size = context.shape[0]
    cond_or_uncond = transformer_options.get("cond_or_uncond")
    if negative_embeddings.ndim == 2:
        negative_embeddings = negative_embeddings.unsqueeze(0)
    negative_embeddings = comfy.utils.repeat_to_batch_size(
        negative_embeddings, batch_size
    ).to(context)
    negative_count = negative_embeddings.shape[1]

    if negative_strength is None:
        negative_strength = torch.full(
            (1, negative_count), -1.0, dtype=torch.float32, device=context.device
        )
    elif negative_strength.ndim == 1:
        negative_strength = negative_strength.unsqueeze(0)
    negative_strength = comfy.utils.repeat_to_batch_size(
        negative_strength, batch_size
    ).to(device=context.device)

    insert_at = original_length - suffix_length
    context_parts = []
    for batch_index in range(batch_size):
        injected = negative_embeddings[batch_index : batch_index + 1]
        if _is_unconditional(batch_index, batch_size, cond_or_uncond):
            injected = torch.zeros_like(injected)
        context_parts.append(
            torch.cat(
                [
                    context[batch_index : batch_index + 1, :insert_at],
                    injected,
                    context[batch_index : batch_index + 1, insert_at:],
                ],
                dim=1,
            )
        )
    new_context = torch.cat(context_parts, dim=0)
    new_length = new_context.shape[1]

    mutable_args = list(args)
    mutable_args[2] = new_context
    if len(mutable_args) > 3 and mutable_args[3] is not None:
        num_tokens = mutable_args[3]
        if isinstance(num_tokens, int):
            mutable_args[3] = new_length
        elif isinstance(num_tokens, list):
            mutable_args[3] = [new_length] * len(num_tokens)
        elif isinstance(num_tokens, tuple):
            mutable_args[3] = (new_length,) * len(num_tokens)
        elif torch.is_tensor(num_tokens):
            mutable_args[3] = torch.full_like(num_tokens, new_length)
    if len(mutable_args) > 4 and mutable_args[4] is not None:
        mutable_args[4] = _extend_attention_mask(
            mutable_args[4], original_length, negative_count, suffix_length
        )

    if "num_tokens" in kwargs:
        num_tokens = kwargs["num_tokens"]
        if isinstance(num_tokens, int):
            kwargs["num_tokens"] = new_length
        elif torch.is_tensor(num_tokens):
            kwargs["num_tokens"] = torch.full_like(num_tokens, new_length)
    if kwargs.get("attention_mask") is not None:
        kwargs["attention_mask"] = _extend_attention_mask(
            kwargs["attention_mask"], original_length, negative_count, suffix_length
        )

    state = {
        _ORIGINAL_TOKEN_COUNT_KEY: original_length,
        _NEGATIVE_TOKEN_COUNT_KEY: negative_count,
        _COND_OR_UNCOND_KEY: cond_or_uncond,
        NEGPIP_STRENGTH_KEY: negative_strength,
        NEGPIP_SUFFIX_LENGTH_KEY: suffix_length,
    }
    transformer_options.update(state)
    kwargs["transformer_options"] = transformer_options

    try:
        return executor(*tuple(mutable_args), **kwargs)
    finally:
        for key in state:
            transformer_options.pop(key, None)


def make_joint_attention_forward_negpip(block: JointAttention):
    q_dim = block.n_local_heads * block.head_dim
    k_dim = block.n_local_kv_heads * block.head_dim
    v_dim = block.n_local_kv_heads * block.head_dim

    def forward(x, x_mask, freqs_cis, transformer_options=None):
        transformer_options = transformer_options or {}
        original_length = transformer_options.get(_ORIGINAL_TOKEN_COUNT_KEY)
        negative_count = transformer_options.get(_NEGATIVE_TOKEN_COUNT_KEY)
        suffix_length = transformer_options.get(NEGPIP_SUFFIX_LENGTH_KEY, 5)
        cond_or_uncond = transformer_options.get(_COND_OR_UNCOND_KEY)
        strength = transformer_options.get(NEGPIP_STRENGTH_KEY)

        batch_size, sequence_length, _ = x.shape
        qkv = block.qkv(x)
        xq, xk, xv = torch.split(qkv, [q_dim, k_dim, v_dim], dim=-1)
        xq = xq.view(batch_size, sequence_length, block.n_local_heads, block.head_dim)
        xk = xk.view(batch_size, sequence_length, block.n_local_kv_heads, block.head_dim)
        xv = xv.view(batch_size, sequence_length, block.n_local_kv_heads, block.head_dim)

        if original_length is not None and negative_count is not None:
            start = original_length - suffix_length
            end = start + negative_count
            if 0 <= start < end <= sequence_length:
                if strength is None:
                    strength = torch.ones(
                        (batch_size, negative_count), dtype=xv.dtype, device=xv.device
                    )
                else:
                    strength = comfy.utils.repeat_to_batch_size(strength, batch_size).to(
                        dtype=xv.dtype, device=xv.device
                    )

                for batch_index in range(batch_size):
                    if not _is_unconditional(batch_index, batch_size, cond_or_uncond):
                        scale = torch.abs(strength[batch_index : batch_index + 1]).view(
                            1, negative_count, 1, 1
                        )
                        xv[batch_index : batch_index + 1, start:end] *= -scale

        xq = block.q_norm(xq)
        xk = block.k_norm(xk)
        xq, xk = apply_rope(xq, xk, freqs_cis)

        repetitions = block.n_local_heads // block.n_local_kv_heads
        if repetitions >= 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, repetitions, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, repetitions, 1).flatten(2, 3)

        output = optimized_attention_masked(
            xq.movedim(1, 2),
            xk.movedim(1, 2),
            xv.movedim(1, 2),
            block.n_local_heads,
            x_mask,
            skip_reshape=True,
            transformer_options=transformer_options,
        )
        return block.out(output)

    return forward
