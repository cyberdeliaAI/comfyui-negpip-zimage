from __future__ import annotations

import copy
from functools import partial
from typing import Any

import comfy.conds
import comfy.patcher_extension
from comfy.ldm.lumina.model import JointAttention, NextDiT
from comfy.model_base import Lumina2
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP
from comfy_api.latest import io

from .negpip_zimage import (
    NEGPIP_STRENGTH_KEY,
    NEGPIP_SUFFIX_LENGTH_KEY,
    NEGPIP_TOKENS_KEY,
    lumina_diffusion_negpip_wrapper,
    make_joint_attention_forward_negpip,
    make_zimage_tokenize_with_weights,
    zimage_encode_token_weights_negpip,
)
from .prompt_merge import merge_prompts

PATCH_KEY = "cyberdelia_negpip_zimage"
QWEN_ENCODER = "qwen3_4b"


class ZImageNegPipPrompt(io.ComfyNode):
    """Patch Z-Image and encode positive/negative text as NegPiP conditioning."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ZImageNegPipPrompt",
            display_name="Z-Image NegPiP Prompt",
            category="conditioning/Z-Image",
            description=(
                "Patches Z-Image and merges positive/negative prompts into NegPiP conditioning."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.String.Input("positive", force_input=True),
                io.String.Input("negative", force_input=True),
            ],
            outputs=[
                io.Model.Output(),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.String.Output(display_name="compiled_prompt"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        clip: CLIP = kwargs["clip"]
        m = model.clone()
        c = clip.clone()

        if not issubclass(type(m.model), Lumina2):
            raise ValueError("Z-Image NegPiP Prompt only supports Z-Image/Lumina2 models.")
        if not hasattr(c.patcher.model, QWEN_ENCODER):
            raise ValueError("The connected CLIP does not contain the Z-Image qwen3_4b encoder.")

        if not c.patcher.model_options.get(PATCH_KEY, False):
            cls._patch_clip(c)
            c.patcher.model_options[PATCH_KEY] = True

        if not m.model_options.get(PATCH_KEY, False):
            cls._patch_model(m)
            m.model_options[PATCH_KEY] = True

        merged_prompt = merge_prompts(kwargs["positive"], kwargs["negative"])
        positive = c.encode_from_tokens_scheduled(c.tokenize(merged_prompt))
        negative = c.encode_from_tokens_scheduled(c.tokenize(""))

        return io.NodeOutput(m, positive, negative, merged_prompt)

    @staticmethod
    def _patch_clip(clip: CLIP) -> None:
        encoder_model = getattr(clip.patcher.model, QWEN_ENCODER)

        # ComfyUI's CLIP.clone() shares its tokenizer with the source CLIP.
        # Copy the lightweight wrapper so this node cannot change the tokenizer
        # used by an unpatched CLIP branch in the same workflow.
        clip.tokenizer = copy.copy(clip.tokenizer)
        inner_tokenizer = getattr(clip.tokenizer, QWEN_ENCODER, None)
        hf_tokenizer = getattr(inner_tokenizer, "tokenizer", None)
        if hf_tokenizer is None:
            raise ValueError("Unable to access the Z-Image Qwen tokenizer.")

        clip.tokenizer.tokenize_with_weights = make_zimage_tokenize_with_weights(
            clip.tokenizer.tokenize_with_weights
        )
        clip.patcher.add_object_patch(
            f"{QWEN_ENCODER}.encode_token_weights",
            partial(zimage_encode_token_weights_negpip, encoder_model, hf_tokenizer),
        )

    @staticmethod
    def _patch_model(model: ModelPatcher) -> None:
        diffusion_model: NextDiT = model.get_model_object("diffusion_model")  # type: ignore
        original_extra_conds = model.model.extra_conds

        def extra_conds_negpip(*args, **kwargs):
            out = original_extra_conds(*args, **kwargs)

            negpip_tokens = kwargs.get(NEGPIP_TOKENS_KEY)
            if negpip_tokens is not None:
                out[NEGPIP_TOKENS_KEY] = comfy.conds.CONDRegular(negpip_tokens)

            negpip_strength = kwargs.get(NEGPIP_STRENGTH_KEY)
            if negpip_strength is not None:
                out[NEGPIP_STRENGTH_KEY] = comfy.conds.CONDRegular(negpip_strength)

            suffix_length = kwargs.get(NEGPIP_SUFFIX_LENGTH_KEY)
            if suffix_length is not None:
                out[NEGPIP_SUFFIX_LENGTH_KEY] = comfy.conds.CONDConstant(int(suffix_length))

            return out

        model.add_object_patch("extra_conds", extra_conds_negpip)
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            PATCH_KEY,
            lumina_diffusion_negpip_wrapper,
        )

        patched_layers = 0
        for block_name, block in diffusion_model.named_modules():
            if isinstance(block, JointAttention) and block_name.startswith("layers."):
                model.add_object_patch(
                    f"diffusion_model.{block_name}.forward",
                    make_joint_attention_forward_negpip(block),
                )
                patched_layers += 1

        if patched_layers == 0:
            raise RuntimeError("No compatible Z-Image JointAttention layers were found.")


NODES = [ZImageNegPipPrompt]
