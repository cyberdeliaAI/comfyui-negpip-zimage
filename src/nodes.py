from __future__ import annotations

import copy
from functools import partial
from typing import Any

import torch

import comfy.conds
import comfy.model_base
import comfy.patcher_extension
from comfy.ldm.lumina.model import JointAttention, NextDiT
from comfy.model_base import Anima, BaseModel, Flux, Lumina2, SDXL, SDXLRefiner
from comfy.model_patcher import ModelPatcher
from comfy.sd import CLIP
from comfy_api.latest import io

from .advanced_encode_compat import patch_advanced_encode
from .cosmos_attention import patch_cosmos_attention
from .negpip_anima import anima_diffusion_negpip_wrapper, anima_extra_conds_negpip
from .negpip_sd import encode_token_weights_negpip, sd_attn2_negpip
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
UPSTREAM_PATCH_KEY = "ppm_negpip"
QWEN_ENCODER = "qwen3_4b"
KREA2_ENCODER = "qwen3vl_4b"
SUPPORTED_ENCODERS = ["clip_g", "clip_l", "t5xxl", "llama", "qwen3_06b"]
ANIMA_ENCODERS = ["qwen3_06b", "qwen3_06b_base", "qwen_3_06b_base"]


class NegPipPrompt(io.ComfyNode):
    """Patch a supported model and encode text as NegPiP conditioning."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            # Retained for backwards compatibility with workflows from v1.x.
            node_id="ZImageNegPipPrompt",
            display_name="NegPiP Prompt (Multi-Model)",
            category="conditioning/NegPiP",
            description=(
                "Patches Z-Image, Krea 2, SD1, SDXL, or Anima and merges "
                "positive/negative prompts into NegPiP conditioning."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.String.Input("positive", force_input=True),
                io.String.Input("negative", force_input=True),
            ],
            outputs=[
                io.Model.Output(display_name="patched_model"),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.String.Output(display_name="compiled_prompt"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs) -> io.NodeOutput:
        model: ModelPatcher = kwargs["model"]
        clip: CLIP = kwargs["clip"]
        family = cls._model_family(model)
        clip_model = getattr(clip, "cond_stage_model", clip.patcher.model)
        encoders = [name for name in SUPPORTED_ENCODERS if hasattr(clip_model, name)]

        if family == "zimage" and not hasattr(clip_model, QWEN_ENCODER):
            raise ValueError("The connected CLIP does not contain the Z-Image qwen3_4b encoder.")
        if family == "krea2" and not hasattr(clip_model, KREA2_ENCODER):
            raise ValueError(
                "The connected CLIP does not contain Krea 2's qwen3vl_4b encoder. "
                "Load it with CLIPLoader type 'krea2'."
            )
        if family == "sd" and not encoders:
            raise ValueError("The connected CLIP has no supported SD1/SDXL text encoder.")
        if family == "anima" and not any(
            hasattr(clip_model, name) for name in ANIMA_ENCODERS
        ):
            raise ValueError(
                "The connected CLIP does not contain Anima's Qwen3-0.6B text encoder."
            )

        if family == "krea2":
            # Imported lazily so older ComfyUI installations can still use the
            # other model families without importing the newer Qwen3-VL stack.
            from .negpip_krea2 import (
                patch_krea2_negpip,
                validate_krea2_negpip_conditioning,
            )

            m, c = patch_krea2_negpip(model, clip)
            uses_upstream_patch = False
        else:
            m = model.clone()
            c = clip.clone()
            uses_upstream_patch = cls._uses_upstream_patch(m, c)

        if not uses_upstream_patch and family != "krea2":
            if family in {"sd", "anima"}:
                patch_advanced_encode()

            if family == "zimage":
                cls._patch_zimage(m, c)
            elif family == "sd":
                cls._patch_sd(m, c, encoders)
            else:
                cls._patch_anima(m, c)

        merged_prompt = merge_prompts(kwargs["positive"], kwargs["negative"])
        positive = c.encode_from_tokens_scheduled(c.tokenize(merged_prompt))
        if family == "zimage" and not uses_upstream_patch and kwargs["negative"].strip():
            cls._validate_zimage_negpip_conditioning(positive)
        if family == "krea2" and kwargs["negative"].strip():
            validate_krea2_negpip_conditioning(positive)
        negative = c.encode_from_tokens_scheduled(c.tokenize(""))

        return io.NodeOutput(m, positive, negative, merged_prompt)

    @staticmethod
    def _model_family(model: ModelPatcher) -> str:
        model_type = type(model.model)
        if issubclass(model_type, Lumina2):
            return "zimage"
        krea2_type = getattr(comfy.model_base, "Krea2", None)
        if krea2_type is not None and issubclass(model_type, krea2_type):
            return "krea2"
        if issubclass(model_type, Anima):
            return "anima"
        if issubclass(model_type, Flux):
            raise ValueError(
                "Flux NegPiP is unmaintained in upstream ComfyUI-ppm and is not "
                "supported by this node."
            )
        if issubclass(model_type, (SDXL, SDXLRefiner)) or model_type is BaseModel:
            return "sd"
        raise ValueError(
            "Unsupported model architecture: "
            f"{model_type.__name__}. Supported: Z-Image, Krea 2, SD1, SDXL, and Anima."
        )

    @staticmethod
    def _uses_upstream_patch(model: ModelPatcher, clip: CLIP) -> bool:
        model_is_patched = bool(model.model_options.get(UPSTREAM_PATCH_KEY))
        clip_is_patched = bool(clip.patcher.model_options.get(UPSTREAM_PATCH_KEY))
        if model_is_patched != clip_is_patched:
            raise ValueError(
                "The model and CLIP have an incomplete ComfyUI-ppm NegPiP patch. "
                "Connect both directly to this node or use matching outputs from CLIP NegPip."
            )
        return model_is_patched and clip_is_patched

    @classmethod
    def _patch_zimage(cls, model: ModelPatcher, clip: CLIP) -> None:
        if not clip.patcher.model_options.get(PATCH_KEY, False):
            cls._patch_zimage_clip(clip)
            clip.patcher.model_options[PATCH_KEY] = True
        if not model.model_options.get(PATCH_KEY, False):
            cls._patch_zimage_model(model)
            model.model_options[PATCH_KEY] = True

    @staticmethod
    def _patch_zimage_clip(clip: CLIP) -> None:
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
    def _patch_zimage_model(model: ModelPatcher) -> None:
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

        patched_main_layers = 0
        patched_context_refiners = 0
        for block_name, block in diffusion_model.named_modules():
            if not isinstance(block, JointAttention):
                continue

            is_main_layer = block_name.startswith("layers.")
            is_context_refiner = block_name.startswith("context_refiner.")
            if is_main_layer or is_context_refiner:
                model.add_object_patch(
                    f"diffusion_model.{block_name}.forward",
                    make_joint_attention_forward_negpip(block),
                )
                if is_main_layer:
                    patched_main_layers += 1
                else:
                    patched_context_refiners += 1

        if patched_main_layers == 0:
            raise RuntimeError("No compatible Z-Image JointAttention layers were found.")
        if patched_context_refiners == 0:
            raise RuntimeError("No compatible Z-Image context-refiner layers were found.")

    @staticmethod
    def _validate_zimage_negpip_conditioning(conditioning) -> None:
        """Fail fast when a negative prompt loses its Z-Image metadata."""

        for item in conditioning:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            metadata = item[1]
            if not isinstance(metadata, dict):
                continue

            tokens = metadata.get(NEGPIP_TOKENS_KEY)
            strength = metadata.get(NEGPIP_STRENGTH_KEY)
            if tokens is None or strength is None:
                continue
            if not torch.is_tensor(strength) or strength.numel() == 0:
                continue
            if torch.any(strength >= 0).item():
                raise RuntimeError("Z-Image NegPiP produced a non-negative strength.")
            return

        raise RuntimeError(
            "The Z-Image negative prompt was compiled, but its NegPiP metadata "
            "was lost during CLIP encoding. Do not route compiled_prompt through "
            "another CLIP Text Encode node."
        )

    @staticmethod
    def _patch_sd(model: ModelPatcher, clip: CLIP, encoders: list[str]) -> None:
        if not clip.patcher.model_options.get(PATCH_KEY, False):
            for encoder_name in encoders:
                encoder = getattr(clip.patcher.model, encoder_name)
                clip.patcher.add_object_patch(
                    f"{encoder_name}.encode_token_weights",
                    partial(encode_token_weights_negpip, encoder),
                )
            clip.patcher.model_options[PATCH_KEY] = True

        if not model.model_options.get(PATCH_KEY, False):
            model.set_model_attn2_patch(sd_attn2_negpip)
            model.model_options[PATCH_KEY] = True

    @staticmethod
    def _patch_anima(model: ModelPatcher, clip: CLIP) -> None:
        if not model.model_options.get(PATCH_KEY, False):
            patch_cosmos_attention(model)
            model.add_object_patch(
                "extra_conds",
                partial(anima_extra_conds_negpip, model.model.extra_conds),
            )
            model.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                PATCH_KEY,
                anima_diffusion_negpip_wrapper,
            )
            model.model_options[PATCH_KEY] = True
        clip.patcher.model_options[PATCH_KEY] = True


NODES = [NegPipPrompt]
