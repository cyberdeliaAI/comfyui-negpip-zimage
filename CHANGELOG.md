# Changelog

## 2.1.1

- Fixed `TypeError: SingleStreamDiT._forward() takes from 4 to 6 positional
  arguments but 7 were given` on ComfyUI versions whose Krea 2 diffusion
  signature does not yet have a separate `ref_latents` parameter.
- Added signature-aware argument binding for both older and current Krea 2
  diffusion wrappers while preserving reference latents, extra keywords, and
  wrapper chaining.

## 2.1.0

- Added Krea 2 support to the existing single multi-model prompt node.
- Added automatic detection of ComfyUI's native Krea 2 model and
  `qwen3vl_4b` text encoder.
- Enabled NegPiP in all 28 Krea 2 main transformer blocks and both text-fusion
  refiner blocks to reduce negative concepts leaking positively between text
  tokens before the main model blocks.
- Serialized Krea 2's temporary attention-forward patches so overlapping
  executions cannot restore one another's model state mid-call.
- Added fail-fast validation for mismatched Krea 2 models, CLIP encoders, and
  missing negative-token metadata.
- Kept the node's four-input workflow and existing outputs unchanged.

## 2.0.2

- Fixed Z-Image negative concepts leaking positively through the unpatched
  context-refiner attention blocks before reaching the main NegPiP blocks.
- Updated the Z-Image JointAttention replacement to match ComfyUI's current
  fused QK-normalization and RoPE path.
- Restored the `transformer_options` fallback used by custom guiders that move
  additional model conditions out of the direct wrapper arguments.
- Added a fail-fast check when a non-empty Z-Image negative prompt loses its
  NegPiP token or strength metadata during CLIP encoding.
- Renamed the model output label to `patched_model` to make the required
  sampler/guider connection explicit.

## 2.0.1

- Fixed false rejection of the official Anima CLIP, whose loaded text encoder
  is exposed as `qwen3_06b` rather than `t5xxl`.
- Kept Anima's internal `t5xxl_weights` metadata path for the NegPiP attention
  mask, matching current ComfyUI and ComfyUI-ppm behavior.

## 2.0.0

- Expanded the single combined prompt node with SD1, SDXL, SDXL Refiner, and
  Anima support alongside Z-Image and Z-Image Turbo.
- Added dedicated SD and Anima patch modules based on the current upstream
  ComfyUI-ppm implementation.
- Updated the Anima path for ComfyUI's current Cosmos rotary-position API.
- Added optional Advanced CLIP Text Encode compatibility for the SD patch.
- Renamed the visible node to `NegPiP Prompt (Multi-Model)` while retaining the
  original `ZImageNegPipPrompt` node ID for saved-workflow compatibility.
- Added detection for an existing ComfyUI-ppm NegPiP patch to avoid applying a
  second patch to the same branch.
- Documented Flux as unsupported because its upstream implementation is
  explicitly unmaintained and no longer matches the current ComfyUI API.

## 1.2.0

- Added a `compiled_prompt` STRING output for Preview Any and debugging.
- Added documentation about prompt length and appropriate negative weights.

## 1.1.0

- Combined the Z-Image patch and NegPiP Prompt Merge into a single node.
- Added separate positive and negative text inputs.
- Made both prompt inputs connectable `STRING` sockets.
- The node now returns the patched model together with positive and empty
  negative conditioning.
- Added prompt parsing derived from ComfyUI-NegPipPromptMerge by Deathspike.

## 1.0.0

- Split the implementation from ComfyUI-ppm into a standalone Z-Image-only
  node.
- Added a unique package name, node ID, and internal patch keys.
- Removed obsolete Anima/Cosmos code and the deleted `apply_rotary_pos_emb`
  import.
- Updated the implementation for the current ComfyUI `Lumina2`, `NextDiT`, and
  `JointAttention` APIs.
- Prevented duplicate attention-mask extension.
- Derived the suffix length dynamically from the active Qwen tokenizer.
- Improved batch handling and kept tokenizer patches local to the node output.
