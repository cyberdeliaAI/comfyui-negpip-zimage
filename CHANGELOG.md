# Changelog

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
