# Changelog

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
