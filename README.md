# ComfyUI-NegPiP-ZImage

A standalone ComfyUI node for negative prompts with **Z-Image** and
**Z-Image Turbo**. It combines the Z-Image NegPiP patch, prompt merging, and
text encoding in a single node. This repository contains only the Z-Image
implementation.

## Installation

Open a terminal in `ComfyUI/custom_nodes` and run:

```bash
git clone https://github.com/cyberdeliaAI/ComfyUI-NegPiP-ZImage
```

Restart ComfyUI afterwards. No additional Python packages are required beyond
the dependencies included with an up-to-date ComfyUI installation.

## Usage

Place **Z-Image NegPiP Prompt** directly after the Z-Image model and CLIP
loaders. The node has four inputs:

- `model`: the loaded Z-Image model;
- `clip`: the matching Z-Image CLIP/Qwen text encoder;
- `positive`: a connected positive `STRING`;
- `negative`: a connected negative `STRING`, optionally containing weights.

`positive` and `negative` are sockets. They can be connected to a multiline
text node, wildcard node, prompt generator, or any other `STRING` output.

The node automatically converts the negative prompt into negative NegPiP
weights. For example:

```text
positive: a sharp portrait, detailed eyes
negative: blurry background, (text:1.3)
```

is internally compiled to:

```text
a sharp portrait, detailed eyes, (blurry background:-1), (text:-1.3)
```

The outputs are:

- the patched `MODEL`;
- positive `CONDITIONING` containing the merged prompt;
- empty negative `CONDITIONING` for the existing guider/sampler workflow;
- `compiled_prompt` as a `STRING`, showing exactly what is sent to the patched
  CLIP encoder.

Connect `compiled_prompt` to **Preview Any** to inspect the conversion. The
negative conditioning is intentionally empty: NegPiP processes the negative
concepts inside the compiled positive conditioning.

## Prompt length and strength

The node has no fixed character limit. The current ComfyUI Qwen3-4B
configuration supports 40,960 token positions; the practical limit also
depends on available memory. Long negative lists can produce an excessively
strong or unpredictable effect well before that limit because every negative
token is added to the diffusion context.

Use lower weights for broad negative lists. For example, enter this in the
negative prompt:

```text
(3D, CGI, render, blender, video game screenshot, illustration, drawing, comic:0.25),
(text, writing, subtitle, watermark, logo:0.7),
(blurry, low quality, jpeg artifacts, grainy:0.4)
```

The node converts these weights to `-0.25`, `-0.7`, and `-0.4` respectively.

Z-Image Turbo normally remains at CFG `1.0`.

## Compatibility

- Z-Image
- Z-Image Turbo
- Z-Image models loaded by ComfyUI as `Lumina2`/`NextDiT` with the
  `qwen3_4b` text encoder

Other architectures such as Anima, SDXL, and Flux are intentionally not
supported. Use the original ComfyUI-ppm package for those models.

This implementation uses the current ComfyUI rotary-position API and does not
import the removed `apply_rotary_pos_emb` function from
`comfy.ldm.cosmos.predict2`.

## Credits and license

This implementation is derived from:

- [ComfyUI-ppm](https://github.com/pamparamm/ComfyUI-ppm) by pamparamm;
- the Z-Image adaptation in
  [BigStationW/ComfyUI-ppm](https://github.com/BigStationW/ComfyUI-ppm);
- [ComfyUI-NegPipPromptMerge](https://github.com/Deathspike/ComfyUI-NegPipPromptMerge)
  by Deathspike;
- NegPiP by laksjdjf and hako-mikan.

Released under the GNU Affero General Public License v3. See [LICENSE](LICENSE).
