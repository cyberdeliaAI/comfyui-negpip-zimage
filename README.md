# ComfyUI NegPiP Prompt

A standalone, single-node NegPiP implementation for **Z-Image**, **Z-Image
Turbo**, **SD1**, **SDXL**, and **Anima**. The node patches the connected
model, merges separate positive and negative text strings, and returns the
conditioning required by a normal ComfyUI sampling workflow.

The package keeps its original Registry/repository name,
`comfyui-negpip-zimage`, but version 2.x is no longer limited to Z-Image.

## Installation

Open a terminal in `ComfyUI/custom_nodes` and run:

```bash
git clone https://github.com/cyberdeliaAI/comfyui-negpip-zimage
```

Restart ComfyUI afterwards. No additional Python packages are required beyond
the dependencies included with an up-to-date ComfyUI installation.

## Usage

Place **NegPiP Prompt (Multi-Model)** directly after the matching model and
CLIP loaders. It has four inputs:

- `model`: the loaded diffusion model;
- `clip`: the matching text encoder;
- `positive`: a connected positive `STRING`;
- `negative`: a connected negative `STRING`, optionally containing weights.

Both prompt inputs are sockets. They can be connected to a multiline text
node, wildcard node, prompt generator, or any other `STRING` output.

The node converts the negative string to negative NegPiP weights. For example:

```text
positive: a sharp portrait, detailed eyes
negative: blurry background, (text:1.3)
```

is compiled to:

```text
a sharp portrait, detailed eyes, (blurry background:-1), (text:-1.3)
```

The outputs are:

- the patched `MODEL`;
- positive `CONDITIONING` containing the compiled prompt;
- empty negative `CONDITIONING` for the existing guider/sampler workflow;
- `compiled_prompt` as a `STRING`, showing exactly what was sent to the
  patched text encoder.

Connect `compiled_prompt` to **Preview Any** to inspect the conversion. The
negative conditioning is intentionally empty: NegPiP processes negative
concepts inside the compiled positive conditioning.

The node consumes and encodes the connected CLIP internally, so a CLIP output
is not required.

## Prompt strength and length

You can enter either plain negative text or positive magnitude weights:

```text
blurry, background blur, bokeh
```

```text
(blurry, background blur, bokeh:0.4)
```

The node changes these to weights of `-1` and `-0.4`, respectively. Do not
enter a negative weight in the negative input unless you deliberately want the
same absolute strength; the node always makes negative-input weights negative.

Large negative lists at strength `1.0` can dominate the prompt and produce
unexpected results. Group related concepts and start with lower strengths:

```text
(3D, CGI, render, blender, video game screenshot, illustration:0.25),
(text, writing, subtitle, watermark, logo:0.7),
(blurry, low quality, jpeg artifacts, grainy:0.4)
```

There is no node-level character limit. The effective token/context limit
depends on the connected model and text encoder. Z-Image's current ComfyUI
Qwen3-4B configuration supports a much longer context than CLIP-based SD
models, but long prompts can still use substantially more memory.

Z-Image Turbo normally remains at CFG `1.0`. Use the normal CFG settings for
SD1/SDXL workflows.

## Compatibility

| Architecture | Status | Patch path |
| --- | --- | --- |
| Z-Image / Z-Image Turbo | Supported | Lumina2 / NextDiT / Qwen3-4B |
| SD1 | Supported | paired CLIP embeddings + cross-attention patch |
| SDXL / SDXL Refiner | Supported | paired CLIP embeddings + cross-attention patch |
| Anima | Supported | Qwen3-0.6B encoder + internal T5 weight mask + current Cosmos API |
| Flux | Not supported | upstream marks its NegPiP path as unmaintained |

The original workflow node ID, `ZImageNegPipPrompt`, is retained so workflows
made with version 1.x continue to load. The visible node name and category are
now model-neutral.

This package can be installed next to the original ComfyUI-ppm because it has a
unique node ID and internal patch keys. Do not stack two NegPiP nodes on the
same model/CLIP branch. If both connected inputs were already patched by
ComfyUI-ppm's **CLIP NegPip**, this node reuses that patch instead of applying a
second one.

The Anima implementation uses the current ComfyUI rotary-position API and does
not import the removed `apply_rotary_pos_emb` function from
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
