# ComfyUI NegPiP Prompt

A standalone, single-node NegPiP implementation for **Z-Image**, **Z-Image
Turbo**, **Krea 2**, **SD1**, **SDXL**, and **Anima**. The node patches the
connected model, merges separate positive and negative text strings, and
returns the conditioning required by a normal ComfyUI sampling workflow.

The package keeps its original Registry/repository name,
`comfyui-negpip-zimage`, but version 2.x is no longer limited to Z-Image.

## Installation

Open a terminal in `ComfyUI/custom_nodes` and run:

```bash
git clone https://github.com/cyberdeliaAI/comfyui-negpip-zimage
```

Restart ComfyUI afterwards. No additional Python packages are required beyond
the dependencies included with an up-to-date ComfyUI installation.

Krea 2 support requires a ComfyUI version that includes the native Krea 2
model and `krea2` CLIP loader type.

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

- `patched_model`: the patched `MODEL` that must be connected to the sampler or
  guider;
- positive `CONDITIONING` containing the compiled prompt;
- empty negative `CONDITIONING` for the existing guider/sampler workflow;
- `compiled_prompt` as a `STRING`, showing exactly what was sent to the
  patched text encoder.

Connect `compiled_prompt` to **Preview Any** to inspect the conversion. The
negative conditioning is intentionally empty: NegPiP processes negative
concepts inside the compiled positive conditioning.

The node consumes and encodes the connected CLIP internally, so a CLIP output
is not required. This also applies to Krea 2: its patched CLIP clone is used
immediately for the node's conditioning output.

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
depends on the connected model and text encoder. The Qwen-based Z-Image and
Krea 2 encoders support a much longer context than CLIP-based SD models, but
long prompts can still use substantially more memory.

Z-Image Turbo normally remains at CFG `1.0`. Use the normal CFG settings for
SD1/SDXL workflows.

For Z-Image, compare prompt changes with the same seed. Broad semantic
categories can react non-linearly, so start with a strength around `0.25` to
`0.5` before trying `1.0`.

## Z-Image troubleshooting

- Connect `patched_model` from this node to the sampler or guider. Do not keep
  that input connected directly to the original model loader.
- Connect the node's positive and negative `CONDITIONING` outputs directly to
  the corresponding sampler/guider inputs.
- Use `compiled_prompt` only for inspection. Do not send it through another
  CLIP Text Encode node; Z-Image's normal tokenizer does not interpret NegPiP
  weights.
- With `Asian` in the negative input, `compiled_prompt` must show
  `(Asian:-1)`. For a weaker test, enter `(Asian:0.4)` in the negative input,
  which compiles to `(Asian:-0.4)`.

## Krea 2 notes

- Load the text encoder with ComfyUI's `CLIPLoader` type `krea2`; a regular
  Qwen, Z-Image, or Flux CLIP is not interchangeable.
- Connect `patched_model` to the sampler and use the node's conditioning
  outputs directly, just as for Z-Image.
- The integrated Krea 2 path applies NegPiP to all 28 main transformer blocks
  and both text-fusion refiner blocks. Enabling the refiners is intentionally
  stronger than the standalone upstream node's default and is intended to
  reduce positive leakage into neighboring text tokens before the main model
  blocks. Start with lower prompt magnitudes if the effect is too strong.
- Prompt magnitudes remain available in the negative input. For example,
  `(blurry:0.4)` compiles to `(blurry:-0.4)`.
- Complex conditioning transforms that normalize or clamp the Krea 2
  conditioning tensor may destroy its embedded NegPiP sidecar. A metadata
  fallback is included, but direct connections from this node remain the
  safest path.

## Compatibility

| Architecture | Status | Patch path |
| --- | --- | --- |
| Z-Image / Z-Image Turbo | Supported | Lumina2 / NextDiT / Qwen3-4B |
| Krea 2 | Supported | SingleStreamDiT / Qwen3-VL-4B / 12-layer text fusion |
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
- [ComfyUI-krea2-negpip](https://github.com/blue-pen5805/ComfyUI-krea2-negpip)
  by blue-pen5805 (adapted from commit
  [`3740add`](https://github.com/blue-pen5805/ComfyUI-krea2-negpip/commit/3740add9dbdc9f254a2befda30e95ba95e3b115d));
- [ComfyUI-NegPipPromptMerge](https://github.com/Deathspike/ComfyUI-NegPipPromptMerge)
  by Deathspike;
- NegPiP by laksjdjf and hako-mikan.

Released under the GNU Affero General Public License v3. See [LICENSE](LICENSE).
