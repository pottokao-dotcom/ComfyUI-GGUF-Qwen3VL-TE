# ComfyUI-GGUF-Qwen3VL-TE

Makes **Qwen-Image-2.1 run fully on GGUF** with [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF),
and lets the **Ideogram 4** and **Boogu** Qwen3-VL-8B text encoders load from GGUF correctly, until upstream
supports it. One add-on, two fixes:

| Fixes | What happens without this add-on |
|---|---|
| **Text encoder** — Qwen3-VL GGUF via `CLIPLoaderGGUF`, type `qwen_image` | `RuntimeError: Given normalized_shape=[4096] ... got input of size [1, 512, 12288]` |
| **Text encoder** — same, type `ideogram4` | **no error, wrong conditioning**: loaded as the text-only Qwen3-8B encoder (cos 0.84–0.87 vs bf16) |
| **Text encoder** — same, type `boogu` | **no error at load, wrong encoder**: loaded as the FLUX.2 klein 9B encoder (512×12288 output) |
| **DiT** — Qwen-Image-2.1 GGUFs without architecture metadata (e.g. [unsloth](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF), [leejet](https://huggingface.co/leejet/Qwen-Image-2.1-GGUF)) via `UnetLoaderGGUF` | `ValueError: This model is not currently supported - (Unknown model architecture!)` |

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/pottokao-dotcom/ComfyUI-GGUF-Qwen3VL-TE
```

Restart ComfyUI. You still need **ComfyUI-GGUF** installed — this is an add-on, not a replacement.
It adds no new nodes; keep using your normal workflow. On startup the console shows
`[GGUF-Qwen3VL-TE] ComfyUI-GGUF patched (qwen_image / ideogram4 / boogu text encoders + Qwen-Image-2.1 DiT detection).`

## Usage

**Text encoder** — put **both** files in `models/text_encoders/`:

- the text encoder, e.g. `qwen3vl_8b_heretic-Q4_K_M.gguf`
- its vision tower, whose name contains the text encoder's name, e.g. `mmproj-qwen3vl_8b_heretic-f16.gguf`

Don't rename them — the mmproj is found by name. If it is missing, loading stops with a
`Missing vision tower` error that names the file to download.

The text-encoder fix only acts when the loader's type is **`qwen_image`**, **`ideogram4`** or **`boogu`**
(the types whose ComfyUI text encoder is the full Qwen3-VL-8B). Other models that use Qwen3-VL GGUFs
(e.g. MiniMax-H3 text encoders) load exactly as they do without this add-on.

**DiT** — put any Qwen-Image-2.1 GGUF in `models/diffusion_models/` (or `models/unet/`) and load it with
`UnetLoaderGGUF` as usual.

Then: `CLIPLoaderGGUF` (type **`qwen_image`**) → `TextEncodeQwenImage21` → `UnetLoaderGGUF` model →
the rest of the official Qwen-Image-2.1 workflow.

The vision tower is used for real, not only to get past the loader: `TextEncodeQwenImage21`'s
reference images (image editing) go through it. Both text-to-image and editing were tested.

## Why it breaks

**Text encoder.** ComfyUI-GGUF only loads the `mmproj` vision tower for `qwen2vl`. For `qwen3vl` it is
skipped, so ComfyUI does not recognise the model as Qwen3-VL (it looks for
`model.visual.deepstack_merger_list.*`) and builds the wrong text encoder, which returns 12288-wide
hidden states. This add-on loads the matching mmproj and renames its tensors to ComfyUI's Qwen3-VL layout:

| mmproj GGUF (llama.cpp) | ComfyUI Qwen3-VL |
|---|---|
| `v.blk.N.{attn_qkv, attn_out, ffn_up, ffn_down, ln1, ln2}` | `model.visual.blocks.N.{attn.qkv, attn.proj, mlp.linear_fc1, mlp.linear_fc2, norm1, norm2}` |
| `v.deepstack.{8,16,24}.{fc1, fc2, norm}` | `model.visual.deepstack_merger_list.{0,1,2}.{linear_fc1, linear_fc2, norm}` |
| `mm.0` / `mm.2` / `v.post_ln` | `model.visual.merger.{linear_fc1, linear_fc2, norm}` |
| `v.patch_embd` / `v.position_embd` | `model.visual.patch_embed.proj` / `model.visual.pos_embed` |

**DiT.** A GGUF without `general.architecture` (stable-diffusion.cpp convention) is identified by its
tensor names, and ComfyUI-GGUF's list has no Qwen-Image entry. This add-on recognises Qwen-Image-2.1
(`img_in`, `txt_in.in_layer`, `txt_in.text_norm`) and loads it as `qwen_image`; ComfyUI itself then
detects the 2.1 architecture as it does for safetensors. DiT GGUFs that already carry
`general.architecture = qwen_image` (e.g. Abiray's) loaded before and are unaffected.

## Tested

ComfyUI 0.36.0 + ComfyUI-GGUF `6ea2651`, NVIDIA GPU:

- **Text encoder:** all 750 keys match the official bf16 safetensors text encoder, the vision tensors
  match it numerically, and Qwen-Image-2.1 text-to-image and reference-image editing match the bf16
  encoder's output for the same seed up to Q4 quantization noise.
- **Ideogram 4 / Boogu text encoders** (ComfyUI `1568e6c`): Qwen3-VL-8B Q8_0 GGUF through `CLIPLoaderGGUF` vs the
  bf16 safetensors through `CLIPLoader`, same prompts: with this add-on the same ComfyUI text encoder is built and
  the conditioning matches (per-token cos 0.9999 for `ideogram4`, 0.9997 for `boogu`); without it, see the table
  at the top. `qwen_image` results are unchanged.
- **DiT:** unsloth, leejet and Abiray Q4 GGUFs are all detected as `qwen_image`; the unsloth Q4_K_M
  (no architecture metadata) generates normally in an all-GGUF pipeline.

Not tested on a Mac. Once ComfyUI-GGUF handles these itself
([city96/ComfyUI-GGUF#485](https://github.com/city96/ComfyUI-GGUF/pull/485) for the text encoder),
this add-on detects it and does nothing.

## License

Apache-2.0
