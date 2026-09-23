# ComfyUI-GGUF-Qwen3VL-TE

Makes **Qwen3-VL GGUF text encoders** (e.g. the Qwen-Image-2.1 text encoder) work with
[city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)'s `CLIPLoaderGGUF`.

Fixes this error when using a GGUF text encoder with **Qwen-Image-2.1**:

```
RuntimeError: Given normalized_shape=[4096], expected input with shape [*4096],
but got input of size[1, 512, 12288]
```

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/pottokao-dotcom/ComfyUI-GGUF-Qwen3VL-TE
```

Restart ComfyUI. You still need **ComfyUI-GGUF** installed — this is an add-on, not a replacement.
It adds no new nodes; keep using your normal workflow.

## Usage

Put **both** files in `models/text_encoders/`:

- the text encoder, e.g. `qwen3vl_8b_heretic-Q4_K_M.gguf`
- its vision tower, whose name contains the text encoder's name, e.g. `mmproj-qwen3vl_8b_heretic-f16.gguf`

Then: `CLIPLoaderGGUF` (type **`qwen_image`**) → `TextEncodeQwenImage21` → the rest of the official
Qwen-Image-2.1 workflow. On startup the console shows
`[GGUF-Qwen3VL-TE] ComfyUI-GGUF patched for qwen3vl text encoders.`

## Why it breaks

ComfyUI-GGUF only loads the `mmproj` vision tower for `qwen2vl`. For `qwen3vl` it is skipped, so
ComfyUI does not recognise the model as Qwen3-VL (it looks for `model.visual.deepstack_merger_list.*`)
and builds the wrong text encoder, which returns 12288-wide hidden states.

This add-on wraps `gguf_clip_loader`: for `qwen3vl` files it loads the matching mmproj and renames
its tensors to ComfyUI's Qwen3-VL layout:

| mmproj GGUF (llama.cpp) | ComfyUI Qwen3-VL |
|---|---|
| `v.blk.N.{attn_qkv, attn_out, ffn_up, ffn_down, ln1, ln2}` | `model.visual.blocks.N.{attn.qkv, attn.proj, mlp.linear_fc1, mlp.linear_fc2, norm1, norm2}` |
| `v.deepstack.{8,16,24}.{fc1, fc2, norm}` | `model.visual.deepstack_merger_list.{0,1,2}.{linear_fc1, linear_fc2, norm}` |
| `mm.0` / `mm.2` / `v.post_ln` | `model.visual.merger.{linear_fc1, linear_fc2, norm}` |
| `v.patch_embd` / `v.position_embd` | `model.visual.patch_embed.proj` / `model.visual.pos_embed` |

Tested with ComfyUI 0.36.0 + ComfyUI-GGUF `6ea2651`: all 750 keys match the official
bf16 safetensors text encoder, the vision tensors match it numerically, and Qwen-Image-2.1 generates
normally. Once ComfyUI-GGUF supports qwen3vl itself, this add-on detects it and does nothing.

## License

Apache-2.0
