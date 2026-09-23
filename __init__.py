"""
Add-on for city96/ComfyUI-GGUF: lets a Qwen3-VL GGUF (e.g. the Qwen-Image-2.1 text encoder)
load through the stock `CLIPLoaderGGUF` node.

Upstream ComfyUI-GGUF only loads the mmproj vision tower for `qwen2vl`. For `qwen3vl` the
vision tower is missing, ComfyUI does not recognise the model as Qwen3-VL, and Qwen-Image-2.1
fails with `Given normalized_shape=[4096] ... got input of size [1, 512, 12288]`.

This add-on wraps ComfyUI-GGUF's `gguf_clip_loader`: for `qwen3vl` files it loads the matching
`mmproj-*.gguf` from the same folder and renames its tensors to ComfyUI's Qwen3-VL layout.
It adds no nodes and does nothing once upstream handles qwen3vl itself.
"""
import logging
import os
import sys

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

TAG = "[GGUF-Qwen3VL-TE]"
_DETECT_KEY = "model.visual.deepstack_merger_list.0.norm.weight"


def _arch(path):
    try:
        import gguf
        reader = gguf.GGUFReader(path)
        field = reader.get_field("general.architecture")
        return str(field.parts[field.data[-1]], encoding="utf-8")
    except Exception as e:
        logging.warning(f"{TAG} could not read architecture from {path}: {e}")
        return None


def _remap_vision(vsd):
    # qwen2vl-style names produced by ComfyUI-GGUF's gguf_mmproj_loader -> ComfyUI Qwen3-VL keys
    ds = sorted({int(k.split(".")[2]) for k in vsd if k.startswith("v.deepstack.")})
    out = {}
    for k, v in vsd.items():
        if k.startswith("v.deepstack."):
            _, _, n, rest = k.split(".", 3)
            k = f"visual.deepstack_merger_list.{ds.index(int(n))}.{rest}"
        k = k.replace("v.position_embd.", "visual.pos_embed.")
        k = k.replace(".attn_qkv.", ".attn.qkv.")
        k = k.replace(".mlp.up_proj.", ".mlp.linear_fc1.").replace(".mlp.down_proj.", ".mlp.linear_fc2.")
        k = k.replace(".fc1.", ".linear_fc1.").replace(".fc2.", ".linear_fc2.")
        k = k.replace("visual.merger.ln_q.", "visual.merger.norm.")
        k = k.replace("visual.merger.mlp.0.", "visual.merger.linear_fc1.")
        k = k.replace("visual.merger.mlp.2.", "visual.merger.linear_fc2.")
        out["model." + k] = v
    return out


def _is_vision_key(k):
    return k.startswith(("v.", "visual."))


def _make_wrapper(orig, loader_mod):
    def gguf_clip_loader(path, *args, **kwargs):
        sd = orig(path, *args, **kwargs)
        if _DETECT_KEY in sd or _arch(path) != "qwen3vl":
            return sd
        # an upstream that loads the mmproj but keeps qwen2vl names: rename in place
        vsd = {k: sd.pop(k) for k in list(sd) if _is_vision_key(k)}
        if not vsd:
            vsd = loader_mod.gguf_mmproj_loader(path)
        if not vsd:
            name = os.path.splitext(os.path.basename(path))[0]
            strip = getattr(loader_mod, "strip_quant_suffix", None)
            base = strip(name.lower()) if strip else name
            raise RuntimeError(
                f"{TAG} Missing vision tower for '{os.path.basename(path)}'.\n"
                f"Qwen3-VL GGUF text encoders need their mmproj file. Put an mmproj GGUF whose name "
                f"contains '{base}' (e.g. 'mmproj-{base}-f16.gguf') in the same folder:\n"
                f"  {os.path.dirname(path)}\n"
                f"Don't rename either file - they are matched by name."
            )
        sd.update(_remap_vision(vsd))
        logging.info(f"{TAG} added {len(vsd)} Qwen3-VL vision tensors from mmproj.")
        return sd
    gguf_clip_loader._qwen3vl_te_patched = True
    return gguf_clip_loader


def _patch():
    loader_mod = nodes_mods = None
    mods = [m for m in list(sys.modules.values()) if m is not None]
    for m in mods:
        if hasattr(m, "gguf_clip_loader") and hasattr(m, "gguf_mmproj_loader"):
            loader_mod = m
    if loader_mod is None:
        return False
    orig = loader_mod.gguf_clip_loader
    if getattr(orig, "_qwen3vl_te_patched", False):
        return True
    wrapped = _make_wrapper(orig, loader_mod)
    loader_mod.gguf_clip_loader = wrapped
    # nodes.py binds the name with `from .loader import gguf_clip_loader`
    for m in mods:
        if getattr(m, "gguf_clip_loader", None) is orig and hasattr(m, "CLIPLoaderGGUF"):
            m.gguf_clip_loader = wrapped
            nodes_mods = m
    if nodes_mods is None:
        logging.warning(f"{TAG} patched ComfyUI-GGUF loader but did not find its nodes module.")
    else:
        logging.info(f"{TAG} ComfyUI-GGUF patched for qwen3vl text encoders.")
    return True


if not _patch():
    # ComfyUI imports custom nodes in directory-listing order; retry once all nodes are loaded
    try:
        from server import PromptServer

        async def _on_startup(app):
            if not _patch():
                logging.warning(f"{TAG} ComfyUI-GGUF not found; this add-on does nothing.")

        PromptServer.instance.app.on_startup.append(_on_startup)
    except Exception as e:
        logging.warning(f"{TAG} could not defer patching: {e}")
