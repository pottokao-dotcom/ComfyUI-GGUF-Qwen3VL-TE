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
import importlib
import logging
import os
import sys
import types

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


def _add_vision(path, sd, loader_mod):
    # Give a Qwen3-VL GGUF text encoder the vision tower Qwen-Image-2.1 needs, in ComfyUI's key layout.
    if _DETECT_KEY in sd or _arch(path) != "qwen3vl":
        return
    # vision tensors already in the file (or loaded by an upstream that keeps other names): rename in place
    vsd = {k: sd.pop(k) for k in list(sd) if _is_vision_key(k)}
    if not vsd:
        vsd = loader_mod.gguf_mmproj_loader(path)
    if not vsd:
        name = os.path.splitext(os.path.basename(path))[0]
        strip = getattr(loader_mod, "strip_quant_suffix", None)
        base = strip(name.lower()) if strip else name
        raise RuntimeError(
            f"{TAG} Missing vision tower for '{os.path.basename(path)}'.\n"
            f"Qwen-Image-2.1 needs the text encoder's mmproj file. Put an mmproj GGUF whose name "
            f"contains '{base}' (e.g. 'mmproj-{base}-f16.gguf') in the same folder:\n"
            f"  {os.path.dirname(path)}\n"
            f"Don't rename either file - they are matched by name."
        )
    sd.update(_remap_vision(vsd))
    logging.info(f"{TAG} added {len(vsd)} Qwen3-VL vision tensors for Qwen-Image-2.1.")


def _make_load_patcher(orig, loader_mod):
    # Act only when the loader's type is qwen_image: other models built on Qwen3-VL GGUFs
    # (e.g. MiniMax-H3 text encoders) must load exactly as before.
    def load_patcher(self, clip_paths, clip_type, clip_data, *args, **kwargs):
        if getattr(clip_type, "name", None) == "QWEN_IMAGE":
            for path, sd in zip(clip_paths, clip_data):
                if str(path).endswith(".gguf"):
                    _add_vision(path, sd, loader_mod)
        return orig(self, clip_paths, clip_type, clip_data, *args, **kwargs)
    load_patcher._qwen3vl_te_patched = True
    return load_patcher


# Qwen-Image-2.1 DiT GGUFs without `general.architecture` metadata (stable-diffusion.cpp style,
# e.g. unsloth/leejet) fail with "Unknown model architecture!": ComfyUI-GGUF's key-based
# detection has no Qwen-Image entry. ComfyUI itself tells 2.1 apart from the state dict later.
_QI21_DIT_KEYS = ("img_in.weight", "txt_in.in_layer.weight", "txt_in.text_norm.weight")


def _find_gguf_modules():
    # Look only at real module attributes (__dict__): some modules, e.g. torch.ops namespaces,
    # answer hasattr() for any name.
    found = {}
    for m in list(sys.modules.values()):
        if not isinstance(m, types.ModuleType):
            continue
        d = m.__dict__
        path = (d.get("__file__") or "").replace("\\", "/")
        if path.endswith("/loader.py") and "gguf_clip_loader" in d and "gguf_mmproj_loader" in d:
            found["loader"] = m
        elif path.endswith("/nodes.py") and "gguf_clip_loader" in d and "CLIPLoaderGGUF" in d:
            found["nodes"] = m
        elif path.endswith("/tools/convert.py") and "detect_arch" in d and "ModelTemplate" in d:
            found["convert"] = m
    return found


def _patch_dit(convert_mod):
    orig = convert_mod.detect_arch
    if getattr(orig, "_qwen3vl_te_patched", False):
        return True

    class ModelQwenImage(convert_mod.ModelTemplate):
        arch = "qwen_image"

    def detect_arch(state_dict):
        try:
            return orig(state_dict)
        except AssertionError:
            if all(k in state_dict for k in _QI21_DIT_KEYS):
                logging.info(f"{TAG} detected a Qwen-Image-2.1 DiT GGUF without architecture metadata.")
                return ModelQwenImage()
            raise

    detect_arch._qwen3vl_te_patched = True
    convert_mod.detect_arch = detect_arch
    return True


def _patch():
    mods = _find_gguf_modules()
    loader_mod = mods.get("loader")
    if loader_mod is None:
        return False

    # DiT: tools.convert is imported lazily by ComfyUI-GGUF, so import it the same way
    convert_mod = mods.get("convert")
    if convert_mod is None:
        try:
            convert_mod = importlib.import_module(loader_mod.__package__ + ".tools.convert")
        except Exception as e:
            logging.warning(f"{TAG} could not patch DiT architecture detection: {e}")
    if convert_mod is not None:
        _patch_dit(convert_mod)

    # text encoder: CLIPLoaderGGUF.load_patcher is where the loader's type is known
    # (Dual/Triple/Quadruple loaders inherit it)
    nodes_mod = mods.get("nodes")
    cls = nodes_mod.__dict__.get("CLIPLoaderGGUF") if nodes_mod is not None else None
    if cls is None:
        logging.warning(f"{TAG} did not find ComfyUI-GGUF's CLIPLoaderGGUF; text encoder fix not applied.")
        return True
    orig = cls.load_patcher
    if not getattr(orig, "_qwen3vl_te_patched", False):
        cls.load_patcher = _make_load_patcher(orig, loader_mod)
    logging.info(f"{TAG} ComfyUI-GGUF patched for Qwen-Image-2.1 (qwen_image text encoder + DiT detection).")
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
