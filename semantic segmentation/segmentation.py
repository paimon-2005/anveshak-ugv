import importlib.util
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval


MEAN_RGB = (0.3257, 0.3690, 0.3223)
STD_RGB = (0.2112, 0.2148, 0.2115)
CLASS_COLORS_BGR = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [5, 54, 244]], np.uint8)
AUXILIARY_PREFIXES = ("aux2.", "aux3.", "aux4.", "aux5_4.")


def fuse_inference_layers(module):
    for child in list(module.children()):
        fuse_inference_layers(child)
    if isinstance(getattr(module, "conv", None), nn.Conv2d) and isinstance(getattr(module, "bn", None), nn.BatchNorm2d):
        module.conv = fuse_conv_bn_eval(module.conv, module.bn)
        module.bn = nn.Identity()
    if isinstance(module, nn.Sequential):
        children = list(module.named_children())
        for (a, first), (b, second) in zip(children, children[1:]):
            if isinstance(first, nn.Conv2d) and isinstance(second, nn.BatchNorm2d):
                module[int(a)] = fuse_conv_bn_eval(first, second)
                module[int(b)] = nn.Identity()
    return module


def load_segmentation_model(model_file, weights_file, device, fuse=True):
    model_file, weights_file = Path(model_file), Path(weights_file)
    if not model_file.is_file() or not weights_file.is_file():
        raise FileNotFoundError(f"Missing model or checkpoint: {model_file}, {weights_file}")
    spec = importlib.util.spec_from_file_location("anveshak_bisenetv2", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(str(model_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = torch.load(weights_file, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for name in ("model_state_dict", "state_dict", "model"):
            if name in state and isinstance(state[name], dict):
                state = state[name]
                break
    if not isinstance(state, dict):
        raise ValueError("Expected a tensor state dictionary")
    state = {k.removeprefix("module."): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith(AUXILIARY_PREFIXES)}
    model = module.BiSeNetV2(4, output_aux=False, return_logits=True)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    if fuse:
        fuse_inference_layers(model)
    return model.to(device=device, memory_format=torch.channels_last)


class SegmentationEngine:
    def __init__(self, model, device, confidence=0.65, scale=1.0, flip_tta=False, fp16=False, floor_fraction=0.45):
        self.model, self.device = model, torch.device(device)
        self.confidence, self.scale = float(confidence), float(scale)
        self.flip_tta, self.fp16 = bool(flip_tta), bool(fp16)
        self.floor_fraction = float(floor_fraction)
        if not 0.25 <= self.confidence <= 1 or not 0.25 <= self.scale <= 2 or not 0 <= self.floor_fraction <= 1:
            raise ValueError("Invalid segmentation confidence, scale, or floor fraction")
        if fp16 and self.device.type != "cuda":
            raise ValueError("FP16 is only supported on CUDA")
        self.mean = torch.tensor(MEAN_RGB, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD_RGB, device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def predict(self, frame):
        height, width = frame.shape[:2]
        if self.scale != 1:
            frame = cv2.resize(frame, (max(32, round(width * self.scale)), max(32, round(height * self.scale))),
                               interpolation=cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(
            device=self.device, dtype=torch.float32, memory_format=torch.channels_last)
        tensor = (tensor / 255 - self.mean) / self.std
        h, w = tensor.shape[-2:]
        tensor = F.pad(tensor, (0, (-w) % 32, 0, (-h) % 32), mode="replicate")
        with torch.autocast(device_type=self.device.type, enabled=self.fp16):
            logits = self.model(tensor)
            if self.flip_tta:
                logits = (logits + self.model(tensor.flip([-1])).flip([-1])) * 0.5
        logits = logits[..., :h, :w].float()
        if (h, w) != (height, width):
            logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        confidence, classes = logits.softmax(dim=1)[0].max(dim=0)
        free = (classes == 1) & (confidence >= self.confidence)
        if self.floor_fraction > 0:
            start = height - int(math.floor(height * self.floor_fraction + 1e-9))
            lower_sky = classes[start:] == 0
            free[start:] |= lower_sky
            classes[start:] = torch.where(lower_sky, 1, classes[start:])
        prediction = classes.to(torch.uint8).cpu().numpy()
        return CLASS_COLORS_BGR[prediction], free.to(torch.uint8).cpu().numpy() * 255
