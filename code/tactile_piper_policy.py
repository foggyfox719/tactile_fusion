"""Piper image/state/action mapping shared by all tactile study branches."""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _image(value):
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC or CHW image, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.dtype != np.uint8:
        image = np.clip(image * 255 if image.max() <= 1 else image, 0, 255).astype(np.uint8)
    return image


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    tactile_mode: str = "none"

    def __call__(self, data: dict) -> dict:
        front = _image(data["image"])
        wrist = _image(data["wrist_image"])
        result = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": front,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(front),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
            "prompt": data["prompt"],
        }
        if self.tactile_mode in ("l2", "h1", "anytouch2"):
            result["tactile"] = np.asarray(data["tactile"], dtype=np.float32)
        if "actions" in data:
            result["actions"] = np.asarray(data["actions"], dtype=np.float32)
        return result


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[:, :7]}
