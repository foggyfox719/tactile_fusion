"""Frozen upstream AnyTouch2 features shared by caching and live inference."""

from collections import deque
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from openpi.policies.tactile_preprocess import FINGERS

SCHEMA = "anytouch2_cls_patchmean_v1"


def history_indices(frame: int) -> tuple[int, ...]:
    if frame < 0:
        raise ValueError("Frame must be nonnegative")
    return tuple(max(0, frame - offset) for offset in (6, 4, 2, 0))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_upstream_model(repo: Path):
    # Only the explicitly selected upstream checkout is imported.
    repo = repo.resolve()
    if not (repo / "model/tactile_mae.py").is_file():
        raise FileNotFoundError(f"AnyTouch2 checkout missing: {repo}")
    sys.path.insert(0, str(repo))
    from model.tactile_mae import TactileVideoMAE
    from transformers import CLIPConfig

    config = CLIPConfig.from_json_file(str(repo / "CLIP-B-16/config.json"))
    return TactileVideoMAE(SimpleNamespace(mask_ratio=0.0), config, 4, 1)


def extract_encoder_state(checkpoint: dict, expected_keys: set[str]) -> dict:
    for container in ("state_dict", "model"):
        if isinstance(checkpoint.get(container), dict):
            checkpoint = checkpoint[container]
            break
    selected = {}
    for original_key, value in checkpoint.items():
        key = original_key.removeprefix("module.")
        for prefix in ("touch_mae_model.", "tactile_model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        if key in expected_keys:
            if key in selected:
                raise ValueError(f"Duplicate encoder weight: {key}")
            selected[key] = value
    # This integer position index is deterministic, not a learned weight.
    missing = expected_keys - set(selected) - {"new_position_ids"}
    if missing:
        raise ValueError(f"Incomplete AnyTouch2 checkpoint: missing {sorted(missing)[:8]}")
    return selected


class FrozenAnyTouch2:
    """Encode [N,4,H,W,3] uint8 BGR windows into [N,2,512]."""

    def __init__(self, repo: Path, checkpoint: Path, sensor_id: int, device: str = "cpu"):
        if not 0 <= sensor_id < 20:
            raise ValueError("sensor_id must identify one of the 20 upstream sensor slots")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        import torch
        from torchvision import transforms

        self.torch = torch
        self.device = torch.device(device)
        self.sensor_id = sensor_id
        self.model = create_upstream_model(repo)
        raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
        selected = extract_encoder_state(raw, set(self.model.state_dict()))
        selected.setdefault("new_position_ids", self.model.new_position_ids.detach())
        self.model.load_state_dict(selected, strict=True)
        self.model.requires_grad_(requires_grad=False).eval().to(self.device)
        self.checkpoint_sha256 = sha256(checkpoint)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def encode(self, windows_bgr: np.ndarray) -> np.ndarray:
        if windows_bgr.ndim != 5 or windows_bgr.shape[1] != 4 or windows_bgr.shape[-1] != 3:
            raise ValueError(f"Expected [N,4,H,W,3], got {windows_bgr.shape}")
        if windows_bgr.dtype != np.uint8:
            raise ValueError("AnyTouch2 input must be uint8 rectified BGR")
        torch = self.torch
        n, t, h, w, _ = windows_bgr.shape
        rgb = np.ascontiguousarray(windows_bgr[..., ::-1])
        x = torch.from_numpy(rgb).permute(0, 1, 4, 2, 3).reshape(n * t, 3, h, w).float() / 255.0
        x = self.preprocess(x).reshape(n, t, 3, 224, 224).permute(0, 2, 1, 3, 4).to(self.device)
        sensors = torch.full((n,), self.sensor_id, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            tokens = self.model(x, sensor_type=sensors)
            if tokens.shape[1:] != (398, 512):
                raise ValueError(f"Unexpected upstream token shape: {tuple(tokens.shape)}")
            # Skip CLS and five sensor tokens when pooling the 392 patch tokens.
            features = torch.stack((tokens[:, 0], tokens[:, 6:].mean(dim=1)), dim=1)
        result = features.cpu().float().numpy()
        if not np.isfinite(result).all():
            raise ValueError("Non-finite AnyTouch2 features")
        return result


class OnlineAnyTouch2History:
    """Feed synchronized dual-finger frames once per control tick; reset per episode."""

    def __init__(self, encoder: FrozenAnyTouch2):
        self.encoder = encoder
        self.frames = {finger: deque(maxlen=7) for finger in FINGERS}
        self.last_timestamp = None

    def update(self, timestamp: float, per_finger_bgr: dict[str, np.ndarray]):
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("Timestamps must increase")
        if set(per_finger_bgr) != set(FINGERS):
            raise ValueError("Expected both configured fingers")
        for finger in FINGERS:
            self.frames[finger].append(np.array(per_finger_bgr[finger], copy=True))
        self.last_timestamp = timestamp

    def observation(self, prompt: str) -> dict:
        if self.last_timestamp is None:
            raise RuntimeError("Call update before observation")
        windows = []
        for finger in FINGERS:
            frames = list(self.frames[finger])
            windows.append(np.stack([frames[i] for i in history_indices(len(frames) - 1)]))
        return {"prompt": prompt, "tactile": self.encoder.encode(np.stack(windows))}
