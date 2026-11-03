"""Attach past-only Xense observations to a LeRobot sample."""

from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np

from openpi.policies.tactile_preprocess import FINGERS
from openpi.policies.tactile_preprocess import TactileFeatures


@lru_cache(maxsize=100)
def _array(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r", allow_pickle=False)


class TactileLeRobotDataset:
    def __init__(
        self,
        dataset,
        root: str,
        mode: str,
        *,
        depth_threshold: float = 2e-4,
        flow_scale: float = 0.05,
        fps: float = 20.0,
    ):
        if mode not in ("l1", "l2", "h1", "anytouch2"):
            raise ValueError(mode)
        if depth_threshold <= 0 or flow_scale <= 0 or fps <= 0:
            raise ValueError("Tactile scales and fps must be positive")
        self.dataset = dataset
        self.root = Path(root)
        self.mode = mode
        self.depth_threshold = depth_threshold
        self.flow_scale = flow_scale
        self.features = TactileFeatures(depth_threshold, flow_scale)
        self.fps = fps
        self._paths = {}
        self.cache_root = Path(os.environ["TACTILE_CACHE_ROOT"]) if os.environ.get("TACTILE_CACHE_ROOT") else None
        self._cache_counts = None
        if mode == "anytouch2" and self.cache_root is None:
            raise RuntimeError("AnyTouch2 requires a frozen-feature cache; set TACTILE_CACHE_ROOT")
        if self.cache_root is not None:
            manifest_path = self.cache_root / self.mode / "manifest.json"
            if not manifest_path.is_file():
                raise RuntimeError(f"Tactile cache is not ready: {manifest_path}")
            manifest = json.loads(manifest_path.read_text())
            if (
                manifest.get("source_root") != str(self.root.resolve())
                or (mode != "anytouch2" and manifest.get("depth_threshold") != self.depth_threshold)
                or (mode != "anytouch2" and manifest.get("flow_scale") != self.flow_scale)
                or manifest.get("fingers") != list(FINGERS)
            ):
                raise ValueError("Tactile cache manifest does not match this dataset/config")
            if mode == "anytouch2" and (
                manifest.get("schema") != "anytouch2_cls_patchmean_v1"
                or manifest.get("num_frames") != 4
                or manifest.get("frame_stride") != 2
                or not manifest.get("checkpoint_sha256")
            ):
                raise ValueError("AnyTouch2 cache has incompatible features or missing weight provenance")
            self._cache_counts = manifest.get("episode_frames")
            if not isinstance(self._cache_counts, dict):
                raise ValueError("Tactile cache manifest is missing episode frame counts")
        info = self.root / "meta/info.json"
        if not info.is_file() or info.stat().st_size == 0:
            raise RuntimeError("Dataset transfer is incomplete: meta/info.json is missing or empty")
        json.loads(info.read_text())

    def __len__(self):
        return len(self.dataset)

    def _episode_paths(self, episode: int, finger: str) -> tuple[Path, Path]:
        key = (episode, finger)
        if key in self._paths:
            return self._paths[key]
        name = f"episode_{episode:06d}"
        timing = self.root / "meta/timing" / f"{name}.json"
        if not timing.is_file() or timing.stat().st_size == 0:
            raise RuntimeError(f"Tactile timing metadata is missing: {timing}")
        json.loads(timing.read_text())
        lossless = self.root / "lossless_tactile" / name
        derived = self.root / "derived_tactile_sdk221" / name / finger
        self._paths[key] = (lossless, derived)
        return self._paths[key]

    def _indices(self, frame: int) -> tuple[int, int, int]:
        return max(frame - 2, 0), max(frame - 1, 0), frame

    def _field(self, path: Path, frame: int) -> np.ndarray:
        array = _array(str(path))
        if frame >= len(array):
            raise IndexError(f"Frame {frame} exceeds {path} ({len(array)} frames)")
        return np.asarray(array[frame])

    def _l2_frame(self, derived: Path, frame: int) -> np.ndarray:
        depth = self._field(derived / "Depth.npy", frame)
        flow = self._field(derived / "Flow.npy", frame)
        return self.features.l2_frame(depth, flow)

    def _h1_frame(self, lossless: Path, finger: str, frame: int) -> np.ndarray:
        current_path = lossless / finger / "rectify" / f"frame_{frame:06d}.npy"
        reference_path = lossless / "reference" / f"{finger}_rectify_bgr.npy"
        current = np.asarray(_array(str(current_path)))
        reference = np.asarray(_array(str(reference_path)))
        return self.features.h1_frame(current, reference)

    def _l1_text(self, derived: Path, frame: int, label: str) -> str:
        depth = self._field(derived / "Depth.npy", frame)
        flow = self._field(derived / "Flow.npy", frame)
        past_depth = self._field(derived / "Depth.npy", max(frame - 2, 0))
        return self.features.l1_summary(depth, flow, past_depth, label)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        episode = int(np.asarray(sample["episode_index"]).item())
        frame = int(np.asarray(sample["frame_index"]).item())
        timestamp = float(np.asarray(sample["timestamp"]).item())
        if abs(timestamp - frame / self.fps) > 0.5 / self.fps:
            raise ValueError(f"Episode {episode} frame {frame}: control timestamp is inconsistent with fps")
        paths = [self._episode_paths(episode, finger) for finger in FINGERS]
        indices = self._indices(frame)
        cached = self.cache_root / self.mode / f"episode_{episode:06d}" if self.cache_root is not None else None
        if cached is not None and frame >= self._cache_counts.get(f"episode_{episode:06d}", 0):
            raise IndexError(f"Episode {episode} frame {frame} is missing from the tactile cache")
        if self.mode == "anytouch2":
            features = np.asarray(_array(str(cached.with_suffix(".npy")))[frame], dtype=np.float32)
            if features.shape != (2, 2, 512) or not np.isfinite(features).all():
                raise ValueError(f"Invalid AnyTouch2 features at episode {episode} frame {frame}")
            sample["tactile"] = features
        elif self.mode == "l1":
            prompt = str(np.asarray(sample["prompt"]).item())
            if cached is not None:
                summaries = self._cached_text(str(cached.with_suffix(".jsonl")))[frame]
            else:
                summaries = "; ".join(
                    self._l1_text(derived, frame, finger)
                    for (_, derived), finger in zip(paths, FINGERS, strict=True)
                )
            sample["prompt"] = prompt + " | " + summaries
        elif cached is not None:
            frames = _array(str(cached.with_suffix(".npy")))[list(indices)]
            sample["tactile"] = np.moveaxis(np.asarray(frames, dtype=np.float32), 0, 1)
        else:
            per_finger = []
            for finger, (lossless, derived) in zip(FINGERS, paths, strict=True):
                if self.mode == "l2":
                    frames = [self._l2_frame(derived, t) for t in indices]
                else:
                    frames = [self._h1_frame(lossless, finger, t) for t in indices]
                per_finger.append(np.stack(frames))
            sample["tactile"] = np.stack(per_finger).astype(np.float32)
        dropout = float(os.environ.get("TACTILE_DROPOUT", "0"))
        if dropout > 0 and "tactile" in sample and np.random.random() < dropout:
            sample["tactile"] = np.zeros_like(np.asarray(sample["tactile"]), dtype=np.float32)
        return sample

    @staticmethod
    @lru_cache(maxsize=25)
    def _cached_text(path: str) -> tuple[str, ...]:
        return tuple(json.loads(line) for line in Path(path).read_text().splitlines())
