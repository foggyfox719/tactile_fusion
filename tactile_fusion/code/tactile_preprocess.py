"""Shared offline/online Xense feature definitions."""

from collections import deque

import cv2
import numpy as np

FINGERS = ("OG001892", "OG001936")


class TactileFeatures:
    def __init__(self, depth_threshold: float = 2e-4, flow_scale: float = 0.05):
        if depth_threshold <= 0 or flow_scale <= 0:
            raise ValueError("Tactile scales must be positive")
        self.depth_threshold = depth_threshold
        self.flow_scale = flow_scale

    def l2_frame(self, depth: np.ndarray, flow: np.ndarray) -> np.ndarray:
        contact = cv2.resize(
            (np.abs(depth) > self.depth_threshold).astype(np.float32), (8, 8), interpolation=cv2.INTER_AREA
        )
        displacement = cv2.resize(flow.astype(np.float32), (8, 8), interpolation=cv2.INTER_AREA)
        displacement = np.clip(displacement / self.flow_scale, -1.0, 1.0)
        return np.concatenate((contact[..., None], displacement), axis=-1)

    def h1_frame(self, current_bgr: np.ndarray, reference_bgr: np.ndarray) -> np.ndarray:
        if current_bgr.shape != reference_bgr.shape or current_bgr.ndim != 3 or current_bgr.shape[-1] != 3:
            raise ValueError("Current and reference tactile images must have matching HWC shapes")
        current = cv2.resize(current_bgr[..., ::-1], (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        reference = (
            cv2.resize(reference_bgr[..., ::-1], (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        )
        return np.concatenate((current, current - reference), axis=-1)

    def l1_summary(self, depth: np.ndarray, flow: np.ndarray, past_depth: np.ndarray, label: str) -> str:
        mask = np.abs(depth) > self.depth_threshold
        area = float(mask.mean())
        contact = area >= 0.001
        if contact:
            yy, xx = np.nonzero(mask)
            center = (float(xx.mean() / depth.shape[1]), float(yy.mean() / depth.shape[0]))
        else:
            center = (0.5, 0.5)
        shear = float(np.linalg.norm(flow, axis=-1).mean() / self.flow_scale)
        area_delta = area - float((np.abs(past_depth) > self.depth_threshold).mean())
        return (
            f"{label}: contact={int(contact)} area={area:.3f} "
            f"cop=({center[0]:.2f},{center[1]:.2f}) "
            f"shear_proxy={shear:.3f} area_change={area_delta:+.3f}"
        )


class OnlineTactileHistory:
    """Call once per control tick using synchronized data from both fingers."""

    def __init__(self, mode: str, features: TactileFeatures | None = None):
        if mode not in ("l1", "l2", "h1"):
            raise ValueError(mode)
        self.mode = mode
        self.features = features or TactileFeatures()
        self.history = {finger: deque(maxlen=3) for finger in FINGERS}
        self.last_timestamp = None

    def update(self, timestamp: float, per_finger: dict[str, dict[str, np.ndarray]]) -> None:
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("Tactile timestamps must increase")
        if set(per_finger) != set(FINGERS):
            raise ValueError(f"Expected both Xense fingers: {FINGERS}")
        for finger in FINGERS:
            item = per_finger[finger]
            if self.mode == "h1":
                frame = self.features.h1_frame(item["rectify_bgr"], item["reference_rectify_bgr"])
            elif self.mode == "l2":
                frame = self.features.l2_frame(item["depth"], item["flow"])
            else:
                frame = (np.array(item["depth"], copy=True), np.array(item["flow"], copy=True))
            self.history[finger].append(frame)
        self.last_timestamp = timestamp

    def observation(self, prompt: str) -> dict:
        if self.last_timestamp is None:
            raise RuntimeError("Call update before requesting an observation")
        if self.mode == "l1":
            summaries = []
            for finger in FINGERS:
                current_depth, current_flow = self.history[finger][-1]
                past_depth = self.history[finger][0][0]
                summaries.append(self.features.l1_summary(current_depth, current_flow, past_depth, finger))
            return {"prompt": prompt + " | " + "; ".join(summaries)}
        fingers = []
        for finger in FINGERS:
            frames = list(self.history[finger])
            frames = [frames[0]] * (3 - len(frames)) + frames
            fingers.append(np.stack(frames))
        return {"prompt": prompt, "tactile": np.stack(fingers).astype(np.float32)}
