"""Small dual-finger encoders with identical four-token output contracts."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class AnyTouchTokenAdapter(nnx.Module):
    """Project frozen CLS/patch-mean features into the pi05 prefix."""

    def __init__(self, output_width: int, rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(512, rngs=rngs)
        self.project = nnx.Linear(512, output_width, rngs=rngs)
        self.finger_embedding = nnx.Embed(2, output_width, rngs=rngs)
        self.token_embedding = nnx.Embed(2, output_width, rngs=rngs)

    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        if features.ndim != 4 or features.shape[1:] != (2, 2, 512):
            raise ValueError(f"Expected [B,2,2,512], got {features.shape}")
        x = self.project(self.norm(features))
        x += self.finger_embedding(jnp.arange(2))[None, :, None, :]
        x += self.token_embedding(jnp.arange(2))[None, None, :, :]
        return x.reshape((x.shape[0], 4, -1))


class TactileEncoder(nnx.Module):
    def __init__(self, mode: str, output_width: int, rngs: nnx.Rngs):
        if mode not in ("l2", "h1"):
            raise ValueError(mode)
        self.mode = mode
        channels = 3 if mode == "l2" else 6
        strides = (1, 2, 2) if mode == "l2" else (2, 2, 2, 2)
        widths = (32, 64, 64) if mode == "l2" else (32, 64, 64, 64)
        self.spatial = [
                nnx.Conv(
                    channels if i == 0 else widths[i - 1],
                    width,
                    kernel_size=(3, 3),
                    strides=(stride, stride),
                    padding="SAME",
                    rngs=rngs,
                )
                for i, (stride, width) in enumerate(zip(strides, widths, strict=True))
            ]
        self.temporal = nnx.Conv(256, 256, kernel_size=(3,), padding="VALID", rngs=rngs)
        self.norm = nnx.LayerNorm(256, rngs=rngs)
        self.project = nnx.Linear(256, 2 * output_width, rngs=rngs)
        self.finger_embedding = nnx.Embed(2, output_width, rngs=rngs)

    def __call__(self, tactile: jnp.ndarray) -> jnp.ndarray:
        # Both encoders receive [B, 2 fingers, 3 past/current frames, H, W, C].
        if tactile.ndim != 6 or tactile.shape[1:3] != (2, 3):
            raise ValueError(f"Expected [B,2,3,H,W,C], got {tactile.shape}")
        expected = (8, 8, 3) if self.mode == "l2" else (96, 96, 6)
        if tactile.shape[3:] != expected:
            raise ValueError(f"Expected spatial/channel shape {expected}, got {tactile.shape[3:]}")
        batch = tactile.shape[0]
        x = tactile.reshape((batch * 2 * 3, *expected))
        for layer in self.spatial:
            x = jax.nn.gelu(layer(x))
        if self.mode == "h1":
            # 96 -> 6; retain a 2x2 spatial layout after 3x3 average pooling.
            x = x.reshape((batch * 6, 2, 3, 2, 3, 64)).mean(axis=(2, 4))
        x = x.reshape((batch * 2, 3, 256))
        x = self.norm(jax.nn.gelu(self.temporal(x))[:, 0, :])
        x = self.project(x).reshape((batch, 2, 2, -1))
        x = x + self.finger_embedding(jnp.arange(2))[None, :, None, :]
        return x.reshape((batch, 4, -1))
