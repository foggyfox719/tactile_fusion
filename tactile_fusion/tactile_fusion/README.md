# Tactile Fusion

Tactile Fusion adds a visuo-tactile branch to an existing pi05 policy.  
It is **not a fork of openpi** and does not include a pi05 checkout. The
repository contains only the tactile modules and the small integration points
needed to connect them to pi05.

The branch takes two fingertip images, forms a six-channel tactile frame
(current RGB + current − no-contact reference), encodes three frames per
finger into four prefix tokens, and conditions the action expert on those
tokens.

## Repository layout

```text
tactile_fusion/
  README.md
  code/
    tactile_encoder.py
    tactile_preprocess.py
    tactile_piper_policy.py
    anytouch2_features.py
    tactile_dataset.py
```

No full openpi tree, model weights, datasets, feature caches, normalization
statistics, or robot deployment scripts are included.

## Modules

### `code/tactile_encoder.py` → `src/openpi/models/`

- H1 encoder: 2 fingers × 3 frames × 96×96×6
- four stride-2 spatial convolutions
- 2×2 spatial pooling
- temporal Conv1D over three frames
- four tactile prefix tokens
- AnyTouch2 token adapter

### `code/tactile_preprocess.py` → `src/openpi/policies/`

- H1 frame construction: `[current_rgb, current_rgb − reference_rgb]`
- L2 depth/flow frame
- online three-frame history per finger

### `code/tactile_piper_policy.py` → `src/openpi/policies/`

- Piper input transform: side + wrist image, state, prompt, tactile
- output transform: first seven action dimensions

### `code/anytouch2_features.py` → `src/openpi/policies/`

- anytouch2 feature extraction for the adapter route

### `code/tactile_dataset.py` → `src/openpi/training/`

- tactile dataset reader and cache adapter

## Integration points

These are the small changes needed in an existing pi05 tree. They are shown as
snippets, not as a copied openpi fork.

### `models/pi0_config.py`

```python
tactile_mode: str = "none"   # none | l2 | h1 | anytouch2
tactile_history: int = 3
tactile_tokens_per_finger: int = 2
```

Validate the mode and history in `__post_init__`, and include the tactile
shape in `fake_obs()`:

```text
l2:        [B, 2, 3, 8, 8, 3]
h1:        [B, 2, 3, 96, 96, 6]
anytouch2: [B, 2, 2, 512]
```

### `models/model.py`

Add an optional tactile field to `Observation`:

```python
tactile: at.Float[ArrayT, "*b ..."] | None = None
```

and pass it through `Observation.from_dict()`.

### `models/pi0.py`

Create the tactile branch:

```python
if config.tactile_mode == "anytouch2":
    self.tactile_encoder = AnyTouchTokenAdapter(width, rngs)
elif config.tactile_mode != "none":
    self.tactile_encoder = TactileEncoder(config.tactile_mode, width, rngs)
```

Append the tactile tokens to the prefix sequence:

```python
if self.tactile_mode != "none":
    tactile_tokens = self.tactile_encoder(obs.tactile)
    tokens.append(tactile_tokens)
    input_mask.append(jnp.ones(tactile_tokens.shape[:2], dtype=jnp.bool_))
    ar_mask += [False] * tactile_tokens.shape[1]
```

Tactile tokens are conditioning tokens: they are not language tokens and are
not autoregressive targets.

### `training/weight_loaders.py`

Three loaders are useful when the tactile branch is added to an existing
checkpoint:

- load base weights, initialize tactile weights locally;
- load everything except the tactile encoder;
- load a complete checkpoint, with H1 spatial-key normalization.

The complete implementations are part of the original training branch; this
showcase keeps only the model/policy/data modules.

### `training/config.py`

The tactile data config wraps the LeRobot Piper dataset and exposes the tactile
key:

```python
data=PiperTactileDataConfig(mode="h1")
model=pi0_config.Pi0Config(
    pi05=True,
    tactile_mode="h1",
    tactile_history=3,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
)
```

The model and data tactile modes must match. The concrete checkpoint and data
paths belong to the training environment and are intentionally not included.

### `scripts/train.py`

Contact-weighted loss uses the three difference channels:

```python
if alpha > 0 and observation.tactile is not None:
    tactile = observation.tactile
    if tactile.ndim >= 5 and tactile.shape[-1] == 6:
        tactile = tactile[..., 3:6]
    strength = jnp.mean(jnp.abs(tactile), axis=axes)
```

`alpha` corresponds to `TACTILE_LOSS_WEIGHT`.

## Input and token shape

```text
[B, 2 fingers, 3 frames, 96, 96, 6]
  channels 0–2: current RGB
  channels 3–5: current − reference
        ↓
4 convolution / pooling / temporal encoder
        ↓
[B, 4, width]
  finger 0 spatial 0
  finger 0 spatial 1
  finger 1 spatial 0
  finger 1 spatial 1
```

## Training outline

The contact-aware variant trains:

```text
tactile encoder
action expert LoRA
action_in_proj / action_out_proj
time_mlp_in / time_mlp_out
```

and freezes the language backbone, language LoRA, vision tower, and action
expert base.

This is a reference description. Dataset paths, checkpoints, normalization
statistics, and hardware setup are environment-specific and are not part of
this showcase repository.
