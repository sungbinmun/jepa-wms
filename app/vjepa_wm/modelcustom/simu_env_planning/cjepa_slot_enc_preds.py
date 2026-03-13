# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""C-JEPA/AC-JEPA wrapper for simu_env_planning evaluation.

This module plugs slot-based C-JEPA checkpoints into the same planning API used by
`vit_enc_preds` (encode/unroll/decode_unroll).

Expected checkpoint format:
- Object checkpoint from `torch.save(pl_module, "..._object.ckpt")`, or
- Lightning checkpoint containing `state_dict` (`epoch=...-step=....ckpt`).

Expected model_kwargs.pretrain_kwargs keys:
- `slotcontrast_checkpoint`: path to SlotContrast .ckpt used for slot extraction.
- `slotcontrast_config`: path to SlotContrast settings.yaml.
- `tubelet_size_enc` (optional, default=1)
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from einops import rearrange
from tensordict.tensordict import TensorDict

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover
    OmegaConf = None

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class _Conv1DEmbedder(nn.Module):
    """Small Conv1D embedder matching cjepa's Embedder implementation."""

    def __init__(self, tubelet_size: int, in_chans: int, emb_dim: int):
        super().__init__()
        self.tubelet_size = int(tubelet_size)
        self.in_chans = int(in_chans)
        self.emb_dim = int(emb_dim)
        self.patch_embed = nn.Conv1d(
            self.in_chans,
            self.emb_dim,
            kernel_size=self.tubelet_size,
            stride=self.tubelet_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(enabled=False, device_type=x.device.type):
            x = x.permute(0, 2, 1)
            x = self.patch_embed(x)
            x = x.permute(0, 2, 1)
        return x


class _SimpleWorldModel(nn.Module):
    """Minimal container exposing the attrs used by planning wrapper."""

    def __init__(self, predictor: nn.Module, action_encoder: nn.Module, proprio_encoder: Optional[nn.Module]):
        super().__init__()
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.proprio_encoder = proprio_encoder


def _torch_load_trusted(path: str | Path, map_location: str | torch.device):
    kwargs = {"map_location": map_location}
    # PyTorch>=2.6 may default to weights_only=True; object checkpoints need False.
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def _resolve_local_checkpoint(folder: str | Path, checkpoint: str) -> Path:
    ckpt_path = Path(checkpoint)
    if ckpt_path.is_absolute():
        return ckpt_path
    return Path(folder) / checkpoint


def _ensure_cjepa_import_path() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    cjepa_root = repo_root / "cjepa"
    if not cjepa_root.exists():
        raise FileNotFoundError(f"cjepa repository not found at {cjepa_root}")
    if str(cjepa_root) not in sys.path:
        sys.path.insert(0, str(cjepa_root))
    return cjepa_root


def _cfg_select(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if OmegaConf is not None:
        try:
            value = OmegaConf.select(cfg, key)
            return default if value is None else value
        except Exception:
            pass
    cur = cfg
    for part in key.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part, None)
        else:
            cur = getattr(cur, part, None)
        if cur is None:
            return default
    return cur


def _find_train_config(checkpoint_path: Path, model_kind: str, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser()
        candidates = [p]
        if not p.is_absolute():
            candidates.append((checkpoint_path.parent / p).resolve())
            candidates.append((Path.cwd() / p).resolve())
        for c in candidates:
            if c.exists():
                return c.resolve()
        raise FileNotFoundError(f"Could not find explicit cjepa training config: {explicit}")

    patterns = []
    if model_kind == "acjepa":
        patterns.extend(
            [
                "config_train_causal_agent_centric*.yaml",
                "*agent*centric*.yaml",
            ]
        )
    elif model_kind == "cjepa":
        patterns.extend(
            [
                "config_train_causal*slot*.yaml",
                "config_train_causal*.yaml",
            ]
        )
    patterns.extend(["config_train*.yaml", "*.yaml"])
    for pattern in patterns:
        found = sorted(checkpoint_path.parent.glob(pattern))
        if found:
            return found[0].resolve()
    return None


def _load_train_config(cfg_path: Optional[Path]):
    if cfg_path is None:
        return None
    if OmegaConf is None:
        logger.warning("OmegaConf unavailable; ignoring training config %s", cfg_path)
        return None
    try:
        return OmegaConf.load(str(cfg_path))
    except Exception as exc:
        logger.warning("Failed to load training config %s (%s). Falling back to inferred hyper-params.", cfg_path, exc)
        return None


def _infer_model_kind(state_dict: dict[str, torch.Tensor]) -> str:
    if any(k.startswith("model.predictor.action_proj.") for k in state_dict):
        return "acjepa"
    if any(k.startswith("model.predictor.mask_token") for k in state_dict):
        return "cjepa"
    return "unknown"


def _infer_depth_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    layers = []
    pat = re.compile(r"model\.predictor\.transformer\.layers\.(\d+)\.")
    for key in state_dict:
        match = pat.search(key)
        if match is not None:
            layers.append(int(match.group(1)))
    return max(layers) + 1 if layers else 6


def _build_cjepa_world_model(state_dict: dict[str, torch.Tensor], cfg) -> _SimpleWorldModel:
    _ensure_cjepa_import_path()
    from src.cjepa_predictor import MaskedSlotPredictor

    action_w = state_dict["model.action_encoder.patch_embed.weight"]
    action_embed_dim, action_in_chans, action_tubelet = map(int, action_w.shape)

    proprio_encoder = None
    proprio_w = state_dict.get("model.proprio_encoder.patch_embed.weight")
    proprio_embed_dim = 0
    if proprio_w is not None:
        proprio_embed_dim, proprio_in_chans, proprio_tubelet = map(int, proprio_w.shape)
        proprio_encoder = _Conv1DEmbedder(
            tubelet_size=proprio_tubelet,
            in_chans=proprio_in_chans,
            emb_dim=proprio_embed_dim,
        )

    predictor_dim = int(state_dict["model.predictor.mask_token"].shape[-1])
    total_frames = int(state_dict["model.predictor.time_pos_embed"].shape[1])
    history_size = int(_cfg_select(cfg, "dinowm.history_size", max(total_frames - 1, 1)))
    pred_frames = int(_cfg_select(cfg, "dinowm.num_preds", 1))
    if history_size + pred_frames != total_frames:
        history_size = max(total_frames - 1, 1)
        pred_frames = 1

    predictor = MaskedSlotPredictor(
        num_slots=int(_cfg_select(cfg, "videosaur.NUM_SLOTS", 7)),
        slot_dim=predictor_dim,
        history_frames=history_size,
        pred_frames=pred_frames,
        num_masked_slots=int(_cfg_select(cfg, "num_masked_slots", 2)),
        seed=int(_cfg_select(cfg, "seed", 42)),
        depth=int(_cfg_select(cfg, "predictor.depth", _infer_depth_from_state_dict(state_dict))),
        heads=int(_cfg_select(cfg, "predictor.heads", 8)),
        dim_head=int(_cfg_select(cfg, "predictor.dim_head", 64)),
        mlp_dim=int(
            _cfg_select(
                cfg,
                "predictor.mlp_dim",
                int(state_dict["model.predictor.transformer.layers.0.1.1.weight"].shape[0]),
            )
        ),
        dropout=float(_cfg_select(cfg, "predictor.dropout", 0.1)),
    )
    action_encoder = _Conv1DEmbedder(
        tubelet_size=action_tubelet,
        in_chans=action_in_chans,
        emb_dim=action_embed_dim,
    )
    model = _SimpleWorldModel(
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
    )
    return model


def _build_acjepa_world_model(state_dict: dict[str, torch.Tensor], cfg) -> _SimpleWorldModel:
    _ensure_cjepa_import_path()
    from src.agent_centric_predictor import AgentCausalSlotPredictor

    action_w = state_dict["model.action_encoder.patch_embed.weight"]
    action_embed_dim, action_in_chans, action_tubelet = map(int, action_w.shape)

    action_proj_w = state_dict["model.predictor.action_proj.weight"]
    slot_dim = int(action_proj_w.shape[0])
    predictor_action_dim = int(action_proj_w.shape[1])

    proprio_encoder = None
    proprio_embed_dim = 0
    proprio_w = state_dict.get("model.proprio_encoder.patch_embed.weight")
    if proprio_w is not None:
        proprio_embed_dim, proprio_in_chans, proprio_tubelet = map(int, proprio_w.shape)
        proprio_encoder = _Conv1DEmbedder(
            tubelet_size=proprio_tubelet,
            in_chans=proprio_in_chans,
            emb_dim=proprio_embed_dim,
        )

    predictor = AgentCausalSlotPredictor(
        slot_dim=slot_dim,
        action_dim=predictor_action_dim,
        proprio_dim=int(_cfg_select(cfg, "agent_centric.proprio_input_dim", proprio_embed_dim)),
        history_len=int(_cfg_select(cfg, "agent_centric.history_len", _cfg_select(cfg, "dinowm.history_size", 3))),
        history_layers=int(_cfg_select(cfg, "agent_centric.history_layers", 2)),
        num_heads=int(_cfg_select(cfg, "agent_centric.heads", 8)),
        mlp_dim=int(_cfg_select(cfg, "agent_centric.mlp_dim", state_dict["model.predictor.agent_ff.0.weight"].shape[0])),
        dropout=float(_cfg_select(cfg, "agent_centric.dropout", 0.1)),
        stop_gradient_agent_to_object=bool(_cfg_select(cfg, "agent_centric.stop_gradient_agent_to_object", True)),
        use_agent_delta=bool(_cfg_select(cfg, "agent_centric.use_agent_delta", True)),
        use_object_self_attn=bool(_cfg_select(cfg, "agent_centric.use_object_self_attn", True)),
    )

    action_encoder = _Conv1DEmbedder(
        tubelet_size=action_tubelet,
        in_chans=action_in_chans,
        emb_dim=action_embed_dim,
    )
    if action_embed_dim != predictor_action_dim:
        raise ValueError(
            f"AC-JEPA mismatch: action_encoder emb_dim={action_embed_dim} "
            f"!= predictor action_dim={predictor_action_dim}"
        )

    model = _SimpleWorldModel(
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
    )
    model.proprio_head = nn.Linear(slot_dim, int(_cfg_select(cfg, "dinowm.proprio_dim", 4)))
    return model


def _filter_loadable_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    target = model.state_dict()
    filtered = {}
    shape_mismatch = []
    for key, value in state_dict.items():
        if not key.startswith("model."):
            continue
        model_key = key[len("model.") :]
        if model_key not in target:
            continue
        if target[model_key].shape != value.shape:
            shape_mismatch.append((model_key, tuple(target[model_key].shape), tuple(value.shape)))
            continue
        filtered[model_key] = value
    if shape_mismatch:
        logger.warning(
            "Skipped %d mismatched keys while loading cjepa checkpoint. Example: %s",
            len(shape_mismatch),
            shape_mismatch[0],
        )
    return filtered


def _load_world_model_from_lightning_ckpt(
    checkpoint_path: Path,
    ckpt_obj: dict[str, Any],
    device: torch.device,
    train_config_path: Optional[str],
):
    state_dict = ckpt_obj.get("state_dict", None)
    if not isinstance(state_dict, dict):
        raise ValueError("Lightning checkpoint is missing `state_dict`.")

    model_kind = _infer_model_kind(state_dict)
    if model_kind not in {"cjepa", "acjepa"}:
        raise ValueError("Could not infer checkpoint type from state_dict (expected C-JEPA or AC-JEPA predictor keys).")

    cfg_path = _find_train_config(checkpoint_path, model_kind=model_kind, explicit=train_config_path)
    if cfg_path is not None:
        logger.info("Using training config for %s: %s", model_kind, cfg_path)
    else:
        logger.warning("No training config found near %s; using inferred hyper-parameters.", checkpoint_path)
    cfg = _load_train_config(cfg_path)

    if model_kind == "acjepa":
        world_model = _build_acjepa_world_model(state_dict, cfg)
    else:
        world_model = _build_cjepa_world_model(state_dict, cfg)

    filtered_state = _filter_loadable_state_dict(world_model, state_dict)
    msg = world_model.load_state_dict(filtered_state, strict=False)
    logger.info(
        "Loaded %d tensor(s) from %s checkpoint (missing=%d, unexpected=%d).",
        len(filtered_state),
        model_kind,
        len(msg.missing_keys),
        len(msg.unexpected_keys),
    )

    world_model = world_model.to(device)
    world_model.eval()
    return world_model


def _load_cjepa_world_model(checkpoint_path: Path, device: torch.device, train_config_path: Optional[str] = None):
    obj = _torch_load_trusted(checkpoint_path, map_location="cpu")

    # Object-save format from cjepa scripts: torch.save(pl_module, ...)
    if hasattr(obj, "model"):
        world_model = obj.model
        world_model = world_model.to(device).eval()
        return world_model
    if hasattr(obj, "predictor"):
        world_model = obj
        world_model = world_model.to(device).eval()
        return world_model

    if isinstance(obj, dict) and "state_dict" in obj:
        return _load_world_model_from_lightning_ckpt(
            checkpoint_path=checkpoint_path,
            ckpt_obj=obj,
            device=device,
            train_config_path=train_config_path,
        )

    raise ValueError(
        "Unsupported C-JEPA checkpoint format. Expected object checkpoint or Lightning checkpoint with `state_dict`."
    )


def _load_slotcontrast_model(
    slotcontrast_checkpoint: Path,
    slotcontrast_config: Path,
    device: torch.device,
    slotcontrast_root: Optional[str] = None,
):
    repo_root = Path(__file__).resolve().parents[4]
    slotcontrast_root = Path(
        slotcontrast_root
        or os.environ.get("SLOTCONTRAST_ROOT", repo_root / "slotcontrast")
    ).expanduser()
    if not slotcontrast_root.exists():
        raise FileNotFoundError(
            f"slotcontrast repository not found at {slotcontrast_root}. "
            "Initialize/add the slotcontrast code before running this eval."
        )

    if (slotcontrast_root / "configuration.py").is_file():
        pkg_parent = slotcontrast_root.parent
    elif (slotcontrast_root / "slotcontrast" / "configuration.py").is_file():
        pkg_parent = slotcontrast_root
    else:
        raise FileNotFoundError(
            "Could not locate SlotContrast source code. Expected either "
            f"`{slotcontrast_root}/configuration.py` or "
            f"`{slotcontrast_root}/slotcontrast/configuration.py`."
        )

    if str(pkg_parent) not in sys.path:
        sys.path.insert(0, str(pkg_parent))

    from slotcontrast import configuration, models
    from slotcontrast.data.transforms import build as build_transforms

    cfg = configuration.load_config(str(slotcontrast_config))
    # We immediately load checkpoint weights below, so avoid network-dependent
    # timm pretrained downloads during model construction.
    try:
        if hasattr(cfg, "model") and hasattr(cfg.model, "encoder") and hasattr(cfg.model.encoder, "backbone"):
            cfg.model.encoder.backbone.pretrained = False
    except Exception:
        pass
    slot_model = models.build(cfg.model, cfg.optimizer)

    ckpt = _torch_load_trusted(slotcontrast_checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError("slotcontrast_checkpoint must be a Lightning checkpoint with key `state_dict`.")

    slot_model.load_state_dict(ckpt["state_dict"], strict=True)
    slot_model = slot_model.to(device).eval()

    train_tf_cfg = cfg.dataset.train_pipeline.transforms
    tf_map = build_transforms(train_tf_cfg)
    if "video" not in tf_map:
        raise ValueError("SlotContrast transform map is missing `video` transform.")

    return slot_model, tf_map["video"]


def init_module(
    folder,
    checkpoint,
    model_kwargs,
    device,
    action_dim,
    proprio_dim,
    preprocessor,
    cfgs_data=None,
    wrapper_kwargs=None,
    **kwargs,
):
    """Initialize C-JEPA world model wrapper for planning eval."""

    if cfgs_data is None:
        cfgs_data = {}
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    custom_cfg = cfgs_data.get("custom", {})
    frameskip = int(custom_cfg.get("frameskip", 1))
    action_skip = int(custom_cfg.get("action_skip", 1))
    tubelet_size_enc = int(model_kwargs.get("tubelet_size_enc", 1))

    model_action_dim = action_dim * tubelet_size_enc * frameskip // action_skip

    slotcontrast_checkpoint = model_kwargs.get("slotcontrast_checkpoint")
    slotcontrast_config = model_kwargs.get("slotcontrast_config")
    if slotcontrast_checkpoint is None or slotcontrast_config is None:
        slot_cfg = model_kwargs.get("slotcontrast", {})
        slotcontrast_checkpoint = slot_cfg.get("checkpoint", slotcontrast_checkpoint)
        slotcontrast_config = slot_cfg.get("config", slotcontrast_config)

    if slotcontrast_checkpoint is None or slotcontrast_config is None:
        raise ValueError(
            "Missing slotcontrast paths in pretrain_kwargs. Provide both "
            "`slotcontrast_checkpoint` and `slotcontrast_config`."
        )

    train_config_path = (
        model_kwargs.get("cjepa_train_config")
        or model_kwargs.get("train_config")
        or model_kwargs.get("model_config")
    )

    cjepa_checkpoint_path = _resolve_local_checkpoint(folder, checkpoint)
    world_model = _load_cjepa_world_model(
        cjepa_checkpoint_path,
        device=device,
        train_config_path=train_config_path,
    )

    slot_model, slot_video_tf = _load_slotcontrast_model(
        slotcontrast_checkpoint=Path(slotcontrast_checkpoint),
        slotcontrast_config=Path(slotcontrast_config),
        device=device,
        slotcontrast_root=model_kwargs.get("slotcontrast_root"),
    )

    model = CJEPAEncPredWM(
        cjepa_model=world_model,
        slotcontrast_model=slot_model,
        slotcontrast_video_transform=slot_video_tf,
        action_dim=model_action_dim,
        preprocessor=preprocessor,
        action_skip=action_skip,
        tubelet_size_enc=tubelet_size_enc,
        ctxt_window=int(wrapper_kwargs.get("ctxt_window", 2)),
        device=device,
    )
    model.eval()
    return model


class CJEPAEncPredWM(nn.Module):
    """Adapter exposing encode/unroll/decode_unroll expected by planning code."""

    def __init__(
        self,
        cjepa_model,
        slotcontrast_model,
        slotcontrast_video_transform,
        action_dim: int,
        preprocessor,
        action_skip: int,
        tubelet_size_enc: int,
        ctxt_window: int,
        device,
    ):
        super().__init__()
        self.model = cjepa_model
        self.slot_model = slotcontrast_model
        self.slot_video_tf = slotcontrast_video_transform

        self.predictor = self.model.predictor
        self.action_encoder = getattr(self.model, "action_encoder", None)
        self.proprio_encoder = getattr(self.model, "proprio_encoder", None)

        self.device = torch.device(device)
        self.action_dim = int(action_dim)
        self.action_skip = int(action_skip)
        self.tubelet_size_enc = int(tubelet_size_enc)
        self.ctxt_window = int(ctxt_window)
        self.preprocessor = preprocessor

        if self.action_encoder is not None and hasattr(self.action_encoder, "in_chans"):
            exp_act_dim = int(getattr(self.action_encoder, "in_chans"))
            if exp_act_dim != self.action_dim:
                raise ValueError(
                    "Action dim mismatch between eval setup and checkpoint: "
                    f"eval action_dim={self.action_dim}, checkpoint expects {exp_act_dim}. "
                    "Align frameskip/action_skip and dataset action format."
                )

        self.heads = {}

        self._feature_conditioned_predictor = not self._is_action_conditioned_predictor(self.predictor)
        self._ap_node_predictor = "AP_Predictor" in self.predictor.__class__.__name__
        self._has_proprio_head = hasattr(self.model, "proprio_head")
        self._action_embed_dim = self._infer_embed_dim(self.action_encoder, self.action_dim)
        self._proprio_embed_dim = self._infer_embed_dim(
            self.proprio_encoder,
            getattr(self.proprio_encoder, "in_chans", 1),
        )

        self._last_proprio_context: Optional[torch.Tensor] = None

        if self._feature_conditioned_predictor and self._ap_node_predictor:
            mode = "feature-conditioned AP-node (C-JEPA)"
        elif self._feature_conditioned_predictor:
            mode = "feature-conditioned (C-JEPA)"
        else:
            mode = "action-conditioned (AC-JEPA)"
        logger.info(f"Loaded predictor mode: {mode}")

    @staticmethod
    def _is_action_conditioned_predictor(predictor) -> bool:
        inference = getattr(predictor, "inference", None)
        if inference is None:
            return False
        sig = inspect.signature(inference)
        # Bound method: `self` already removed.
        return len(sig.parameters) >= 2

    def _infer_embed_dim(self, encoder, in_chans: int) -> int:
        if encoder is None:
            return 0
        if hasattr(encoder, "emb_dim"):
            return int(encoder.emb_dim)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, int(in_chans), device=self.device, dtype=torch.float32)
            out = encoder(dummy)
        return int(out.shape[-1])

    def _extract_visual_and_proprio(self, obs):
        if isinstance(obs, dict) or hasattr(obs, "keys"):
            visual = obs["visual"]
            proprio = obs.get("proprio", None)
            return visual, proprio
        return obs, None

    def _to_uint8(self, visual: torch.Tensor) -> torch.Tensor:
        if visual.dtype == torch.uint8:
            return visual
        if torch.is_floating_point(visual):
            if visual.numel() > 0 and float(visual.max()) <= 1.0 + 1e-6:
                visual = visual * 255.0
            return visual.clamp(0.0, 255.0).to(torch.uint8)
        return visual.to(torch.uint8)

    @torch.no_grad()
    def _encode_slots(self, visual: torch.Tensor) -> torch.Tensor:
        # visual: [B, T, C, H, W]
        visual = self._to_uint8(visual)
        bsz = visual.shape[0]

        transformed = []
        for i in range(bsz):
            frames_np = visual[i].permute(0, 2, 3, 1).cpu().numpy()
            transformed.append(self.slot_video_tf(frames_np))

        video = torch.stack(transformed, dim=0).to(self.device)
        enc_out = self.slot_model.encoder(video)
        features = enc_out["features"]
        slots_init = self.slot_model.initializer(batch_size=bsz).to(self.device)
        proc_out = self.slot_model.processor(slots_init, features)
        return proc_out["state"].to(torch.float32)

    @torch.no_grad()
    def _encode_proprio(self, proprio: torch.Tensor) -> Optional[torch.Tensor]:
        if proprio is None:
            return None
        if self.preprocessor is not None and hasattr(self.preprocessor, "normalize_proprios"):
            proprio = self.preprocessor.normalize_proprios(proprio.cpu()).to(self.device, dtype=torch.float32)
        else:
            proprio = proprio.to(self.device, dtype=torch.float32)
        if self.proprio_encoder is None:
            if self._has_proprio_head:
                return proprio
            return None
        return self.proprio_encoder(proprio)

    @staticmethod
    def _squeeze_proprio_node(proprio: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if proprio is None:
            return None
        if proprio.ndim == 4 and proprio.shape[2] == 1:
            return proprio[:, :, 0, :]
        return proprio

    def _pack_outputs(
        self,
        visual: torch.Tensor,
        proprio: Optional[torch.Tensor],
        batch_first: bool,
    ):
        if proprio is None:
            return visual
        if proprio.ndim == 3:
            proprio = proprio.unsqueeze(2)
        if batch_first:
            return TensorDict({"visual": visual, "proprio": proprio}, device=visual.device)
        return TensorDict(
            {
                "visual": rearrange(visual, "b t ... -> t b ..."),
                "proprio": rearrange(proprio, "b t ... -> t b ..."),
            },
            device=visual.device,
        )

    def _compose_apnode_embedding(
        self,
        visual_slots: torch.Tensor,
        proprio_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, t, _, dim = visual_slots.shape
        pieces = [visual_slots]

        if self.proprio_encoder is not None:
            if proprio_embed is None:
                proprio_embed = torch.zeros(bsz, t, dim, device=visual_slots.device, dtype=visual_slots.dtype)
            pieces.append(proprio_embed.unsqueeze(2))

        if self.action_encoder is not None:
            action_zeros = torch.zeros(bsz, t, dim, device=visual_slots.device, dtype=visual_slots.dtype)
            pieces.append(action_zeros.unsqueeze(2))

        return torch.cat(pieces, dim=2)

    def _inject_action_node(
        self,
        full_embed: torch.Tensor,
        action_embed: torch.Tensor,
    ) -> torch.Tensor:
        if self.action_encoder is None:
            return full_embed

        if full_embed.shape[0] != action_embed.shape[0]:
            if full_embed.shape[0] == 1:
                full_embed = full_embed.expand(action_embed.shape[0], *full_embed.shape[1:])
            elif action_embed.shape[0] == 1:
                action_embed = action_embed.expand(full_embed.shape[0], *action_embed.shape[1:])
            else:
                raise ValueError(
                    "Batch mismatch between latent features and action node: "
                    f"{tuple(full_embed.shape)} vs {tuple(action_embed.shape)}"
                )

        out = full_embed.clone()
        out[:, :, -1, :] = action_embed
        return out

    def _compose_feature_conditioned_embedding(
        self,
        visual_slots: torch.Tensor,
        proprio_embed: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # visual_slots: [B, T, S, Dv]
        bsz, t, num_slots, _ = visual_slots.shape
        pieces = [visual_slots]

        if self.proprio_encoder is not None:
            if proprio_embed is None:
                proprio_embed = torch.zeros(
                    bsz, t, self._proprio_embed_dim, device=visual_slots.device, dtype=visual_slots.dtype
                )
            pieces.append(proprio_embed.unsqueeze(2).expand(-1, -1, num_slots, -1))

        if self.action_encoder is not None:
            action_zeros = torch.zeros(
                bsz, t, self._action_embed_dim, device=visual_slots.device, dtype=visual_slots.dtype
            )
            pieces.append(action_zeros.unsqueeze(2).expand(-1, -1, num_slots, -1))

        return torch.cat(pieces, dim=-1)

    def _inject_action_features(
        self,
        full_embed: torch.Tensor,
        action_embed: torch.Tensor,
        visual_dim: int,
    ) -> torch.Tensor:
        if self.action_encoder is None:
            return full_embed

        # Planning passes batched action candidates (e.g. CEM num_samples),
        # while context encoding is often computed for a single observation.
        # Align the batch dimension before action injection.
        if full_embed.shape[0] != action_embed.shape[0]:
            if full_embed.shape[0] == 1:
                full_embed = full_embed.expand(action_embed.shape[0], *full_embed.shape[1:])
            elif action_embed.shape[0] == 1:
                action_embed = action_embed.expand(full_embed.shape[0], *action_embed.shape[1:])
            else:
                raise ValueError(
                    "Batch mismatch between latent features and action features: "
                    f"{tuple(full_embed.shape)} vs {tuple(action_embed.shape)}"
                )

        start = visual_dim
        if self.proprio_encoder is not None:
            start += self._proprio_embed_dim

        out = full_embed.clone()
        action_tiled = action_embed.unsqueeze(2).expand(-1, -1, full_embed.shape[2], -1)
        out[..., start : start + self._action_embed_dim] = action_tiled
        return out

    @staticmethod
    def _match_batch_dim(x: Optional[torch.Tensor], batch_size: int, name: str) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if x.shape[0] == batch_size:
            return x
        if x.shape[0] == 1:
            return x.expand(batch_size, *x.shape[1:])
        raise ValueError(f"Cannot broadcast `{name}` batch from {x.shape[0]} to {batch_size}.")

    @torch.no_grad()
    def encode(self, obs, act=True):
        """Encode raw observations into slot latents.

        Returns tensor [B, T, S, D_slot].
        """
        visual, proprio = self._extract_visual_and_proprio(obs)
        visual = visual.to(self.device, dtype=visual.dtype, non_blocking=True)

        slots = self._encode_slots(visual)
        proprio_ctx = self._encode_proprio(proprio) if proprio is not None else None
        self._last_proprio_context = proprio_ctx
        return self._pack_outputs(slots, proprio_ctx, batch_first=True)

    @torch.no_grad()
    def unroll(self, z_ctxt, act_suffix: torch.Tensor = None, debug: bool = False):
        """Autoregressive latent rollout.

        Args:
            z_ctxt: [B, T_ctx, S, D_slot]
            act_suffix: [T_plan, B, A]

        Returns:
            [T_ctx + T_plan, B, S, D_slot]
        """
        if act_suffix is None:
            raise ValueError("act_suffix is required for unroll.")

        t_plan, bsz, _ = act_suffix.shape
        act_bt = rearrange(act_suffix, "t b a -> b t a")

        if self.action_encoder is not None:
            act_feats = self.action_encoder(act_bt)
        else:
            act_feats = act_bt

        if isinstance(z_ctxt, dict) or hasattr(z_ctxt, "keys"):
            visual_feats = z_ctxt["visual"]
            proprio_input = self._squeeze_proprio_node(z_ctxt.get("proprio", None))
        else:
            visual_feats = z_ctxt
            proprio_input = None

        visual_feats = self._match_batch_dim(visual_feats, bsz, "z_ctxt.visual")
        proprio_input = self._match_batch_dim(proprio_input, bsz, "z_ctxt.proprio")

        if self._feature_conditioned_predictor:
            visual_dim = int(visual_feats.shape[-1])
            proprio_ctx = proprio_input
            if proprio_ctx is None:
                proprio_ctx = self._last_proprio_context
                if proprio_ctx is not None:
                    if proprio_ctx.shape[1] != visual_feats.shape[1]:
                        proprio_ctx = None
                    else:
                        proprio_ctx = self._match_batch_dim(proprio_ctx, bsz, "proprio_context")

            if self._ap_node_predictor:
                full_feats = self._compose_apnode_embedding(visual_feats, proprio_ctx)
            else:
                full_feats = self._compose_feature_conditioned_embedding(visual_feats, proprio_ctx)

            for h in range(t_plan):
                pred_full = self.predictor.inference(full_feats[:, -self.ctxt_window :])
                next_full = pred_full[:, -1:]
                if self._ap_node_predictor:
                    next_full = self._inject_action_node(next_full, act_feats[:, h : h + 1])
                else:
                    next_full = self._inject_action_features(next_full, act_feats[:, h : h + 1], visual_dim)
                full_feats = torch.cat([full_feats, next_full], dim=1)

            if self._ap_node_predictor:
                num_visual_slots = int(visual_feats.shape[2])
                visual_out = full_feats[:, :, :num_visual_slots, :]
                proprio_out = None
                if self.proprio_encoder is not None:
                    proprio_out = full_feats[:, :, num_visual_slots : num_visual_slots + 1, :]
            else:
                visual_out = full_feats[..., :visual_dim]
                proprio_out = None
                if self.proprio_encoder is not None:
                    proprio_slice = full_feats[..., visual_dim : visual_dim + self._proprio_embed_dim]
                    proprio_out = proprio_slice[:, :, :1, :]
            return self._pack_outputs(visual_out, proprio_out, batch_first=False)

        # AC-JEPA: predictor takes slot history + action embedding + proprio history.
        proprio_hist = proprio_input
        for h in range(t_plan):
            pred_next = self.predictor.inference(
                visual_feats[:, -self.ctxt_window :],
                act_feats[:, h : h + 1],
                history_proprio=proprio_hist[:, -self.ctxt_window :] if proprio_hist is not None else None,
            )
            next_visual = pred_next[:, -1:]
            visual_feats = torch.cat([visual_feats, next_visual], dim=1)
            if self._has_proprio_head:
                agent_slots = next_visual[:, :, -1, :]
                next_proprio_raw = self.model.proprio_head(agent_slots)
                if self.proprio_encoder is not None:
                    next_proprio = self.proprio_encoder(next_proprio_raw)
                else:
                    next_proprio = next_proprio_raw
                if proprio_hist is None:
                    proprio_hist = next_proprio
                else:
                    proprio_hist = torch.cat([proprio_hist, next_proprio], dim=1)

        return self._pack_outputs(visual_feats, proprio_hist, batch_first=False)

    @torch.no_grad()
    def decode_unroll(self, predicted_encs, batch: bool = False):
        # C-JEPA/AC-JEPA wrapper has no decoder head in this integration.
        return None

    def eval(self):
        super().eval()
        self.model.eval()
        self.slot_model.eval()
        return self
