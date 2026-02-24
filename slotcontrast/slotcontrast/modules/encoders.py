import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import einops
import logging
import timm
import torch
import torchvision
from torch import nn

from slotcontrast.modules import utils
from slotcontrast.utils import config_as_kwargs, make_build_fn


@make_build_fn(__name__, "encoder")
def build(config, name: str):
    if name == "FrameEncoder":
        pos_embed = None
        if config.get("pos_embed"):
            pos_embed = utils.build_module(config.pos_embed)

        output_transform = None
        if config.get("output_transform"):
            output_transform = utils.build_module(config.output_transform)
        return FrameEncoder(
            backbone=utils.build_module(config.backbone, default_group="encoders"),
            pos_embed=pos_embed,
            output_transform=output_transform,
            **config_as_kwargs(config, ("backbone", "pos_embed", "output_transform")),
        )
    else:
        return None


class FrameEncoder(nn.Module):
    """Module reducing image to set of features."""

    def __init__(
        self,
        backbone: nn.Module,
        pos_embed: Optional[nn.Module] = None,
        output_transform: Optional[nn.Module] = None,
        spatial_flatten: bool = False,
        main_features_key: str = "vit_block12",
        feat_upsample: bool = False,
        feat_upsample_model: str = "anyup_multi_backbone",
        feat_upsample_checkpoint: Optional[str] = None,
        feat_upsample_output_size: Optional[Union[int, List[int]]] = None,
        feat_upsample_use_natten: bool = False,
        feat_upsample_q_chunk_size: Optional[int] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.pos_embed = pos_embed
        self.output_transform = output_transform
        self.spatial_flatten = spatial_flatten
        self.main_features_key = main_features_key
        self.feat_upsample = feat_upsample
        self.feat_upsample_output_size = feat_upsample_output_size
        self.feat_upsample_q_chunk_size = feat_upsample_q_chunk_size
        self.patch_size = None
        self.feat_upsampler = None
        if self.feat_upsample:
            self.patch_size = self._get_patch_size()
            self.feat_upsampler = self._load_anyup(
                feat_upsample_model, feat_upsample_checkpoint, feat_upsample_use_natten
            )
            self.feat_upsampler.eval()
            self.feat_upsampler.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # images: batch x n_channels x height x width
        backbone_features = self.backbone(images)
        if isinstance(backbone_features, dict):
            features = backbone_features[self.main_features_key].clone()
        else:
            features = backbone_features.clone()

        if self.feat_upsample:
            features_map = self._to_map(features, images)
            out_size = self._get_output_size(images)
            with torch.no_grad():
                features_up = self.feat_upsampler(
                    images, features_map, output_size=out_size, q_chunk_size=self.feat_upsample_q_chunk_size
                )
            features = features_up.flatten(2).transpose(1, 2).contiguous()
            if isinstance(backbone_features, dict):
                backbone_features = dict(backbone_features)
                backbone_features[f"{self.main_features_key}_lr"] = backbone_features[
                    self.main_features_key
                ]
                backbone_features[self.main_features_key] = features
                backbone_features["backbone_features_lr"] = backbone_features[
                    f"{self.main_features_key}_lr"
                ]
            else:
                backbone_features = {
                    "backbone_features_lr": backbone_features,
                    self.main_features_key: features,
                }

        if self.pos_embed:
            features = self.pos_embed(features)

        if self.spatial_flatten:
            features = einops.rearrange(features, "b c h w -> b (h w) c")
        if self.output_transform:
            features = self.output_transform(features)

        assert (
            features.ndim == 3
        ), f"Expect output shape (batch, tokens, dims), but got {features.shape}"
        if isinstance(backbone_features, dict):
            for k, backbone_feature in backbone_features.items():
                bf = backbone_feature
                if self.spatial_flatten:
                    bf = einops.rearrange(bf, "b c h w -> b (h w) c")
                    backbone_features[k] = bf
                assert (
                    bf.ndim == 3
                ), f"Expect output shape (batch, tokens, dims), but got {bf.shape}"
            main_backbone_features = backbone_features[self.main_features_key]

            return {
                "features": features,
                "backbone_features": main_backbone_features,
                **backbone_features,
            }
        else:
            if self.spatial_flatten:
                backbone_features = einops.rearrange(backbone_features, "b c h w -> b (h w) c")
            assert (
                backbone_features.ndim == 3
            ), f"Expect output shape (batch, tokens, dims), but got {backbone_features.shape}"

            return {
                "features": features,
                "backbone_features": backbone_features,
            }

    def _get_patch_size(self) -> int:
        name = getattr(self.backbone, "model_name", "") or ""
        match = re.search(r"patch(\\d+)", name)
        if match:
            return int(match.group(1))
        base = getattr(self.backbone, "model", None)
        base = getattr(base, "base", base)
        patch = getattr(getattr(base, "patch_embed", None), "patch_size", None)
        if patch is not None:
            if isinstance(patch, (list, tuple)):
                return int(patch[0])
            return int(patch)
        raise ValueError("feat_upsample needs a ViT-style backbone with a patch size")

    def _to_map(self, tokens: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 4:
            return tokens
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens to be 3D or 4D, got {tokens.shape}")
        b, n, c = tokens.shape
        h = images.shape[-2] // self.patch_size
        w = images.shape[-1] // self.patch_size
        if h * w != n:
            side = int(math.sqrt(n))
            if side * side != n:
                raise ValueError(f"Can not reshape {n} tokens into a grid for AnyUp")
            h = side
            w = side
        return tokens.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    def _get_output_size(self, images: torch.Tensor) -> List[int]:
        out_size = self.feat_upsample_output_size
        if out_size is None:
            return [int(images.shape[-2]), int(images.shape[-1])]
        if isinstance(out_size, int):
            return [int(out_size), int(out_size)]
        return [int(out_size[0]), int(out_size[1])]

    def _load_anyup(self, model_name: str, checkpoint: Optional[str], use_natten: bool):
        repo_dir = Path(__file__).resolve().parents[2] / "anyup"
        if checkpoint:
            if repo_dir.is_dir():
                model = torch.hub.load(
                    str(repo_dir),
                    model_name,
                    source="local",
                    pretrained=False,
                    use_natten=use_natten,
                )
            else:
                model = torch.hub.load(
                    "wimmerth/anyup",
                    model_name,
                    pretrained=False,
                    use_natten=use_natten,
                )
            state = torch.load(checkpoint, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            return model
        if repo_dir.is_dir():
            return torch.hub.load(
                str(repo_dir),
                model_name,
                source="local",
                pretrained=True,
                use_natten=use_natten,
            )
        return torch.hub.load(
            "wimmerth/anyup",
            model_name,
            pretrained=True,
            use_natten=use_natten,
        )

class TimmExtractor(nn.Module):
    """Feature extractor utilizing models from timm library."""

    FEATURE_ALIASES = {
        **{f"resnet_block{i}": f"layer{i}" for i in range(1, 5)},
        **{f"vit_block{i + 1}": f"blocks.{i}" for i in range(12)},
        **{f"vit_block_values{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_queries{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_keys{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        "vit_output": "norm",
    }
    FEATURE_MAPPING = {
        **{f"layer{i}": f"resnet_block{i}" for i in range(1, 5)},
        **{f"blocks.{i}": f"vit_block{i + 1}" for i in range(12)},
        **{f"blocks.{i}.attn.qkv": f"vit_block_keys{i + 1}" for i in range(12)},
        "norm": "vit_output",
    }

    def __init__(
        self,
        model: str,
        pretrained: bool = False,
        frozen: bool = False,
        features: Optional[Union[str, List[str]]] = None,
        checkpoint_path: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model_name = model
        self.frozen = frozen
        self.features = [features] if isinstance(features, str) else features

        base = TimmExtractor._create_model(self.model_name, pretrained, checkpoint_path, model_kwargs)

        # Determine ViT-like models by structure (not by name prefix).
        # This is robust for models such as dinov3_* / eva_* that do not start with "vit".
        self.is_vit = self._is_vit_like(base)

        self._uses_fx = False
        if self.features is not None:
            if self._should_use_hook_extractor(self.model_name, base):
                hook_nodes = self._resolve_feature_nodes(base, self.features)
                self.model = self._create_hook_extractor(base, hook_nodes)
                logging.info(
                    "Using hook-based extractor for %s (nodes=%s) to avoid FX.",
                    self.model_name,
                    hook_nodes,
                )
            else:
                self._uses_fx = True
                nodes = torchvision.models.feature_extraction.get_graph_node_names(base)[0]
                feats = []
                for name in self.features:
                    name = self.FEATURE_ALIASES.get(name, name)
                    if not any(node.startswith(name) for node in nodes):
                        raise ValueError(
                            f"Requested features under node {name}, but this node does "
                            f"not exist in model {self.model_name}. Available nodes: {nodes}"
                        )
                    feats.append(name)
                self.model = torchvision.models.feature_extraction.create_feature_extractor(base, feats)
        else:
            self.model = base

        if self.frozen:
            self.requires_grad_(False)

        logging.info(
            "Loaded timm backbone %s (pretrained=%s, frozen=%s, features=%s, uses_fx=%s, is_vit_like=%s)",
            self.model_name,
            pretrained,
            frozen,
            self.features,
            self._uses_fx,
            self.is_vit,
        )

    # --------- Core: automatically infer the number of prefix tokens (CLS + register/storage) ---------
    def _num_prefix_tokens(self) -> int:
        """
        Many timm VisionTransformer/EVA variants store num_prefix_tokens (= CLS + register tokens).
        If unavailable, fall back to assuming CLS only (1).
        """
        base = self.model
        # If wrapped by HookExtractor, the underlying model is in .base
        if hasattr(base, "base"):
            base = base.base

        if hasattr(base, "num_prefix_tokens"):
            n = int(getattr(base, "num_prefix_tokens"))
            return max(n, 0)

        # Some models may store register count separately
        if hasattr(base, "num_reg_tokens"):
            nreg = int(getattr(base, "num_reg_tokens"))
            # 1 CLS + reg tokens
            return 1 + max(nreg, 0)

        # Final fallback: assume CLS only
        return 1

    def _strip_prefix(self, t: torch.Tensor) -> torch.Tensor:
        """Remove prefix tokens from (B, N, C) so that only patch tokens remain."""
        if not (torch.is_tensor(t) and t.ndim >= 2):
            return t
        n_prefix = self._num_prefix_tokens()
        if t.shape[1] <= n_prefix:
            return t[:, 0:0]  # empty
        return t[:, n_prefix:]

    # --------- Move FX-captured tensor constants to the correct device (FX path only) ---------
    def _move_tensor_constants_if_needed(self, device: torch.device):
        if not self._uses_fx:
            return
        for module in self.model.modules():
            for name, val in list(module.__dict__.items()):
                if torch.is_tensor(val) and val.device != device:
                    setattr(module, name, val.to(device))
            for name, buf in list(module._buffers.items()):
                if torch.is_tensor(buf) and buf.device != device:
                    module._buffers[name] = buf.to(device)

    @staticmethod
    def _unwrap_base(m: nn.Module) -> nn.Module:
        """Unwrap a HookExtractor-like wrapper if present."""
        if hasattr(m, "base"):
            return m.base
        return m

    @classmethod
    def _is_vit_like(cls, m: nn.Module) -> bool:
        """
        Detect ViT/EVA-like models by structure.
        timm ViT/EVA typically expose patch_embed + blocks.
        """
        base = cls._unwrap_base(m)
        return hasattr(base, "patch_embed") and hasattr(base, "blocks")

    def _should_use_hook_extractor(self, model_name: str, base_model: nn.Module) -> bool:
        """
        Prefer hook-based extraction for dinov3/eva models to avoid FX quirks.
        Only use it when the model is ViT-like and feature nodes can be resolved.
        """
        name = model_name.lower()
        if ("dinov3" in name) or ("eva" in name):
            return self._is_vit_like(base_model)
        # Otherwise keep previous behavior (FX is fine for most models)
        return False

    def _resolve_feature_nodes(self, model: nn.Module, features: List[str]) -> List[str]:
        modules = dict(model.named_modules())
        resolved = []
        for name in features:
            target = self.FEATURE_ALIASES.get(name, name)
            if target not in modules:
                raise ValueError(
                    f"Requested feature node {target} not found in model. "
                    f"Example available modules: {list(modules.keys())[:80]}"
                )
            resolved.append(target)
        return resolved

    def _create_hook_extractor(self, model: nn.Module, hook_nodes: List[str]) -> nn.Module:
        class HookExtractor(nn.Module):
            def __init__(self, base: nn.Module, nodes: List[str]):
                super().__init__()
                self.base = base
                self.nodes = nodes
                self.cache: Dict[str, Any] = {}
                self.hooks = []

                available = dict(base.named_modules())
                missing = [n for n in nodes if n not in available]
                if missing:
                    raise ValueError(f"Hook nodes not found: {missing}")

                for n in nodes:
                    m = available[n]
                    self.hooks.append(m.register_forward_hook(self._make_hook(n)))

            def _make_hook(self, name: str):
                def hook(_module, _inp, out):
                    self.cache[name] = out

                return hook

            def forward(self, x):
                self.cache = {}
                # For EVA/ViT models, forward_features is safer than forward (it avoids heads)
                if hasattr(self.base, "forward_features"):
                    _ = self.base.forward_features(x)
                else:
                    _ = self.base(x)

                missing = [k for k in self.nodes if k not in self.cache]
                if missing:
                    raise RuntimeError(
                        f"Missing hooked features: {missing}. Got keys={list(self.cache.keys())}"
                    )
                return {k: self.cache[k] for k in self.nodes}

        return HookExtractor(model, hook_nodes)

    @staticmethod
    def _create_model(
        model_name: str,
        pretrained: bool,
        checkpoint_path: Optional[str],
        model_kwargs: Optional[Dict[str, Any]],
        trials: int = 0,
    ) -> nn.Module:
        if model_kwargs is None:
            model_kwargs = {}
        try:
            model = timm.create_model(
                model_name,
                pretrained=pretrained,
                checkpoint_path=checkpoint_path,
                **model_kwargs,
            )
        except (FileExistsError, FileNotFoundError):
            if trials == 2:
                raise
            model = None

        if model is None:
            model = TimmExtractor._create_model(
                model_name,
                pretrained,
                checkpoint_path,
                model_kwargs,
                trials=trials + 1,
            )
        return model

    def forward(self, inp):
        # Move FX-captured tensor constants only in the FX path
        if isinstance(inp, torch.Tensor):
            self._move_tensor_constants_if_needed(inp.device)
        elif isinstance(inp, (list, tuple)) and len(inp) > 0 and isinstance(inp[0], torch.Tensor):
            self._move_tensor_constants_if_needed(inp[0].device)

        if self.frozen:
            with torch.no_grad():
                outputs = self.model(inp)
        else:
            outputs = self.model(inp)

        # --------- feature extraction path ---------
        if self.features is not None:
            # If this is a ViT-like model, drop CLS + register/storage tokens to keep patch tokens only.
            if self.is_vit:
                outputs = {k: self._strip_prefix(v) for k, v in outputs.items()}

            # Map FX/hook module keys -> slotcontrast alias keys
            outputs = {self.FEATURE_MAPPING[key]: value for key, value in outputs.items()}

            # Split qkv tensor for keys/queries/values requests
            for name in self.features:
                if ("keys" in name) or ("queries" in name) or ("values" in name):
                    # Internally, qkv is stored under vit_block_keysX in the original logic
                    feature_name = name.replace("queries", "keys").replace("values", "keys")

                    t = outputs[feature_name]
                    # t: (B, N, 3*C)
                    if t.ndim != 3:
                        raise ValueError(
                            f"Expected qkv tensor of shape (B,N,3C) but got {t.shape} for {feature_name}"
                        )

                    B, N, C3 = t.shape
                    if C3 % 3 != 0:
                        raise ValueError(
                            f"qkv last dim must be divisible by 3, got {C3} for {feature_name}"
                        )

                    qkv = t.reshape(B, N, 3, C3 // 3)
                    q, k, v = qkv.unbind(2)
                    if "keys" in name:
                        outputs[name] = k
                    elif "queries" in name:
                        outputs[name] = q
                    elif "values" in name:
                        outputs[name] = v

            # If there is only one output, return the tensor directly
            if len(outputs) == 1:
                return next(iter(outputs.values()))
            return outputs

        # --------- no-features path ---------
        return outputs
'''
class TimmExtractor(nn.Module):
    """Feature extractor utilizing models from timm library."""

    FEATURE_ALIASES = {
        **{f"resnet_block{i}": f"layer{i}" for i in range(1, 5)},
        **{f"vit_block{i + 1}": f"blocks.{i}" for i in range(12)},
        **{f"vit_block_values{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_queries{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        **{f"vit_block_keys{i + 1}": f"blocks.{i}.attn.qkv" for i in range(12)},
        "vit_output": "norm",
    }
    FEATURE_MAPPING = {
        **{f"layer{i}": f"resnet_block{i}" for i in range(1, 5)},
        **{f"blocks.{i}": f"vit_block{i + 1}" for i in range(12)},
        **{f"blocks.{i}.attn.qkv": f"vit_block_keys{i + 1}" for i in range(12)},
        "norm": "vit_output",
    }

    def __init__(
        self,
        model: str,
        pretrained: bool = False,
        frozen: bool = False,
        features: Optional[Union[str, List[str]]] = None,
        checkpoint_path: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model_name = model
        self.frozen = frozen
        self.features = [features] if isinstance(features, str) else features
        self.is_vit = self.model_name.startswith("vit")

        base = TimmExtractor._create_model(self.model_name, pretrained, checkpoint_path, model_kwargs)

        self._uses_fx = False
        if self.features is not None:
            if self._should_use_hook_extractor(self.model_name):
                hook_nodes = self._resolve_feature_nodes(base, self.features)
                self.model = self._create_hook_extractor(base, hook_nodes)
                logging.info("Using hook-based extractor for %s (nodes=%s) to avoid FX.", self.model_name, hook_nodes)
            else:
                self._uses_fx = True
                nodes = torchvision.models.feature_extraction.get_graph_node_names(base)[0]
                feats = []
                for name in self.features:
                    name = self.FEATURE_ALIASES.get(name, name)
                    if not any(node.startswith(name) for node in nodes):
                        raise ValueError(
                            f"Requested features under node {name}, but this node does "
                            f"not exist in model {self.model_name}. Available nodes: {nodes}"
                        )
                    feats.append(name)
                self.model = torchvision.models.feature_extraction.create_feature_extractor(base, feats)
        else:
            self.model = base

        if self.frozen:
            self.requires_grad_(False)

        logging.info(
            "Loaded timm backbone %s (pretrained=%s, frozen=%s, features=%s, uses_fx=%s)",
            self.model_name,
            pretrained,
            frozen,
            self.features,
            self._uses_fx,
        )

    # --------- 핵심: prefix token(= CLS + register 등) 개수 자동 추정 ---------
    def _num_prefix_tokens(self) -> int:
        """
        timm VisionTransformer/EVA 계열은 보통 num_prefix_tokens(= cls + reg)을 들고 있음.
        없으면 1(= cls only)로 fallback.
        """
        base = self.model
        # HookExtractor로 감싼 경우 base는 .base에 있음
        if hasattr(base, "base"):
            base = base.base

        for attr in ("num_prefix_tokens",):
            if hasattr(base, attr):
                n = int(getattr(base, attr))
                return max(n, 0)

        # 일부 모델은 num_tokens / num_reg_tokens를 따로 들고 있을 수 있음
        if hasattr(base, "num_reg_tokens"):
            nreg = int(getattr(base, "num_reg_tokens"))
            # cls 1개 + reg
            return 1 + max(nreg, 0)

        # 최후 fallback: cls만 있다고 가정
        return 1

    def _strip_prefix(self, t: torch.Tensor) -> torch.Tensor:
        """(B, N, C)에서 prefix token들을 제거해 패치 토큰만 남김."""
        if not (torch.is_tensor(t) and t.ndim >= 2):
            return t
        n_prefix = self._num_prefix_tokens()
        if t.shape[1] <= n_prefix:
            return t[:, 0:0]  # empty
        return t[:, n_prefix:]

    # --------- FX 상수 이동(이제 FX 경로에서만) ---------
    def _move_tensor_constants_if_needed(self, device: torch.device):
        if not self._uses_fx:
            return
        for module in self.model.modules():
            for name, val in list(module.__dict__.items()):
                if torch.is_tensor(val) and val.device != device:
                    setattr(module, name, val.to(device))
            for name, buf in list(module._buffers.items()):
                if torch.is_tensor(buf) and buf.device != device:
                    module._buffers[name] = buf.to(device)

    def _should_use_hook_extractor(self, model_name: str) -> bool:
        name = model_name.lower()
        return self.is_vit and ("eva" in name or "dinov3" in name)

    def _resolve_feature_nodes(self, model: nn.Module, features: List[str]) -> List[str]:
        modules = dict(model.named_modules())
        resolved = []
        for name in features:
            target = self.FEATURE_ALIASES.get(name, name)
            if target not in modules:
                raise ValueError(
                    f"Requested feature node {target} not found in model. "
                    f"Example available modules: {list(modules.keys())[:80]}"
                )
            resolved.append(target)
        return resolved

    def _create_hook_extractor(self, model: nn.Module, hook_nodes: List[str]) -> nn.Module:
        class HookExtractor(nn.Module):
            def __init__(self, base: nn.Module, nodes: List[str]):
                super().__init__()
                self.base = base
                self.nodes = nodes
                self.cache: Dict[str, Any] = {}
                self.hooks = []

                available = dict(base.named_modules())
                missing = [n for n in nodes if n not in available]
                if missing:
                    raise ValueError(f"Hook nodes not found: {missing}")

                for n in nodes:
                    m = available[n]
                    self.hooks.append(m.register_forward_hook(self._make_hook(n)))

            def _make_hook(self, name: str):
                def hook(_module, _inp, out):
                    self.cache[name] = out
                return hook

            def forward(self, x):
                self.cache = {}
                # EVA/ViT는 forward_features가 더 안전 (head를 안 거침)
                if hasattr(self.base, "forward_features"):
                    _ = self.base.forward_features(x)
                else:
                    _ = self.base(x)

                missing = [k for k in self.nodes if k not in self.cache]
                if missing:
                    raise RuntimeError(f"Missing hooked features: {missing}. Got keys={list(self.cache.keys())}")
                return {k: self.cache[k] for k in self.nodes}

        return HookExtractor(model, hook_nodes)

    @staticmethod
    def _create_model(
        model_name: str,
        pretrained: bool,
        checkpoint_path: Optional[str],
        model_kwargs: Optional[Dict[str, Any]],
        trials: int = 0,
    ) -> nn.Module:
        if model_kwargs is None:
            model_kwargs = {}
        try:
            model = timm.create_model(
                model_name, pretrained=pretrained, checkpoint_path=checkpoint_path, **model_kwargs
            )
        except (FileExistsError, FileNotFoundError):
            if trials == 2:
                raise
            model = None

        if model is None:
            model = TimmExtractor._create_model(
                model_name, pretrained, checkpoint_path, model_kwargs, trials=trials + 1
            )
        return model

    def forward(self, inp):
        # FX 경로에서만 stray tensor 상수 이동
        if isinstance(inp, torch.Tensor):
            self._move_tensor_constants_if_needed(inp.device)
        elif isinstance(inp, (list, tuple)) and len(inp) > 0 and isinstance(inp[0], torch.Tensor):
            self._move_tensor_constants_if_needed(inp[0].device)

        if self.frozen:
            with torch.no_grad():
                outputs = self.model(inp)
        else:
            outputs = self.model(inp)

        # --------- feature extraction path ---------
        if self.features is not None:
            # 1) vit이면 CLS + reg(prefix) 토큰을 전부 제거해서 patch 토큰만 남김
            if self.is_vit:
                outputs = {k: self._strip_prefix(v) for k, v in outputs.items()}

            # 2) FX/hook에서 나온 module key -> slotcontrast alias key로 매핑
            outputs = {self.FEATURE_MAPPING[key]: value for key, value in outputs.items()}

            # 3) qkv 분해 (keys/queries/values 요청 대응)
            for name in self.features:
                if ("keys" in name) or ("queries" in name) or ("values" in name):
                    # 내부적으로는 vit_block_keysX에 qkv 전체가 들어있음 (기존 로직 유지)
                    feature_name = name.replace("queries", "keys").replace("values", "keys")

                    t = outputs[feature_name]
                    # t: (B, N, 3*C)
                    if t.ndim != 3:
                        raise ValueError(f"Expected qkv tensor of shape (B,N,3C) but got {t.shape} for {feature_name}")

                    B, N, C3 = t.shape
                    if C3 % 3 != 0:
                        raise ValueError(f"qkv last dim must be divisible by 3, got {C3} for {feature_name}")

                    qkv = t.reshape(B, N, 3, C3 // 3)
                    q, k, v = qkv.unbind(2)
                    if "keys" in name:
                        outputs[name] = k
                    elif "queries" in name:
                        outputs[name] = q
                    elif "values" in name:
                        outputs[name] = v

            # 4) 단일 출력이면 tensor로 언팩 (기존 동작 유지)
            if len(outputs) == 1:
                return next(iter(outputs.values()))
            return outputs

        # --------- no-features path ---------
        return outputs
'''
