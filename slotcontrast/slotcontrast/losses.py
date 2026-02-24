from typing import Any, Dict, Optional, Sequence, Tuple, Union
import pathlib

import einops
import torch
from torch import nn
from PIL import Image

from slotcontrast import modules, utils


@utils.make_build_fn(__name__, "loss")
def build(config, name: str):
    target_transform = None
    if config.get("target_transform"):
        target_transform = modules.build_module(config.get("target_transform"))

    cls = utils.get_class_by_name(__name__, name)
    if cls is not None:
        return cls(
            target_transform=target_transform,
            **utils.config_as_kwargs(config, ("target_transform",)),
        )
    else:
        raise ValueError(f"Unknown loss `{name}`")


class Loss(nn.Module):
    """Base class for loss functions.

    Args:
        video_inputs: If true, assume inputs contain a time dimension.
        patch_inputs: If true, assume inputs have a one-dimensional patch dimension. If false,
            assume inputs have height, width dimensions.
        pred_dims: Dimensions [from, to) of prediction tensor to slice. Useful if only a
            subset of the predictions should be used in the loss, i.e. because the other dimensions
            are used in other losses.
        remove_last_n_frames: Number of frames to remove from the prediction before computing the
            loss. Only valid with video inputs. Useful if the last frame does not have a
            correspoding target.
        target_transform: Transform that can optionally be applied to the target.
    """

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        video_inputs: bool = False,
        patch_inputs: bool = True,
        keep_input_dim: bool = False,
        pred_dims: Optional[Tuple[int, int]] = None,
        remove_last_n_frames: int = 0,
        target_transform: Optional[nn.Module] = None,
        input_key: Optional[str] = None,
    ):
        super().__init__()
        self.pred_path = pred_key.split(".")
        self.target_path = target_key.split(".")
        self.video_inputs = video_inputs
        self.patch_inputs = patch_inputs
        self.keep_input_dim = keep_input_dim
        self.input_key = input_key
        self.n_expected_dims = (
            2 + (1 if patch_inputs or keep_input_dim else 2) + (1 if video_inputs else 0)
        )

        if pred_dims is not None:
            assert len(pred_dims) == 2
            self.pred_dims = slice(pred_dims[0], pred_dims[1])
        else:
            self.pred_dims = None

        self.remove_last_n_frames = remove_last_n_frames
        if remove_last_n_frames > 0 and not video_inputs:
            raise ValueError("`remove_last_n_frames > 0` only valid with `video_inputs==True`")

        self.target_transform = target_transform
        self.to_canonical_dims = self.get_dimension_canonicalizer()

    def get_dimension_canonicalizer(self) -> torch.nn.Module:
        """Return a module which reshapes tensor dimensions to (batch, n_positions, n_dims)."""
        if self.video_inputs:
            if self.patch_inputs:
                pattern = "B F P D -> B (F P) D"
            elif self.keep_input_dim:
                return torch.nn.Identity()
            else:
                pattern = "B F D H W -> B (F H W) D"
        else:
            if self.patch_inputs:
                return torch.nn.Identity()
            else:
                pattern = "B D H W -> B (H W) D"

        return einops.layers.torch.Rearrange(pattern)

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        target = utils.read_path(outputs, elements=self.target_path, error=False)
        if target is None:
            target = utils.read_path(inputs, elements=self.target_path)

        target = target.detach()

        if self.target_transform:
            with torch.no_grad():
                if self.input_key is not None:
                    target = self.target_transform(target, inputs[self.input_key])
                else:
                    target = self.target_transform(target)

        # Convert to dimension order (batch, positions, dims)
        target = self.to_canonical_dims(target)

        return target

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        prediction = utils.read_path(outputs, elements=self.pred_path)
        if prediction.ndim != self.n_expected_dims:
            raise ValueError(
                f"Prediction has {prediction.ndim} dimensions (and shape {prediction.shape}), but "
                f"expected it to have {self.n_expected_dims} dimensions."
            )

        if self.video_inputs and self.remove_last_n_frames > 0:
            prediction = prediction[:, : -self.remove_last_n_frames]

        # Convert to dimension order (batch, positions, dims)
        prediction = self.to_canonical_dims(prediction)

        if self.pred_dims:
            prediction = prediction[..., self.pred_dims]

        return prediction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement in subclasses")


class TorchLoss(Loss):
    """Wrapper around PyTorch loss functions."""

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        loss: str,
        loss_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        loss_kwargs = loss_kwargs if loss_kwargs is not None else {}
        if hasattr(torch.nn, loss):
            self.loss_fn = getattr(torch.nn, loss)(reduction="mean", **loss_kwargs)
        else:
            raise ValueError(f"Loss function torch.nn.{loss} not found")

        # Cross entropy loss wants dimension order (batch, classes, positions)
        self.positions_last = loss == "CrossEntropyLoss"

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.positions_last:
            prediction = prediction.transpose(-2, -1)
            target = target.transpose(-2, -1)

        return self.loss_fn(prediction, target)


class MSELoss(TorchLoss):
    def __init__(self, pred_key: str, target_key: str, **kwargs):
        super().__init__(pred_key, target_key, loss="MSELoss", **kwargs)


class CrossEntropyLoss(TorchLoss):
    def __init__(self, pred_key: str, target_key: str, **kwargs):
        super().__init__(pred_key, target_key, loss="CrossEntropyLoss", **kwargs)


class SlotMaskMSELoss(Loss):
    """MSE loss between decoder masks and GT segmentation grouped as BG/FG/agent."""

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        bg_ids: Tuple[int, ...] = (0, 65535),
        agent_ids: Tuple[int, ...] = (1,),
        bg_slot: Union[int, Sequence[int]] = 0,
        agent_slot: Union[int, Sequence[int]] = -1,
        fg_slots: Optional[Tuple[int, ...]] = None,
        pred_is_spatial: Optional[bool] = None,
        debug_save_dir: Optional[str] = None,
        debug_save_once: bool = True,
        debug_max_views: int = 2,
        debug_frame_idx: int = 0,
        **kwargs,
    ):
        kwargs.setdefault("keep_input_dim", True)
        super().__init__(pred_key, target_key, **kwargs)
        self.bg_ids = tuple(int(x) for x in bg_ids)
        self.agent_ids = tuple(int(x) for x in agent_ids)
        self.bg_slot = self._normalize_slot_config(bg_slot)
        self.agent_slot = self._normalize_slot_config(agent_slot)
        self.fg_slots = fg_slots  # accepted for backward compatibility; not used
        self.pred_is_spatial = pred_is_spatial
        self.pred_resizer_patch = modules.Resizer(
            patch_inputs=True,
            video_inputs=self.video_inputs,
            resize_mode="nearest-exact",
        )
        self.pred_resizer_spatial = modules.Resizer(
            patch_inputs=False,
            video_inputs=self.video_inputs,
            resize_mode="nearest-exact",
        )
        self.debug_save_dir = debug_save_dir
        self.debug_save_once = bool(debug_save_once)
        self.debug_max_views = int(debug_max_views)
        self.debug_frame_idx = int(debug_frame_idx)
        self._debug_saved = False

    @staticmethod
    def _normalize_slot_config(slot_spec: Union[int, Sequence[int]]) -> Tuple[int, ...]:
        if isinstance(slot_spec, Sequence) and not isinstance(slot_spec, (str, bytes)):
            slots = tuple(int(s) for s in slot_spec)
        else:
            slots = (int(slot_spec),)
        if len(slots) == 0:
            raise ValueError("Slot spec must contain at least one index.")
        return slots

    @staticmethod
    def _resolve_slot_indices(slot_spec: Tuple[int, ...], n_slots: int, name: str) -> Tuple[int, ...]:
        resolved = []
        for s in slot_spec:
            idx = s if s >= 0 else n_slots + s
            if idx < 0 or idx >= n_slots:
                raise ValueError(f"{name} index {s} out of range for n_slots={n_slots}")
            resolved.append(idx)
        # keep order while removing duplicates
        return tuple(dict.fromkeys(resolved))

    def _infer_pred_is_spatial(self, prediction: torch.Tensor) -> bool:
        if self.pred_is_spatial is not None:
            return bool(self.pred_is_spatial)
        if prediction.ndim == 5:
            return True
        if prediction.ndim == 4:
            return not self.video_inputs
        return False

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        seg = utils.read_path(outputs, elements=self.target_path, error=False)
        if seg is None:
            seg = utils.read_path(inputs, elements=self.target_path)
        seg = seg.detach()
        if seg.ndim >= 4 and seg.shape[-1] == 1:
            seg = seg.squeeze(-1)

        if self.video_inputs and seg.ndim == 3:
            seg = seg.unsqueeze(1)

        seg = seg.long()
        bg_any = torch.zeros_like(seg, dtype=torch.bool)
        for idx in self.bg_ids:
            bg_any |= seg == idx
        agent = torch.zeros_like(seg, dtype=torch.bool)
        for idx in self.agent_ids:
            agent |= seg == idx
        fg = ~(bg_any | agent)
        masks = torch.stack([bg_any, fg, agent], dim=2).to(torch.float32)
        return masks

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        prediction = utils.read_path(outputs, elements=self.pred_path)
        if prediction.ndim not in (3, 4, 5):
            raise ValueError(
                "SlotMaskMSELoss expects masks of shape (B,S,P), (B,T,S,P), "
                "or spatial (B,S,H,W)/(B,T,S,H,W), "
                f"but got {prediction.shape}"
            )
        if self.video_inputs and self.remove_last_n_frames > 0 and prediction.ndim == 4:
            prediction = prediction[:, : -self.remove_last_n_frames]
        if self.pred_dims is not None:
            raise ValueError("SlotMaskMSELoss does not support pred_dims slicing.")
        return prediction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = prediction
        targ = target
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if targ.ndim == 4:
            targ = targ.unsqueeze(1)

        size_tensor = targ
        if self._infer_pred_is_spatial(pred):
            pred_sp = self.pred_resizer_spatial(pred, size_tensor)
        else:
            pred_sp = self.pred_resizer_patch(pred, size_tensor)

        n_slots = pred_sp.shape[2]
        bg_slots = self._resolve_slot_indices(self.bg_slot, n_slots, "bg_slot")
        agent_slots = self._resolve_slot_indices(self.agent_slot, n_slots, "agent_slot")
        used = set((*bg_slots, *agent_slots))
        fg_slots = [i for i in range(n_slots) if i not in used]

        pred_bg = pred_sp[:, :, list(bg_slots)].sum(dim=2)
        pred_agent = pred_sp[:, :, list(agent_slots)].sum(dim=2)
        pred_fg = (
            pred_sp[:, :, fg_slots].sum(dim=2)
            if fg_slots
            else pred_bg * 0.0
        )

        gt_bg = targ[:, :, 0]
        gt_fg = targ[:, :, 1]
        gt_agent = targ[:, :, 2]

        loss_terms = [(pred_bg - gt_bg) ** 2]
        loss_terms.append((pred_fg - gt_fg) ** 2)
        loss_terms.append((pred_agent - gt_agent) ** 2)
        loss = sum(loss_terms).mean()
        return loss

    def _save_masks(
        self,
        gt_bg: torch.Tensor,
        gt_fg: torch.Tensor,
        gt_agent: torch.Tensor,
    ) -> None:
        if not self.debug_save_dir:
            return
        if self.debug_save_once and self._debug_saved:
            return
        if gt_fg.ndim != 4:
            return
        out_dir = pathlib.Path(self.debug_save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        t_idx = min(self.debug_frame_idx, gt_fg.shape[1] - 1)
        max_views = min(self.debug_max_views, gt_fg.shape[0])
        for view_idx in range(max_views):
            self._save_mask_png(
                gt_bg[view_idx, t_idx],
                out_dir / f"gt_bg_view{view_idx}.png",
            )
            self._save_mask_png(
                gt_fg[view_idx, t_idx],
                out_dir / f"gt_fg_view{view_idx}.png",
            )
            self._save_mask_png(
                gt_agent[view_idx, t_idx],
                out_dir / f"gt_agent_view{view_idx}.png",
            )
        self._debug_saved = True

    @staticmethod
    def _save_mask_png(mask: torch.Tensor, path: pathlib.Path) -> None:
        if mask.ndim != 2:
            mask = mask.squeeze()
        mask = (mask > 0.5).to(torch.uint8) * 255
        Image.fromarray(mask.cpu().numpy()).save(path)


class SlotMaskCategoryMSELoss(SlotMaskMSELoss):
    """Category-specific MSE loss for decoder masks.

    Categories:
      - bg: background
      - fg: foreground (everything except bg/agent ids)
      - agent: agent
    """

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        category: str = "bg",
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        category = str(category).lower()
        if category not in ("bg", "fg", "agent"):
            raise ValueError("`category` should be one of ['bg', 'fg', 'agent']")
        self.category = category

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = prediction
        targ = target
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if targ.ndim == 4:
            targ = targ.unsqueeze(1)

        size_tensor = targ
        if self._infer_pred_is_spatial(pred):
            pred_sp = self.pred_resizer_spatial(pred, size_tensor)
        else:
            pred_sp = self.pred_resizer_patch(pred, size_tensor)

        n_slots = pred_sp.shape[2]
        bg_slots = self._resolve_slot_indices(self.bg_slot, n_slots, "bg_slot")
        agent_slots = self._resolve_slot_indices(self.agent_slot, n_slots, "agent_slot")
        used = set((*bg_slots, *agent_slots))
        fg_slots = [i for i in range(n_slots) if i not in used]

        pred_bg = pred_sp[:, :, list(bg_slots)].sum(dim=2)
        pred_agent = pred_sp[:, :, list(agent_slots)].sum(dim=2)
        pred_fg = (
            pred_sp[:, :, fg_slots].sum(dim=2) if fg_slots else pred_bg * 0.0
        )

        gt_bg = targ[:, :, 0]
        gt_fg = targ[:, :, 1]
        gt_agent = targ[:, :, 2]

        if self.category == "bg":
            return ((pred_bg - gt_bg) ** 2).mean()
        if self.category == "fg":
            return ((pred_fg - gt_fg) ** 2).mean()
        return ((pred_agent - gt_agent) ** 2).mean()


class SlotInstanceMSELoss(Loss):
    """MSE loss between decoder masks and per-instance GT segmentation masks.

    Uses fixed slots for background/agent and assigns remaining instances to fg_slots in order.
    Unmatched fg slots are ignored (no loss contribution).
    """

    def __init__(
        self,
        pred_key: str,
        target_key: str,
        bg_ids: Tuple[int, ...] = (0, 65535),
        agent_ids: Tuple[int, ...] = (1,),
        bg_slot: Union[int, Sequence[int]] = 0,
        agent_slot: Union[int, Sequence[int]] = -1,
        fg_slots: Optional[Tuple[int, ...]] = None,
        pred_is_spatial: Optional[bool] = None,
        **kwargs,
    ):
        kwargs.setdefault("keep_input_dim", True)
        super().__init__(pred_key, target_key, **kwargs)
        self.bg_ids = tuple(int(x) for x in bg_ids)
        self.agent_ids = tuple(int(x) for x in agent_ids)
        self.bg_slot = SlotMaskMSELoss._normalize_slot_config(bg_slot)
        self.agent_slot = SlotMaskMSELoss._normalize_slot_config(agent_slot)
        self.fg_slots = tuple(int(x) for x in fg_slots) if fg_slots is not None else None
        self.pred_is_spatial = pred_is_spatial
        self.pred_resizer_patch = modules.Resizer(
            patch_inputs=True,
            video_inputs=self.video_inputs,
            resize_mode="nearest-exact",
        )
        self.pred_resizer_spatial = modules.Resizer(
            patch_inputs=False,
            video_inputs=self.video_inputs,
            resize_mode="nearest-exact",
        )

    def _infer_pred_is_spatial(self, prediction: torch.Tensor) -> bool:
        if self.pred_is_spatial is not None:
            return bool(self.pred_is_spatial)
        if prediction.ndim == 5:
            return True
        if prediction.ndim == 4:
            return not self.video_inputs
        return False

    def get_target(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> torch.Tensor:
        seg = utils.read_path(outputs, elements=self.target_path, error=False)
        if seg is None:
            seg = utils.read_path(inputs, elements=self.target_path)
        seg = seg.detach()
        if seg.ndim >= 4 and seg.shape[-1] == 1:
            seg = seg.squeeze(-1)
        if self.video_inputs and seg.ndim == 3:
            seg = seg.unsqueeze(1)
        return seg.long()

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        prediction = utils.read_path(outputs, elements=self.pred_path)
        if prediction.ndim not in (3, 4, 5):
            raise ValueError(
                "SlotInstanceMSELoss expects masks of shape (B,S,P), (B,T,S,P), "
                "or spatial (B,S,H,W)/(B,T,S,H,W), "
                f"but got {prediction.shape}"
            )
        if self.video_inputs and self.remove_last_n_frames > 0 and prediction.ndim == 4:
            prediction = prediction[:, : -self.remove_last_n_frames]
        if self.pred_dims is not None:
            raise ValueError("SlotInstanceMSELoss does not support pred_dims slicing.")
        return prediction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = prediction
        targ = target
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if targ.ndim == 3:
            targ = targ.unsqueeze(1)

        size_tensor = targ
        if self._infer_pred_is_spatial(pred):
            pred_sp = self.pred_resizer_spatial(pred, size_tensor)
        else:
            pred_sp = self.pred_resizer_patch(pred, size_tensor)
        # pred_sp[0, 0].unsqueeze(1)
        if targ.shape[1] != pred_sp.shape[1]:
            t = min(targ.shape[1], pred_sp.shape[1])
            targ = targ[:, :t]
            pred_sp = pred_sp[:, :t]
        B, T, S, H, W = pred_sp.shape

        bg_slots = SlotMaskMSELoss._resolve_slot_indices(self.bg_slot, S, "bg_slot")
        agent_slots = SlotMaskMSELoss._resolve_slot_indices(self.agent_slot, S, "agent_slot")
        used_slots = set((*bg_slots, *agent_slots))
        if self.fg_slots is None:
            fg_slots = [i for i in range(S) if i not in used_slots]
        else:
            fg_slots = [i for i in self.fg_slots if 0 <= i < S and i not in used_slots]

        target_masks = torch.zeros_like(pred_sp)
        slot_weights = torch.zeros((B, S), device=pred_sp.device, dtype=pred_sp.dtype)

        for b in range(B):
            seg_b = targ[b]  # (T,H,W)
            bg = torch.zeros_like(seg_b, dtype=torch.bool)
            for idx in self.bg_ids:
                bg |= seg_b == idx
            agent = torch.zeros_like(seg_b, dtype=torch.bool)
            for idx in self.agent_ids:
                agent |= seg_b == idx

            for s in bg_slots:
                target_masks[b, :, s] = bg.float()
                slot_weights[b, s] = 1.0
            for s in agent_slots:
                target_masks[b, :, s] = agent.float()
                slot_weights[b, s] = 1.0

            inst_ids = torch.unique(seg_b)
            inst_ids = [
                int(i)
                for i in inst_ids.tolist()
                if i not in self.bg_ids and i not in self.agent_ids
            ]
            inst_ids.sort()

            for slot_idx, inst_id in zip(fg_slots, inst_ids):
                target_masks[b, :, slot_idx] = (seg_b == inst_id).float()
                slot_weights[b, slot_idx] = 1.0

        weight = slot_weights[:, None, :, None, None]
        denom = weight.sum() * T * H * W
        if denom <= 0:
            return pred_sp.new_tensor(0.0)

        #target_masks[0, 0].unsqueeze(1)
        #pred_sp[0, 0].unsqueeze(1)
        loss = ((pred_sp - target_masks) ** 2 * weight).sum() / (denom + 1e-6)
        return loss


class Slot_Slot_Contrastive_Loss(Loss):
    def __init__(
        self,
        pred_key: str,
        target_key: str,
        temperature: float = 0.1,
        batch_contrast: bool = True,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.batch_contrast = batch_contrast

    def forward(self, slots, _):
        slots = nn.functional.normalize(slots, p=2.0, dim=-1)
        if self.batch_contrast:
            slots = slots.split(1)  # [1xTxKxD]
            slots = torch.cat(slots, dim=-2)  # 1xTxK*BxD
        s1 = slots[:, :-1, :, :]
        s2 = slots[:, 1:, :, :]
        ss = torch.matmul(s1, s2.transpose(-2, -1)) / self.temperature
        B, T, S, D = ss.shape
        ss = ss.reshape(B * T, S, S)
        target = torch.eye(S).expand(B * T, S, S).to(ss.device)
        loss = self.criterion(ss, target)
        return loss


class View_Slot_Contrastive_Loss(Loss):
    def __init__(
        self,
        pred_key: str,
        target_key: str,
        temperature: float = 0.1,
        batch_contrast: bool = True,
        view_indices: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        super().__init__(pred_key, target_key, **kwargs)
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.batch_contrast = batch_contrast
        self.view_indices = view_indices

    def get_prediction(self, outputs: Dict[str, Any]) -> torch.Tensor:
        prediction = utils.read_path(outputs, elements=self.pred_path)
        if prediction is None:
            raise ValueError("View_Slot_Contrastive_Loss requires `pred_key` for view slots.")
        if prediction.ndim not in (4, 5):
            raise ValueError(
                "View_Slot_Contrastive_Loss expects slots of shape (B,V,S,D) or (B,V,T,S,D), "
                f"but got {prediction.shape}"
            )
        return prediction

    def forward(self, slots: torch.Tensor, _) -> torch.Tensor:
        if slots.ndim == 4:
            slots = slots.unsqueeze(2)
        slots = nn.functional.normalize(slots, p=2.0, dim=-1)
        _, V, _, S, D = slots.shape
        if V < 2:
            raise ValueError("View_Slot_Contrastive_Loss needs at least two views.")
        if self.view_indices is None:
            view_a, view_b = 0, 1
        else:
            view_a, view_b = self.view_indices
            if max(view_a, view_b) >= V:
                raise ValueError(
                    f"View indices {self.view_indices} out of range for V={V}."
                )

        s1 = slots[:, view_a]
        s2 = slots[:, view_b]
        if self.batch_contrast:
            s1 = s1.reshape(1, s1.shape[1], s1.shape[0] * S, D)
            s2 = s2.reshape(1, s2.shape[1], s2.shape[0] * S, D)

        ss = torch.matmul(s1, s2.transpose(-2, -1)) / self.temperature
        B, T, K, _ = ss.shape
        ss = ss.reshape(B * T, K, K)
        target = torch.eye(K, device=ss.device).expand(B * T, K, K)
        loss = self.criterion(ss, target)
        return loss


class DynamicsLoss(Loss):
    def __init__(self, pred_key: str, target_key: str, **kwargs):
        super().__init__(pred_key, target_key, **kwargs)
        self.criterion = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        rollout_length = prediction.shape[1]
        target = target[:, -rollout_length:]
        loss = self.criterion(prediction, target)
        return loss
