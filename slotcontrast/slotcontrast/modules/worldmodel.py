import contextlib
from typing import Any, Dict, Optional, Sequence

import pytorch_lightning as pl
import torch
from torch import nn
from slotcontrast import schedulers

class ActionCondSlotFormer(nn.Module):
    """
    Action-conditioned SlotFormer-style predictor.
    Predict next slots using history slots + current action (a_{t-1} -> slots_t).
    """
    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        action_dim: int,
        history_len: int = 2,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        norm_first: bool = True,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.history_len = history_len
        self.d_model = d_model

        self.in_proj = nn.Linear(slot_dim, d_model)
        self.act_proj = nn.Linear(action_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            norm_first=norm_first,
            batch_first=True,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, slot_dim)

        # simple learnable time embedding for history steps
        self.t_pe = nn.Parameter(torch.zeros(1, history_len, d_model))

    def step(self, hist_slots: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        hist_slots: (B,H,N,D) with H=history_len
        action: (B,A)
        returns next_slots: (B,N,D)
        """
        B, H, N, D = hist_slots.shape
        assert H == self.history_len and N == self.num_slots and D == self.slot_dim

        x = hist_slots.reshape(B, H * N, D)              # (B, HN, D)
        x = self.in_proj(x)                              # (B, HN, d_model)

        pe = self.t_pe.unsqueeze(2).repeat(B, 1, N, 1).reshape(B, H * N, self.d_model)
        x = x + pe

        a_tok = self.act_proj(action).unsqueeze(1)       # (B,1,d_model)
        x = torch.cat([a_tok, x], dim=1)                 # (B,1+HN,d_model)

        y = self.tr(x)                                   # (B,1+HN,d_model)
        next_block = y[:, -N:]                           # (B,N,d_model)
        return self.out_proj(next_block)                 # (B,N,D)

    def forward(
        self,
        hist_slots: torch.Tensor,
        action: torch.Tensor = None,
        actions: torch.Tensor = None,
        slots: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Compatible forward wrapper.

        Accepts any of:
          - forward(hist_slots, action)
          - forward(hist_slots, actions=action)
          - forward(slots=hist_slots, actions=action)
        Returns:
          {"next_state": (B,N,D)}
        """
        # Allow calling with keyword 'slots'
        if slots is not None:
            hist_slots = slots

        # Allow keyword 'actions'
        if actions is not None:
            action = actions

        if hist_slots is None or action is None:
            raise ValueError("Need hist_slots and action (or actions=...).")

        next_slots = self.step(hist_slots, action)  # (B,N,D)
        return {"next_state": next_slots}
    
def extract_gt_from_slot_model(slot_model, inputs: dict, *, freeze: bool = True):
    """
    Returns:
      slots_gt: (B,T,N,D) from outputs["processor"]["state"]
      feats_gt: (B,T,P,Df) from outputs["encoder"]["backbone_features"] or ["features"]
    """
    context = torch.no_grad() if freeze else contextlib.nullcontext()
    with context:
        outputs = slot_model(inputs)

    slots_gt = outputs["processor"]["state"]
    enc = outputs.get("encoder", {})
    feats_gt = enc.get("backbone_features", enc.get("features", None))
    if feats_gt is None:
        raise KeyError("No encoder features found in outputs['encoder'].")

    if feats_gt.dim() == 3:
        feats_gt = feats_gt.unsqueeze(1)  # (B,1,P,Df)

    return slots_gt, feats_gt, outputs

def decode_slots_frozen(slot_model, slots_btnd: torch.Tensor, *, freeze: bool = True) -> dict:
    """
    slots_btnd: (B,T,N,D)
    returns decoder dict with (B,T,...) shapes, using real decoder behind MapOverTime.
    """
    dec = slot_model.decoder
    real_dec = dec.module if hasattr(dec, "module") else dec

    B, T, N, D = slots_btnd.shape
    flat = slots_btnd.flatten(0, 1)  # (B*T, N, D)
    context = torch.no_grad() if freeze else contextlib.nullcontext()
    with context:
        out = real_dec(flat)
    return {k: v.unflatten(0, (B, T)) for k, v in out.items()}

import torch.nn.functional as F

def mse_loss(a, b):
    return F.mse_loss(a, b)

def rollout_and_loss(
    slot_model,
    predictor: ActionCondSlotFormer,
    inputs: dict,
    actions: torch.Tensor,
    seed_len: int = 2,
    freeze_slot_model: bool = True,
):
    """
    SOLD-like:
      - Get GT slots and GT encoder features from frozen slot_model
      - Seed with first S GT slots
      - Predict next slots open-loop (no teacher forcing) using actions
      - Decode predicted slots with frozen decoder
      - Hybrid loss: joint embedding + reconstruction
    actions: (B,T,A), using a_{t-1} -> predict slot at t
    """
    slots_gt, feats_gt, slot_outputs = extract_gt_from_slot_model(
        slot_model, inputs, freeze=freeze_slot_model
    )
    B, T, N, D = slots_gt.shape
    assert actions.shape[0] == B and actions.shape[1] == T

    # --- open-loop rollout ---
    assert T >= seed_len + 1, f"T={T} too short for seed_len={seed_len}"
    hist = slots_gt[:, :seed_len].detach()  # (B,S,N,D), seed uses GT only
    slots_hat = [slots_gt[:, 0].detach()]   # keep length T; t=0 uses GT for alignment/plot
    grad_flags = [slots_hat[0].requires_grad]

    for t in range(1, T):
        if t < seed_len:
            # still in seed: use GT
            slots_hat.append(slots_gt[:, t].detach())
            grad_flags.append(False)
        else:
            a_prev = actions[:, t - 1]                # (B,A)
            pred = predictor.step(hist, a_prev)       # (B,N,D)
            slots_hat.append(pred)
            grad_flags.append(pred.requires_grad)

            # update history with predicted slots (no teacher forcing)
            hist = torch.cat([hist[:, 1:], pred.unsqueeze(1)], dim=1)

    assert all(not flag for flag in grad_flags[:seed_len]), "Seed slots should be detached GT."
    assert all(grad_flags[seed_len:]), "Predicted slots after seed must require grad (no teacher forcing)."
    slots_hat = torch.stack(slots_hat, dim=1)  # (B,T,N,D)

    # --- decode predicted slots to feature recon (frozen decoder) ---
    dec_hat = decode_slots_frozen(slot_model, slots_hat, freeze=freeze_slot_model)
    feats_hat = dec_hat["reconstruction"]  # (B,T,P,Df)

    # align feature length if needed
    T2 = min(feats_hat.shape[1], feats_gt.shape[1])
    feats_hat = feats_hat[:, :T2]
    feats_gt2 = feats_gt[:, :T2]
    slots_hat2 = slots_hat[:, :T2]
    slots_gt2 = slots_gt[:, :T2]
    assert feats_hat.shape[2:] == feats_gt2.shape[2:], (
        f"Decoder recon shape {feats_hat.shape} does not match encoder feats {feats_gt2.shape}"
    )

    # --- SOLD hybrid dynamics loss ---
    # joint embedding loss in slot space
    loss_embed = mse_loss(slots_hat2, slots_gt2)

    # reconstruction loss in feature space (SOLD uses L2; keep it L2 for now)
    loss_recon = mse_loss(feats_hat, feats_gt2)

    metrics = {
        "loss_embed": loss_embed.detach(),
        "loss_recon": loss_recon.detach(),
    }
    losses = {"loss_embed": loss_embed, "loss_recon": loss_recon}
    aux = {
        "slots_gt": slots_gt,
        "slots_hat": slots_hat,
        "feats_gt": feats_gt,
        "feats_hat": feats_hat,
        "slot_outputs": slot_outputs,
    }
    return losses, metrics, aux


class SOLDWorldModel(pl.LightningModule):
    """
    LightningModule wrapper to train an action-conditioned SlotFormer world model
    with the SOLD hybrid loss, keeping the slot encoder/decoder frozen.
    """

    def __init__(
        self,
        slot_model: nn.Module,
        predictor: ActionCondSlotFormer,
        optimizer_builder,
        seed_len: int = 2,
        w_embed: float = 1.0,
        w_recon: float = 1.0,
        freeze_slot_model: bool = True,
        trainable_slot_modules: Optional[Sequence[str]] = None,
        lambda_schedules: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        if predictor.history_len != seed_len:
            raise ValueError(
                f"Predictor history_len ({predictor.history_len}) must match seed_len ({seed_len})"
            )

        # Frozen (teacher) perception + decoder tower unless finetuning is requested
        self.slot_model = slot_model
        self.freeze_slot_model = freeze_slot_model
        valid = {"initializer", "encoder", "processor", "decoder"}
        trainable_slot_modules = trainable_slot_modules or tuple(sorted(valid))
        invalid = set(trainable_slot_modules) - valid
        if invalid:
            raise ValueError(
                f"Unknown slot modules to finetune: {invalid}. " f"Valid options: {sorted(valid)}"
            )
        self.trainable_slot_modules = set(trainable_slot_modules)

        if self.freeze_slot_model:
            self.slot_model = slot_model.eval()
            for p in self.slot_model.parameters():
                p.requires_grad_(False)
        else:
            self._set_slot_requires_grad()

        self.predictor = predictor
        self.optimizer_builder = optimizer_builder
        self.seed_len = seed_len
        self.w_embed = w_embed
        self.w_recon = w_recon

        self.lambda_schedules = {}
        if lambda_schedules:
            for key, cfg in lambda_schedules.items():
                self.lambda_schedules[key] = schedulers.build(cfg)

        slot_loss_weights = getattr(self.slot_model, "loss_weights", {}) or {}
        self.loss_weights = {
            "loss_embed": w_embed,
            "loss_recon": w_recon,
            **slot_loss_weights,
        }

        # Reuse input key if available (image vs video)
        self.input_key = getattr(self.slot_model, "input_key", "video")

    def _slot_modules(self) -> Dict[str, nn.Module]:
        return {
            "initializer": self.slot_model.initializer,
            "encoder": self.slot_model.encoder,
            "processor": self.slot_model.processor,
            "decoder": self.slot_model.decoder,
        }

    def _set_slot_requires_grad(self) -> None:
        if self.freeze_slot_model:
            return

        for name, module in self._slot_modules().items():
            requires_grad = name in self.trainable_slot_modules
            if name == "encoder" and hasattr(module, "backbone"):
                backbone = module.backbone
                backbone_frozen = getattr(backbone, "frozen", True)
                # Respect backbone.frozen flag even when encoder is selected for finetuning
                for p in backbone.parameters():
                    p.requires_grad_(False if backbone_frozen else requires_grad)
                # Remaining encoder parts (pos_embed, output_transform, etc.)
                for child_name, child in module.named_children():
                    if child is backbone:
                        continue
                    for p in child.parameters():
                        p.requires_grad_(requires_grad)
            else:
                for p in module.parameters():
                    p.requires_grad_(requires_grad)

    def _compute_slot_losses(self, slot_outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if not hasattr(self.slot_model, "loss_fns") or not self.slot_model.loss_fns:
            return {}

        losses = {}
        for name, loss_fn in self.slot_model.loss_fns.items():
            prediction = loss_fn.get_prediction(slot_outputs)
            target = slot_outputs["targets"][name]
            losses[name] = loss_fn(prediction, target)
        return losses

    def _current_lambda(self, loss_name: str) -> float:
        weight = self.loss_weights.get(loss_name, 1.0)
        schedule = self.lambda_schedules.get(loss_name)
        if schedule is not None:
            weight = weight * schedule(self.global_step)
        return weight

    def configure_optimizers(self):
        modules = {"predictor": self.predictor}
        requested_modules = set()
        if self.optimizer_builder.param_groups:
            for group in self.optimizer_builder.param_groups:
                mods = group.get("modules", [])
                if isinstance(mods, str):
                    mods = [mods]
                requested_modules.update(mods)

        if self.freeze_slot_model:
            needed = requested_modules - {"predictor"}
            if needed:
                for name, module in self._slot_modules().items():
                    if name in needed:
                        modules[name] = module
        else:
            for name, module in self._slot_modules().items():
                if (not requested_modules) or (name in requested_modules) or any(
                    p.requires_grad for p in module.parameters()
                ):
                    modules[name] = module

        return self.optimizer_builder(modules)

    def _strip_padding(self, batch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        padding_mask = batch.get("batch_padding_mask")
        if padding_mask is None:
            return batch

        if torch.all(padding_mask):
            return None

        mask = ~padding_mask
        mask_as_idxs = torch.arange(len(mask))[mask.cpu()]

        output = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                output[key] = value[mask]
            elif isinstance(value, list):
                output[key] = [value[idx] for idx in mask_as_idxs]
        return output

    def _shared_step(self, batch: Dict[str, Any], train: bool):
        batch = self._strip_padding(batch)
        if batch is None:
            return None

        actions = batch.get("actions", None)
        if actions is None:
            raise KeyError("Batch is missing 'actions' needed for action-conditioned rollout.")

        if self.freeze_slot_model:
            self.slot_model.eval()
        else:
            self.slot_model.train(mode=train)
        world_losses, _, aux = rollout_and_loss(
            self.slot_model,
            self.predictor,
            batch,
            actions,
            seed_len=self.seed_len,
            freeze_slot_model=self.freeze_slot_model,
        )
        losses = dict(world_losses)

        if not self.freeze_slot_model:
            slot_outputs = aux.get("slot_outputs", {})
            losses.update(self._compute_slot_losses(slot_outputs))

        total_loss = None
        for name, loss in losses.items():
            weight = self._current_lambda(name)
            if weight == 0:
                continue
            if total_loss is None:
                total_loss = loss * weight
            else:
                total_loss = total_loss + loss * weight

        if total_loss is None:
            raise ValueError("No losses are active to optimize.")

        prefix = "train" if train else "val"
        log_dict = {f"{prefix}/{name}": loss.detach() for name, loss in losses.items()}
        log_dict[f"{prefix}/loss"] = total_loss.detach()

        return total_loss, log_dict, aux, int(actions.shape[0])

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        shared = self._shared_step(batch, train=True)
        if shared is None:
            return None
        loss, logs, _, batch_size = shared
        self.log("train/loss", logs["train/loss"], on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        remaining = {k: v for k, v in logs.items() if k != "train/loss"}
        if remaining:
            self.log_dict(remaining, on_step=True, on_epoch=False, batch_size=batch_size)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        shared = self._shared_step(batch, train=False)
        if shared is None:
            return None
        loss, logs, _, batch_size = shared
        self.log("val/loss", logs["val/loss"], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        remaining = {k: v for k, v in logs.items() if k != "val/loss"}
        if remaining:
            self.log_dict(remaining, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss
