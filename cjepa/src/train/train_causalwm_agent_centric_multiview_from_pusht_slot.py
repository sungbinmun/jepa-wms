"""Train multi-view agent-centric C-JEPA world model from pre-extracted slots.

Design:
- Input slots are concatenated across views: (B, T, V*S, D)
- Each view's last slot is treated as that view's agent slot
- Agent transition: agent_v(t), action(t) -> agent_v(t+1)
- Object transition: objects_v(t), [all views' agents(t), stopgrad(agents(t+1))]
"""

from pathlib import Path
import os
from datetime import datetime

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from einops import rearrange
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.custom_codes.custom_dataset import PushTSlotMultiViewDataset
from src.custom_codes.hungarian import hungarian_matching_loss_AP
from src.multi_view_agent_centric_predictor import MultiViewAgentCausalSlotPredictor
from src.third_party.videosaur.videosaur import models
from src.world_models.dinowm_causal import CausalWM, Embedder

import pickle as pkl


def _get_view_slot_paths(cfg):
    view_names = list(cfg.multiview.get("view_names", ["gripper", "third"]))
    paths = {}
    for view_name in view_names:
        key = f"embedding_dir_{view_name}"
        if key not in cfg:
            raise KeyError(
                f"Missing config key `{key}` for view `{view_name}`. "
                "Add embedding_dir_<view_name> to config/overrides."
            )
        paths[view_name] = cfg[key]
    return view_names, paths


def get_data(cfg):
    """Setup multi-view dataset with pre-extracted slot representations."""
    view_names, view_paths = _get_view_slot_paths(cfg)
    slot_data_by_view = {}
    for view_name in view_names:
        with open(view_paths[view_name], "rb") as f:
            slot_data_by_view[view_name] = pkl.load(f)
        logging.info(f"Loaded {view_name} slot embeddings from {view_paths[view_name]}")

    train_slot_data = {view_name: slot_data_by_view[view_name]["train"] for view_name in view_names}
    val_slot_data = {view_name: slot_data_by_view[view_name]["val"] for view_name in view_names}

    train_dataset = PushTSlotMultiViewDataset(
        slot_data_views=train_slot_data,
        split="train",
        history_size=cfg.dinowm.history_size,
        num_preds=cfg.dinowm.num_preds,
        action_dir=cfg.action_dir,
        proprio_dir=cfg.proprio_dir,
        state_dir=cfg.get("state_dir", None),
        frameskip=cfg.frameskip,
        seed=cfg.seed,
    )
    val_dataset = PushTSlotMultiViewDataset(
        slot_data_views=val_slot_data,
        split="val",
        history_size=cfg.dinowm.history_size,
        num_preds=cfg.dinowm.num_preds,
        action_dir=cfg.action_dir,
        proprio_dir=cfg.proprio_dir,
        state_dir=cfg.get("state_dir", None),
        frameskip=cfg.frameskip,
        seed=cfg.seed,
    )

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    logging.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
        shuffle=True,
        generator=rnd_gen,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return spt.data.DataModule(train=train_loader, val=val_loader)


def get_world_model(cfg):
    """Build multi-view agent-centric world model."""
    view_names, _ = _get_view_slot_paths(cfg)
    num_views = int(cfg.multiview.get("num_views", len(view_names)))
    if num_views != len(view_names):
        raise ValueError(
            f"multiview.num_views={num_views} does not match len(view_names)={len(view_names)}"
        )
    slots_per_view = int(cfg.videosaur.NUM_SLOTS)

    def _sample_object_mask_indices(num_object_slots: int, device: torch.device):
        num_masked_slots = int(cfg.get("num_masked_slots", 0))
        if num_masked_slots <= 0 or num_object_slots <= 0:
            return None
        n_mask = min(num_masked_slots, num_object_slots)
        rng = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
        return torch.randperm(num_object_slots, generator=rng)[:n_mask].to(device)

    def _reshape_view_slots(x: torch.Tensor, num_steps: int) -> torch.Tensor:
        # (B, T, V*S, D) -> (B, T, V, S, D)
        bsz, _, total_slots, dim = x.shape
        expected = num_views * slots_per_view
        if total_slots != expected:
            raise ValueError(
                f"Expected {expected} slots (num_views={num_views}, slots_per_view={slots_per_view}), "
                f"got {total_slots}"
            )
        return x.reshape(bsz, num_steps, num_views, slots_per_view, dim)

    def forward(self, batch, stage):
        action = torch.nan_to_num(batch["action"].float(), 0.0)  # (B, T, action_dim*frameskip)
        pixels_embed = batch["pixels_embed"].float()  # (B, T, V*S, D)

        history_size = int(cfg.dinowm.history_size)
        num_preds = int(cfg.dinowm.num_preds)
        if history_size < 1:
            raise ValueError("dinowm.history_size must be >= 1.")
        if action.size(1) < history_size + num_preds - 1:
            raise ValueError(
                f"Insufficient action horizon: got {action.size(1)}, "
                f"need at least {history_size + num_preds - 1}."
            )

        history_slots = pixels_embed[:, :history_size, :, :]
        target_future = pixels_embed[:, history_size : history_size + num_preds, :, :]
        target_future_view = _reshape_view_slots(target_future, num_preds)
        target_obj = target_future_view[:, :, :, :-1, :]  # (B, T_pred, V, S-1, D)
        target_agent = target_future_view[:, :, :, -1, :]  # (B, T_pred, V, D)

        action_embed = self.model.action_encoder(action)
        future_action_embed = action_embed[:, history_size - 1 : history_size - 1 + num_preds, :]
        pred_future, aux = self.model.predictor(history_slots, future_action_embed)
        pred_obj = aux["pred_object_slots"]  # (B, T_pred, V, S-1, D)
        pred_agent = aux["pred_agent_slots"]  # (B, T_pred, V, D)

        batch["pred_future"] = pred_future
        batch["pred_object_slots"] = pred_obj
        batch["pred_agent_slots"] = pred_agent

        # C-JEPA-style masked history loss on masked object slots.
        loss_masked_history = pixels_embed.new_tensor(0.0)
        if history_size > 1:
            history_action_embed = action_embed[:, : history_size - 1, :]
            _, hist_aux = self.model.predictor.rollout(history_slots[:, 0, :, :], history_action_embed)
            hist_pred_obj = hist_aux["pred_object_slots"]  # (B, T_hist-1, V, S-1, D)

            hist_target = history_slots[:, 1:history_size, :, :]
            hist_target_view = _reshape_view_slots(hist_target, history_size - 1)
            hist_target_obj = hist_target_view[:, :, :, :-1, :]

            mask_indices = _sample_object_mask_indices(hist_target_obj.size(3), hist_target_obj.device)
            if mask_indices is not None and mask_indices.numel() > 0:
                loss_masked_history = F.mse_loss(
                    hist_pred_obj[:, :, :, mask_indices, :],
                    hist_target_obj[:, :, :, mask_indices, :].detach(),
                )
            else:
                loss_masked_history = F.mse_loss(hist_pred_obj, hist_target_obj.detach())
        batch["loss_masked_history"] = loss_masked_history

        use_hungarian = bool(cfg.get("use_hungarian_matching", True))
        hungarian_cost_type = str(cfg.get("hungarian_cost_type", "mse"))

        object_losses = []
        direct_object_losses = []
        for v_idx, view_name in enumerate(view_names):
            pred_obj_v = pred_obj[:, :, v_idx, :, :]
            target_obj_v = target_obj[:, :, v_idx, :, :]
            if use_hungarian:
                loss_obj_v = hungarian_matching_loss_AP(
                    pred=pred_obj_v,
                    target=target_obj_v.detach(),
                    cost_type=hungarian_cost_type,
                    reduction="mean",
                )["pixels_loss"]
            else:
                loss_obj_v = F.mse_loss(pred_obj_v, target_obj_v.detach())
            object_losses.append(loss_obj_v)
            batch[f"object_loss_{view_name}"] = loss_obj_v

            with torch.no_grad():
                direct_v = F.mse_loss(pred_obj_v, target_obj_v.detach())
            direct_object_losses.append(direct_v)
            batch[f"direct_mse_object_loss_{view_name}"] = direct_v

        object_loss = torch.stack(object_losses).mean()
        batch["object_loss"] = object_loss
        batch["direct_mse_object_loss"] = torch.stack(direct_object_losses).mean()

        agent_losses = []
        for v_idx, view_name in enumerate(view_names):
            agent_loss_v = F.mse_loss(pred_agent[:, :, v_idx, :], target_agent[:, :, v_idx, :].detach())
            agent_losses.append(agent_loss_v)
            batch[f"agent_loss_{view_name}"] = agent_loss_v
        agent_loss = torch.stack(agent_losses).mean()
        batch["agent_loss"] = agent_loss

        object_weight = float(cfg.agent_centric.get("object_loss_weight", 1.0))
        agent_weight = float(cfg.agent_centric.get("agent_loss_weight", 1.0))
        loss_future = object_weight * object_loss + agent_weight * agent_loss
        batch["loss_future"] = loss_future

        if "proprio" in batch and hasattr(self.model, "proprio_head"):
            proprio = torch.nan_to_num(batch["proprio"].float(), 0.0)
            target_proprio = proprio[:, history_size : history_size + num_preds, :]
            pred_proprio = self.model.proprio_head(pred_agent)  # (B, T_pred, V, proprio_dim)
            target_proprio = target_proprio.unsqueeze(2).expand(-1, -1, num_views, -1)
            proprio_loss = F.mse_loss(pred_proprio, target_proprio.detach())
            batch["proprio_loss"] = proprio_loss
        else:
            proprio_loss = None

        masked_weight = float(cfg.agent_centric.get("masked_history_loss_weight", 1.0))
        proprio_weight = float(cfg.agent_centric.get("proprio_loss_weight", 1.0))
        total_loss = masked_weight * loss_masked_history + loss_future
        if proprio_loss is not None:
            total_loss = total_loss + proprio_weight * proprio_loss
        batch["loss"] = total_loss

        pred_flat = rearrange(pred_future, "b t s d -> (b t) (s d)")
        batch["predictor_embed"] = pred_flat

        prefix = "train/" if self.training else "val/"
        losses_dict = {f"{prefix}{k}": v.detach() for k, v in batch.items() if "loss" in k}
        self.log_dict(losses_dict, on_step=True, sync_dist=True)
        return batch

    placeholder_model = models.build(cfg.model, cfg.dummy_optimizer, None, None)
    encoder = placeholder_model.encoder
    slot_attention = placeholder_model.processor
    initializer = placeholder_model.initializer

    slot_dim = int(cfg.videosaur.SLOT_DIM)
    effective_act_dim = int(cfg.frameskip * cfg.dinowm.action_dim)
    action_embed_dim = int(cfg.agent_centric.get("action_embed_dim", slot_dim))
    predictor = MultiViewAgentCausalSlotPredictor(
        slot_dim=slot_dim,
        action_dim=action_embed_dim,
        num_views=num_views,
        slots_per_view=slots_per_view,
        num_heads=int(cfg.agent_centric.get("heads", 8)),
        mlp_dim=int(cfg.agent_centric.get("mlp_dim", 256)),
        dropout=float(cfg.agent_centric.get("dropout", 0.1)),
        stop_gradient_agent_to_object=bool(
            cfg.agent_centric.get("stop_gradient_agent_to_object", True)
        ),
        use_view_pe=bool(cfg.multiview.get("use_view_pe", True)),
        use_slot_pe=bool(cfg.multiview.get("use_slot_pe", True)),
    )
    action_encoder = Embedder(in_chans=effective_act_dim, emb_dim=action_embed_dim)

    world_model = CausalWM(
        encoder=spt.backbone.EvalOnly(encoder),
        slot_attention=spt.backbone.EvalOnly(slot_attention),
        initializer=spt.backbone.EvalOnly(initializer),
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=None,
        history_size=cfg.dinowm.history_size,
        num_pred=cfg.dinowm.num_preds,
    )
    world_model.proprio_head = torch.nn.Linear(slot_dim, int(cfg.dinowm.proprio_dim))

    def add_opt(module_name, lr):
        return {"modules": str(module_name), "optimizer": {"type": "AdamW", "lr": lr}}

    return spt.Module(
        model=world_model,
        forward=forward,
        optim={
            "predictor_opt": add_opt("model.predictor", cfg.predictor_lr),
            "action_opt": add_opt("model.action_encoder", cfg.action_encoder_lr),
            "proprio_head_opt": add_opt(
                "model.proprio_head",
                float(cfg.get("proprio_head_lr", cfg.action_encoder_lr)),
            ),
        },
    )


def setup_pl_logger(cfg):
    if not cfg.wandb.enable:
        return None
    wandb_run_id = cfg.wandb.get("run_id", None)
    wandb_entity = os.environ.get("WANDB_ENTITY", cfg.wandb.get("entity", None))
    wandb_project = os.environ.get("WANDB_PROJECT", cfg.wandb.get("project", "ocjepa"))
    wandb_name = os.environ.get("WANDB_NAME", cfg.wandb.get("name", "mv_acjepa"))
    wandb_logger = WandbLogger(
        name=wandb_name,
        project=wandb_project,
        entity=wandb_entity,
        resume="allow" if wandb_run_id else None,
        id=wandb_run_id,
        log_model=False,
    )
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))
    return wandb_logger


class ModelObjectCallBack(Callback):
    """Save model object each epoch for downstream compatibility."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(trainer, pl_module)
        if not trainer.is_global_zero:
            return
        if (trainer.current_epoch + 1) % self.epoch_interval == 0:
            output_path = self.dirpath / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
            torch.save(pl_module, output_path)
            logging.info(f"Saved world model object to {output_path}")
        if (trainer.current_epoch + 1) == trainer.max_epochs:
            final_path = self.dirpath / f"{self.filename}_object.ckpt"
            torch.save(pl_module, final_path)
            logging.info(f"Saved final world model object to {final_path}")


class ConsoleLossCallback(Callback):
    """Print training progress/loss to terminal during an epoch."""

    def __init__(self, every_n_steps: int = 20, single_line: bool = False):
        super().__init__()
        self.every_n_steps = max(1, int(every_n_steps))
        self.single_line = bool(single_line)

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return float(value.detach().mean().cpu())
        try:
            return float(value)
        except Exception:
            return None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not trainer.is_global_zero:
            return
        global_step = int(trainer.global_step)
        if global_step == 0 or global_step % self.every_n_steps != 0:
            return

        tracked = {}
        if isinstance(outputs, dict):
            for key in (
                "loss",
                "loss_future",
                "loss_masked_history",
                "proprio_loss",
                "object_loss",
                "agent_loss",
            ):
                val = self._to_float(outputs.get(key))
                if val is not None:
                    tracked[key] = val

        if "loss" not in tracked:
            for key in (
                "train/loss",
                "train/loss_future",
                "train/loss_masked_history",
                "train/proprio_loss",
                "train/object_loss",
                "train/agent_loss",
            ):
                val = self._to_float(trainer.callback_metrics.get(key))
                if val is not None:
                    tracked[key.split("/", 1)[-1]] = val

        num_batches = trainer.num_training_batches
        batch_prog = (
            f"{batch_idx + 1}/{num_batches}"
            if isinstance(num_batches, int) and num_batches > 0
            else f"{batch_idx + 1}"
        )
        msg = f"[train] epoch={trainer.current_epoch + 1} step={global_step} batch={batch_prog}"
        for key in ("loss", "loss_future", "loss_masked_history", "proprio_loss", "object_loss", "agent_loss"):
            if key in tracked:
                msg += f" {key}={tracked[key]:.6f}"
        if self.single_line:
            print(f"\r{msg}", end="", flush=True)
        else:
            logging.info(msg)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self.single_line and trainer.is_global_zero:
            print("", flush=True)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="config_train_causal_agent_centric_metaworld_multiview_slot",
)
def run(cfg):
    cache_dir_raw = swm.data.utils.get_cache_dir() if cfg.cache_dir is None else cfg.cache_dir
    cache_dir = Path(cache_dir_raw).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cache_dir / f"{cfg.output_model_name}_weights.ckpt"
    resume_from_existing_ckpt = bool(cfg.get("resume_from_existing_ckpt", False))
    if ckpt_path.is_file() and not resume_from_existing_ckpt:
        backup = ckpt_path.with_name(
            f"{ckpt_path.stem}.backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.ckpt"
        )
        ckpt_path.rename(backup)
        logging.warning(
            f"Found existing checkpoint at {ckpt_path}. "
            f"resume_from_existing_ckpt=false, moved to backup: {backup}"
        )

    wandb_logger = setup_pl_logger(cfg)
    data = get_data(cfg)
    world_model = get_world_model(cfg)

    callbacks = [
        ModelObjectCallBack(
            dirpath=cache_dir,
            filename=cfg.output_model_name,
            epoch_interval=1,
        ),
        ConsoleLossCallback(
            every_n_steps=int(
                cfg.get("console_log_every_n_steps", cfg.trainer.get("log_every_n_steps", 20))
            ),
            single_line=bool(cfg.get("console_log_single_line", True)),
        ),
    ]

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=wandb_logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data,
        ckpt_path=str(ckpt_path),
        seed=cfg.seed,
    )
    manager()


if __name__ == "__main__":
    run()
