"""Train agent-centric C-JEPA world model from pre-extracted slots.

This variant assumes the last slot is the agent slot (from segmentation-supervised
SlotContrast training) and applies the following causal bias:
1) agent(t+1) is predicted from agent(t) conditioned on action(t)
2) non-agent(t+1) is predicted from non-agent(t), agent(t), and stopgrad(agent(t+1))
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

from src.agent_centric_predictor import AgentCausalSlotPredictor
from src.custom_codes.custom_dataset import PushTSlotDataset
from src.custom_codes.hungarian import hungarian_matching_loss_AP
from src.third_party.videosaur.videosaur import models
from src.world_models.dinowm_causal import CausalWM, Embedder

import pickle as pkl


def get_data(cfg):
    """Setup dataset with pre-extracted slot representations."""
    with open(cfg.embedding_dir, "rb") as f:
        slot_data = pkl.load(f)
    logging.info(f"Loaded slot embeddings from {cfg.embedding_dir}")

    train_dataset = PushTSlotDataset(
        slot_data=slot_data["train"],
        split="train",
        history_size=cfg.dinowm.history_size,
        num_preds=cfg.dinowm.num_preds,
        action_dir=cfg.action_dir,
        proprio_dir=cfg.proprio_dir,
        state_dir=cfg.get("state_dir", None),
        frameskip=cfg.frameskip,
        seed=cfg.seed,
    )
    val_dataset = PushTSlotDataset(
        slot_data=slot_data["val"],
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
    """Build AC-JEPA v2 with history/proprio conditioning and rollout training."""

    def _get_object_mask_indices(num_object_slots: int, device: torch.device) -> torch.Tensor | None:
        num_masked_slots = int(cfg.get("num_masked_slots", 0))
        if num_masked_slots <= 0 or num_object_slots <= 0:
            return None
        n_mask = min(num_masked_slots, num_object_slots)
        return torch.randperm(num_object_slots, device=device)[:n_mask]

    def _set_stop_gradient_schedule(module):
        warmup_epochs = int(cfg.agent_centric.get("stop_gradient_warmup_epochs", 0))
        if warmup_epochs <= 0:
            module.model.predictor.stop_gradient_agent_to_object = bool(
                cfg.agent_centric.get("stop_gradient_agent_to_object", True)
            )
            return
        current_epoch = int(getattr(module, "current_epoch", 0))
        module.model.predictor.stop_gradient_agent_to_object = current_epoch < warmup_epochs

    def _compute_object_loss(pred_obj, target_obj, use_hungarian, hungarian_cost_type):
        if use_hungarian:
            result = hungarian_matching_loss_AP(
                pred=pred_obj,
                target=target_obj.detach(),
                cost_type=hungarian_cost_type,
                reduction="mean",
            )
            return result["pixels_loss"]
        return F.mse_loss(pred_obj, target_obj.detach())

    def forward(self, batch, stage):
        _set_stop_gradient_schedule(self)

        action = torch.nan_to_num(batch["action"].float(), 0.0)  # (B, T, act_dim*frameskip)
        pixels_embed = batch["pixels_embed"].float()  # (B, T, S, D)
        proprio = None
        proprio_embed = None
        if "proprio" in batch:
            proprio = torch.nan_to_num(batch["proprio"].float(), 0.0)
            proprio_embed = self.model.proprio_encoder(proprio)

        history_size = int(cfg.dinowm.history_size)
        num_preds = int(cfg.dinowm.num_preds)
        rollout_steps = int(cfg.agent_centric.get("rollout_steps", num_preds))
        rollout_steps = min(rollout_steps, num_preds)
        if history_size < 1:
            raise ValueError("dinowm.history_size must be >= 1.")
        if action.size(1) < history_size + rollout_steps - 1:
            raise ValueError(
                f"Insufficient action horizon: got {action.size(1)}, "
                f"need at least {history_size + rollout_steps - 1}."
            )

        history_slots = pixels_embed[:, :history_size, :, :]
        target_future = pixels_embed[:, history_size : history_size + rollout_steps, :, :]
        target_obj = target_future[:, :, :-1, :]
        target_agent = target_future[:, :, -1, :]

        action_embed = self.model.action_encoder(action)
        future_action_embed = action_embed[:, history_size - 1 : history_size - 1 + rollout_steps, :]
        history_proprio = proprio_embed[:, :history_size, :] if proprio_embed is not None else None
        target_future_proprio = proprio[:, history_size : history_size + rollout_steps, :] if proprio is not None else None

        pred_future, aux = self.model.predictor(
            history_slots,
            future_action_embed,
            history_proprio=history_proprio,
        )
        batch["pred_future"] = pred_future
        batch["pred_object_slots"] = aux["pred_object_slots"]
        batch["pred_agent_slots"] = aux["pred_agent_slots"]

        pred_proprio = self.model.proprio_head(aux["pred_agent_slots"])
        batch["pred_proprio"] = pred_proprio

        # History losses with teacher-forced prefixes.
        use_hungarian = bool(cfg.get("use_hungarian_matching", True))
        hungarian_cost_type = str(cfg.get("hungarian_cost_type", "mse"))
        hist_object_losses = []
        hist_agent_losses = []
        hist_proprio_losses = []
        if history_size > 1:
            for step in range(history_size - 1):
                prefix_slots = history_slots[:, : step + 1, :, :]
                prefix_actions = action_embed[:, step : step + 1, :]
                prefix_proprio = history_proprio[:, : step + 1, :] if history_proprio is not None else None
                _, hist_aux = self.model.predictor(
                    prefix_slots,
                    prefix_actions,
                    history_proprio=prefix_proprio,
                )
                target_hist = history_slots[:, step + 1 : step + 2, :, :]
                target_hist_obj = target_hist[:, :, :-1, :]
                target_hist_agent = target_hist[:, :, -1, :]

                mask_indices = _get_object_mask_indices(target_hist_obj.size(2), target_hist_obj.device)
                pred_hist_obj = hist_aux["pred_object_slots"]
                target_hist_obj_loss = target_hist_obj
                if mask_indices is not None and mask_indices.numel() > 0:
                    pred_hist_obj = pred_hist_obj[:, :, mask_indices, :]
                    target_hist_obj_loss = target_hist_obj_loss[:, :, mask_indices, :]
                hist_object_losses.append(F.mse_loss(pred_hist_obj, target_hist_obj_loss.detach()))
                hist_agent_losses.append(F.mse_loss(hist_aux["pred_agent_slots"], target_hist_agent.detach()))

                if proprio is not None:
                    pred_hist_prop = self.model.proprio_head(hist_aux["pred_agent_slots"])
                    target_hist_prop = proprio[:, step + 1 : step + 2, :]
                    hist_proprio_losses.append(F.mse_loss(pred_hist_prop, target_hist_prop.detach()))

        loss_masked_history = (
            torch.stack(hist_object_losses).mean() if hist_object_losses else pixels_embed.new_tensor(0.0)
        )
        history_agent_loss = (
            torch.stack(hist_agent_losses).mean() if hist_agent_losses else pixels_embed.new_tensor(0.0)
        )
        history_proprio_loss = (
            torch.stack(hist_proprio_losses).mean() if hist_proprio_losses else pixels_embed.new_tensor(0.0)
        )
        batch["loss_masked_history"] = loss_masked_history
        batch["history_agent_loss"] = history_agent_loss
        batch["history_proprio_loss"] = history_proprio_loss

        object_loss = _compute_object_loss(
            aux["pred_object_slots"],
            target_obj,
            use_hungarian=use_hungarian,
            hungarian_cost_type=hungarian_cost_type,
        )
        with torch.no_grad():
            batch["direct_mse_object_loss"] = F.mse_loss(aux["pred_object_slots"], target_obj.detach())

        agent_loss = F.mse_loss(aux["pred_agent_slots"], target_agent.detach())
        if rollout_steps > 0:
            target_agent_delta = target_agent[:, :, :] - torch.cat(
                [history_slots[:, -1:, -1, :], target_agent[:, :-1, :]],
                dim=1,
            )
            agent_delta_loss = F.mse_loss(aux["pred_agent_deltas"], target_agent_delta.detach())
        else:
            agent_delta_loss = pixels_embed.new_tensor(0.0)

        if target_future_proprio is not None:
            proprio_loss = F.mse_loss(pred_proprio, target_future_proprio.detach())
        else:
            proprio_loss = pixels_embed.new_tensor(0.0)

        object_weight = float(cfg.agent_centric.get("object_loss_weight", 1.0))
        agent_weight = float(cfg.agent_centric.get("agent_loss_weight", 1.0))
        proprio_weight = float(cfg.agent_centric.get("proprio_loss_weight", 1.0))
        masked_weight = float(cfg.agent_centric.get("masked_history_loss_weight", 1.0))
        history_agent_weight = float(cfg.agent_centric.get("history_agent_loss_weight", 1.0))
        history_proprio_weight = float(cfg.agent_centric.get("history_proprio_loss_weight", 1.0))
        agent_delta_weight = float(cfg.agent_centric.get("agent_delta_loss_weight", 1.0))

        loss_future = (
            object_weight * object_loss
            + agent_weight * agent_loss
            + proprio_weight * proprio_loss
            + agent_delta_weight * agent_delta_loss
        )
        batch["object_loss"] = object_loss
        batch["agent_loss"] = agent_loss
        batch["proprio_loss"] = proprio_loss
        batch["agent_delta_loss"] = agent_delta_loss
        batch["loss_future"] = loss_future

        total_loss = (
            masked_weight * loss_masked_history
            + history_agent_weight * history_agent_loss
            + history_proprio_weight * history_proprio_loss
            + loss_future
        )
        batch["loss"] = total_loss

        pred_flat = rearrange(pred_future, "b t s d -> (b t) (s d)")
        batch["predictor_embed"] = pred_flat

        prefix = "train/" if self.training else "val/"
        losses_dict = {f"{prefix}{k}": v.detach() for k, v in batch.items() if "loss" in k}
        self.log_dict(losses_dict, on_step=True, sync_dist=True)
        return batch

    # Build placeholders for checkpoint compatibility with existing CausalWM object format.
    placeholder_model = models.build(cfg.model, cfg.dummy_optimizer, None, None)
    encoder = placeholder_model.encoder
    slot_attention = placeholder_model.processor
    initializer = placeholder_model.initializer

    slot_dim = int(cfg.videosaur.SLOT_DIM)
    effective_act_dim = int(cfg.frameskip * cfg.dinowm.action_dim)
    action_embed_dim = int(cfg.agent_centric.get("action_embed_dim", slot_dim))

    proprio_embed_dim = int(cfg.agent_centric.get("proprio_input_dim", cfg.dinowm.get("proprio_embed_dim", slot_dim)))
    predictor = AgentCausalSlotPredictor(
        slot_dim=slot_dim,
        action_dim=action_embed_dim,
        proprio_dim=proprio_embed_dim,
        history_len=int(cfg.agent_centric.get("history_len", cfg.dinowm.history_size)),
        history_layers=int(cfg.agent_centric.get("history_layers", 2)),
        num_heads=int(cfg.agent_centric.get("heads", 8)),
        mlp_dim=int(cfg.agent_centric.get("mlp_dim", 256)),
        dropout=float(cfg.agent_centric.get("dropout", 0.1)),
        stop_gradient_agent_to_object=bool(
            cfg.agent_centric.get("stop_gradient_agent_to_object", True)
        ),
        use_agent_delta=bool(cfg.agent_centric.get("use_agent_delta", True)),
        use_object_self_attn=bool(cfg.agent_centric.get("use_object_self_attn", True)),
    )
    action_encoder = Embedder(in_chans=effective_act_dim, emb_dim=action_embed_dim)
    proprio_encoder = Embedder(in_chans=int(cfg.dinowm.proprio_dim), emb_dim=proprio_embed_dim)

    world_model = CausalWM(
        encoder=spt.backbone.EvalOnly(encoder),
        slot_attention=spt.backbone.EvalOnly(slot_attention),
        initializer=spt.backbone.EvalOnly(initializer),
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
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
            "proprio_opt": add_opt(
                "model.proprio_encoder",
                float(cfg.get("proprio_encoder_lr", cfg.action_encoder_lr)),
            ),
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
    wandb_name = os.environ.get("WANDB_NAME", cfg.wandb.get("name", "cjepa_agent_centric"))
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
                "history_agent_loss",
                "history_proprio_loss",
                "proprio_loss",
                "object_loss",
                "agent_loss",
                "agent_delta_loss",
            ):
                val = self._to_float(outputs.get(key))
                if val is not None:
                    tracked[key] = val

        if "loss" not in tracked:
            for key in (
                "train/loss",
                "train/loss_future",
                "train/loss_masked_history",
                "train/history_agent_loss",
                "train/history_proprio_loss",
                "train/proprio_loss",
                "train/object_loss",
                "train/agent_loss",
                "train/agent_delta_loss",
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
        for key in (
            "loss",
            "loss_future",
            "loss_masked_history",
            "history_agent_loss",
            "history_proprio_loss",
            "proprio_loss",
            "object_loss",
            "agent_loss",
            "agent_delta_loss",
        ):
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
    config_name="config_train_causal_agent_centric_metaworld_slot",
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
