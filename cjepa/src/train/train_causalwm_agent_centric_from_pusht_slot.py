"""Train agent-centric C-JEPA world model from pre-extracted slots.

This variant assumes the last slot is the agent slot (from segmentation-supervised
SlotContrast training) and applies the following causal bias:
1) agent(t+1) is predicted from agent(t) conditioned on action(t)
2) non-agent(t+1) is predicted from non-agent(t), agent(t), and stopgrad(agent(t+1))
"""

from pathlib import Path

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
from src.world_models.dinowm_causal import CausalWM

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
    """Build world model with agent-centric transition structure."""

    def forward(self, batch, stage):
        action = torch.nan_to_num(batch["action"].float(), 0.0)  # (B, T, act_dim*frameskip)
        pixels_embed = batch["pixels_embed"].float()  # (B, T, S, D)

        history_size = int(cfg.dinowm.history_size)
        num_preds = int(cfg.dinowm.num_preds)
        if history_size < 1:
            raise ValueError("dinowm.history_size must be >= 1.")

        target_future = pixels_embed[:, history_size : history_size + num_preds, :, :]
        history_slots = pixels_embed[:, :history_size, :, :]

        # Action at source step t drives transition t -> t+1.
        # For targets [history_size ... history_size+num_preds-1], source steps are
        # [history_size-1 ... history_size+num_preds-2].
        action_embed = self.model.action_encoder(action)
        future_action_embed = action_embed[:, history_size - 1 : history_size - 1 + num_preds, :]

        pred_future, aux = self.model.predictor(history_slots, future_action_embed)
        batch["pred_future"] = pred_future
        batch["pred_object_slots"] = aux["pred_object_slots"]
        batch["pred_agent_slots"] = aux["pred_agent_slots"]

        target_obj = target_future[:, :, :-1, :]
        target_agent = target_future[:, :, -1, :]

        use_hungarian = bool(cfg.get("use_hungarian_matching", True))
        hungarian_cost_type = str(cfg.get("hungarian_cost_type", "mse"))
        if use_hungarian:
            object_loss = hungarian_matching_loss_AP(
                pred=aux["pred_object_slots"],
                target=target_obj.detach(),
                cost_type=hungarian_cost_type,
                reduction="mean",
            )["pixels_loss"]
            with torch.no_grad():
                batch["direct_mse_object_loss"] = F.mse_loss(aux["pred_object_slots"], target_obj.detach())
        else:
            object_loss = F.mse_loss(aux["pred_object_slots"], target_obj.detach())

        agent_loss = F.mse_loss(aux["pred_agent_slots"], target_agent.detach())
        object_weight = float(cfg.agent_centric.get("object_loss_weight", 1.0))
        agent_weight = float(cfg.agent_centric.get("agent_loss_weight", 1.0))
        total_loss = object_weight * object_loss + agent_weight * agent_loss

        batch["object_loss"] = object_loss
        batch["agent_loss"] = agent_loss
        batch["loss"] = total_loss

        # Keep same monitoring key pattern as existing training scripts.
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

    predictor = AgentCausalSlotPredictor(
        slot_dim=slot_dim,
        action_dim=action_embed_dim,
        num_heads=int(cfg.agent_centric.get("heads", 8)),
        mlp_dim=int(cfg.agent_centric.get("mlp_dim", 256)),
        dropout=float(cfg.agent_centric.get("dropout", 0.1)),
        stop_gradient_agent_to_object=bool(
            cfg.agent_centric.get("stop_gradient_agent_to_object", True)
        ),
    )
    action_encoder = swm.wm.dinowm.Embedder(in_chans=effective_act_dim, emb_dim=action_embed_dim)

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

    def add_opt(module_name, lr):
        return {"modules": str(module_name), "optimizer": {"type": "AdamW", "lr": lr}}

    return spt.Module(
        model=world_model,
        forward=forward,
        optim={
            "predictor_opt": add_opt("model.predictor", cfg.predictor_lr),
            "action_opt": add_opt("model.action_encoder", cfg.action_encoder_lr),
        },
    )


def setup_pl_logger(cfg):
    if not cfg.wandb.enable:
        return None
    wandb_run_id = cfg.wandb.get("run_id", None)
    wandb_logger = WandbLogger(
        name=cfg.wandb.get("name", "cjepa_agent_centric"),
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
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


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="config_train_causal_agent_centric_pusht_slot",
)
def run(cfg):
    cache_dir = Path(swm.data.utils.get_cache_dir() if cfg.cache_dir is None else cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    wandb_logger = setup_pl_logger(cfg)
    data = get_data(cfg)
    world_model = get_world_model(cfg)

    callbacks = [
        ModelObjectCallBack(
            dirpath=cache_dir,
            filename=cfg.output_model_name,
            epoch_interval=1,
        )
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
        ckpt_path=str(cache_dir / f"{cfg.output_model_name}_weights.ckpt"),
        seed=cfg.seed,
    )
    manager()


if __name__ == "__main__":
    run()
