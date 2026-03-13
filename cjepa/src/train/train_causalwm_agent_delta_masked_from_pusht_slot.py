from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.cjepa_predictor import MaskedSlotAgentDeltaPredictor
from src.custom_codes.custom_dataset import PushTSlotDataset
from src.custom_codes.hungarian import hungarian_matching_loss_AP
from src.third_party.videosaur.videosaur import models
from src.world_models.dinowm_causal_AP_node import CausalWM_AP

import pickle as pkl


def get_data(cfg):
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
    def forward(self, batch, stage):
        del stage
        proprio_key = "proprio" if "proprio" in batch else None

        if proprio_key is not None:
            batch[proprio_key] = torch.nan_to_num(batch[proprio_key], 0.0)
        if "action" in batch:
            batch["action"] = torch.nan_to_num(batch["action"], 0.0)

        pixels_embed = batch["pixels_embed"].float()
        batch["pixels_embed"] = pixels_embed

        total_frames = cfg.dinowm.history_size + cfg.dinowm.num_preds
        if pixels_embed.size(1) != total_frames:
            raise ValueError(f"Expected {total_frames} frames, got {pixels_embed.size(1)}")

        action = batch["action"].float()
        proprio = batch[proprio_key].float() if proprio_key is not None else None
        action_embed = self.model.action_encoder(action)
        batch["action_embed"] = action_embed

        if proprio is not None:
            proprio_embed = self.model.proprio_encoder(proprio)
        else:
            proprio_embed = torch.zeros_like(action_embed)
        batch["proprio_embed"] = proprio_embed

        agent_slot = pixels_embed[:, :, -1, :]
        hist_len = cfg.dinowm.history_size
        fut_len = cfg.dinowm.num_preds

        future_agent_context = agent_slot[:, hist_len - 1 : hist_len, :].expand(-1, fut_len, -1)
        future_prop_context = proprio_embed[:, hist_len - 1 : hist_len, :].expand(-1, fut_len, -1)
        agent_context = torch.cat([agent_slot[:, :hist_len, :], future_agent_context], dim=1)
        proprio_context = torch.cat([proprio_embed[:, :hist_len, :], future_prop_context], dim=1)

        delta_input = torch.cat([agent_context, action_embed, proprio_context], dim=-1)
        delta_embed = self.model.delta_encoder(delta_input)
        batch["agent_delta_embed"] = delta_embed

        embedding = torch.cat(
            [
                pixels_embed,
                delta_embed.unsqueeze(2),
                action_embed.unsqueeze(2),
                proprio_embed.unsqueeze(2),
            ],
            dim=2,
        )
        batch["embed"] = embedding

        pred_embedding, mask_indices = self.model.predict(embedding)

        slot_num = pixels_embed.shape[2]
        object_slots = slot_num - 1
        agent_idx = object_slots
        delta_idx = slot_num
        action_idx = slot_num + 1
        proprio_idx = slot_num + 2

        pred_history = pred_embedding[:, :hist_len, :, :]
        pred_future = pred_embedding[:, hist_len : hist_len + fut_len, :, :]
        gt_history = embedding[:, :hist_len, :, :]
        gt_future = embedding[:, hist_len : hist_len + fut_len, :, :]
        gt_future_pixels = pixels_embed[:, hist_len : hist_len + fut_len, :, :]

        loss_masked_history = F.mse_loss(
            pred_history[:, :, mask_indices, :],
            gt_history[:, :, mask_indices, :].detach(),
        )

        use_hungarian = cfg.get("use_hungarian_matching", False)
        if use_hungarian:
            hungarian_result = hungarian_matching_loss_AP(
                pred=pred_future[:, :, :object_slots, :],
                target=gt_future[:, :, :object_slots, :].detach(),
                cost_type=cfg.get("hungarian_cost_type", "mse"),
                reduction="mean",
            )
            loss_future_object = hungarian_result["pixels_loss"]
        else:
            loss_future_object = F.mse_loss(
                pred_future[:, :, :object_slots, :],
                gt_future[:, :, :object_slots, :].detach(),
            )

        loss_future_agent = F.mse_loss(
            pred_future[:, :, agent_idx : agent_idx + 1, :],
            gt_future[:, :, agent_idx : agent_idx + 1, :].detach(),
        )
        loss_future_delta = F.mse_loss(
            pred_future[:, :, delta_idx : delta_idx + 1, :],
            gt_future[:, :, delta_idx : delta_idx + 1, :].detach(),
        )
        loss_future_proprio_token = F.mse_loss(
            pred_future[:, :, proprio_idx : proprio_idx + 1, :],
            gt_future[:, :, proprio_idx : proprio_idx + 1, :].detach(),
        )

        pred_proprio = self.model.proprio_head(pred_future[:, :, agent_idx, :])
        loss_future_proprio_head = F.mse_loss(
            pred_proprio,
            proprio[:, hist_len : hist_len + fut_len, :].detach(),
        )

        batch["loss_masked_history"] = loss_masked_history
        batch["loss_future_object"] = loss_future_object
        batch["loss_future_agent"] = loss_future_agent
        batch["loss_future_delta"] = loss_future_delta
        batch["loss_future_proprio_token"] = loss_future_proprio_token
        batch["loss_future_proprio_head"] = loss_future_proprio_head
        batch["loss"] = (
            cfg.agent_delta_masked.masked_history_loss_weight * loss_masked_history
            + cfg.agent_delta_masked.object_loss_weight * loss_future_object
            + cfg.agent_delta_masked.agent_loss_weight * loss_future_agent
            + cfg.agent_delta_masked.delta_loss_weight * loss_future_delta
            + cfg.agent_delta_masked.proprio_token_loss_weight * loss_future_proprio_token
            + cfg.agent_delta_masked.proprio_head_loss_weight * loss_future_proprio_head
        )

        with torch.no_grad():
            direct_mse_future = F.mse_loss(
                pred_future[:, :, :slot_num, :],
                gt_future[:, :, :slot_num, :].detach(),
            )
            batch["direct_mse_future_loss"] = direct_mse_future
            batch["future_action_token_mse"] = F.mse_loss(
                pred_future[:, :, action_idx : action_idx + 1, :],
                gt_future[:, :, action_idx : action_idx + 1, :].detach(),
            )
            batch["future_agent_pixel_mse"] = F.mse_loss(
                pred_future[:, :, agent_idx : agent_idx + 1, :],
                gt_future_pixels[:, :, agent_idx : agent_idx + 1, :].detach(),
            )

        B, T, S, D = pred_embedding.shape
        batch["predictor_embed"] = pred_embedding.reshape(B * T, S * D)

        prefix = "train/" if self.training else "val/"
        losses_dict = {f"{prefix}{k}": v.detach() for k, v in batch.items() if "loss" in k}
        self.log_dict(losses_dict, on_step=True, sync_dist=True)
        return batch

    model = models.build(cfg.model, cfg.dummy_optimizer, None, None)
    encoder = model.encoder
    slot_attention = model.processor
    initializer = model.initializer
    embedding_dim = cfg.videosaur.SLOT_DIM
    num_slots = cfg.videosaur.NUM_SLOTS
    logging.info(f"Slots: {num_slots}, Embedding dim: {embedding_dim}")

    predictor = MaskedSlotAgentDeltaPredictor(
        num_slots=num_slots + 3,
        slot_dim=embedding_dim,
        history_frames=cfg.dinowm.history_size,
        pred_frames=cfg.dinowm.num_preds,
        num_masked_slots=cfg.get("num_masked_slots", 2),
        seed=cfg.seed,
        depth=cfg.predictor.get("depth", 6),
        heads=cfg.predictor.get("heads", 8),
        dim_head=cfg.predictor.get("dim_head", 64),
        mlp_dim=cfg.predictor.get("mlp_dim", 2048),
        dropout=cfg.predictor.get("dropout", 0.1),
        num_special_slots=4,
        future_conditioning_indices=(num_slots, num_slots + 1),
    )

    effective_act_dim = cfg.frameskip * cfg.dinowm.action_dim
    action_encoder = swm.wm.dinowm.Embedder(in_chans=effective_act_dim, emb_dim=embedding_dim)
    proprio_encoder = swm.wm.dinowm.Embedder(in_chans=cfg.dinowm.proprio_dim, emb_dim=embedding_dim)
    delta_encoder = swm.wm.dinowm.Embedder(in_chans=embedding_dim * 3, emb_dim=embedding_dim)
    proprio_head = torch.nn.Linear(embedding_dim, cfg.dinowm.proprio_dim)

    world_model = CausalWM_AP(
        encoder=spt.backbone.EvalOnly(encoder),
        slot_attention=spt.backbone.EvalOnly(slot_attention),
        initializer=spt.backbone.EvalOnly(initializer),
        predictor=predictor,
        action_encoder=action_encoder,
        proprio_encoder=proprio_encoder,
        history_size=cfg.dinowm.history_size,
        num_pred=cfg.dinowm.num_preds,
    )
    world_model.delta_encoder = delta_encoder
    world_model.proprio_head = proprio_head

    def add_opt(module_name, lr):
        return {"modules": str(module_name), "optimizer": {"type": "AdamW", "lr": lr}}

    world_model = spt.Module(
        model=world_model,
        forward=forward,
        optim={
            "predictor_opt": add_opt("model.predictor", cfg.predictor_lr),
            "action_opt": add_opt("model.action_encoder", cfg.action_encoder_lr),
            "proprio_opt": add_opt("model.proprio_encoder", cfg.proprio_encoder_lr),
            "delta_opt": add_opt("model.delta_encoder", cfg.agent_delta_masked.delta_encoder_lr),
            "proprio_head_opt": add_opt("model.proprio_head", cfg.agent_delta_masked.proprio_head_lr),
        },
    )
    return world_model


def setup_pl_logger(cfg):
    if not cfg.wandb.enable:
        return None

    wandb_run_id = cfg.wandb.get("run_id", None)
    wandb_logger = WandbLogger(
        name="acjepa_masked_slot",
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        resume="allow" if wandb_run_id else None,
        id=wandb_run_id,
        log_model=False,
    )
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))
    return wandb_logger


class ModelObjectCallBack(Callback):
    def __init__(self, dirpath, filename="model_object", epoch_interval=1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
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
    config_name="config_train_causal_agent_delta_masked_metaworld_slot",
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
