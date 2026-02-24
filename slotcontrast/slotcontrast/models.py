from copy import copy, deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import pathlib

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn
from torch.serialization import add_safe_globals
from torchvision.utils import make_grid
from PIL import Image

from slotcontrast import configuration, losses, modules, optimizers, utils, visualizations
from slotcontrast.modules.worldmodel import ActionCondSlotFormer, SOLDWorldModel
from slotcontrast.data.transforms import Denormalize


def _build_object_centric_model(
    model_config: configuration.ModelConfig,
    optimizer_builder: Callable,
    train_metrics: Optional[Dict[str, torchmetrics.Metric]],
    val_metrics: Optional[Dict[str, torchmetrics.Metric]],
):
    initializer = modules.build_initializer(model_config.initializer)
    encoder = modules.build_encoder(model_config.encoder, "FrameEncoder")
    grouper = modules.build_grouper(model_config.grouper)
    decoder = modules.build_decoder(model_config.decoder)

    target_encoder = None
    if model_config.target_encoder:
        target_encoder = modules.build_encoder(model_config.target_encoder, "FrameEncoder")
        assert (
            model_config.target_encoder_input is not None
        ), "Please specify `target_encoder_input`."

    dynamics_predictor = None
    if model_config.dynamics_predictor:
        dynamics_predictor = modules.build_dynamics_predictor(model_config.dynamics_predictor)

    input_type = model_config.get("input_type", "image")
    if input_type == "image":
        processor = modules.LatentProcessor(grouper, predictor=None)
    elif input_type == "video":
        encoder = modules.MapOverTime(encoder)
        decoder = modules.MapOverTime(decoder)
        if target_encoder:
            target_encoder = modules.MapOverTime(target_encoder)
        if model_config.predictor is not None:
            predictor = modules.build_module(model_config.predictor)
        else:
            predictor = None
        if model_config.latent_processor:
            processor = modules.build_video(
                model_config.latent_processor,
                "LatentProcessor",
                corrector=grouper,
                predictor=predictor,
            )
        else:
            processor = modules.LatentProcessor(grouper, predictor)
        processor = modules.ScanOverTime(processor)
    else:
        raise ValueError(f"Unknown input type {input_type}")

    target_type = model_config.get("target_type", "features")
    if target_type == "input":
        default_target_key = input_type
    elif target_type == "features":
        if model_config.target_encoder_input is not None:
            default_target_key = "target_encoder.backbone_features"
        else:
            default_target_key = "encoder.backbone_features"
    else:
        raise ValueError(f"Unknown target type {target_type}. Should be `input` or `features`.")

    loss_defaults = {
        "pred_key": "decoder.reconstruction",
        "target_key": default_target_key,
        "video_inputs": input_type == "video",
        "patch_inputs": target_type == "features",
    }
    if model_config.losses is None:
        loss_fns = {"mse": losses.build(dict(**loss_defaults, name="MSELoss"))}
    else:
        loss_fns = {
            name: losses.build({**loss_defaults, **loss_config})
            for name, loss_config in model_config.losses.items()
        }

    if model_config.mask_resizers:
        mask_resizers = {
            name: modules.build_utils(resizer_config, "Resizer")
            for name, resizer_config in model_config.mask_resizers.items()
        }
    else:
        mask_resizers = {
            "decoder": modules.build_utils(
                {
                    "name": "Resizer",
                    # When using features as targets, assume patch-shaped outputs. With other
                    # targets, assume spatial outputs.
                    "patch_inputs": target_type == "features",
                    "video_inputs": input_type == "video",
                    "resize_mode": "bilinear",
                }
            ),
            "grouping": modules.build_utils(
                {
                    "name": "Resizer",
                    "patch_inputs": True,
                    "video_inputs": input_type == "video",
                    "resize_mode": "bilinear",
                }
            ),
        }

    if model_config.masks_to_visualize:
        masks_to_visualize = model_config.masks_to_visualize
    else:
        masks_to_visualize = "decoder"

    model = ObjectCentricModel(
        optimizer_builder,
        initializer,
        encoder,
        processor,
        decoder,
        loss_fns,
        loss_weights=model_config.get("loss_weights", None),
        target_encoder=target_encoder,
        dynamics_predictor=dynamics_predictor,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        mask_resizers=mask_resizers,
        input_type=input_type,
        target_encoder_input=model_config.get("target_encoder_input", None),
        visualize=model_config.get("visualize", False),
        visualize_every_n_steps=model_config.get("visualize_every_n_steps", 1000),
        visualize_now=model_config.get("visualize_now", False),
        visualize_inputs=model_config.get("visualize_inputs", True),
        masks_to_visualize=masks_to_visualize,
    )

    if model_config.load_weights:
        model.load_weights_from_checkpoint(model_config.load_weights, model_config.modules_to_load)

    return model


def _build_world_model(
    model_config: configuration.ModelConfig,
    optimizer_config,
    train_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
    val_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
):
    """Build SOLD-style world model trainer, optionally finetuning the slot encoder/decoder."""
    world_cfg = getattr(model_config, "world_model", {}) or {}

    # Frozen teacher slots/decoder; optimizer builder is unused for it.
    frozen_builder = lambda modules: None  # noqa: E731
    slot_model = _build_object_centric_model(model_config, frozen_builder, None, None)
    slot_model.dynamics_predictor = None  # skip extra SlotFormer branch during teacher forward

    num_slots = getattr(slot_model.initializer, "n_slots", None)
    slot_dim = getattr(slot_model.initializer, "dim", None)
    if num_slots is None or slot_dim is None:
        raise ValueError("slot_model.initializer must expose `n_slots` and `dim` for world model training.")

    predictor_cfg = world_cfg.get("predictor", {}) or {}
    def _cfg_val(key: str, default):
        val = predictor_cfg.get(key, default)
        return default if val is None else val

    action_dim = world_cfg.get("action_dim", predictor_cfg.get("action_dim", None))
    if action_dim is None:
        raise ValueError("Specify `model.world_model.action_dim` for the action-conditioned predictor.")

    seed_len_cfg = world_cfg.get("seed_len", predictor_cfg.get("history_len", 2))
    seed_len = 2 if seed_len_cfg is None else seed_len_cfg
    predictor = ActionCondSlotFormer(
        num_slots=num_slots,
        slot_dim=slot_dim,
        action_dim=action_dim,
        history_len=_cfg_val("history_len", seed_len),
        d_model=_cfg_val("d_model", 128),
        num_layers=_cfg_val("num_layers", 4),
        num_heads=_cfg_val("num_heads", 8),
        ffn_dim=_cfg_val("ffn_dim", 512),
        dropout=_cfg_val("dropout", 0.1),
        norm_first=_cfg_val("norm_first", True),
    )
    optimizer_cfg = deepcopy(optimizer_config)

    finetune_slot_model = world_cfg.get("finetune_slot_model", False)
    trainable_slot_modules = world_cfg.get(
        "slot_modules_to_finetune", ["initializer", "encoder", "processor", "decoder"]
    )
    lambda_schedules = world_cfg.get("lambda_schedules", None)
    lr_groups = world_cfg.get("lr_groups")
    if lr_groups:
        base_lr = optimizer_cfg.get("lr")
        optimizer_cfg["lr"] = lr_groups.get("predictor", base_lr)
        param_groups = [
            {
                "modules": "predictor",
                "lr": lr_groups.get("predictor", base_lr),
            }
        ]
        if finetune_slot_model:
            param_groups = [
                {
                    "modules": ["initializer", "encoder", "processor"],
                    "lr": lr_groups.get("encoder", lr_groups.get("slot_model", base_lr)),
                },
                {
                    "modules": "decoder",
                    "lr": lr_groups.get("decoder", lr_groups.get("slot_model", base_lr)),
                },
                *param_groups,
            ]

        optimizer_cfg["param_groups"] = param_groups
    optimizer_builder = optimizers.OptimizerBuilder(**optimizer_cfg)

    return SOLDWorldModel(
        slot_model=slot_model,
        predictor=predictor,
        optimizer_builder=optimizer_builder,
        seed_len=seed_len,
        w_embed=world_cfg.get("w_embed", 1.0),
        w_recon=world_cfg.get("w_recon", 1.0),
        freeze_slot_model=not finetune_slot_model,
        trainable_slot_modules=trainable_slot_modules,
        lambda_schedules=lambda_schedules,
    )


def build(
    model_config: configuration.ModelConfig,
    optimizer_config,
    train_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
    val_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
):
    if getattr(model_config, "world_model", None):
        return _build_world_model(model_config, optimizer_config, train_metrics, val_metrics)

    optimizer_builder = optimizers.OptimizerBuilder(**optimizer_config)
    return _build_object_centric_model(model_config, optimizer_builder, train_metrics, val_metrics)


class ObjectCentricModel(pl.LightningModule):
    def __init__(
        self,
        optimizer_builder: Callable,
        initializer: nn.Module,
        encoder: nn.Module,
        processor: nn.Module,
        decoder: nn.Module,
        loss_fns: Dict[str, losses.Loss],
        *,
        loss_weights: Optional[Dict[str, float]] = None,
        target_encoder: Optional[nn.Module] = None,
        dynamics_predictor: Optional[nn.Module] = None,
        train_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        val_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        mask_resizers: Optional[Dict[str, modules.Resizer]] = None,
        input_type: str = "image",
        target_encoder_input: Optional[str] = None,
        visualize: bool = False,
        visualize_every_n_steps: Optional[int] = None,
        visualize_now: bool = False,
        visualize_inputs: bool = True,
        masks_to_visualize: Union[str, List[str]] = "decoder",
    ):
        super().__init__()
        self.optimizer_builder = optimizer_builder
        self.initializer = initializer
        self.encoder = encoder
        self.processor = processor
        self.decoder = decoder
        self.target_encoder = target_encoder
        self.dynamics_predictor = dynamics_predictor

        if loss_weights is not None:
            # Filter out losses that are not used
            assert (
                loss_weights.keys() == loss_fns.keys()
            ), f"Loss weight keys {loss_weights.keys()} != {loss_fns.keys()}"
            loss_fns_filtered = {k: loss for k, loss in loss_fns.items() if loss_weights[k] != 0.0}
            loss_weights_filtered = {
                k: loss for k, loss in loss_weights.items() if loss_weights[k] != 0.0
            }
            self.loss_fns = nn.ModuleDict(loss_fns_filtered)
            self.loss_weights = loss_weights_filtered
        else:
            self.loss_fns = nn.ModuleDict(loss_fns)
            self.loss_weights = {}

        self.mask_resizers = mask_resizers if mask_resizers else {}
        self.mask_resizers["segmentation"] = modules.Resizer(
            video_inputs=input_type == "video", resize_mode="nearest-exact"
        )
        self.mask_soft_to_hard = modules.SoftToHardMask()
        self.train_metrics = torch.nn.ModuleDict(train_metrics)
        self.val_metrics = torch.nn.ModuleDict(val_metrics)

        self.visualize = visualize
        if visualize:
            assert visualize_every_n_steps is not None
        self.visualize_every_n_steps = visualize_every_n_steps
        self.visualize_now = visualize_now
        self.visualize_inputs = visualize_inputs
        if isinstance(masks_to_visualize, str):
            masks_to_visualize = [masks_to_visualize]
        for key in masks_to_visualize:
            if key not in ("decoder", "grouping", "dynamics_predictor"):
                raise ValueError(f"Unknown mask type {key}. Should be `decoder` or `grouping`.")
        self.mask_keys_to_visualize = [f"{key}_masks" for key in masks_to_visualize]

        if input_type == "image":
            self.input_key = "image"
            self.expected_input_dims = 4
        elif input_type == "video":
            self.input_key = "video"
            self.expected_input_dims = 5
        else:
            raise ValueError(f"Unknown input type {input_type}. Should be `image` or `video`.")

        self.target_encoder_input_key = (
            target_encoder_input if target_encoder_input else self.input_key
        )

    def _flatten_view_batch(
        self, batch: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, int]]]:
        if self.input_key not in batch:
            return batch, None
        video = batch[self.input_key]
        if not torch.is_tensor(video):
            return batch, None
        if video.ndim != self.expected_input_dims + 1:
            return batch, None

        batch_size, num_views = video.shape[:2]
        flat_batch: Dict[str, Any] = {}
        for key, value in batch.items():
            if (
                torch.is_tensor(value)
                and value.ndim >= 2
                and value.shape[0] == batch_size
                and value.shape[1] == num_views
            ):
                flat_batch[key] = value.reshape(batch_size * num_views, *value.shape[2:])
            else:
                flat_batch[key] = value
        actions = flat_batch.get("actions")
        if torch.is_tensor(actions) and actions.ndim == 3 and actions.shape[0] == batch_size:
            flat_batch["actions"] = actions.repeat_interleave(num_views, dim=0)

        return flat_batch, {"num_views": num_views, "batch_size": batch_size}

    def configure_optimizers(self):
        modules = {
            "initializer": self.initializer,
            "encoder": self.encoder,
            "processor": self.processor,
            "decoder": self.decoder,
        }
        if self.dynamics_predictor:
            modules["dynamics_predictor"] = self.dynamics_predictor
        return self.optimizer_builder(modules)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        inputs, view_info = self._flatten_view_batch(inputs)
        encoder_input = inputs[self.input_key]  # batch [x n_frames] x n_channels x height x width
        assert encoder_input.ndim == self.expected_input_dims
        batch_size = len(encoder_input)

        encoder_output = self.encoder(encoder_input)
        features = encoder_output["features"]

        slots_initial = self.initializer(batch_size=batch_size)
        processor_output = self.processor(slots_initial, features)
        slots = processor_output["state"]
        decoder_output = self.decoder(slots)

        outputs = {
            "batch_size": batch_size,
            "encoder": encoder_output,
            "processor": processor_output,
            "decoder": decoder_output,
        }
        if view_info is not None:
            outputs["view"] = view_info
            outputs["_flat_inputs"] = inputs
            state = processor_output.get("state")
            if torch.is_tensor(state) and state.dim() == 4:
                outputs["processor"]["state_view"] = state.view(
                    view_info["batch_size"], view_info["num_views"], *state.shape[1:]
                )

        if self.dynamics_predictor:
            outputs["dynamics_predictor"] = self.dynamics_predictor(slots)
            predicted_slots = outputs["dynamics_predictor"].get("next_state")
            decoded_predicted_slots = self.decoder(predicted_slots)
            decoded_predicted_slots = {
                f"predicted_{key}": value for key, value in decoded_predicted_slots.items()
            }
            outputs["decoder"].update(decoded_predicted_slots)

        outputs["targets"] = self.get_targets(inputs, outputs)

        return outputs

    def process_masks(
        self,
        masks: torch.Tensor,
        inputs: Dict[str, Any],
        resizer: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if masks is None:
            return None, None, None

        if isinstance(resizer, modules.Resizer) and resizer.patch_inputs:
            if masks.ndim == resizer.n_expected_dims + 1:
                resizer = modules.Resizer(
                    size=resizer.size,
                    patch_inputs=False,
                    patch_outputs=resizer.patch_outputs,
                    video_inputs=resizer.video_inputs,
                    channels_last=resizer.channels_last,
                    resize_mode=resizer.resize_mode,
                )

        if resizer is None:
            masks_for_vis = masks
            masks_for_vis_hard = self.mask_soft_to_hard(masks)
            masks_for_metrics_hard = masks_for_vis_hard
        else:
            masks_for_vis = resizer(masks, inputs[self.input_key])
            masks_for_vis_hard = self.mask_soft_to_hard(masks_for_vis)
            target_masks = inputs.get("segmentations")
            if target_masks is not None and masks_for_vis.shape[-2:] != target_masks.shape[-2:]:
                size_tensor = target_masks
                if target_masks.ndim >= 3 and target_masks.shape[-1] == 1:
                    size_tensor = target_masks.squeeze(-1)
                masks_for_metrics = resizer(masks, size_tensor)
                masks_for_metrics_hard = self.mask_soft_to_hard(masks_for_metrics)
            else:
                masks_for_metrics_hard = masks_for_vis_hard

        return masks_for_vis, masks_for_vis_hard, masks_for_metrics_hard

    @torch.no_grad()
    def aux_forward(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute auxilliary outputs only needed for metrics and visualisations."""
        decoder_masks = outputs["decoder"].get("masks")
        decoder_masks, decoder_masks_hard, decoder_masks_metrics_hard = self.process_masks(
            decoder_masks, inputs, self.mask_resizers.get("decoder")
        )

        grouping_masks = outputs["processor"]["corrector"].get("masks")
        grouping_masks, grouping_masks_hard, grouping_masks_metrics_hard = self.process_masks(
            grouping_masks, inputs, self.mask_resizers.get("grouping")
        )

        aux_outputs = {}
        if decoder_masks is not None:
            aux_outputs["decoder_masks"] = decoder_masks
        if decoder_masks_hard is not None:
            aux_outputs["decoder_masks_vis_hard"] = decoder_masks_hard
        if decoder_masks_metrics_hard is not None:
            aux_outputs["decoder_masks_hard"] = decoder_masks_metrics_hard
        if grouping_masks is not None:
            aux_outputs["grouping_masks"] = grouping_masks
        if grouping_masks_hard is not None:
            aux_outputs["grouping_masks_vis_hard"] = grouping_masks_hard
        if grouping_masks_metrics_hard is not None:
            aux_outputs["grouping_masks_hard"] = grouping_masks_metrics_hard

        if self.dynamics_predictor:
            dynamics_predictor_masks = outputs["decoder"].get("predicted_masks")
            (
                dynamics_predictor_masks,
                dynamics_predictor_masks_hard,
                dynamics_predictor_masks_metrics_hard,
            ) = self.process_masks(
                dynamics_predictor_masks, inputs, self.mask_resizers.get("decoder")
            )
            if dynamics_predictor_masks is not None:
                aux_outputs["dynamics_predictor_masks"] = dynamics_predictor_masks
            if dynamics_predictor_masks_hard is not None:
                aux_outputs["dynamics_predictor_masks_vis_hard"] = dynamics_predictor_masks_hard
            if dynamics_predictor_masks_metrics_hard is not None:
                aux_outputs["dynamics_predictor_masks_hard"] = dynamics_predictor_masks_metrics_hard

        return aux_outputs

    def get_targets(
        self, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        if self.target_encoder:
            target_encoder_input = inputs[self.target_encoder_input_key]
            assert target_encoder_input.ndim == self.expected_input_dims

            with torch.no_grad():
                encoder_output = self.target_encoder(target_encoder_input)

            outputs["target_encoder"] = encoder_output

        targets = {}
        for name, loss_fn in self.loss_fns.items():
            targets[name] = loss_fn.get_target(inputs, outputs)

        return targets

    def compute_loss(self, outputs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = {}
        for name, loss_fn in self.loss_fns.items():
            prediction = loss_fn.get_prediction(outputs)
            target = outputs["targets"][name]
            losses[name] = loss_fn(prediction, target)

        losses_weighted = [loss * self.loss_weights.get(name, 1.0) for name, loss in losses.items()]
        total_loss = torch.stack(losses_weighted).sum()

        return total_loss, losses

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        outputs = self.forward(batch)
        batch_for_model = outputs.get("_flat_inputs", batch)
        should_visualize = False
        if self.visualize and self.visualize_every_n_steps is not None:
            should_visualize = (
                self.trainer.global_step > 0
                and self.trainer.global_step % self.visualize_every_n_steps == 0
            )
        force_visualize = self.visualize_now
        if self.train_metrics or should_visualize or force_visualize:
            aux_outputs = self.aux_forward(batch_for_model, outputs)

        total_loss, losses = self.compute_loss(outputs)
        if len(losses) == 1:
            loss_logs = {"train/loss": total_loss}  # Log only total loss if only one loss configured
        else:
            loss_logs = {f"train/{name}": loss for name, loss in losses.items()}
            loss_logs["train/loss"] = total_loss
        to_log = dict(loss_logs)

        if self.train_metrics and self.dynamics_predictor:
            prediction_batch = copy.deepcopy(batch_for_model)
            for k, v in prediction_batch.items():
                if isinstance(v, torch.Tensor) and v.dim() == 5:
                    prediction_batch[k] = v[:, self.dynamics_predictor.history_len :]

        if self.train_metrics:
            for key, metric in self.train_metrics.items():
                if "predicted" in key.lower():
                    values = metric(**prediction_batch, **outputs, **aux_outputs)
                else:
                    values = metric(**batch_for_model, **outputs, **aux_outputs)
                self._add_metric_to_log(to_log, f"train/{key}", values)
                metric.reset()

        # Show per-loss terms in the progress bar every step.
        self.log_dict(
            loss_logs, on_step=True, on_epoch=False, prog_bar=True, batch_size=outputs["batch_size"]
        )

        # Log additional metrics without cluttering the progress bar.
        extra_logs = {k: v for k, v in to_log.items() if k not in loss_logs}
        if extra_logs:
            self.log_dict(extra_logs, on_step=True, on_epoch=False, batch_size=outputs["batch_size"])

        if (should_visualize or force_visualize) and self.global_rank == 0:
            logged_multi = False
            if outputs.get("view") is not None:
                logged_multi = self._log_view_axis_visualizations(
                    batch, outputs, aux_outputs, mode="train"
                )
            else:
                logged_multi = self._log_multi_view_visualizations(
                    batch_for_model, outputs, aux_outputs, mode="train"
                )
            if not logged_multi:
                if self.visualize_inputs:
                    self._log_inputs(
                        batch_for_model[self.input_key],
                        self._collect_masks_for_input_visualization(aux_outputs),
                        mode="train",
                    )
                    self._log_masks(aux_outputs, self.mask_keys_to_visualize, mode="train")
                self._log_pca_grid(
                    batch_for_model[self.input_key], outputs, aux_outputs, mode="train"
                )
        if force_visualize:
            self.visualize_now = False

        del outputs  # Explicitly delete to save memory

        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        if "batch_padding_mask" in batch:
            batch = self._remove_padding(batch, batch["batch_padding_mask"])
            if batch is None:
                return

        outputs = self.forward(batch)
        batch_for_model = outputs.get("_flat_inputs", batch)
        aux_outputs = self.aux_forward(batch_for_model, outputs)

        total_loss, losses = self.compute_loss(outputs)
        if len(losses) == 1:
            to_log = {"val/loss": total_loss}  # Log only total loss if only one loss configured
        else:
            to_log = {f"val/{name}": loss for name, loss in losses.items()}
            to_log["val/loss"] = total_loss

        if self.dynamics_predictor:
            prediction_batch = deepcopy(batch_for_model)
            for k, v in prediction_batch.items():
                if isinstance(v, torch.Tensor) and v.dim() == 5:
                    prediction_batch[k] = v[:, self.dynamics_predictor.history_len :]

        if self.val_metrics:
            for key, metric in self.val_metrics.items():
                if "predicted" in key.lower():
                    metric.update(**prediction_batch, **outputs, **aux_outputs)
                else:
                    metric.update(**batch_for_model, **outputs, **aux_outputs)

        self.log_dict(
            to_log, on_step=False, on_epoch=True, batch_size=outputs["batch_size"], prog_bar=True
        )

        if (
            self.visualize
            and batch_idx == 0
            and self.global_rank == 0
            and (self.trainer.global_step > 0 or self.visualize_now)
        ):
            if outputs.get("view") is not None:
                self._log_view_axis_visualizations(batch, outputs, aux_outputs, mode="val")
            else:
                if self.visualize_inputs:
                    masks_to_vis = {
                        key: aux_outputs[f"{key}_vis_hard"] for key in self.mask_keys_to_visualize
                    }
                    if (
                        batch_for_model["segmentations"].shape[-2:]
                        != batch_for_model[self.input_key].shape[-2:]
                    ):
                        masks_to_vis["segmentations"] = self.mask_resizers["segmentation"](
                            batch_for_model["segmentations"], batch_for_model[self.input_key]
                        )
                    else:
                        masks_to_vis["segmentations"] = batch_for_model["segmentations"]
                    self._log_inputs(
                        batch_for_model[self.input_key],
                        masks_to_vis,
                        mode="val",
                    )
                    self._log_masks(aux_outputs, self.mask_keys_to_visualize, mode="val")
                self._log_pca_grid(
                    batch_for_model[self.input_key], outputs, aux_outputs, mode="val"
                )

    def on_validation_epoch_end(self):
        if self.val_metrics:
            to_log = {}
            for key, metric in self.val_metrics.items():
                self._add_metric_to_log(to_log, f"val/{key}", metric.compute())
                metric.reset()
            self.log_dict(to_log, prog_bar=True)

    @staticmethod
    def _add_metric_to_log(
        log_dict: Dict[str, Any], name: str, values: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ):
        if isinstance(values, dict):
            for k, v in values.items():
                log_dict[f"{name}/{k}"] = v
        else:
            log_dict[name] = values

    def _collect_masks_for_input_visualization(self, aux_outputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Use input-resolution masks for overlays when available."""
        masks_to_vis: Dict[str, torch.Tensor] = {}
        missing: List[str] = []
        for key in self.mask_keys_to_visualize:
            vis_key = f"{key}_vis_hard"
            metric_key = f"{key}_hard"
            if vis_key in aux_outputs:
                masks_to_vis[key] = aux_outputs[vis_key]
            elif metric_key in aux_outputs:
                masks_to_vis[key] = aux_outputs[metric_key]
            else:
                missing.append(vis_key)
        if missing:
            raise KeyError(f"Missing masks for visualization: {missing}.")
        return masks_to_vis

    def _log_inputs(
        self,
        inputs: torch.Tensor,
        masks_by_name: Dict[str, torch.Tensor],
        mode: str,
        step: Optional[int] = None,
    ):
        denorm = Denormalize(input_type=self.input_key)
        if step is None:
            step = self.trainer.global_step

        if self.input_key == "video":
            video = torch.stack([denorm(video) for video in inputs])
            self._log_video(f"{mode}/{self.input_key}", video, global_step=step)
            for mask_name, masks in masks_by_name.items():
                if "dynamics_predictor" in mask_name:
                    rollout_length = masks.shape[1]
                    trimmed_video = video[:, -rollout_length:]
                    if masks.shape[-2:] != trimmed_video.shape[-2:]:
                        bsz, n_frames, n_slots, h_m, w_m = masks.shape
                        masks = F.interpolate(
                            masks.float().reshape(bsz * n_frames, n_slots, h_m, w_m),
                            size=trimmed_video.shape[-2:],
                            mode="nearest",
                        ).reshape(bsz, n_frames, n_slots, *trimmed_video.shape[-2:])
                    video_with_masks = visualizations.mix_videos_with_masks(trimmed_video, masks)
                else:
                    if masks.shape[-2:] != video.shape[-2:]:
                        bsz, n_frames, n_slots, h_m, w_m = masks.shape
                        masks = F.interpolate(
                            masks.float().reshape(bsz * n_frames, n_slots, h_m, w_m),
                            size=video.shape[-2:],
                            mode="nearest",
                        ).reshape(bsz, n_frames, n_slots, *video.shape[-2:])
                    video_with_masks = visualizations.mix_videos_with_masks(video, masks)
                self._log_video(
                    f"{mode}/video_with_{mask_name}",
                    video_with_masks,
                    global_step=step,
                )
        elif self.input_key == "image":
            image = denorm(inputs)
            self._log_images(f"{mode}/{self.input_key}", image, global_step=step)
            for mask_name, masks in masks_by_name.items():
                if masks.shape[-2:] != image.shape[-2:]:
                    masks = F.interpolate(masks.float(), size=image.shape[-2:], mode="nearest")
                image_with_masks = visualizations.mix_images_with_masks(image, masks)
                self._log_images(
                    f"{mode}/image_with_{mask_name}",
                    image_with_masks,
                    global_step=step,
                )
        else:
            raise ValueError(f"input_type should be 'image' or 'video', but got '{self.input_key}'")

    def _get_vis_frames(self, inputs: torch.Tensor) -> Optional[torch.Tensor]:
        denorm = Denormalize(input_type=self.input_key)
        if self.input_key == "video":
            video = torch.stack([denorm(video) for video in inputs])
            return video[:, 0]
        if self.input_key == "image":
            return denorm(inputs)
        return None

    def _prep_masks_for_grid(
        self, masks: Optional[torch.Tensor], out_hw: Tuple[int, int]
    ) -> Optional[torch.Tensor]:
        if masks is None:
            return None
        if masks.dim() == 5:
            masks = masks[:, 0]
        if masks.dim() == 4 and masks.shape[-2:] != out_hw:
            masks = masks[:, 0]
        if masks.dim() == 3:
            p = masks.shape[-1]
            side = int(math.sqrt(p))
            if side * side != p:
                return None
            masks = masks.view(masks.shape[0], masks.shape[1], side, side)
        if masks.dim() == 4 and masks.shape[-2:] != out_hw:
            masks = F.interpolate(masks.float(), size=out_hw, mode="nearest")
        if masks.dtype != torch.bool:
            idx = masks.argmax(dim=1)
            masks = torch.nn.functional.one_hot(idx, masks.shape[1]).permute(0, 3, 1, 2).bool()
        return masks

    def _pca_rgb(self, tokens: torch.Tensor, out_hw: Tuple[int, int]) -> Optional[np.ndarray]:
        if tokens.dim() != 2:
            return None
        p = tokens.shape[0]
        side = int(math.sqrt(p))
        if side * side != p:
            return None
        feats = tokens.view(side, side, -1)
        proj = visualizations.pca_proj(tokens)
        return visualizations.colorize_map(feats, proj, out_hw=out_hw)

    def _overlay_masks(
        self, image: torch.Tensor, masks: torch.Tensor, alpha: float = 0.5
    ) -> np.ndarray:
        image = image.detach().cpu()
        masks = masks.detach().cpu()
        mixed = visualizations.mix_images_with_masks(
            image.unsqueeze(0), masks.unsqueeze(0), alpha=alpha
        )[0]
        return (mixed.float() / 255).permute(1, 2, 0).cpu().numpy()

    def _save_grid_image(self, name: str, grid: np.ndarray, step: int):
        if self.trainer is None or self.trainer.default_root_dir is None:
            return
        out_dir = pathlib.Path(self.trainer.default_root_dir) / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "pca_grid"
        if name:
            safe_name = name.replace("/", "_")
        Image.fromarray(grid).save(out_dir / f"{safe_name}_{step}.png")

    @staticmethod
    def _extract_meta_value(value, idx: int = 0):
        if torch.is_tensor(value):
            return value[idx].item()
        if isinstance(value, (list, tuple)):
            return value[idx]
        return value

    def _get_batch_meta(self, batch: Dict[str, Any], idx: int = 0) -> Optional[Dict[str, Any]]:
        required = ("__demo__", "__chunk__", "__file_idx__", "__view__")
        if not all(key in batch for key in required):
            return None
        return {
            "demo_name": self._extract_meta_value(batch["__demo__"], idx),
            "chunk_idx": int(self._extract_meta_value(batch["__chunk__"], idx)),
            "file_idx": int(self._extract_meta_value(batch["__file_idx__"], idx)),
            "view_key": self._extract_meta_value(batch["__view__"], idx),
        }

    def _get_dataset_for_mode(self, mode: str):
        if self.trainer is None:
            return None
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is None:
            return None
        if mode == "train":
            return getattr(datamodule, "train_set", None)
        if mode == "val":
            return getattr(datamodule, "val_set", None)
        return None

    def _prepare_view_batch(self, sample: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        batch: Dict[str, Any] = {}
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if torch.is_tensor(value):
                value = value.unsqueeze(0).to(device)
            batch[key] = value
        return batch

    def _slice_view_value(
        self, value: Any, view_idx: int, view_info: Dict[str, int]
    ) -> Any:
        if torch.is_tensor(value):
            expected = view_info["batch_size"] * view_info["num_views"]
            if value.shape[0] == expected:
                return value[view_idx:: view_info["num_views"]]
        if isinstance(value, dict):
            return {k: self._slice_view_value(v, view_idx, view_info) for k, v in value.items()}
        return value

    def _normalize_view_keys(
        self, view_value: Any, view_info: Dict[str, int]
    ) -> Optional[List[str]]:
        if isinstance(view_value, (list, tuple)):
            if view_value and all(isinstance(v, str) for v in view_value):
                return list(view_value)
            if view_value and all(isinstance(v, (list, tuple)) for v in view_value):
                if len(view_value) == view_info["num_views"]:
                    first = [v[0] for v in view_value if v]
                    if len(first) == view_info["num_views"] and all(
                        isinstance(v, str) for v in first
                    ):
                        return first
                if len(view_value) == view_info["batch_size"]:
                    first = view_value[0]
                    if isinstance(first, (list, tuple)) and all(
                        isinstance(v, str) for v in first
                    ):
                        return list(first)
        return None

    def _log_view_axis_visualizations(
        self,
        batch: Dict[str, Any],
        outputs: Dict[str, Any],
        aux_outputs: Dict[str, Any],
        mode: str,
    ) -> bool:
        view_info = outputs.get("view")
        if view_info is None or self.input_key not in batch:
            return False
        view_keys = None
        if "__view__" in batch:
            view_keys = self._normalize_view_keys(batch["__view__"], view_info)
        if view_keys is None:
            dataset = self._get_dataset_for_mode(mode)
            view_keys = getattr(dataset, "rgb_key", None) if dataset is not None else None
        if not isinstance(view_keys, (list, tuple)) or len(view_keys) < 2:
            return False
        if view_info["num_views"] != len(view_keys):
            raise ValueError(
                f"view_info num_views={view_info['num_views']} does not match "
                f"view_keys length={len(view_keys)}."
            )

        inputs = batch[self.input_key]
        if not getattr(self, "_view_vis_debug_logged", False):
            shape = tuple(inputs.shape) if torch.is_tensor(inputs) else None
            print(
                f"[view_vis] mode={mode} input_shape={shape} "
                f"view_keys={list(view_keys)} view_info={view_info} "
                f"mask_keys={list(self.mask_keys_to_visualize)}"
            )
            if isinstance(aux_outputs, dict):
                aux_keys = [k for k in aux_outputs.keys() if isinstance(k, str)]
                print(f"[view_vis] aux_keys={sorted(aux_keys)}")
            self._view_vis_debug_logged = True
        if not torch.is_tensor(inputs) or inputs.ndim != self.expected_input_dims + 1:
            raise ValueError(
                f"Expected view-axis inputs with dims {self.expected_input_dims + 1}, "
                f"but got shape {getattr(inputs, 'shape', None)}."
            )
        if inputs.shape[1] != len(view_keys):
            raise ValueError(
                f"Input view dimension {inputs.shape[1]} does not match view_keys length "
                f"{len(view_keys)}."
            )

        for view_idx, view_key in enumerate(view_keys):
            view_mode = f"{mode}/{view_key}"
            view_inputs = inputs[:, view_idx]
            view_outputs = self._slice_view_value(outputs, view_idx, view_info)
            view_aux = self._slice_view_value(aux_outputs, view_idx, view_info)

            if self.visualize_inputs:
                self._log_inputs(
                    view_inputs,
                    self._collect_masks_for_input_visualization(view_aux),
                    mode=view_mode,
                )
                self._log_masks(view_aux, self.mask_keys_to_visualize, mode=view_mode)
            self._log_pca_grid(view_inputs, view_outputs, view_aux, mode=view_mode)

        return True

    def _log_multi_view_visualizations(
        self,
        batch: Dict[str, Any],
        outputs: Dict[str, Any],
        aux_outputs: Dict[str, Any],
        mode: str,
    ) -> bool:
        dataset = self._get_dataset_for_mode(mode)
        if dataset is None:
            return False
        if not hasattr(dataset, "get_view_sample"):
            return False
        view_keys = getattr(dataset, "rgb_key", None)
        if not isinstance(view_keys, list) or len(view_keys) < 2:
            return False
        if getattr(dataset, "rgb_key_sampling", None) != "all":
            return False
        meta = self._get_batch_meta(batch, idx=0)
        if meta is None:
            return False

        device = batch[self.input_key].device
        for view_key in view_keys:
            if view_key == meta["view_key"]:
                view_batch = batch
                view_outputs = outputs
                view_aux = aux_outputs
            else:
                view_sample = dataset.get_view_sample(
                    meta["file_idx"], meta["demo_name"], meta["chunk_idx"], view_key
                )
                view_batch = self._prepare_view_batch(view_sample, device)
                with torch.no_grad():
                    view_outputs = self.forward(view_batch)
                    view_aux = self.aux_forward(view_batch, view_outputs)

            view_mode = f"{mode}/{view_key}"
            if self.visualize_inputs:
                self._log_inputs(
                    view_batch[self.input_key],
                    self._collect_masks_for_input_visualization(view_aux),
                    mode=view_mode,
                )
                self._log_masks(view_aux, self.mask_keys_to_visualize, mode=view_mode)
            self._log_pca_grid(view_batch[self.input_key], view_outputs, view_aux, mode=view_mode)

        return True

    def _log_pca_grid(
        self,
        inputs: torch.Tensor,
        outputs: Dict[str, Any],
        aux_outputs: Optional[Dict[str, Any]],
        mode: str,
        step: Optional[int] = None,
        max_examples: int = 1,
    ):
        frames = self._get_vis_frames(inputs)
        if frames is None:
            return
        if step is None:
            step = self.trainer.global_step

        enc_out = outputs.get("encoder", {})
        inp_tokens = enc_out.get("backbone_features", enc_out.get("features", None))
        if inp_tokens is None:
            return
        if inp_tokens.dim() == 4:
            inp_tokens = inp_tokens[:, 0]
        dec_out = outputs.get("decoder", {})
        recon_tokens = dec_out.get("reconstruction", None)
        if recon_tokens is None:
            return
        if recon_tokens.dim() == 4:
            recon_tokens = recon_tokens[:, 0]

        masks = None
        if aux_outputs is not None:
            masks = aux_outputs.get("decoder_masks_vis_hard", None)
            if masks is None:
                masks = aux_outputs.get("decoder_masks_hard", None)
        if masks is None:
            masks = dec_out.get("masks")
        out_hw = tuple(frames.shape[-2:])
        masks = self._prep_masks_for_grid(masks, out_hw)
        if masks is None:
            return

        n = min(max_examples, frames.shape[0])
        panels = []
        for i in range(n):
            orig = frames[i]
            orig_np = orig.permute(1, 2, 0).cpu().numpy()

            inp_rgb = self._pca_rgb(inp_tokens[i], out_hw)
            rec_rgb = self._pca_rgb(recon_tokens[i], out_hw)
            if inp_rgb is None or rec_rgb is None:
                return

            orig_mask = self._overlay_masks(orig, masks[i])
            rec_tensor = torch.from_numpy(rec_rgb).permute(2, 0, 1)
            rec_mask = self._overlay_masks(rec_tensor, masks[i])

            panels.extend([orig_np, inp_rgb, rec_rgb, orig_mask, rec_mask])

        grid = visualizations.create_grid_frame_rgb(
            panels, grid_size=(n, 5), image_size=out_hw, padding=2
        )
        grid_uint8 = (grid * 255).astype(np.uint8)
        grid_tensor = torch.from_numpy(grid_uint8).permute(2, 0, 1).unsqueeze(0).float() / 255
        self._log_images(f"{mode}/pca_grid", grid_tensor, global_step=step, n_examples=1)
        self._save_grid_image(f"{mode}_pca_grid", grid_uint8, step)

    def _log_masks(
        self,
        aux_outputs,
        mask_keys=("decoder_masks",),
        mode="val",
        types: tuple = ("frames",),
        step: Optional[int] = None,
    ):
        if step is None:
            step = self.trainer.global_step
        for mask_key in mask_keys:
            if mask_key in aux_outputs:
                masks = aux_outputs[mask_key]
                if self.input_key == "video":
                    _, f, n_obj, H, W = masks.shape
                    first_masks = masks[0].permute(1, 0, 2, 3)
                    first_masks_inverted = 1 - first_masks.reshape(n_obj, f, 1, H, W)
                    self._log_video(
                        f"{mode}/{mask_key}",
                        first_masks_inverted,
                        global_step=step,
                        n_examples=n_obj,
                        types=types,
                    )
                elif self.input_key == "image":
                    _, n_obj, H, W = masks.shape
                    first_masks_inverted = 1 - masks[0].reshape(n_obj, 1, H, W)
                    self._log_images(
                        f"{mode}/{mask_key}",
                        first_masks_inverted,
                        global_step=step,
                        n_examples=n_obj,
                    )
                else:
                    raise ValueError(
                        f"input_type should be 'image' or 'video', but got '{self.input_key}'"
                    )

    def _log_video(
        self,
        name: str,
        data: torch.Tensor,
        global_step: int,
        n_examples: int = 8,
        max_frames: int = 8,
        types: tuple = ("frames",),
    ):
        data = data[:n_examples]
        tb_logger = self._get_tensorboard_logger()
        wandb_logger = self._get_wandb_logger()

        if tb_logger is not None:
            if "video" in types:
                tb_logger.experiment.add_video(f"{name}/video", data, global_step=global_step)
            if "frames" in types:
                _, num_frames, _, _, _ = data.shape
                num_frames = min(max_frames, num_frames)
                frames = data[:, :num_frames]
                frames = frames.flatten(0, 1)
                tb_logger.experiment.add_image(
                    f"{name}/frames", make_grid(frames, nrow=num_frames), global_step=global_step
                )

        if wandb_logger is not None:
            self._log_video_wandb(
                wandb_logger,
                name,
                data,
                global_step=global_step,
                max_frames=max_frames,
                types=types,
            )

    def _log_video_wandb(
        self,
        wandb_logger,
        name: str,
        data: torch.Tensor,
        global_step: int,
        max_frames: int,
        types: tuple = ("frames",),
    ):
        try:
            import wandb
        except ImportError:
            return

        outputs = {}
        video_data = data.detach().cpu()

        if "video" in types:
            videos = []
            for video in video_data:
                video = video[:max_frames]
                video = torch.clamp(video, 0.0, 1.0)
                video_uint8 = (video * 255).byte()
                videos.append(wandb.Video(video_uint8.numpy(), fps=4, format="mp4"))
            if videos:
                outputs[f"{name}/video"] = videos

        if "frames" in types:
            num_frames = min(max_frames, video_data.shape[1])
            frames = video_data[:, :num_frames].flatten(0, 1)
            grid = make_grid(frames, nrow=num_frames)
            grid_np = grid.detach().cpu().numpy()
            grid_np = np.transpose(grid_np, (1, 2, 0))
            outputs[f"{name}/frames"] = wandb.Image(grid_np)

        if outputs:
            wandb_logger.experiment.log(outputs, step=global_step)

    def _save_video(self, name: str, data: torch.Tensor, global_step: int):
        assert (
            data.shape[0] == 1
        ), f"Only single videos saving are supported, but shape is: {data.shape}"
        data = data.cpu().numpy()[0].transpose(0, 2, 3, 1)
        data_dir = self.save_data_dir / name
        data_dir.mkdir(parents=True, exist_ok=True)
        np.save(data_dir / f"{global_step}.npy", data)

    def _log_images(
        self,
        name: str,
        data: torch.Tensor,
        global_step: int,
        n_examples: int = 8,
    ):
        n_examples = min(n_examples, data.shape[0])
        data = data[:n_examples]
        tb_logger = self._get_tensorboard_logger()
        wandb_logger = self._get_wandb_logger()

        if tb_logger is not None:
            tb_logger.experiment.add_image(
                f"{name}/images", make_grid(data, nrow=n_examples), global_step=global_step
            )

        if wandb_logger is not None:
            self._log_images_wandb(
                wandb_logger, name, data, global_step=global_step, n_examples=n_examples
            )

    def _log_images_wandb(
        self,
        wandb_logger,
        name: str,
        data: torch.Tensor,
        global_step: int,
        n_examples: int = 8,
    ):
        try:
            import wandb
        except ImportError:
            return

        grid = make_grid(data.detach().cpu(), nrow=n_examples)
        grid_np = grid.numpy()
        grid_np = np.transpose(grid_np, (1, 2, 0))
        wandb_logger.experiment.log(
            {f"{name}/images": wandb.Image(grid_np)}, step=global_step
        )

    @staticmethod
    def _remove_padding(
        batch: Dict[str, Any], padding_mask: torch.Tensor
    ) -> Optional[Dict[str, Any]]:
        if torch.all(padding_mask):
            # Batch consists only of padding
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

    def _get_logger(self, logger_cls):
        loggers = getattr(self, "loggers", None)
        if loggers:
            for logger in loggers:
                if isinstance(logger, logger_cls):
                    return logger
        if isinstance(self.logger, logger_cls):
            return self.logger
        return None

    def _get_tensorboard_logger(self):
        return self._get_logger(pl.loggers.tensorboard.TensorBoardLogger)

    def _get_wandb_logger(self):
        try:
            wandb_logger_cls = pl.loggers.wandb.WandbLogger
        except AttributeError:
            return None
        return self._get_logger(wandb_logger_cls)

    def on_load_checkpoint(self, checkpoint):
        # Reset timer during loading of the checkpoint
        # as timer is used to track time from the start
        # of the current run.
        if "callbacks" in checkpoint and "Timer" in checkpoint["callbacks"]:
            checkpoint["callbacks"]["Timer"]["time_elapsed"] = {
                "train": 0.0,
                "sanity_check": 0.0,
                "validate": 0.0,
                "test": 0.0,
                "predict": 0.0,
            }

    def load_weights_from_checkpoint(
        self, checkpoint_path: str, module_mapping: Optional[Dict[str, str]] = None
    ):
        """Load weights from a checkpoint into the specified modules."""
        # Allow PosixPath objects saved in older checkpoints and explicitly allow code pickling.
        add_safe_globals([pathlib.PosixPath])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        if module_mapping is None:
            module_mapping = {
                key.split(".")[0]: key.split(".")[0]
                for key in checkpoint
                if hasattr(self, key.split(".")[0])
            }

        for dest_module, source_module in module_mapping.items():
            try:
                module = utils.read_path(self, dest_module)
            except ValueError:
                raise ValueError(f"Module {dest_module} could not be retrieved from model") from None

            state_dict = {}
            for key, weights in checkpoint.items():
                if key.startswith(source_module):
                    if key != source_module:
                        key = key[len(source_module + ".") :]  # Remove prefix
                    state_dict[key] = weights
            if len(state_dict) == 0:
                raise ValueError(
                    f"No weights for module {source_module} found in checkpoint {checkpoint_path}."
                )

            module.load_state_dict(state_dict)
