import argparse
import inspect
import pathlib
import re
import torch
import torch.nn.functional as F
from torchvision.io import read_video, read_image
from omegaconf import OmegaConf
from slotcontrast import configuration, models
from slotcontrast.data.transforms import CropResize, build_inference_transform
from slotcontrast.data.libero_h5_datamodule import LiberoH5Dataset
import os
import imageio
import numpy as np
from torchvision import transforms as tvt
from slotcontrast.visualizations import (
    create_grid_frame_rgb,
    colorize_map,
    mix_inputs_with_masks,
    draw_segmentation_masks_on_image,
    color_map,
    pca_proj,
)
import matplotlib.pyplot as plt


def _torch_load_trusted(checkpoint_path: str, device: torch.device):
    kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(checkpoint_path, **kwargs)


def load_model_from_checkpoint(checkpoint_path: str, config_path: str, device: torch.device):
    config = configuration.load_config(config_path)
    model = models.build(config.model, config.optimizer)
    checkpoint = _torch_load_trusted(checkpoint_path, device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, config


def _resolve_inference_paths(config, config_path: str) -> None:
    config_file = pathlib.Path(config_path)
    run_dir = config_file.parent
    model_config = config.get("model_config")
    if not model_config:
        train_config = run_dir / "train_config.yaml"
        if not train_config.exists():
            raise FileNotFoundError(
                f"Missing model_config and no train_config.yaml found in {run_dir}"
            )
        config.model_config = str(train_config)

    checkpoint = config.get("checkpoint")
    if checkpoint is None:
        raise ValueError("Missing `checkpoint` in inference config.")
    if isinstance(checkpoint, (int, float)) or (
        isinstance(checkpoint, str) and checkpoint.isdigit()
    ):
        step = int(checkpoint)
        ckpt_path = run_dir / "checkpoints" / f"step={step}.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        config.checkpoint = str(ckpt_path)
        config._checkpoint_step = step
    elif isinstance(checkpoint, str) and not os.path.isabs(checkpoint):
        ckpt_path = (run_dir / checkpoint).resolve()
        if ckpt_path.exists():
            config.checkpoint = str(ckpt_path)

    output_config = config.get("output")
    if output_config is None:
        output_config = OmegaConf.create()
        config.output = output_config
    input_config = config.get("input") or {}

    def _safe_token(value, default: str) -> str:
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            value = "+".join(str(v) for v in value if v)
        value = str(value)
        value = value.replace("/", "_")
        value = value.replace("[", "").replace("]", "")
        value = value.replace("'", "").replace('"', "")
        value = value.replace(", ", "+").replace(",", "+").replace(" ", "")
        return value

    def _checkpoint_token() -> str:
        step = getattr(config, "_checkpoint_step", None)
        if step is not None:
            return str(step)
        ckpt = config.get("checkpoint")
        if isinstance(ckpt, (int, float)):
            return str(int(ckpt))
        if isinstance(ckpt, str):
            match = re.search(r"step=(\\d+)", ckpt)
            if match:
                return match.group(1)
            return pathlib.Path(ckpt).stem
        return "checkpoint"

    prefix = _safe_token(input_config.get("h5_prefixes"), "unknown")
    demo = _safe_token(input_config.get("demo"), "demo")
    rgb = _safe_token(input_config.get("rgb_key") or input_config.get("view_key"), "rgb")
    ckpt_token = _checkpoint_token()
    output_config.slot_eval_path = str(
        run_dir / "inference" / f"{prefix}_{demo}_{rgb}_{ckpt_token}.mp4"
    )


def prepare_video(video_path: str, transfom_config=None):
    # Load video
    video, _, _ = read_video(video_path)
    video = video.float() / 255.0
    # change size of the video to 224x224
    video_vis = video.permute(0, 3, 1, 2)
    video_vis = tvt.Resize((transfom_config.input_size, transfom_config.input_size))(video_vis)
    video_vis = video_vis.permute(1, 0, 2, 3)

    if transfom_config:
        tfs = build_inference_transform(transfom_config)
        video = video.permute(3, 0, 1, 2)
        video = tfs(video).permute(1, 0, 2, 3)
    # Add batch dimension
    inputs = {"video": video.unsqueeze(0), "video_visualization": video_vis.unsqueeze(0)}
    return inputs


def prepare_image(image_path: str, transfom_config=None):
    image = read_image(image_path)
    image = image.float() / 255.0
    resize = CropResize(
        dataset_type="image",
        crop_type="short_side_resize_central",
        size=transfom_config.input_size,
        resize_mode="bilinear",
    )
    image_vis = resize(image)

    if transfom_config:
        tfs = build_inference_transform(transfom_config)
        image = tfs(image)
    # Add batch dimension
    inputs = {"image": image.unsqueeze(0), "image_visualization": image_vis.unsqueeze(0)}
    return inputs


def _select_libero_index(dataset: LiberoH5Dataset, input_config) -> int:
    sample_index = input_config.get("sample_index")
    if sample_index is not None:
        return int(sample_index)

    demo = input_config.get("demo")
    file_index = input_config.get("file_index")
    chunk_index = input_config.get("chunk_index")

    if demo is None and file_index is None and chunk_index is None:
        return 0

    requested_view = input_config.get("view_key")
    if requested_view is None:
        rgb_key = input_config.get("rgb_key")
        if isinstance(rgb_key, str):
            requested_view = rgb_key

    for idx, entry in enumerate(dataset.index):
        if len(entry) == 3:
            file_idx, demo_name, ch_idx = entry
            view_key = None
        else:
            file_idx, demo_name, ch_idx, view_key = entry
        if demo is not None and demo_name != demo:
            continue
        if file_index is not None and file_idx != file_index:
            continue
        if chunk_index is not None and ch_idx != chunk_index:
            continue
        if requested_view is not None and view_key is not None and view_key != requested_view:
            continue
        return idx

    raise ValueError(
        "No matching Libero sample found. "
        "Check demo/file_index/chunk_index or use sample_index."
    )


def prepare_libero_h5(input_config, include_segmentations=None):
    if input_config.get("h5_path") is None:
        raise ValueError("Libero input requires `input.h5_path` in the config.")
    if input_config.get("transforms") is None:
        raise ValueError("Libero input requires `input.transforms` to set input_size.")

    if include_segmentations is None:
        include_segmentations = bool(input_config.get("include_segmentations", False))

    dataset = LiberoH5Dataset(
        h5_path=input_config.h5_path,
        transforms_dict=None,
        rgb_key=input_config.get("rgb_key"),
        rgb_key_sampling=input_config.get("rgb_key_sampling", "first"),
        segmentation_key=input_config.get("segmentation_key", "segmentation"),
        include_segmentations=include_segmentations,
        include_actions=input_config.get("include_actions", False),
        h5_prefixes=input_config.get("h5_prefixes"),
        recursive_scan=input_config.get("recursive_scan", False),
        h5_extensions=input_config.get("h5_extensions"),
        use_chunks=input_config.get("use_chunks", False),
        chunk_size=input_config.get("chunk_size", 4),
        sample_one_chunk_per_video=input_config.get("sample_one_chunk_per_video", False),
        temporal_stride=input_config.get("temporal_stride", 1),
    )

    sample_idx = _select_libero_index(dataset, input_config)
    sample = dataset[sample_idx]
    video = sample["video"]
    if isinstance(video, np.ndarray):
        video = torch.from_numpy(video)
    if video.dtype == torch.uint8:
        video = video.float() / 255.0

    frame_start = int(input_config.get("frame_start", 0))
    frame_end = input_config.get("frame_end")
    if frame_end is not None:
        frame_end = int(frame_end)
    if frame_start or frame_end is not None:
        video = video[frame_start:frame_end]

    video_vis = video.permute(0, 3, 1, 2)
    input_size = input_config.transforms.input_size
    video_vis = tvt.Resize((input_size, input_size))(video_vis)
    video_vis = video_vis.permute(1, 0, 2, 3)

    tfs = build_inference_transform(input_config.transforms)
    video = video.permute(3, 0, 1, 2)
    video = tfs(video).permute(1, 0, 2, 3)

    inputs = {"video": video.unsqueeze(0), "video_visualization": video_vis.unsqueeze(0)}
    if include_segmentations and "segmentations" in sample:
        segmentations = sample["segmentations"]
        if isinstance(segmentations, np.ndarray):
            segmentations = torch.from_numpy(segmentations)
        if frame_start or frame_end is not None:
            segmentations = segmentations[frame_start:frame_end]
        inputs["segmentations"] = segmentations.unsqueeze(0)
    return inputs


def _ensure_video_vis_layout(video_vis: torch.Tensor) -> torch.Tensor:
    if video_vis.ndim != 5:
        raise ValueError(f"Expected video_visualization with 5 dims, got {video_vis.shape}")
    if video_vis.shape[1] in (1, 3) and video_vis.shape[2] > 3:
        return video_vis.permute(0, 2, 1, 3, 4)
    return video_vis


def _features_to_map(frame_features: torch.Tensor) -> torch.Tensor:
    if frame_features.ndim == 2:
        num_patches, dim = frame_features.shape
        grid = int(np.sqrt(num_patches))
        if grid * grid != num_patches:
            raise ValueError(f"Expected square patch grid, got {num_patches} patches")
        return frame_features.reshape(grid, grid, dim)
    if frame_features.ndim == 3:
        if (
            frame_features.shape[0] > frame_features.shape[1]
            and frame_features.shape[0] > frame_features.shape[2]
        ):
            return frame_features.permute(1, 2, 0)
        return frame_features
    raise ValueError(f"Unsupported feature shape {frame_features.shape}")


def _ensure_time_dim(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 3:
        return features.unsqueeze(1)
    return features


def _to_patch_features(reconstruction: torch.Tensor, masks: torch.Tensor):
    reconstruction = _ensure_time_dim(reconstruction)
    masks = _ensure_time_dim(masks)

    if reconstruction.ndim == 5:
        if reconstruction.shape[-1] <= reconstruction.shape[2]:
            recon_spatial = reconstruction
        else:
            recon_spatial = reconstruction.permute(0, 1, 3, 4, 2)
        reconstruction = recon_spatial.reshape(
            recon_spatial.shape[0], recon_spatial.shape[1], -1, recon_spatial.shape[-1]
        )
    if reconstruction.ndim != 4:
        raise ValueError(f"Expected reconstruction with 4 dims, got {reconstruction.shape}")

    if masks.ndim == 5:
        masks = masks.reshape(masks.shape[0], masks.shape[1], masks.shape[2], -1)
    if masks.ndim != 4:
        raise ValueError(f"Expected masks with 4 dims, got {masks.shape}")

    return reconstruction, masks


def _resize_mask_to_frame(mask: torch.Tensor, out_hw):
    mask = mask.float().unsqueeze(0).unsqueeze(0)
    mask = F.interpolate(mask, size=out_hw, mode="nearest")
    return mask.squeeze(0).squeeze(0)


def _resize_rgb_to_frame(rgb: np.ndarray, out_hw, mode: str = "nearest"):
    if rgb.shape[:2] == tuple(out_hw):
        return rgb
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    if mode == "bilinear":
        tensor = F.interpolate(tensor, size=out_hw, mode="bilinear", align_corners=False)
    else:
        tensor = F.interpolate(tensor, size=out_hw, mode=mode)
    return tensor.squeeze(0).permute(1, 2, 0).numpy()


def _save_slot_eval_video(inputs, outputs, aux_outputs, output_config):
    output_path = output_config.get("slot_eval_path")
    if not output_path:
        return

    fps = int(output_config.get("slot_eval_fps", 10))
    video_vis = _ensure_video_vis_layout(inputs["video_visualization"])
    frame_size = tuple(video_vis.shape[-2:])

    enc_out = outputs.get("encoder", {})
    enc_features = enc_out.get("backbone_features")
    if enc_features is None:
        enc_features = enc_out.get("features")
    if enc_features is None:
        raise KeyError("No encoder features found for PCA visualization.")
    dec_recon = outputs.get("decoder", {}).get("reconstruction")
    if dec_recon is None:
        raise KeyError("No decoder reconstruction found for PCA visualization.")

    enc_features = _ensure_time_dim(enc_features)
    dec_recon = _ensure_time_dim(dec_recon)
    num_frames = min(video_vis.shape[1], enc_features.shape[1], dec_recon.shape[1])

    enc_first_map = _features_to_map(enc_features[0, 0])
    proj = pca_proj(enc_first_map.reshape(-1, enc_first_map.shape[-1]))

    save_dir = os.path.dirname(output_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    masks = outputs.get("decoder", {}).get("masks")
    if masks is None:
        raise KeyError("No decoder masks found for slot visualization.")
    dec_recon, masks = _to_patch_features(dec_recon, masks)
    num_slots = masks.shape[2]
    vis_masks = None
    if aux_outputs is not None:
        vis_masks = aux_outputs.get("decoder_masks_vis_hard")
        if vis_masks is None:
            vis_masks = aux_outputs.get("decoder_masks_hard")

    with imageio.get_writer(output_path, fps=fps) as writer:
        video_cpu = video_vis.detach().cpu()
        for t in range(num_frames):
            input_frame_unflipped = (
                video_cpu[0, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy()
            )

            enc_map = _features_to_map(enc_features[0, t])
            enc_rgb = colorize_map(enc_map, proj, out_hw=frame_size)

            recon_map = _features_to_map(dec_recon[0, t])
            recon_rgb_patch = colorize_map(recon_map, proj, out_hw=None)
            recon_rgb_unflipped = _resize_rgb_to_frame(recon_rgb_patch, frame_size)

            input_frame = np.flipud(input_frame_unflipped)
            enc_rgb = np.flipud(enc_rgb)
            recon_rgb = np.flipud(recon_rgb_unflipped)

            black = np.zeros((*frame_size, 3), dtype=np.float32)
            top_row = [input_frame, enc_rgb, recon_rgb]
            top_row += [black] * max(0, num_slots - len(top_row))

            slot_frames = []
            slot_input_frames = []
            hard_idx = torch.argmax(masks[0, t], dim=0)
            for s in range(num_slots):
                mask_bin = (hard_idx == s).float()
                mask_map = _features_to_map(mask_bin.unsqueeze(-1)).squeeze(-1)
                slot_rgb_patch = recon_rgb_patch * mask_map.cpu().numpy()[..., None]
                slot_rgb = _resize_rgb_to_frame(slot_rgb_patch, frame_size)
                slot_rgb = np.flipud(slot_rgb)
                slot_frames.append(slot_rgb)
                if vis_masks is not None:
                    input_mask = vis_masks[0, t, s]
                    input_mask_up = input_mask.float().cpu().numpy()
                    if input_mask_up.shape[:2] != frame_size:
                        input_mask_up = _resize_mask_to_frame(
                            torch.from_numpy(input_mask_up), frame_size
                        ).cpu().numpy()
                else:
                    input_mask = (hard_idx == s).float()
                    input_mask_map = _features_to_map(input_mask.unsqueeze(-1)).squeeze(-1)
                    input_mask_up = _resize_mask_to_frame(input_mask_map, frame_size).cpu().numpy()
                slot_input = input_frame_unflipped * input_mask_up[..., None]
                slot_input = np.flipud(slot_input)
                slot_input_frames.append(slot_input)

            grid = create_grid_frame_rgb(
                top_row + slot_frames + slot_input_frames,
                grid_size=(3, num_slots),
                image_size=frame_size,
                padding=2,
            )
            writer.append_data((grid * 255).astype(np.uint8))


def _move_inputs_to_device(inputs, device: torch.device):
    inputs_device = {}
    for key, value in inputs.items():
        if torch.is_tensor(value) and key not in ("video_visualization", "image_visualization"):
            inputs_device[key] = value.to(device)
        else:
            inputs_device[key] = value
    return inputs_device


def _merge_window_output(dst, src, start: int, end: int, total_frames: int):
    if torch.is_tensor(src):
        if src.ndim >= 2 and src.shape[1] == end - start:
            if dst is None:
                dst = torch.zeros(
                    (src.shape[0], total_frames, *src.shape[2:]),
                    device=src.device,
                    dtype=src.dtype,
                )
            dst[:, start:end] = src
            return dst
        return src if dst is None else dst
    if isinstance(src, dict):
        if dst is None:
            dst = {}
        for key, value in src.items():
            dst[key] = _merge_window_output(dst.get(key), value, start, end, total_frames)
        return dst
    return src if dst is None else dst


def _run_sliding_window_inference(model, model_inputs, input_config):
    if "video" not in model_inputs:
        raise ValueError("Sliding-window inference requires `video` inputs.")
    video = model_inputs["video"]
    total_frames = int(video.shape[1])
    window_size = input_config.get("window_size")
    if window_size is None:
        window_size = input_config.get("chunk_size")
    if window_size is None:
        raise ValueError("Sliding-window inference requires input.window_size or input.chunk_size.")
    window_size = int(window_size)
    stride = int(input_config.get("window_stride", window_size))
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and window_stride must be positive.")

    combined_outputs = None
    combined_aux = None
    for start in range(0, total_frames, stride):
        end = min(start + window_size, total_frames)
        if end <= start:
            break
        window_inputs = {"video": video[:, start:end]}
        if "segmentations" in model_inputs:
            window_inputs["segmentations"] = model_inputs["segmentations"][:, start:end]
        outputs = model(window_inputs)
        aux_outputs = model.aux_forward(window_inputs, outputs)
        combined_outputs = _merge_window_output(
            combined_outputs, outputs, start, end, total_frames
        )
        combined_aux = _merge_window_output(
            combined_aux, aux_outputs, start, end, total_frames
        )
    return combined_outputs, combined_aux


def _resolve_device(config, cli_gpu):
    if cli_gpu is not None:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{cli_gpu}")
        return torch.device("cpu")
    device = config.get("device")
    if device:
        return torch.device(device)
    gpu_index = config.get("gpu_index")
    if gpu_index is not None:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{int(gpu_index)}")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_n_slots(inference_config, model_config):
    n_slots = inference_config.get("n_slots")
    if n_slots is not None:
        return int(n_slots)
    model_slots = model_config.get("model", {}).get("initializer", {}).get("n_slots")
    if model_slots is not None:
        return int(model_slots)
    global_slots = model_config.get("globals", {}).get("NUM_SLOTS")
    if global_slots is not None:
        return int(global_slots)
    raise KeyError("Missing n_slots (not found in inference config or model config)")


def _maybe_disable_losses(model, config):
    if config.get("skip_losses", True):
        model.loss_fns = torch.nn.ModuleDict()
        model.loss_weights = {}


def main(config, config_path: str):
    # Load the model from checkpoint
    _resolve_inference_paths(config, config_path)
    device = _resolve_device(config, getattr(config, "_cli_gpu", None))
    model, model_config = load_model_from_checkpoint(config.checkpoint, config.model_config, device)
    model.initializer.n_slots = _resolve_n_slots(config, model_config)
    _maybe_disable_losses(model, config)
    # Prepare the video dict
    if config.input.type == "video":
        inputs = prepare_video(config.input.path, config.input.transforms)
    elif config.input.type == "image":
        inputs = prepare_image(config.input.path, config.input.transforms)
    elif config.input.type in ("libero", "libero_h5"):
        needs_segmentations = any(
            "segmentations" in loss_fn.target_path for loss_fn in model.loss_fns.values()
        )
        include_segmentations = config.input.get("include_segmentations", None)
        if include_segmentations is None:
            include_segmentations = needs_segmentations
        inputs = prepare_libero_h5(config.input, include_segmentations=include_segmentations)
    else:
        raise ValueError(f"Unknown input type `{config.input.type}`")
    model_inputs = _move_inputs_to_device(inputs, device)
    # Perform inference
    use_sliding = bool(config.input.get("sliding_window", False))
    with torch.no_grad():
        if use_sliding:
            outputs, aux_outputs = _run_sliding_window_inference(model, model_inputs, config.input)
        else:
            outputs = model(model_inputs)
            aux_outputs = model.aux_forward(model_inputs, outputs)
    if config.input.type in ("video", "libero", "libero_h5"):
        _save_slot_eval_video(inputs, outputs, aux_outputs, config.output)
    if config.input.type == "video" and config.output.get("save_path"):
        # Save the results
        save_dir = os.path.dirname(config.output.save_path)
        os.makedirs(save_dir, exist_ok=True)
        masked_video_frames = mix_inputs_with_masks(inputs, outputs)
        with imageio.get_writer(config.output.save_path, fps=10) as writer:
            for frame in masked_video_frames:
                writer.append_data(frame)
        writer.close()
    elif config.input.type == "image" and config.output.save_path:
        save_dir = os.path.dirname(config.output.save_path)
        os.makedirs(save_dir, exist_ok=True)
        masks = aux_outputs["decoder_masks_hard"][0].bool()
        cmap = color_map(masks.shape[0])
        image = (inputs["image_visualization"] * 256)[0].type(torch.uint8)
        mixed_image = draw_segmentation_masks_on_image(image, masks, colors=cmap)
        # Save the results
        plt.imsave(config.output.save_path, mixed_image.permute(1, 2, 0).numpy())
    print("Inference completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference on a single MP4 video.")
    parser.add_argument(
        "--config", default="configs/inference/movi_c.yml", help="Configuration to run"
    )
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device index (overrides config)")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config._cli_gpu = args.gpu
    main(config, args.config)
