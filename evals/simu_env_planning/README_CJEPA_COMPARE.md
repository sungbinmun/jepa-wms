# Metaworld JEPA/CJEPA/ACJEPA Comparison (No Decoder Plots)

This setup evaluates all three models with the same `simu_env_planning` pipeline and CEM planner, while disabling decoder-dependent plotting (`planner.decode_each_iteration: false`).

## Configs

- JEPA-WM: `configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48_nodecode.yaml`
- C-JEPA: `configs/evals/simu_env_planning/mw/cjepa/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0_ep48_nodecode.yaml`
- AC-JEPA: `configs/evals/simu_env_planning/mw/acjepa/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0_ep48_nodecode.yaml`

## Required environment variables

```bash
export SLOTCONTRAST_CKPT=/home/sungbin/jepa-wms/_local/checkpoints/slotcontrast/metaworld_dinov3_512_mv/checkpoints/step=93000.ckpt
export SLOTCONTRAST_CFG=/home/sungbin/jepa-wms/_local/checkpoints/slotcontrast/metaworld_dinov3_512_mv/settings.yaml
export SLOTCONTRAST_ROOT=/path/to/slotcontrast/source   # optional if not in /home/sungbin/jepa-wms/slotcontrast

export CJEPA_OBJECT_CKPT=/home/sungbin/jepa-wms/cjepa/checkpoints/cjepa_metaworld/epoch=29-step=701640.ckpt
export ACJEPA_OBJECT_CKPT=/home/sungbin/jepa-wms/cjepa/checkpoints/acjepa_metaworld/epoch=29-step=701640.ckpt
```

Notes:
- `CJEPA_OBJECT_CKPT` / `ACJEPA_OBJECT_CKPT` can be either:
  - `*_object.ckpt` from `torch.save(pl_module, ...)`, or
  - Lightning checkpoints like `epoch=29-step=701640.ckpt`.
- If a Lightning checkpoint is used, the wrapper auto-detects a training yaml in the same folder (for heads/depth/dropout). You can also pass it explicitly via `model_kwargs.pretrain_kwargs.cjepa_train_config`.
- If SlotContrast source is elsewhere, set `SLOTCONTRAST_ROOT` or `model_kwargs.pretrain_kwargs.slotcontrast_root`.
- The wrapper used for slot-based models is `app.vjepa_wm.modelcustom.simu_env_planning.cjepa_slot_enc_preds`.

## Run

```bash
python -m evals.main --fname configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48_nodecode.yaml --debug
python -m evals.main --fname configs/evals/simu_env_planning/mw/cjepa/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0_ep48_nodecode.yaml --debug
python -m evals.main --fname configs/evals/simu_env_planning/mw/acjepa/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0_ep48_nodecode.yaml --debug
```

Use distributed launcher (`evals.main_distributed`) if needed.
