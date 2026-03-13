# Repo-Local Paths

Use repo-local paths so datasets, logs, checkpoints, and pretrained assets live under this repository instead of `~/`.

## Enable repo-local env vars

From the repo root:

```bash
source /home/sungbin/jepa-wms/use_repo_local_env.sh
```

This sets:

- `JEPAWM_HOME=/home/sungbin/jepa-wms`
- `JEPAWM_DSET=/home/sungbin/jepa-wms/_local/datasets`
- `JEPAWM_LOGS=/home/sungbin/jepa-wms/_local/logs`
- `JEPAWM_CKPT=/home/sungbin/jepa-wms/_local/checkpoints`
- `JEPAWM_OSSCKPT=/home/sungbin/jepa-wms/_local/oss_ckpt`

and regenerates `macros.py`.

## Recommended local layout

```text
jepa-wms/
  _local/
    datasets/
    logs/
    checkpoints/
      slotcontrast/
        metaworld_dinov3_512_mv/
          settings.yaml
          checkpoints/
            step=93000.ckpt
    oss_ckpt/
```

## Current SlotContrast artifact

The provided SlotContrast config/checkpoint were organized into:

```text
/home/sungbin/jepa-wms/_local/checkpoints/slotcontrast/metaworld_dinov3_512_mv/
```

Training scripts under `cjepa/scripts/metaworld/` now default to that location.
