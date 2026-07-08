# RL Stage (III) — GRPO

Group Relative Policy Optimization on top of the **Stage-II (VGR-SFT)** checkpoint, built on [verl](https://github.com/volcengine/verl) (vendored under `verl/`).

> RL must be warm-started from the VGR-SFT model. Starting from the Stage-I (OMS-SFT) model leads to unstable rewards and training collapse.

## Launch

```bash
bash run_qwen3_vl-4b-strict.sh   # exact-match reward
bash run_qwen3_vl-4b-slack.sh    # distance-graded reward
```

Set `TRAIN_MODEL`, `TRAIN_DATA`, and `TEST_DATA` at the top of each script.

## Reward

- Custom reward function: `verl/verl/utils/reward_score/3viewsense.py` → `compute_score`.
- Reward manager: `percept3view_batch` (`verl/verl/workers/reward_manager/percept3view_batch.py`).
- **Strict** mode: exact match. **Slack** mode: graded by numeric distance (counting) or partial axis match (direction).

## Data preparation

```bash
python verl/examples/data_preprocess/3viewsense_datacollect.py \
    --local_dataset_path <jsonl_dir> --local_save_dir OrthoMind-3D-RL --test_ratio 0.2
```
