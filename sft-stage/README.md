# SFT Stages (I & II)

Supervised fine-tuning for the first two stages of 3ViewSense, built on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (vendored under `LLaMA-Factory/`). Base model: **Qwen3-VL-4B-Instruct**.

## Stage I — Orthographic Mental Simulation (OMS)

Teach the model to generate a structured **front / left / top** orthographic view description from a single egocentric image.

```bash
cd OMS-stage
# scene JSON → three-view text descriptions
python json2viewdescription/blockcount_json2abstract.py   # block counting
python json2viewdescription/geometrt_json2abstract.py     # object reasoning
# fine-tune
bash sft-scripts/oms-sft.sh
```

## Stage II — View-Grounded Reasoning (VGR)

Teach the model to solve spatial queries by integrating the inferred views into first-person reasoning traces (`front → left → top`). Traces are distilled from a teacher model and filtered by answer correctness.

```bash
cd VGR-stage
export API_KEY=...   # teacher model API key
export API_URL=...   # teacher model endpoint
python VGR_generation/vgr_generation.py
# fine-tune from the Stage-I checkpoint
bash sft_scripts/vgr-sft.sh
```

`VGR_generation/prompt_template.py` holds the reasoning-trace prompt templates.
