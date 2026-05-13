# AGENTS

This repository has one goal only:

- Add Turkish support to Orpheus using 100 hours of high-quality Turkish audio.

All work in this repo should be evaluated against that goal.

Priorities:

- Improve Turkish text-to-speech quality, naturalness, pronunciation, and robustness.
- Prefer changes that directly support Turkish data preparation, training, finetuning, evaluation, and inference.
- Avoid unrelated feature work unless it is required to enable Turkish support.

Non-goals:

- General repo cleanup that does not help Turkish support.
- New features unrelated to Turkish training or Turkish inference.
- Broad multilingual expansion beyond what is necessary to ship Turkish support well.

---

## Current Status (2026-05-10)

### Pretraining — complete
- Pretrain checkpoint saved to `vsqrd/orpheus-3b-turkish-pretrain` (HF)
- Final checkpoint: `/workspace/Orpheus-TTS/pretrain/checkpoints/tr_orpheus_pretrain_stage1/checkpoint-38799`

### Finetuning — interrupted, resuming
- LoRA finetune ran on `vsqrd/100h_turkish`, interrupted mid-run
- LoRA checkpoint pushed to HF: `vsqrd/orpheus-3b-turkish-finetune-lora`
- Config updated: **3 epochs**, save every ~19,400 steps (twice per epoch), resuming from HF LoRA checkpoint
- Resume command:
  ```bash
  cd /workspace/Orpheus-TTS/finetune
  ORPHEUS_FINETUNE_CONFIG=config.yaml \
    accelerate launch --num_processes 2 --mixed_precision bf16 lora.py
  ```

### What we trained on
- Dataset: `vsqrd/100h_turkish` — 77,598 samples, 100h of Turkish speech
- Pretrain base: `canopylabs/orpheus-3b-0.1-pretrained`
- Finetune base: pretrain checkpoint-38799
- LoRA: r=32, alpha=64, rslora, targets all projection + MLP layers + lm_head/embed_tokens
- LR: 5e-5, cosine schedule, bf16, 2 GPUs

### Key decisions & lessons
- Do NOT use `canopylabs/orpheus-tts-0.1-finetune-prod` (English-specialized) or raw Llama as base — only `orpheus-3b-0.1-pretrained`
- RunPod images have torch pre-installed — never pip install torch, use `pip install --no-deps snac` since snac hard-pins an older torch version
- No venv needed on RunPod — install directly into system Python
- Pre-download all models (base model, SNAC `hubertsiuzdak/snac_24khz`) into HF cache before training starts
- `transformers` version conflict with torch 2.8.0 on newer pods — if you see `infer_schema` errors, pin `transformers==4.46.3`
- FSDP requires multi-GPU — use accelerate for multi-GPU, disable FSDP
- Log file: `/workspace/orpheus_tr/orpheus_setup.log`
- To resume LoRA finetuning from an existing checkpoint: set `resume_from_lora` in config.yaml

### Next steps
1. Resume finetuning for 3 epochs total (from `vsqrd/orpheus-3b-turkish-finetune-lora`)
2. Evaluate naturalness and pronunciation manually after epoch 1 completes
3. If quality is good after epoch 2, stop early rather than overfitting
