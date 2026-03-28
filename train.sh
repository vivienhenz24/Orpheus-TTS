#!/usr/bin/env bash
set -euo pipefail

# README-aligned source dataset contract:
# - Hugging Face dataset with at least `audio` and `text`
# - optional `speaker_id`; falls back to DEFAULT_SPEAKER if absent
#
# The official README points to an external Colab preprocessing notebook that is
# not present in this repository. This script mirrors the notebook logic locally,
# including SNAC tokenization, duplicate-frame removal, and Orpheus token layout.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

: "${HF_TOKEN:?Set HF_TOKEN to your Hugging Face token before running this script.}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
DATASET_ID="${HF_DATASET_ID:-vsqrd/turkish-tts-dataset-temp}"
DATASET_SPLIT="${HF_DATASET_SPLIT:-train}"
MODEL_ID="${HF_MODEL_ID:-canopylabs/orpheus-tts-0.1-pretrained}"
TOKENIZER_ID="${HF_TOKENIZER_ID:-canopylabs/orpheus-3b-0.1-pretrained}"
PROCESSED_DIR="${PROCESSED_DIR:-$ROOT_DIR/artifacts/datasets/orpheus_tr_processed}"
RUN_NAME="${RUN_NAME:-turkish-finetune-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/artifacts/runs/$RUN_NAME}"

EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
MIN_DURATION_S="${MIN_DURATION_S:-0.5}"
MAX_DURATION_S="${MAX_DURATION_S:-20.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SAVE_STEPS="${SAVE_STEPS:-500}"
RESUME_FROM="${RESUME_FROM:-}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-16}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"

PHONEMIZE="${PHONEMIZE:-1}"
PHONEMIZE_LANG="${PHONEMIZE_LANG:-tr}"
WARM_START_FROM="${WARM_START_FROM:-vsqrd/orpheus-turkish-finetune-v1}"   # path to existing LoRA checkpoint to warm-start from
HF_PUSH_REPO="${HF_PUSH_REPO:-}"        # e.g. vsqrd/orpheus-turkish-warmstart — set to push to HF
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"
TEXT_COLUMN="${TEXT_COLUMN:-text}"
SPEAKER_COLUMN="${SPEAKER_COLUMN:-speaker_id}"
DEFAULT_SPEAKER="${DEFAULT_SPEAKER:-speaker}"
AUDIO_COLUMN="${AUDIO_COLUMN:-audio}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-{speaker_id}: {text}}"

export HF_TOKEN
export TOKENIZERS_PARALLELISM=false

mkdir -p "$ROOT_DIR/artifacts/datasets" "$ROOT_DIR/artifacts/runs"

if [[ "$PHONEMIZE" == "1" ]] && ! command -v espeak-ng >/dev/null 2>&1; then
  echo "Installing espeak-ng..."
  apt-get update -q && apt-get install -y -q espeak-ng
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found, installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python "$PYTHON_BIN" "$VENV_DIR"
RUN_PYTHON="$VENV_DIR/bin/python"

uv pip install --python "$RUN_PYTHON" --upgrade pip setuptools wheel
uv pip install --python "$RUN_PYTHON" \
  "transformers>=4.49.0,<5" \
  "datasets[audio]>=3.2.0" \
  "accelerate>=1.2.0" \
  "huggingface_hub>=0.27.0" \
  "torch>=2.2.0" \
  "torchaudio>=2.2.0" \
  "soundfile>=0.12.1" \
  "snac>=1.2.1" \
  "wandb>=0.19.0" \
  "sentencepiece>=0.2.0" \
  "protobuf>=5.29.0" \
  "peft>=0.10.0"

# ── Warm-start: extend tokenizer + migrate checkpoint ─────────────────────────
if [[ "$PHONEMIZE" == "1" && -n "$WARM_START_FROM" ]]; then
  PHONEME_TOK_DIR="$ROOT_DIR/artifacts/tokenizer_phoneme"
  WARMSTART_DIR="$ROOT_DIR/artifacts/warmstart"

  if [[ ! -d "$WARMSTART_DIR" ]]; then
    echo "Running tokenizer extension + warm-start from $WARM_START_FROM..."
    "$RUN_PYTHON" "$ROOT_DIR/scripts/extend_tokenizer.py" \
      --checkpoint "$WARM_START_FROM" \
      --output-tokenizer "$PHONEME_TOK_DIR" \
      --output-checkpoint "$WARMSTART_DIR" \
      --hf-token "$HF_TOKEN"
  else
    echo "Warm-start checkpoint already exists at $WARMSTART_DIR"
  fi

  if [[ -n "$HF_PUSH_REPO" ]]; then
    echo "Pushing warm-start checkpoint to $HF_PUSH_REPO..."
    huggingface-cli upload "$HF_PUSH_REPO" "$WARMSTART_DIR" . \
      --token "$HF_TOKEN" --repo-type model
    echo "Pushed to https://huggingface.co/$HF_PUSH_REPO"
    MODEL_ID="$HF_PUSH_REPO"
  else
    MODEL_ID="$WARMSTART_DIR"
  fi
  TOKENIZER_ID="$PHONEME_TOK_DIR"
fi

if [[ "$FORCE_PREPROCESS" == "1" || ! -d "$PROCESSED_DIR" ]]; then
  "$RUN_PYTHON" "$ROOT_DIR/scripts/prepare_orpheus_dataset.py" \
    --dataset "$DATASET_ID" \
    --split "$DATASET_SPLIT" \
    --output-dir "$PROCESSED_DIR" \
    --model-name "$MODEL_ID" \
    --tokenizer-name "$TOKENIZER_ID" \
    --hf-token "$HF_TOKEN" \
    --audio-column "$AUDIO_COLUMN" \
    --text-column "$TEXT_COLUMN" \
    --speaker-column "$SPEAKER_COLUMN" \
    --default-speaker "$DEFAULT_SPEAKER" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --min-duration-s "$MIN_DURATION_S" \
    --max-duration-s "$MAX_DURATION_S" \
    --prefetch-workers "$PREFETCH_WORKERS" \
    --prefetch-factor "$PREFETCH_FACTOR" \
    --progress-every "$PROGRESS_EVERY" \
    ${PROMPT_TEMPLATE:+--prompt-template "$PROMPT_TEMPLATE"} \
    $([[ "$PHONEMIZE" == "1" ]] && echo "--phonemize --phonemize-lang $PHONEMIZE_LANG")
else
  echo "Using existing processed dataset at $PROCESSED_DIR"
fi

mkdir -p "$OUTPUT_DIR"

NUM_GPUS="${NUM_GPUS:-$(python3 -c "import torch; print(max(1, torch.cuda.device_count()))" 2>/dev/null || echo 1)}"
echo "Launching training on $NUM_GPUS GPU(s)"

if [[ "$NUM_GPUS" -gt 1 ]]; then
  LAUNCHER="$RUN_PYTHON -m torch.distributed.run --nproc_per_node=$NUM_GPUS --master_port=29500"
else
  LAUNCHER="$RUN_PYTHON"
fi

$LAUNCHER "$ROOT_DIR/scripts/run_orpheus_finetune.py" \
  --dataset-dir "$PROCESSED_DIR" \
  --model-name "$MODEL_ID" \
  --output-dir "$OUTPUT_DIR" \
  --hf-token "$HF_TOKEN" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --learning-rate "$LEARNING_RATE" \
  --lora-r "$LORA_R" \
  --tokenizer-name "$TOKENIZER_ID" \
  --lora-alpha "$LORA_ALPHA" \
  --save-steps "$SAVE_STEPS" \
  --logging-steps "$LOGGING_STEPS" \
  --warmup-ratio "$WARMUP_RATIO" \
  --bf16 \
  ${RESUME_FROM:+--resume-from-checkpoint "$RESUME_FROM"} \
  --prompt-template "$PROMPT_TEMPLATE" \
  --sample-text "Merhaba, bu modeli Turkce ses uretimi icin test ediyoruz." \
  --sample-text "Bugun hava oldukca guzel, disari cikmak istiyorum." \
  --sample-text "Toplanti yarin sabah dokuzda baslayacak, lutfen hazir olun."

echo
echo "Training artifacts:"
echo "  Processed dataset: $PROCESSED_DIR"
echo "  Run output:        $OUTPUT_DIR"
echo "  Final weights:     $OUTPUT_DIR/final"
echo "  Samples:           $OUTPUT_DIR/samples"
