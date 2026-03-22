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

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
MIN_DURATION_S="${MIN_DURATION_S:-0.5}"
MAX_DURATION_S="${MAX_DURATION_S:-20.0}"
SAVE_STEPS="${SAVE_STEPS:-0}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-16}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"

FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"
TEXT_COLUMN="${TEXT_COLUMN:-text}"
SPEAKER_COLUMN="${SPEAKER_COLUMN:-speaker_id}"
DEFAULT_SPEAKER="${DEFAULT_SPEAKER:-speaker}"
AUDIO_COLUMN="${AUDIO_COLUMN:-audio}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-{speaker_id}: {text}}"

export HF_TOKEN
export TOKENIZERS_PARALLELISM=false

mkdir -p "$ROOT_DIR/artifacts/datasets" "$ROOT_DIR/artifacts/runs"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found on PATH." >&2
  exit 1
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
  "protobuf>=5.29.0"

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
    ${PROMPT_TEMPLATE:+--prompt-template "$PROMPT_TEMPLATE"}
else
  echo "Using existing processed dataset at $PROCESSED_DIR"
fi

mkdir -p "$OUTPUT_DIR"

"$RUN_PYTHON" "$ROOT_DIR/scripts/run_orpheus_finetune.py" \
  --dataset-dir "$PROCESSED_DIR" \
  --model-name "$MODEL_ID" \
  --output-dir "$OUTPUT_DIR" \
  --hf-token "$HF_TOKEN" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --learning-rate "$LEARNING_RATE" \
  --save-steps "$SAVE_STEPS" \
  --logging-steps "$LOGGING_STEPS" \
  --bf16 \
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
