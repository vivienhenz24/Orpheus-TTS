#!/usr/bin/env bash
set -euo pipefail

# Runpod setup script for Orpheus Turkish training.
#
# What it does:
# 1. Installs Python dependencies.
# 2. Downloads the base model and dataset from Hugging Face.
# 3. Decodes audio + SNAC-tokenizes in one parallel pass (no intermediate WAV files).
# 4. Writes a continued-pretraining config and launches training.
#
# Assumptions:
# - You are running on a CUDA/PyTorch Runpod image.
# - HF_TOKEN is already exported in the environment.
# - Optional: WANDB_API_KEY is exported if you want wandb logging.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_ID="${DATASET_ID:-vsqrd/100h_turkish}"
RUNPOD_WORKDIR="${RUNPOD_WORKDIR:-/workspace/orpheus_tr}"
RAW_DIR="${RAW_DIR:-$RUNPOD_WORKDIR/raw_dataset}"
HF_HOME_DIR="${HF_HOME_DIR:-$RUNPOD_WORKDIR/.hf}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
SKIP_APT="${SKIP_APT:-0}"
LAUNCH_TRAINING="${LAUNCH_TRAINING:-1}"
TRAINING_MODE="${TRAINING_MODE:-continued_pretrain}"
BASE_MODEL="${BASE_MODEL:-canopylabs/orpheus-3b-0.1-pretrained}"
TOKENIZED_LOCAL_DIR="${TOKENIZED_LOCAL_DIR:-$RUNPOD_WORKDIR/tokenized_tts_dataset}"
TOKENIZATION_LIMIT="${TOKENIZATION_LIMIT:-0}"
TOKENIZATION_DEVICE="${TOKENIZATION_DEVICE:-cuda}"
TOKENIZATION_MAX_LENGTH="${TOKENIZATION_MAX_LENGTH:-8192}"
TOKENIZATION_BATCH_SIZE="${TOKENIZATION_BATCH_SIZE:-48}"
TOKENIZATION_WRITE_SHARD_SIZE="${TOKENIZATION_WRITE_SHARD_SIZE:-512}"
TOKENIZATION_NUM_WORKERS="${TOKENIZATION_NUM_WORKERS:-16}"
TTS_TOKENIZED_DATASET="${TTS_TOKENIZED_DATASET:-$TOKENIZED_LOCAL_DIR}"
TEXT_QA_DATASET="${TEXT_QA_DATASET:-}"
PRETRAIN_RATIO="${PRETRAIN_RATIO:-0}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-1}"
PRETRAIN_NUM_PROCESSES="${PRETRAIN_NUM_PROCESSES:-8}"
PRETRAIN_SAVE_STEPS="${PRETRAIN_SAVE_STEPS:-2000}"
PRETRAIN_LR="${PRETRAIN_LR:-2.0e-5}"
PRETRAIN_SAVE_FOLDER="${PRETRAIN_SAVE_FOLDER:-checkpoints/tr_orpheus_pretrain_stage1}"
PRETRAIN_PROJECT_NAME="${PRETRAIN_PROJECT_NAME:-orpheus-tr}"
PRETRAIN_RUN_NAME="${PRETRAIN_RUN_NAME:-tr-stage1}"
PRETRAIN_CONFIG_PATH="${PRETRAIN_CONFIG_PATH:-$RUNPOD_WORKDIR/pretrain_turkish_generated.yaml}"
LOG_FILE="${LOG_FILE:-$RUNPOD_WORKDIR/orpheus_setup.log}"

export DATASET_ID
export RAW_DIR
export BASE_MODEL
export PRETRAIN_CONFIG_PATH
export TOKENIZED_LOCAL_DIR

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required." >&2
  exit 1
fi

mkdir -p "$RUNPOD_WORKDIR" "$RAW_DIR" "$HF_HOME_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'echo "[CRASH] $(date +%H:%M:%S) line $LINENO exit $?: $BASH_COMMAND" >&2' ERR
echo "==> Log: $LOG_FILE  ($(date))"
export HF_HOME="$HF_HOME_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_HOME_DIR/hub"
export HF_HUB_CACHE="$HF_HOME_DIR/hub"
export HF_XET_CACHE="$HF_HOME_DIR/xet"

echo "==> Repo root: $REPO_ROOT"
echo "==> Workdir: $RUNPOD_WORKDIR"
echo "==> Dataset: $DATASET_ID"

if [[ "$SKIP_APT" != "1" ]] && command -v apt-get >/dev/null 2>&1; then
  echo "==> Installing system packages"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    git-lfs \
    libsndfile1 \
    libsndfile1-dev \
    build-essential
  git lfs install
fi

echo "==> Installing Python dependencies"
pip install accelerate datasets huggingface_hub librosa pandas pyarrow soundfile "transformers==4.46.3" wandb
pip install --no-deps snac

if [[ "$INSTALL_FLASH_ATTN" == "1" ]]; then
  echo "==> Installing flash-attn (best effort)"
  pip install flash-attn --no-build-isolation || {
    echo "flash-attn install failed. Continuing without it."
  }
fi

echo "==> Environment summary"
python - <<'PY'
import json
import platform
import torch

summary = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
}
if torch.cuda.is_available():
    summary["cuda_name_0"] = torch.cuda.get_device_name(0)
print(json.dumps(summary, indent=2))
PY

echo "==> Pre-downloading base model into HF cache"
_t0=$SECONDS
python - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ.get("BASE_MODEL", "canopylabs/orpheus-3b-0.1-pretrained")
token = os.environ["HF_TOKEN"]

path = snapshot_download(repo_id=model_id, repo_type="model", token=token)
print(f"Model cached at {path}")
PY
echo "==> model download done ($((SECONDS - _t0))s)"

echo "==> Pre-downloading SNAC model into HF cache"
python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="hubertsiuzdak/snac_24khz", repo_type="model")
print(f"SNAC cached at {path}")
PY

echo "==> Downloading dataset snapshot"
_t1=$SECONDS
python - <<'PY'
import os
from huggingface_hub import snapshot_download

dataset_id = os.environ.get("DATASET_ID", "vsqrd/100h_turkish")
raw_dir = os.environ["RAW_DIR"]
token = os.environ["HF_TOKEN"]

snapshot_download(
    repo_id=dataset_id,
    repo_type="dataset",
    local_dir=raw_dir,
    token=token,
)
print(f"Downloaded dataset snapshot to {raw_dir}")
PY
echo "==> dataset download done ($((SECONDS - _t1))s)"

echo "==> Preparing dataset (decode + SNAC tokenize in one pass)"
_t2=$SECONDS
PREP_CMD=(
  python "$REPO_ROOT/scripts/prepare_dataset.py"
  --raw-dir "$RAW_DIR"
  --output-dir "$TOKENIZED_LOCAL_DIR"
  --model-name "$BASE_MODEL"
  --device "$TOKENIZATION_DEVICE"
  --max-length "$TOKENIZATION_MAX_LENGTH"
  --batch-size "$TOKENIZATION_BATCH_SIZE"
  --write-shard-size "$TOKENIZATION_WRITE_SHARD_SIZE"
  --num-workers "$TOKENIZATION_NUM_WORKERS"
)
if [[ "$TOKENIZATION_LIMIT" != "0" ]]; then
  PREP_CMD+=(--limit "$TOKENIZATION_LIMIT")
fi
"${PREP_CMD[@]}"
echo "==> dataset prep done ($((SECONDS - _t2))s)"

if [[ "$LAUNCH_TRAINING" == "1" ]]; then
  if [[ -z "$TTS_TOKENIZED_DATASET" ]]; then
    echo "LAUNCH_TRAINING=1 requires TTS_TOKENIZED_DATASET." >&2
    exit 1
  fi
  if [[ "$PRETRAIN_RATIO" != "0" && -z "$TEXT_QA_DATASET" ]]; then
    echo "PRETRAIN_RATIO=$PRETRAIN_RATIO requires TEXT_QA_DATASET." >&2
    exit 1
  fi

  echo "==> Writing generated continued-pretraining config"
  cat > "$PRETRAIN_CONFIG_PATH" <<EOF
model_name: "$BASE_MODEL"
tokenizer_name: "$BASE_MODEL"

epochs: $PRETRAIN_EPOCHS
batch_size: $PRETRAIN_BATCH_SIZE
number_processes: $PRETRAIN_NUM_PROCESSES
pad_token: 128263
save_steps: $PRETRAIN_SAVE_STEPS
learning_rate: $PRETRAIN_LR
ratio: $PRETRAIN_RATIO

text_QA_dataset: ${TEXT_QA_DATASET:-null}
TTS_dataset: "$TTS_TOKENIZED_DATASET"

save_folder: "$PRETRAIN_SAVE_FOLDER"
project_name: "$PRETRAIN_PROJECT_NAME"
run_name: "$PRETRAIN_RUN_NAME"
EOF

  echo "==> Generated config"
  cat "$PRETRAIN_CONFIG_PATH"

  if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "==> Configuring Weights & Biases"
    wandb login "$WANDB_API_KEY"
  fi

  if [[ "$TRAINING_MODE" != "continued_pretrain" ]]; then
    echo "Unsupported TRAINING_MODE: $TRAINING_MODE" >&2
    exit 1
  fi

  echo "==> Launching continued pretraining ($(date))"
  _t4=$SECONDS
  (
    cd "$REPO_ROOT/pretrain"
    ORPHEUS_PRETRAIN_CONFIG="$PRETRAIN_CONFIG_PATH" \
      accelerate launch --num_processes "$PRETRAIN_NUM_PROCESSES" --mixed_precision bf16 train.py
  )
  echo "==> training done ($((SECONDS - _t4))s)"
fi

cat <<EOF

==> Setup complete

Prepared outputs:
  Tokenized dataset: $TOKENIZED_LOCAL_DIR/parquet/
  Stats:             $TOKENIZED_LOCAL_DIR/stats.json
EOF
