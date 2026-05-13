#!/usr/bin/env bash
set -euo pipefail

# RunPod setup script — resumes Turkish LoRA finetuning from vsqrd/orpheus-3b-turkish-finetune-lora.
#
# What it does:
# 1. Installs Python dependencies (including peft).
# 2. Downloads the pretrain base model, the LoRA checkpoint, SNAC, and the raw dataset.
# 3. Decodes audio + SNAC-tokenizes the dataset (skips if already done).
# 4. Writes a finetune config and launches lora.py via accelerate.
#
# Assumptions:
# - You are running on a CUDA/PyTorch RunPod image.
# - HF_TOKEN is already exported in the environment.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET_ID="${DATASET_ID:-vsqrd/100h_turkish}"
RUNPOD_WORKDIR="${RUNPOD_WORKDIR:-/workspace/orpheus_tr}"
RAW_DIR="${RAW_DIR:-$RUNPOD_WORKDIR/raw_dataset}"
HF_HOME_DIR="${HF_HOME_DIR:-$RUNPOD_WORKDIR/.hf}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
SKIP_APT="${SKIP_APT:-0}"
LAUNCH_TRAINING="${LAUNCH_TRAINING:-1}"

BASE_MODEL="${BASE_MODEL:-vsqrd/orpheus-3b-turkish-pretrain}"
TOKENIZER_NAME="${TOKENIZER_NAME:-canopylabs/orpheus-3b-0.1-pretrained}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-vsqrd/orpheus-3b-turkish-finetune-lora}"

TOKENIZED_LOCAL_DIR="${TOKENIZED_LOCAL_DIR:-$RUNPOD_WORKDIR/tokenized_tts_dataset}"
TOKENIZATION_LIMIT="${TOKENIZATION_LIMIT:-0}"
TOKENIZATION_DEVICE="${TOKENIZATION_DEVICE:-cuda}"
TOKENIZATION_MAX_LENGTH="${TOKENIZATION_MAX_LENGTH:-8192}"
TOKENIZATION_BATCH_SIZE="${TOKENIZATION_BATCH_SIZE:-48}"
TOKENIZATION_WRITE_SHARD_SIZE="${TOKENIZATION_WRITE_SHARD_SIZE:-512}"
TOKENIZATION_NUM_WORKERS="${TOKENIZATION_NUM_WORKERS:-16}"

FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-3}"
FINETUNE_BATCH_SIZE="${FINETUNE_BATCH_SIZE:-1}"
FINETUNE_NUM_PROCESSES="${FINETUNE_NUM_PROCESSES:-2}"
FINETUNE_GRAD_ACCUM="${FINETUNE_GRAD_ACCUM:-8}"
FINETUNE_WARMUP_STEPS="${FINETUNE_WARMUP_STEPS:-500}"
FINETUNE_LOGGING_STEPS="${FINETUNE_LOGGING_STEPS:-500}"
# save twice per epoch: 38799 samples / 2 GPUs / 8 accum = ~2425 optimizer steps/epoch
FINETUNE_SAVE_STEPS="${FINETUNE_SAVE_STEPS:-1200}"
FINETUNE_LR="${FINETUNE_LR:-5.0e-5}"
FINETUNE_SAVE_FOLDER="${FINETUNE_SAVE_FOLDER:-checkpoints/tr_orpheus_finetune_lora_resume}"
FINETUNE_PROJECT_NAME="${FINETUNE_PROJECT_NAME:-orpheus-tr}"
FINETUNE_RUN_NAME="${FINETUNE_RUN_NAME:-tr-lora-resume}"
FINETUNE_CONFIG_PATH="${FINETUNE_CONFIG_PATH:-$RUNPOD_WORKDIR/finetune_lora_resume_generated.yaml}"

LOG_FILE="${LOG_FILE:-$RUNPOD_WORKDIR/orpheus_lora_resume.log}"

export DATASET_ID RAW_DIR BASE_MODEL TOKENIZER_NAME LORA_CHECKPOINT TOKENIZED_LOCAL_DIR FINETUNE_CONFIG_PATH

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

echo "==> Repo root:   $REPO_ROOT"
echo "==> Workdir:     $RUNPOD_WORKDIR"
echo "==> Base model:  $BASE_MODEL"
echo "==> LoRA resume: $LORA_CHECKPOINT"
echo "==> Dataset:     $DATASET_ID"

# ── 1. System packages ────────────────────────────────────────────────────────

if [[ "$SKIP_APT" != "1" ]] && command -v apt-get >/dev/null 2>&1; then
  echo "==> Installing system packages"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends \
    ffmpeg git git-lfs libsndfile1 libsndfile1-dev build-essential
  git lfs install
fi

# ── 2. Python dependencies ────────────────────────────────────────────────────

echo "==> Installing Python dependencies"
pip install accelerate datasets huggingface_hub librosa pandas peft pyarrow soundfile "transformers==4.46.3" wandb
pip install --no-deps snac

if [[ "$INSTALL_FLASH_ATTN" == "1" ]]; then
  echo "==> Installing flash-attn (best effort)"
  pip install flash-attn --no-build-isolation || {
    echo "flash-attn install failed — continuing without it."
  }
fi

# ── 3. Environment summary ────────────────────────────────────────────────────

python - <<'PY'
import json, platform, torch
print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    **( {"cuda_name_0": torch.cuda.get_device_name(0)} if torch.cuda.is_available() else {} ),
}, indent=2))
PY

# ── 4. Download models ────────────────────────────────────────────────────────

echo "==> Pre-downloading base model ($BASE_MODEL)"
_t0=$SECONDS
python - <<'PY'
import os
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id=os.environ["BASE_MODEL"], repo_type="model", token=os.environ["HF_TOKEN"])
print(f"Base model cached at {path}")
PY
echo "==> base model download done ($((SECONDS - _t0))s)"

echo "==> Pre-downloading tokenizer ($TOKENIZER_NAME)"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id=os.environ["TOKENIZER_NAME"], repo_type="model", token=os.environ["HF_TOKEN"])
print(f"Tokenizer cached at {path}")
PY

echo "==> Pre-downloading LoRA checkpoint ($LORA_CHECKPOINT)"
_t1=$SECONDS
python - <<'PY'
import os
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id=os.environ["LORA_CHECKPOINT"], repo_type="model", token=os.environ["HF_TOKEN"])
print(f"LoRA checkpoint cached at {path}")
PY
echo "==> LoRA checkpoint download done ($((SECONDS - _t1))s)"

echo "==> Pre-downloading SNAC model"
python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="hubertsiuzdak/snac_24khz", repo_type="model")
print(f"SNAC cached at {path}")
PY

# ── 5. Download + tokenize dataset ───────────────────────────────────────────

echo "==> Downloading dataset snapshot ($DATASET_ID)"
_t2=$SECONDS
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["DATASET_ID"],
    repo_type="dataset",
    local_dir=os.environ["RAW_DIR"],
    token=os.environ["HF_TOKEN"],
)
print(f"Downloaded to {os.environ['RAW_DIR']}")
PY
echo "==> dataset download done ($((SECONDS - _t2))s)"

if [[ -f "$TOKENIZED_LOCAL_DIR/stats.json" ]]; then
  echo "==> Dataset already prepared — skipping (found $TOKENIZED_LOCAL_DIR/stats.json)"
else
  echo "==> Preparing dataset (decode + SNAC tokenize)"
  _t3=$SECONDS
  PREP_CMD=(
    python "$REPO_ROOT/scripts/prepare_dataset.py"
    --raw-dir "$RAW_DIR"
    --output-dir "$TOKENIZED_LOCAL_DIR"
    --model-name "$TOKENIZER_NAME"
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
  echo "==> dataset prep done ($((SECONDS - _t3))s)"
fi

# ── 6. Write finetune config ──────────────────────────────────────────────────

if [[ "$LAUNCH_TRAINING" == "1" ]]; then
  echo "==> Writing generated LoRA finetune config to $FINETUNE_CONFIG_PATH"
  cat > "$FINETUNE_CONFIG_PATH" <<EOF
TTS_dataset: "$TOKENIZED_LOCAL_DIR"

model_name: "$BASE_MODEL"
tokenizer_name: "$TOKENIZER_NAME"

# Resume LoRA weights from this checkpoint (local path or HF repo ID).
resume_from_lora: "$LORA_CHECKPOINT"

epochs: $FINETUNE_EPOCHS
batch_size: $FINETUNE_BATCH_SIZE
number_processes: $FINETUNE_NUM_PROCESSES
pad_token: 128263
save_steps: $FINETUNE_SAVE_STEPS
learning_rate: $FINETUNE_LR
gradient_accumulation_steps: $FINETUNE_GRAD_ACCUM
warmup_steps: $FINETUNE_WARMUP_STEPS
logging_steps: $FINETUNE_LOGGING_STEPS

save_folder: "$FINETUNE_SAVE_FOLDER"
project_name: "$FINETUNE_PROJECT_NAME"
run_name: "$FINETUNE_RUN_NAME"
EOF

  echo "==> Generated config:"
  cat "$FINETUNE_CONFIG_PATH"

  # ── 7. Launch LoRA finetuning ─────────────────────────────────────────────

  echo "==> Launching LoRA finetuning resume ($(date))"
  _t4=$SECONDS
  (
    cd "$REPO_ROOT/finetune"
    ORPHEUS_FINETUNE_CONFIG="$FINETUNE_CONFIG_PATH" \
      accelerate launch \
        --num_processes "$FINETUNE_NUM_PROCESSES" \
        --mixed_precision bf16 \
        lora.py
  )
  echo "==> finetuning done ($((SECONDS - _t4))s)"
fi

cat <<EOF

==> Setup complete

Outputs:
  Tokenized dataset : $TOKENIZED_LOCAL_DIR/parquet/
  Finetune config   : $FINETUNE_CONFIG_PATH
  Checkpoints       : $REPO_ROOT/finetune/$FINETUNE_SAVE_FOLDER/
  Log               : $LOG_FILE
EOF
