#!/usr/bin/env bash
set -euo pipefail

# RunPod setup script for Turkish inference.
#
# What it does:
# 1. Installs inference dependencies.
# 2. Downloads the Turkish pretrain model, tokenizer, LoRA adapter, and SNAC.
# 3. Merges the LoRA adapter into the Turkish pretrained checkpoint.
# 4. Starts the repo's realtime streaming server with HTTP + WebSocket TTS on port 8000 by default.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUNPOD_WORKDIR="${RUNPOD_WORKDIR:-/workspace/orpheus_infer}"
HF_HOME_DIR="${HF_HOME_DIR:-$RUNPOD_WORKDIR/.hf}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-$RUNPOD_WORKDIR/models/orpheus-tr-merged}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_APT="${SKIP_APT:-0}"
START_SERVER="${START_SERVER:-1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEFAULT_SPEAKER="${DEFAULT_SPEAKER:-Metin}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

BASE_MODEL="${BASE_MODEL:-vsqrd/orpheus-3b-turkish-pretrain}"
TOKENIZER_NAME="${TOKENIZER_NAME:-canopylabs/orpheus-3b-0.1-pretrained}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-vsqrd/orpheus-3b-turkish-finetune-lora}"

LOG_FILE="${LOG_FILE:-$RUNPOD_WORKDIR/orpheus_inference_setup.log}"
SERVER_LOG_FILE="${SERVER_LOG_FILE:-$RUNPOD_WORKDIR/orpheus_tts_server.log}"
SERVER_PID_FILE="${SERVER_PID_FILE:-$RUNPOD_WORKDIR/orpheus_tts_server.pid}"

export HF_HOME="$HF_HOME_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_HOME_DIR/hub"
export HF_HUB_CACHE="$HF_HOME_DIR/hub"
export HF_XET_CACHE="$HF_HOME_DIR/xet"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export BASE_MODEL TOKENIZER_NAME LORA_CHECKPOINT MERGED_MODEL_DIR HOST PORT DEFAULT_SPEAKER MAX_MODEL_LEN

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required." >&2
  exit 1
fi

mkdir -p "$RUNPOD_WORKDIR" "$HF_HOME_DIR" "$(dirname "$MERGED_MODEL_DIR")"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'echo "[CRASH] $(date +%H:%M:%S) line $LINENO exit $?: $BASH_COMMAND" >&2' ERR
echo "==> Log: $LOG_FILE  ($(date))"
echo "==> Repo root: $REPO_ROOT"
echo "==> Workdir: $RUNPOD_WORKDIR"
echo "==> Base model: $BASE_MODEL"
echo "==> LoRA adapter: $LORA_CHECKPOINT"
echo "==> Merged output: $MERGED_MODEL_DIR"

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
# Training in this repo pins 4.46.3 for a separate torch/transformers issue, but
# vLLM 0.7.3 requires transformers >= 4.48.2. Inference uses the vLLM path, so
# we install a compatible transformers release here.
pip install accelerate fastapi huggingface_hub peft soundfile "transformers==4.48.3" "uvicorn[standard]" "vllm==0.7.3"
pip install --no-deps snac

echo "==> Environment summary"
"$PYTHON_BIN" - <<'PY'
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

echo "==> Pre-downloading base model"
"$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=os.environ["BASE_MODEL"],
    repo_type="model",
    token=os.environ["HF_TOKEN"],
)
print(f"Base model cached at {path}")
PY

echo "==> Pre-downloading tokenizer"
"$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=os.environ["TOKENIZER_NAME"],
    repo_type="model",
    token=os.environ["HF_TOKEN"],
)
print(f"Tokenizer cached at {path}")
PY

echo "==> Pre-downloading LoRA adapter"
"$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=os.environ["LORA_CHECKPOINT"],
    repo_type="model",
    token=os.environ["HF_TOKEN"],
)
print(f"LoRA cached at {path}")
PY

echo "==> Pre-downloading SNAC"
"$PYTHON_BIN" - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(repo_id="hubertsiuzdak/snac_24khz", repo_type="model")
print(f"SNAC cached at {path}")
PY

if [[ ! -f "$MERGED_MODEL_DIR/config.json" ]]; then
  echo "==> Merging LoRA into base model"
  "$PYTHON_BIN" - <<'PY'
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = os.environ["BASE_MODEL"]
tokenizer_name = os.environ["TOKENIZER_NAME"]
lora_checkpoint = os.environ["LORA_CHECKPOINT"]
merged_model_dir = os.environ["MERGED_MODEL_DIR"]

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
num_custom_tokens = 7 * 4096 + 10
tokenizer.add_tokens([f"<custom_token_{i}>" for i in range(num_custom_tokens + 1)])

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=dtype,
)
model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(model, lora_checkpoint)
merged = model.merge_and_unload()
merged.save_pretrained(merged_model_dir, safe_serialization=True)
tokenizer.save_pretrained(merged_model_dir)
print(f"Merged model saved to {merged_model_dir}")
PY
else
  echo "==> Reusing existing merged model at $MERGED_MODEL_DIR"
fi

if [[ "$START_SERVER" == "1" ]]; then
  if [[ -f "$SERVER_PID_FILE" ]] && kill -0 "$(cat "$SERVER_PID_FILE")" >/dev/null 2>&1; then
    echo "==> Server already running with PID $(cat "$SERVER_PID_FILE")"
  else
    echo "==> Starting TTS server on $HOST:$PORT"
    nohup env \
      MODEL_NAME="$MERGED_MODEL_DIR" \
      TOKENIZER_PATH="$MERGED_MODEL_DIR" \
      HOST="$HOST" \
      PORT="$PORT" \
      DEFAULT_VOICE="$DEFAULT_SPEAKER" \
      MAX_MODEL_LEN="$MAX_MODEL_LEN" \
      PYTHONPATH="$REPO_ROOT/orpheus_tts_pypi${PYTHONPATH:+:$PYTHONPATH}" \
      "$PYTHON_BIN" "$REPO_ROOT/realtime_streaming_example/main.py" \
      >"$SERVER_LOG_FILE" 2>&1 &
    echo $! > "$SERVER_PID_FILE"
    sleep 5
    echo "==> Server PID: $(cat "$SERVER_PID_FILE")"
    echo "==> Server log: $SERVER_LOG_FILE"
  fi
fi

POD_ID="${RUNPOD_POD_ID:-${POD_ID:-}}"
PUBLIC_IP="${RUNPOD_PUBLIC_IP:-${PUBLIC_IP:-}}"
if [[ -z "$PUBLIC_IP" ]]; then
  PUBLIC_IP="$(curl -fsS https://api.ipify.org 2>/dev/null || true)"
fi

PUBLIC_HTTP_URL=""
PUBLIC_WS_URL=""
if [[ -n "$PUBLIC_IP" ]]; then
  PUBLIC_HTTP_URL="http://$PUBLIC_IP:$PORT"
  PUBLIC_WS_URL="ws://$PUBLIC_IP:$PORT/v1/audio/speech/stream"
fi

cat <<EOF

==> Inference setup complete

Artifacts:
  Merged model : $MERGED_MODEL_DIR
  Setup log    : $LOG_FILE
  Server log   : $SERVER_LOG_FILE
  Server pid   : $SERVER_PID_FILE

Endpoints:
  Health    : http://$HOST:$PORT/health
  HTTP TTS  : http://$HOST:$PORT/tts?text=Merhaba%20dunya&speaker=$DEFAULT_SPEAKER
  WS TTS    : ws://$HOST:$PORT/v1/audio/speech/stream
  Local     : http://127.0.0.1:$PORT

RunPod:
  Pod ID    : ${POD_ID:-unknown}
  Public IP : ${PUBLIC_IP:-unknown}
  Public    : ${PUBLIC_HTTP_URL:-unknown}
  Public WS : ${PUBLIC_WS_URL:-unknown}

Example:
  curl "http://127.0.0.1:$PORT/tts?text=Merhaba%20dunya&speaker=$DEFAULT_SPEAKER" --output sample.wav
EOF
