import asyncio
import json
import os
import struct
import sys
from pathlib import Path
import queue
import threading
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "orpheus_tts_pypi"))

from orpheus_tts import OrpheusModel

MODEL_NAME = os.environ.get("MODEL_NAME", "canopylabs/orpheus-tts-0.1-finetune-prod")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_NAME)
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "Metin")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))

DEFAULT_REPETITION_PENALTY = float(os.environ.get("DEFAULT_REPETITION_PENALTY", "1.1"))
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", "2000"))
DEFAULT_TEMPERATURE = float(os.environ.get("DEFAULT_TEMPERATURE", "0.4"))
DEFAULT_TOP_P = float(os.environ.get("DEFAULT_TOP_P", "0.9"))
SAMPLE_RATE = 24000
DEBUG_DIR = os.environ.get("ORPHEUS_DEBUG_DIR")

engine = OrpheusModel(
    model_name=MODEL_NAME,
    tokenizer=TOKENIZER_PATH,
    max_model_len=MAX_MODEL_LEN,
)

app = FastAPI(title="Orpheus Turkish Realtime Server")


def create_wav_header(sample_rate: int = SAMPLE_RATE, bits_per_sample: int = 16, channels: int = 1) -> bytes:
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        0xFFFFFFFF,
    )


class IncrementalTextBuffer:
    def __init__(self, split_granularity: str = "sentence") -> None:
        self.buffer = ""
        self.split_granularity = split_granularity

    def push(self, text: str) -> list[str]:
        self.buffer += text
        return self._pop_ready(final=False)

    def finish(self) -> list[str]:
        ready = self._pop_ready(final=True)
        self.buffer = ""
        return ready

    def _pop_ready(self, final: bool) -> list[str]:
        ready: list[str] = []
        while True:
            boundary = self._find_boundary(self.buffer, final=final)
            if boundary <= 0:
                break
            chunk = self.buffer[:boundary].strip()
            self.buffer = self.buffer[boundary:]
            if chunk:
                ready.append(chunk)
        return ready

    def _find_boundary(self, text: str, final: bool) -> int:
        if not text:
            return -1

        punctuation = ".!?…" if self.split_granularity == "sentence" else ".!?…,:;\n"
        closing = "\"')]}”’ "
        for idx, char in enumerate(text):
            if char not in punctuation:
                continue
            end = idx + 1
            while end < len(text) and text[end] in closing:
                end += 1
            if end < len(text):
                return end
            if final:
                return end

        if final and text.strip():
            return len(text)
        return -1


def start_audio_generation(
    prompt: str,
    voice: str,
    repetition_penalty: float,
    max_tokens: int,
    temperature: float,
    top_p: float,
    request_id: str,
    debug_meta: dict[str, Any] | None = None,
) -> queue.Queue[Any]:
    audio_queue: queue.Queue[Any] = queue.Queue()

    def worker() -> None:
        total_bytes = 0
        total_chunks = 0
        started_at = time.time()
        try:
            for chunk in engine.generate_speech(
                prompt=prompt,
                voice=voice,
                request_id=request_id,
                repetition_penalty=repetition_penalty,
                stop_token_ids=[128258],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ):
                total_bytes += len(chunk)
                total_chunks += 1
                audio_queue.put(chunk)
            if DEBUG_DIR:
                payload = {
                    "request_id": request_id,
                    "prompt": prompt,
                    "voice": voice,
                    "total_bytes": total_bytes,
                    "total_chunks": total_chunks,
                    "elapsed_s": time.time() - started_at,
                }
                if debug_meta:
                    payload.update(debug_meta)
                debug_dir = Path(DEBUG_DIR)
                debug_dir.mkdir(parents=True, exist_ok=True)
                with (debug_dir / f"{request_id}.transport.json").open("w") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
            audio_queue.put({"done": True, "total_bytes": total_bytes})
        except Exception as exc:  # pragma: no cover - transport path only
            audio_queue.put({"error": str(exc)})

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return audio_queue


async def stream_pcm_chunks(
    prompt: str,
    voice: str,
    repetition_penalty: float,
    max_tokens: int,
    temperature: float,
    top_p: float,
    request_id: str,
    debug_meta: dict[str, Any] | None = None,
):
    audio_queue = start_audio_generation(
        prompt=prompt,
        voice=voice,
        repetition_penalty=repetition_penalty,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        request_id=request_id,
        debug_meta=debug_meta,
    )
    while True:
        item = await asyncio.to_thread(audio_queue.get)
        if isinstance(item, dict):
            if item.get("error"):
                raise RuntimeError(item["error"])
            break
        yield item


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "model_name": MODEL_NAME,
            "tokenizer_path": TOKENIZER_PATH,
            "default_voice": DEFAULT_VOICE,
            "transport": {
                "http_tts": "/tts",
                "websocket_tts": "/v1/audio/speech/stream",
            },
        }
    )


@app.get("/")
async def root() -> JSONResponse:
    return JSONResponse(
        {
            "endpoints": ["/health", "/tts", "/v1/audio/speech/stream"],
            "websocket_protocol": {
                "client": [
                    {"type": "session.config", "voice": "Metin", "stream_audio": True, "response_format": "pcm"},
                    {"type": "input.text", "text": "Merhaba dunya. Nasil gidiyor? "},
                    {"type": "input.done"},
                ],
                "server": [
                    {"type": "audio.start", "sentence_index": 0, "sentence_text": "Merhaba dunya.", "format": "pcm", "sample_rate": SAMPLE_RATE},
                    "binary pcm frame(s)",
                    {"type": "audio.done", "sentence_index": 0, "total_bytes": 0, "error": False},
                    {"type": "session.done", "total_sentences": 1},
                ],
            },
        }
    )


@app.api_route("/tts", methods=["GET", "POST"])
async def tts_http(request: Request):
    if request.method == "POST":
        payload = await request.json()
        chosen_text = payload.get("text") or payload.get("prompt") or ""
        chosen_voice = payload.get("speaker") or payload.get("voice") or DEFAULT_VOICE
    else:
        chosen_text = request.query_params.get("text") or request.query_params.get("prompt") or ""
        chosen_voice = request.query_params.get("speaker") or request.query_params.get("voice") or DEFAULT_VOICE

    if not chosen_text.strip():
        return JSONResponse({"error": "text is required"}, status_code=400)

    async def generate_audio_stream():
        yield create_wav_header()
        request_id = f"http-{uuid.uuid4().hex}"
        async for chunk in stream_pcm_chunks(
            prompt=chosen_text,
            voice=chosen_voice,
            repetition_penalty=DEFAULT_REPETITION_PENALTY,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            request_id=request_id,
            debug_meta={"source": "http"},
        ):
            yield chunk

    return StreamingResponse(generate_audio_stream(), media_type="audio/wav")


@app.websocket("/v1/audio/speech/stream")
async def speech_stream(websocket: WebSocket) -> None:
    await websocket.accept()

    config = {
        "voice": DEFAULT_VOICE,
        "stream_audio": True,
        "response_format": "pcm",
        "split_granularity": "sentence",
        "repetition_penalty": DEFAULT_REPETITION_PENALTY,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
    }
    text_buffer = IncrementalTextBuffer()
    ready_chunks: asyncio.Queue[str | None] = asyncio.Queue()
    send_lock = asyncio.Lock()

    async def send_json(payload: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def send_bytes(payload: bytes) -> None:
        async with send_lock:
            await websocket.send_bytes(payload)

    async def receiver() -> None:
        try:
            while True:
                payload = json.loads(await websocket.receive_text())
                msg_type = payload.get("type")

                if msg_type == "session.config":
                    config["voice"] = payload.get("voice") or payload.get("speaker") or config["voice"]
                    config["stream_audio"] = bool(payload.get("stream_audio", True))
                    config["response_format"] = payload.get("response_format", "pcm")
                    config["split_granularity"] = payload.get("split_granularity", "sentence")
                    config["repetition_penalty"] = float(payload.get("repetition_penalty", config["repetition_penalty"]))
                    config["max_tokens"] = int(payload.get("max_tokens", config["max_tokens"]))
                    config["temperature"] = float(payload.get("temperature", config["temperature"]))
                    config["top_p"] = float(payload.get("top_p", config["top_p"]))
                    text_buffer.split_granularity = config["split_granularity"]
                    await send_json(
                        {
                            "type": "session.configured",
                            "voice": config["voice"],
                            "response_format": config["response_format"],
                            "split_granularity": config["split_granularity"],
                            "sample_rate": SAMPLE_RATE,
                        }
                    )
                elif msg_type == "input.text":
                    for chunk in text_buffer.push(payload.get("text", "")):
                        await ready_chunks.put(chunk)
                elif msg_type == "input.done":
                    for chunk in text_buffer.finish():
                        await ready_chunks.put(chunk)
                    await ready_chunks.put(None)
                    return
                else:
                    await send_json({"type": "error", "message": f"unsupported message type: {msg_type}"})
        except WebSocketDisconnect:
            await ready_chunks.put(None)

    async def sender() -> None:
        sentence_index = 0
        total_sentences = 0
        while True:
            chunk = await ready_chunks.get()
            if chunk is None:
                break

            await send_json(
                {
                    "type": "audio.start",
                    "sentence_index": sentence_index,
                    "sentence_text": chunk,
                    "format": config["response_format"],
                    "sample_rate": SAMPLE_RATE,
                }
            )

            total_bytes = 0
            try:
                request_id = f"ws-{sentence_index}-{uuid.uuid4().hex}"
                async for audio_chunk in stream_pcm_chunks(
                    prompt=chunk,
                    voice=config["voice"],
                    repetition_penalty=config["repetition_penalty"],
                    max_tokens=config["max_tokens"],
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    request_id=request_id,
                    debug_meta={
                        "source": "websocket",
                        "sentence_index": sentence_index,
                        "sentence_text": chunk,
                    },
                ):
                    total_bytes += len(audio_chunk)
                    if config["stream_audio"]:
                        await send_bytes(audio_chunk)
                await send_json(
                    {
                        "type": "audio.done",
                        "sentence_index": sentence_index,
                        "total_bytes": total_bytes,
                        "error": False,
                    }
                )
            except Exception as exc:  # pragma: no cover - runtime path only
                await send_json(
                    {
                        "type": "audio.done",
                        "sentence_index": sentence_index,
                        "total_bytes": total_bytes,
                        "error": True,
                        "message": str(exc),
                    }
                )
            sentence_index += 1
            total_sentences += 1

        await send_json({"type": "session.done", "total_sentences": total_sentences})

    await asyncio.gather(receiver(), sender())


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
