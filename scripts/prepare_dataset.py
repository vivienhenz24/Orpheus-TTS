#!/usr/bin/env python3
"""
Single-pass dataset preparation: raw audio parquet -> tokenized Orpheus parquet.
No intermediate WAV files. Parallel shard decoding feeds a GPU SNAC batching loop.
"""

import argparse
import io
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
from snac import SNAC
from transformers import AutoTokenizer

START_TOKEN_ID = 128259
END_PROMPT_TOKEN_IDS = [128009, 128260, 128261, 128257]
AUDIO_EOS_TOKEN_ID = 128258
PAD_TOKEN_ID = 128263
SNAC_PAD_MULTIPLE = 2048


def decode_shard(parquet_path, text_by_file):
    print(f"  [shard] reading {parquet_path.name}...", flush=True)
    table = pq.read_table(parquet_path, columns=["filepath", "audio", "speaker", "speaker_id"])
    print(f"  [shard] {parquet_path.name} read ({len(table)} rows), decoding audio...", flush=True)
    samples = []
    for row in table.to_pylist():
        text = text_by_file.get(row["filepath"])
        if text is None:
            continue
        try:
            data, sr = sf.read(io.BytesIO(row["audio"]), dtype="float32", always_2d=False)
        except Exception as e:
            print(f"  [warn] decode failed {row['filepath']}: {e}", flush=True)
            continue
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 24000:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=24000)
        samples.append({
            "filepath": row["filepath"],
            "speaker": row["speaker"],
            "speaker_id": row["speaker_id"],
            "text": text,
            "audio": data,
        })
    print(f"  [shard] {parquet_path.name} done ({len(samples)} samples decoded)", flush=True)
    return samples


def format_prompt_ids(tokenizer, speaker, text):
    prompt = f"{speaker}: {text}" if speaker else text
    tokens = tokenizer(prompt, return_tensors="pt").input_ids
    return torch.cat([
        torch.tensor([[START_TOKEN_ID]]),
        tokens,
        torch.tensor([END_PROMPT_TOKEN_IDS]),
    ], dim=1)[0].tolist()


def expected_code_lengths(n_samples):
    padded = math.ceil(n_samples / SNAC_PAD_MULTIPLE) * SNAC_PAD_MULTIPLE
    return padded // 2048, padded // 1024, padded // 512


def interleave_codes(codes, idx, code_lengths, base_id):
    n0, n1, n2 = code_lengths
    c0 = codes[0][idx, :n0].tolist()
    c1 = codes[1][idx, :n1].tolist()
    c2 = codes[2][idx, :n2].tolist()
    flat = []
    for i in range(len(c0)):
        for j, code in enumerate([c0[i], c1[2*i], c2[4*i], c2[4*i+1], c1[2*i+1], c2[4*i+2], c2[4*i+3]]):
            flat.append(int(base_id + code + 10 + j * 4096))
    return flat


def encode_batch(batch, snac_model, tokenizer, base_id, max_length, device):
    audio_tensors = [torch.tensor(s["audio"], dtype=torch.float32) for s in batch]
    code_lengths = [expected_code_lengths(t.shape[0]) for t in audio_tensors]
    max_len = max(t.shape[0] for t in audio_tensors)
    padded = [torch.nn.functional.pad(t, (0, max_len - t.shape[0])) for t in audio_tensors]
    audio_batch = torch.stack(padded).unsqueeze(1).to(device)

    print(f"  [encode] batch={len(batch)} audio_shape={audio_batch.shape} device={audio_batch.device}", flush=True)
    with torch.inference_mode():
        codes = snac_model.encode(audio_batch)
    print(f"  [encode] done", flush=True)

    out, skipped = [], 0
    for i, s in enumerate(batch):
        prompt_ids = format_prompt_ids(tokenizer, s["speaker"], s["text"])
        audio_ids = interleave_codes(codes, i, code_lengths[i], base_id)
        input_ids = prompt_ids + audio_ids + [AUDIO_EOS_TOKEN_ID]
        if max_length and len(input_ids) > max_length:
            skipped += 1
            continue
        out.append({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "speaker": s["speaker"],
            "filename": s["filepath"],
            "seq_len": len(input_ids),
        })
    return out, skipped


def write_shard(rows, path):
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="canopylabs/orpheus-3b-0.1-pretrained")
    parser.add_argument("--snac-model", default="hubertsiuzdak/snac_24khz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--write-shard-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    shards_dir = output_dir / "parquet"
    shards_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(raw_dir / "metadata.csv", sep="|").drop_duplicates(subset=["filename"])
    text_by_file = dict(zip(metadata["filename"], metadata["text"]))

    parquet_files = sorted((raw_dir / "audio_parquet").glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet shards in {raw_dir / 'audio_parquet'}")
    if args.limit:
        parquet_files = parquet_files[:args.limit]

    print(f"  {len(parquet_files)} input shards | {len(text_by_file)} metadata rows | workers={args.num_workers}", flush=True)

    print(f"  loading tokenizer from {args.model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"  tokenizer loaded", flush=True)
    base_id = tokenizer.convert_tokens_to_ids("<custom_token_0>")
    print(f"  custom_token_0 id={base_id}", flush=True)

    print(f"  loading SNAC from {args.snac_model}...", flush=True)
    snac_model = SNAC.from_pretrained(args.snac_model)
    print(f"  SNAC loaded, moving to {args.device}...", flush=True)
    snac_model = snac_model.eval().to(args.device)
    print(f"  SNAC on {args.device}", flush=True)

    stats = {"rows_seen": 0, "rows_written": 0, "rows_skipped_length": 0}
    pending, batch_buf, shard_idx = [], [], 0
    n_total = len(parquet_files)
    milestone = max(1, n_total // 10)

    def flush_batch():
        nonlocal shard_idx
        if not batch_buf:
            return
        out_rows, skipped = encode_batch(batch_buf, snac_model, tokenizer, base_id, args.max_length, args.device)
        stats["rows_skipped_length"] += skipped
        stats["rows_written"] += len(out_rows)
        pending.extend(out_rows)
        batch_buf.clear()
        while len(pending) >= args.write_shard_size:
            wp = shards_dir / f"train-{shard_idx:05d}.parquet"
            write_shard(pending[:args.write_shard_size], wp)
            del pending[:args.write_shard_size]
            shard_idx += 1

    # Submit shards in chunks to bound memory usage
    chunk_size = args.num_workers * 2
    done = 0
    print(f"  starting parallel decode, chunk_size={chunk_size}", flush=True)
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        for chunk_start in range(0, n_total, chunk_size):
            chunk = parquet_files[chunk_start:chunk_start + chunk_size]
            futures = {pool.submit(decode_shard, p, text_by_file): p for p in chunk}
            for future in as_completed(futures):
                samples = future.result()
                stats["rows_seen"] += len(samples)
                done += 1
                for s in samples:
                    batch_buf.append(s)
                    if len(batch_buf) >= args.batch_size:
                        flush_batch()
                if done % milestone == 0 or done == n_total:
                    pct = 100 * done // n_total
                    print(f"  [{pct:3d}%] {done}/{n_total} shards | {stats['rows_written']} rows tokenized", flush=True)

    flush_batch()
    if pending:
        write_shard(pending, shards_dir / f"train-{shard_idx:05d}.parquet")

    with (output_dir / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
