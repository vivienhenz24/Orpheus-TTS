#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

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


def load_manifest(path, limit):
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            if limit and idx >= limit:
                break
            row["frames"] = int(float(row["frames"]))
            row["duration_sec"] = float(row["duration_sec"])
            rows.append(row)
    return rows


def format_prompt_ids(tokenizer, voice, text):
    prompt = f"{voice}: {text}" if voice else text
    prompt_tokens = tokenizer(prompt, return_tensors="pt")
    all_ids = torch.cat(
        [
            torch.tensor([[START_TOKEN_ID]], dtype=torch.int64),
            prompt_tokens.input_ids,
            torch.tensor([END_PROMPT_TOKEN_IDS], dtype=torch.int64),
        ],
        dim=1,
    )
    return all_ids[0].tolist()


def padded_audio_frames(num_samples):
    return math.ceil(num_samples / SNAC_PAD_MULTIPLE) * SNAC_PAD_MULTIPLE


def expected_code_lengths(num_samples):
    padded = padded_audio_frames(num_samples)
    return padded // 2048, padded // 1024, padded // 512


def interleave_codes_for_sample(codes, sample_idx, code_lengths, custom_token_0_id):
    n0, n1, n2 = code_lengths
    code0 = codes[0][sample_idx, :n0].tolist()
    code1 = codes[1][sample_idx, :n1].tolist()
    code2 = codes[2][sample_idx, :n2].tolist()

    if len(code1) != len(code0) * 2 or len(code2) != len(code0) * 4:
        raise ValueError(
            f"Unexpected SNAC code lengths for sample {sample_idx}: "
            f"{len(code0)}, {len(code1)}, {len(code2)}"
        )

    flat = []
    for i in range(len(code0)):
        frame = [
            code0[i],
            code1[2 * i],
            code2[4 * i],
            code2[4 * i + 1],
            code1[2 * i + 1],
            code2[4 * i + 2],
            code2[4 * i + 3],
        ]
        for j, code in enumerate(frame):
            if code < 0 or code > 4096:
                raise ValueError(f"SNAC code out of range: {code}")
            flat.append(int(custom_token_0_id + code + 10 + j * 4096))
    return flat


def read_audio(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 24000:
        raise ValueError(f"Expected 24k audio, got {sr} for {path}")
    return torch.tensor(audio, dtype=torch.float32)


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def write_parquet_shard(rows, out_path):
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path, compression="zstd")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="canopylabs/orpheus-3b-0.1-pretrained")
    parser.add_argument("--snac-model", default="hubertsiuzdak/snac_24khz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--write-shard-size", type=int, default=512)
    parser.add_argument("--sort-by-length", action="store_true", default=True)
    parser.add_argument("--no-sort-by-length", dest="sort_by_length", action="store_false")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "parquet"
    shards_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest(args.manifest, args.limit)
    if args.sort_by_length:
        manifest_rows.sort(key=lambda r: r["frames"])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    custom_token_0_id = tokenizer.convert_tokens_to_ids("<custom_token_0>")
    snac_model = SNAC.from_pretrained(args.snac_model).eval().to(args.device)

    stats = {
        "rows_seen": len(manifest_rows),
        "rows_written": 0,
        "rows_skipped_over_length": 0,
        "output_dir": str(output_dir),
        "batch_size": args.batch_size,
        "write_shard_size": args.write_shard_size,
        "max_seq_len": 0,
        "min_seq_len": None,
        "total_seq_len": 0,
        "pad_token_id": PAD_TOKEN_ID,
        "audio_eos_token_id": AUDIO_EOS_TOKEN_ID,
        "parquet_files": [],
    }

    pending_rows = []
    shard_idx = 0
    total_rows = len(manifest_rows)
    milestone = max(1, total_rows // 10)

    for batch in chunked(manifest_rows, args.batch_size):
        audio_tensors = [read_audio(row["wav_path"]) for row in batch]
        code_lengths = [expected_code_lengths(t.shape[0]) for t in audio_tensors]
        max_len = max(t.shape[0] for t in audio_tensors)

        padded = []
        for t in audio_tensors:
            if t.shape[0] < max_len:
                t = torch.nn.functional.pad(t, (0, max_len - t.shape[0]))
            padded.append(t)

        audio_batch = torch.stack(padded, dim=0).unsqueeze(1).to(args.device)
        with torch.inference_mode():
            codes = snac_model.encode(audio_batch)

        for sample_idx, row in enumerate(batch):
            prompt_ids = format_prompt_ids(tokenizer, row["speaker"], row["text"])
            audio_ids = interleave_codes_for_sample(
                codes,
                sample_idx,
                code_lengths[sample_idx],
                custom_token_0_id,
            )
            input_ids = prompt_ids + audio_ids + [AUDIO_EOS_TOKEN_ID]
            if args.max_length and len(input_ids) > args.max_length:
                stats["rows_skipped_over_length"] += 1
                continue

            seq_len = len(input_ids)
            stats["rows_written"] += 1
            stats["max_seq_len"] = max(stats["max_seq_len"], seq_len)
            stats["min_seq_len"] = seq_len if stats["min_seq_len"] is None else min(stats["min_seq_len"], seq_len)
            stats["total_seq_len"] += seq_len

            pending_rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * seq_len,
                    "speaker": row["speaker"],
                    "filename": row["filename"],
                    "seq_len": seq_len,
                }
            )

            if stats["rows_written"] % milestone == 0:
                pct = 100 * stats["rows_written"] // total_rows
                print(f"  [{pct:3d}%] {stats['rows_written']}/{total_rows} rows tokenized", flush=True)

            if len(pending_rows) >= args.write_shard_size:
                shard_path = shards_dir / f"train-{shard_idx:05d}.parquet"
                write_parquet_shard(pending_rows, shard_path)
                stats["parquet_files"].append(str(shard_path))
                pending_rows = []
                shard_idx += 1

    if pending_rows:
        shard_path = shards_dir / f"train-{shard_idx:05d}.parquet"
        write_parquet_shard(pending_rows, shard_path)
        stats["parquet_files"].append(str(shard_path))

    stats["min_seq_len"] = stats["min_seq_len"] or 0
    stats["mean_seq_len"] = (stats["total_seq_len"] / stats["rows_written"]) if stats["rows_written"] else 0
    del stats["total_seq_len"]

    with (output_dir / "tokenization_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    with (output_dir / "dataset_pointer.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "format": "parquet",
                "data_dir": str(shards_dir),
                "num_shards": len(stats["parquet_files"]),
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
