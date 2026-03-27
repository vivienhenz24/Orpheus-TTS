import argparse
import concurrent.futures
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


import soundfile as sf
import torch
import torchaudio.transforms as T
from datasets import Audio, Dataset, DatasetDict, Features, Sequence, Value, load_dataset, load_from_disk
from snac import SNAC
from transformers import AutoTokenizer


TOKENIZER_LENGTH = 128256
END_OF_TEXT = 128009
START_OF_SPEECH = TOKENIZER_LENGTH + 1
END_OF_SPEECH = TOKENIZER_LENGTH + 2
START_OF_HUMAN = TOKENIZER_LENGTH + 3
END_OF_HUMAN = TOKENIZER_LENGTH + 4
START_OF_AI = TOKENIZER_LENGTH + 5
END_OF_AI = TOKENIZER_LENGTH + 6
PAD_TOKEN = TOKENIZER_LENGTH + 7
AUDIO_TOKEN_BASE = TOKENIZER_LENGTH + 10


def interleave_and_offset(codes_l0, codes_l1, codes_l2):
    tokens = []
    n_frames = min(len(codes_l0), len(codes_l1) // 2, len(codes_l2) // 4)
    for i in range(n_frames):
        tokens.extend(
            [
                int(codes_l0[i]) + AUDIO_TOKEN_BASE,
                int(codes_l1[2 * i]) + AUDIO_TOKEN_BASE + 4096,
                int(codes_l2[4 * i]) + AUDIO_TOKEN_BASE + (2 * 4096),
                int(codes_l2[4 * i + 1]) + AUDIO_TOKEN_BASE + (3 * 4096),
                int(codes_l1[2 * i + 1]) + AUDIO_TOKEN_BASE + (4 * 4096),
                int(codes_l2[4 * i + 2]) + AUDIO_TOKEN_BASE + (5 * 4096),
                int(codes_l2[4 * i + 3]) + AUDIO_TOKEN_BASE + (6 * 4096),
            ]
        )
    return tokens


def load_source_dataset(dataset_name_or_path, split, token):
    path = Path(dataset_name_or_path)
    if path.exists():
        dataset = load_from_disk(str(path))
        if isinstance(dataset, DatasetDict):
            return dataset[split]
        return dataset
    return load_dataset(dataset_name_or_path, split=split, token=token)


def load_audio(audio_dict):
    audio_bytes = audio_dict.get("bytes")
    audio_path = audio_dict.get("path")

    if audio_bytes is not None:
        waveform_np, sample_rate = sf.read(io.BytesIO(audio_bytes), always_2d=False)
    elif audio_path:
        waveform_np, sample_rate = sf.read(audio_path, always_2d=False)
    else:
        raise ValueError("Audio example does not contain bytes or path.")

    waveform = torch.tensor(waveform_np, dtype=torch.float32)
    return waveform, int(sample_rate)


def normalize_waveform(audio_dict):
    waveform, sample_rate = load_audio(audio_dict)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 2:
        waveform = waveform.transpose(0, 1)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    peak = waveform.abs().max()
    if peak > 1.0:
        waveform = waveform / peak

    return waveform, sample_rate


def remove_duplicate_frames(audio_tokens):
    if len(audio_tokens) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")
    if not audio_tokens:
        return audio_tokens

    result = audio_tokens[:7]
    for i in range(7, len(audio_tokens), 7):
        current_first = audio_tokens[i]
        previous_first = result[-7]
        if current_first != previous_first:
            result.extend(audio_tokens[i : i + 7])
    return result


def render_prompt(template, text, speaker):
    if not template:
        return text
    return template.replace("{speaker_id}", speaker).replace("{text}", text)


def prepare_example_audio(example, args):
    text = str(example[args.text_column]).strip()
    speaker_value = example.get(args.speaker_column, args.default_speaker)
    speaker = str(speaker_value).strip() if speaker_value is not None else args.default_speaker
    if not text or not speaker:
        return None

    waveform, sample_rate = normalize_waveform(example[args.audio_column])
    duration_s = waveform.shape[-1] / sample_rate
    if duration_s < args.min_duration_s or duration_s > args.max_duration_s:
        return None

    resample_transform = T.Resample(orig_freq=sample_rate, new_freq=args.target_sample_rate)
    waveform = resample_transform(waveform)
    return {
        "waveform": waveform,
        "text": text,
        "speaker_id": speaker,
        "duration_s": float(duration_s),
    }


def build_processor(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, token=args.hf_token)
    snac_model = SNAC.from_pretrained(args.snac_model).to(device).eval()

    def process_prepared(prepared):
        if prepared is None:
            return None

        waveform = prepared["waveform"].unsqueeze(0).to(device)
        with torch.inference_mode():
            codes = snac_model.encode(waveform)

        codes_l0 = codes[0].squeeze(0).detach().cpu().tolist()
        codes_l1 = codes[1].squeeze(0).detach().cpu().tolist()
        codes_l2 = codes[2].squeeze(0).detach().cpu().tolist()
        audio_tokens = remove_duplicate_frames(interleave_and_offset(codes_l0, codes_l1, codes_l2))
        if not audio_tokens:
            return None

        text = prepared["text"]
        speaker = prepared["speaker_id"]

        if args.phonemize:
            from phonemize import phonemize as ph
            phonemized = ph(text, lang=args.phonemize_lang)
            prompt = render_prompt(args.prompt_template, phonemized, speaker)
        else:
            prompt = render_prompt(args.prompt_template, text, speaker)

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_ids.append(END_OF_TEXT)
        input_ids = [START_OF_HUMAN] + prompt_ids + [END_OF_HUMAN] + [START_OF_AI] + [START_OF_SPEECH] + audio_tokens + [END_OF_SPEECH] + [END_OF_AI]

        if len(input_ids) > args.max_seq_len:
            return None

        prompt_length = 1 + len(prompt_ids) + 1  # START_OF_HUMAN + prompt_ids + END_OF_HUMAN
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "speaker_id": speaker,
            "text": text,
            "duration_s": prepared["duration_s"],
            "audio_token_count": len(audio_tokens),
            "sequence_length": len(input_ids),
        }

    return process_prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="canopylabs/orpheus-tts-0.1-pretrained")
    parser.add_argument("--tokenizer-name", default="canopylabs/orpheus-3b-0.1-pretrained")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--audio-column", default="audio")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--speaker-column", default="speaker_id")
    parser.add_argument("--default-speaker", default="speaker")
    parser.add_argument("--phonemize", action="store_true", help="Convert text to phoneme tokens via eSpeak-NG")
    parser.add_argument("--phonemize-lang", default="tr", help="eSpeak-NG language code (default: tr)")
    parser.add_argument("--prompt-template", default=None)
    parser.add_argument("--target-sample-rate", type=int, default=24000)
    parser.add_argument("--min-duration-s", type=float, default=0.5)
    parser.add_argument("--max-duration-s", type=float, default=20.0)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--snac-model", default="hubertsiuzdak/snac_24khz")
    parser.add_argument("--device", default=None)
    parser.add_argument("--prefetch-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_dataset = load_source_dataset(args.dataset, args.split, args.hf_token)
    raw_dataset = raw_dataset.cast_column(args.audio_column, Audio(decode=False))

    required_columns = {args.audio_column, args.text_column}
    missing = [col for col in required_columns if col not in raw_dataset.column_names]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns {missing}. "
            f"Expected README-style source format with at least '{args.audio_column}' and '{args.text_column}'."
        )

    process_prepared = build_processor(args)
    features = Features(
        {
            "input_ids": Sequence(Value("int32")),
            "labels": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
            "speaker_id": Value("string"),
            "text": Value("string"),
            "duration_s": Value("float32"),
            "audio_token_count": Value("int32"),
            "sequence_length": Value("int32"),
        }
    )

    stats = {
        "source_dataset": args.dataset,
        "split": args.split,
        "target_sample_rate": args.target_sample_rate,
        "kept_examples": 0,
        "dropped_examples": 0,
        "speakers": {},
    }

    processed_rows = []
    total_examples = len(raw_dataset)
    started_at = time.time()
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.prefetch_workers)) as executor:
        future_to_index = {}
        next_index = 0
        max_in_flight = max(1, args.prefetch_workers * args.prefetch_factor)

        while next_index < total_examples and len(future_to_index) < max_in_flight:
            future = executor.submit(prepare_example_audio, raw_dataset[next_index], args)
            future_to_index[future] = next_index
            next_index += 1

        while future_to_index:
            done, _ = concurrent.futures.wait(
                future_to_index,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                future_to_index.pop(future)
                prepared = future.result()
                processed = process_prepared(prepared)
                completed += 1

                if processed is None:
                    stats["dropped_examples"] += 1
                else:
                    stats["kept_examples"] += 1
                    speaker = processed["speaker_id"]
                    speaker_stats = stats["speakers"].setdefault(speaker, {"examples": 0, "hours": 0.0})
                    speaker_stats["examples"] += 1
                    speaker_stats["hours"] += processed["duration_s"] / 3600.0
                    processed_rows.append(processed)

                if completed % max(1, args.progress_every) == 0 or completed == total_examples:
                    elapsed = max(1e-6, time.time() - started_at)
                    rate = completed / elapsed
                    eta_s = (total_examples - completed) / rate if rate > 0 else 0.0
                    print(
                        f"[prep] {completed}/{total_examples} "
                        f"kept={stats['kept_examples']} dropped={stats['dropped_examples']} "
                        f"rate={rate:.2f} ex/s eta={eta_s/60:.1f}m",
                        flush=True,
                    )

                while next_index < total_examples and len(future_to_index) < max_in_flight:
                    future = executor.submit(prepare_example_audio, raw_dataset[next_index], args)
                    future_to_index[future] = next_index
                    next_index += 1

    processed_dataset = Dataset.from_list(processed_rows, features=features)
    processed_dataset.save_to_disk(str(output_dir))

    lengths = processed_dataset["sequence_length"] if len(processed_dataset) else []
    if lengths:
        stats["max_sequence_length"] = max(lengths)
        stats["avg_sequence_length"] = sum(lengths) / len(lengths)
    else:
        stats["max_sequence_length"] = 0
        stats["avg_sequence_length"] = 0.0

    with (output_dir / "prep_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
