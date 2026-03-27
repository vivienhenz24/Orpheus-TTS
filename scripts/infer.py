#!/usr/bin/env python3
"""
Local inference with a LoRA-finetuned Orpheus checkpoint.

Usage:
    python scripts/infer.py \
        --checkpoint artifacts/checkpoints/checkpoint-4506 \
        --text "Merhaba, nasılsınız?" \
        --speaker "female_speaker" \
        --output output.wav
"""
import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
END_OF_AI = 128262
PAD_TOKEN = 128263
END_OF_TEXT = 128009

AUDIO_TOKEN_BASE = 128266
LAYER_1_OFFSET = AUDIO_TOKEN_BASE + 4096
LAYER_2_OFFSET = AUDIO_TOKEN_BASE + 8192


def deinterleave(token_ids):
    codes_l0, codes_l1, codes_l2 = [], [], []
    for i in range(0, len(token_ids), 7):
        if i + 7 > len(token_ids):
            break
        f = token_ids[i : i + 7]
        codes_l0.append(f[0] - AUDIO_TOKEN_BASE)
        codes_l1.append(f[1] - LAYER_1_OFFSET)
        codes_l2.append(f[2] - LAYER_2_OFFSET)
        codes_l2.append(f[3] - LAYER_2_OFFSET - 4096)
        codes_l1.append(f[4] - LAYER_1_OFFSET - 12288)
        codes_l2.append(f[5] - LAYER_2_OFFSET - 12288)
        codes_l2.append(f[6] - LAYER_2_OFFSET - 16384)
    return codes_l0, codes_l1, codes_l2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to LoRA checkpoint directory")
    parser.add_argument("--base-model", default="canopylabs/orpheus-tts-0.1-pretrained")
    parser.add_argument("--tokenizer", default="canopylabs/orpheus-3b-0.1-pretrained")
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker", default="female_speaker")
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument("--snac-model", default="hubertsiuzdak/snac_24khz")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"  vocab size: {tokenizer.vocab_size}")

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32,
    )
    print(f"  base model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    print(f"Loading LoRA adapter from {args.checkpoint}...")
    model = PeftModel.from_pretrained(model, args.checkpoint)
    print("  merging LoRA weights into base model...")
    model = model.merge_and_unload()
    model = model.to(device).eval()
    print(f"  merged model params: {sum(p.numel() for p in model.parameters()):,}")

    prompt = args.text
    emb_size = model.get_input_embeddings().weight.shape[0]
    print(f"\nEmbedding table size: {emb_size}")
    print(f"Max token ID we use: {LAYER_2_OFFSET + 16384} (should be < {emb_size})")
    print(f"\nPrompt: {prompt!r}")
    text_ids = tokenizer.encode(prompt, add_special_tokens=True)
    text_ids.append(END_OF_TEXT)
    input_ids = [START_OF_HUMAN] + text_ids + [END_OF_HUMAN] + [START_OF_AI] + [START_OF_SPEECH]
    print(f"Input token count: {len(input_ids)}")
    print(f"Input token IDs: {input_ids}")
    print(f"Max input token ID: {max(input_ids)}")
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    print("\nGenerating audio tokens...")
    with torch.inference_mode():
        output = model.generate(
            input_tensor,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=END_OF_AI,
            pad_token_id=PAD_TOKEN,
        )

    generated = output[0][len(input_ids):].detach().cpu().tolist()
    print(f"Generated {len(generated)} tokens total")
    print(f"First 14 tokens: {generated[:14]}")
    print(f"Last 14 tokens: {generated[-14:]}")
    print(f"Token range: min={min(generated)}, max={max(generated)}")

    # Check for key special tokens
    for name, tok_id in [("END_OF_SPEECH", END_OF_SPEECH), ("END_OF_AI", END_OF_AI),
                          ("START_OF_SPEECH", START_OF_SPEECH), ("PAD", PAD_TOKEN)]:
        count = generated.count(tok_id)
        if count:
            print(f"  {name} ({tok_id}) appears {count} times, first at pos {generated.index(tok_id)}")

    # Check how many tokens are in valid audio range
    # Max valid token: AUDIO_TOKEN_BASE + 6*4096 + 4095 = 156937
    MAX_AUDIO_TOKEN = AUDIO_TOKEN_BASE + 6 * 4096 + 4095
    audio_tokens = [t for t in generated if AUDIO_TOKEN_BASE <= t <= MAX_AUDIO_TOKEN]
    non_audio = [t for t in generated if t < AUDIO_TOKEN_BASE or t > MAX_AUDIO_TOKEN]
    print(f"Valid audio tokens: {len(audio_tokens)}/{len(generated)}")
    if non_audio:
        print(f"Non-audio tokens in output: {non_audio[:20]}")

    if END_OF_SPEECH in generated:
        eos_pos = generated.index(END_OF_SPEECH)
        print(f"END_OF_SPEECH found at position {eos_pos}")
        generated = generated[:eos_pos]
    else:
        print("WARNING: END_OF_SPEECH not found — output may be truncated")
    print(f"Audio tokens after trim: {len(generated)}")

    if len(generated) < 7:
        print("Error: generated sequence too short.", file=sys.stderr)
        sys.exit(1)

    print("\nDecoding audio with SNAC...")
    from snac import SNAC
    snac = SNAC.from_pretrained(args.snac_model).to(device).eval()

    codes_l0, codes_l1, codes_l2 = deinterleave(generated)
    print(f"  layer 0 frames: {len(codes_l0)}, layer 1: {len(codes_l1)}, layer 2: {len(codes_l2)}")
    print(f"  layer 0 range: min={min(codes_l0)}, max={max(codes_l0)}")
    print(f"  layer 1 range: min={min(codes_l1)}, max={max(codes_l1)}")
    print(f"  layer 2 range: min={min(codes_l2)}, max={max(codes_l2)}")
    invalid_l0 = [c for c in codes_l0 if not (0 <= c <= 4095)]
    invalid_l1 = [c for c in codes_l1 if not (0 <= c <= 4095)]
    invalid_l2 = [c for c in codes_l2 if not (0 <= c <= 4095)]
    if invalid_l0 or invalid_l1 or invalid_l2:
        print(f"  WARNING: invalid codes — l0:{len(invalid_l0)}, l1:{len(invalid_l1)}, l2:{len(invalid_l2)}")
    codes = [
        torch.tensor([codes_l0], dtype=torch.long, device=device),
        torch.tensor([codes_l1], dtype=torch.long, device=device),
        torch.tensor([codes_l2], dtype=torch.long, device=device),
    ]
    with torch.inference_mode():
        audio = snac.decode(codes)

    audio_np = audio.squeeze().detach().cpu().numpy()
    duration = len(audio_np) / 24000
    print(f"  audio shape: {audio_np.shape}, duration: {duration:.2f}s")
    sf.write(args.output, audio_np, 24000)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
