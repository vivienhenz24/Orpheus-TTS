"""
Quick inference script for the Turkish pretrained checkpoint.
Generates 5 Turkish sentences and saves them as WAV files.
"""
import os
import sys
import torch
import numpy as np
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC

CHECKPOINT = os.environ.get(
    "CHECKPOINT",
    "/workspace/Orpheus-TTS/pretrain/checkpoints/tr_orpheus_pretrain_stage1/checkpoint-38799",
)
BASE_TOKENIZER = "canopylabs/orpheus-3b-0.1-pretrained"
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/orpheus_tr/inference_samples")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SENTENCES = [
    "Merhaba, nasılsınız?",
    "Bugün hava çok güzel, dışarı çıkmak için mükemmel bir gün.",
    "Türkçe konuşmayı öğrenmek istiyorum, çok ilginç bir dil.",
    "İstanbul, Türkiye'nin en büyük ve en kalabalık şehridir.",
    "Yarın saat üçte parkta buluşup birlikte yürüyüş yapalım.",
]

SNAC_OFFSET = 10
TOKENS_PER_FRAME = 7
CODEBOOK_SIZE = 4096
START_TOKEN = 128259
END_TOKENS = [128009, 128260, 128261, 128257]
AUDIO_EOS_TOKEN_ID = 128258


def format_prompt(text, tokenizer, speaker="Metin"):
    prompt = f"{speaker}: {text}" if speaker else text
    tokens = tokenizer(prompt, return_tensors="pt")
    start = torch.tensor([[START_TOKEN]], dtype=torch.int64)
    end = torch.tensor([END_TOKENS], dtype=torch.int64)
    return torch.cat([start, tokens.input_ids, end], dim=1)


def extract_snac_ids(token_ids, tokenizer):
    ids = []
    count = 0
    for tid in token_ids:
        text = tokenizer.decode([tid]).strip()
        if text.startswith("<custom_token_") and text.endswith(">"):
            try:
                n = int(text[14:-1])
                val = n - SNAC_OFFSET - (count % TOKENS_PER_FRAME) * CODEBOOK_SIZE
                if 0 < val <= CODEBOOK_SIZE:
                    ids.append(val)
                    count += 1
            except ValueError:
                pass
    return ids


def snac_decode(snac_model, token_ids):
    num_frames = len(token_ids) // TOKENS_PER_FRAME
    if num_frames == 0:
        return None
    token_ids = token_ids[: num_frames * TOKENS_PER_FRAME]

    c0, c1, c2 = [], [], []
    for j in range(num_frames):
        i = j * TOKENS_PER_FRAME
        c0.append(token_ids[i])
        c1 += [token_ids[i + 1], token_ids[i + 4]]
        c2 += [token_ids[i + 2], token_ids[i + 3], token_ids[i + 5], token_ids[i + 6]]

    codes = [
        torch.tensor(c0, dtype=torch.int32, device=DEVICE).unsqueeze(0),
        torch.tensor(c1, dtype=torch.int32, device=DEVICE).unsqueeze(0),
        torch.tensor(c2, dtype=torch.int32, device=DEVICE).unsqueeze(0),
    ]
    if any(
        torch.any(c < 0) or torch.any(c > CODEBOOK_SIZE) for c in codes
    ):
        return None

    with torch.inference_mode():
        audio = snac_model.decode(codes)
    return audio.squeeze().cpu().numpy()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading tokenizer from {BASE_TOKENIZER}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    number_add_tokens = 7 * 4096 + 10
    new_tokens = [f"<custom_token_{i}>" for i in range(number_add_tokens + 1)]
    tokenizer.add_tokens(new_tokens)

    print(f"Loading model from {CHECKPOINT}")
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    print("Loading SNAC")
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(DEVICE)

    for i, sentence in enumerate(SENTENCES):
        print(f"\n[{i+1}/5] {sentence}")
        input_ids = format_prompt(sentence, tokenizer).to(DEVICE)

        with torch.inference_mode():
            out = model.generate(
                input_ids,
                max_new_tokens=1200,
                do_sample=True,
                temperature=0.6,
                top_p=0.8,
                repetition_penalty=1.3,
                eos_token_id=AUDIO_EOS_TOKEN_ID,
            )

        generated = out[0][input_ids.shape[1]:].tolist()
        decoded_sample = tokenizer.decode(generated[:50])
        print(f"  first 50 tokens decoded: {repr(decoded_sample[:200])}")
        snac_ids = extract_snac_ids(generated, tokenizer)
        print(f"  generated {len(generated)} tokens → {len(snac_ids)} SNAC ids")

        audio = snac_decode(snac, snac_ids)
        if audio is None:
            print("  decode failed — skipping")
            continue

        path = os.path.join(OUTPUT_DIR, f"sample_{i+1:02d}.wav")
        sf.write(path, audio, samplerate=24000)
        print(f"  saved → {path}")

    print(f"\nDone. Files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
