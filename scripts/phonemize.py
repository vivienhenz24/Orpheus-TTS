"""
Turkish phonemizer using eSpeak-NG.

Converts Turkish text to a sequence of phoneme tokens that can be added
to the tokenizer as special tokens, giving the model 1 token per phoneme
and eliminating all Turkish character fragmentation issues.

Phoneme token format: <|ph_X|>
"""
import re
import subprocess
from functools import lru_cache

# ── Phoneme token definitions ──────────────────────────────────────────────────
# Map IPA symbol → token string. Order matters: digraphs must come first.
IPA_TO_TOKEN = {
    # Digraphs (must be checked before single chars)
    "dʒ": "<|ph_dj|>",
    "tʃ": "<|ph_ch|>",
    # Vowels — special IPA
    "æ":  "<|ph_ae|>",
    "ø":  "<|ph_oe|>",
    "œ":  "<|ph_OE|>",
    "ɔ":  "<|ph_O|>",
    "ɛ":  "<|ph_E|>",
    "ɪ":  "<|ph_I|>",
    "ɯ":  "<|ph_ih|>",
    "ʊ":  "<|ph_U|>",
    # Consonants — special IPA
    "ɟ":  "<|ph_gj|>",
    "ɡ":  "<|ph_g|>",
    "ɫ":  "<|ph_L|>",
    "ɾ":  "<|ph_R|>",
    "ʃ":  "<|ph_sh|>",
    "ʒ":  "<|ph_zh|>",
    # Prosody markers
    "ˈ":  "<|ph_stress|>",
    "ː":  "<|ph_long|>",
    # ASCII phonemes — kept as explicit tokens for consistency
    "a":  "<|ph_a|>",
    "b":  "<|ph_b|>",
    "c":  "<|ph_c|>",
    "d":  "<|ph_d|>",
    "e":  "<|ph_e|>",
    "f":  "<|ph_f|>",
    "h":  "<|ph_h|>",
    "i":  "<|ph_i|>",
    "j":  "<|ph_j|>",
    "k":  "<|ph_k|>",
    "l":  "<|ph_l|>",
    "m":  "<|ph_m|>",
    "n":  "<|ph_n|>",
    "o":  "<|ph_o|>",
    "p":  "<|ph_p|>",
    "r":  "<|ph_r|>",
    "s":  "<|ph_s|>",
    "t":  "<|ph_t|>",
    "u":  "<|ph_u|>",
    "v":  "<|ph_v|>",
    "y":  "<|ph_y|>",
    "z":  "<|ph_z|>",
    # Word boundary
    " ":  "<|ph_sp|>",
}

# All tokens as a list (for adding to tokenizer)
PHONEME_TOKENS = sorted(set(IPA_TO_TOKEN.values()))

# Build regex for parsing IPA — digraphs first
_IPA_PATTERN = re.compile(
    "|".join(re.escape(k) for k in sorted(IPA_TO_TOKEN, key=len, reverse=True))
)


def ipa_to_tokens(ipa: str) -> str:
    """Convert a raw IPA string from eSpeak to a phoneme token sequence."""
    # Collapse multiple spaces
    ipa = re.sub(r" +", " ", ipa.strip())
    result = []
    i = 0
    while i < len(ipa):
        matched = False
        # Try digraphs first (longer matches)
        for sym in sorted(IPA_TO_TOKEN, key=len, reverse=True):
            if ipa[i:i+len(sym)] == sym:
                result.append(IPA_TO_TOKEN[sym])
                i += len(sym)
                matched = True
                break
        if not matched:
            # Skip unknown chars silently (punctuation, etc.)
            i += 1
    return "".join(result)


@lru_cache(maxsize=4096)
def phonemize(text: str, lang: str = "tr") -> str:
    """
    Phonemize text using eSpeak-NG and return a phoneme token string.
    Result is space-joined phoneme tokens, e.g.:
      '<|ph_l|> <|ph_stress|> <|ph_oe|> <|ph_t|> ...'
    """
    result = subprocess.run(
        ["espeak-ng", "-v", lang, "--ipa", "-q"],
        input=text,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"eSpeak-NG failed: {result.stderr.strip()}")
    return ipa_to_tokens(result.stdout)


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Lütfen bu numarayı kayıt edin."
    ipa_raw = subprocess.run(
        ["espeak-ng", "-v", "tr", "--ipa", "-q"],
        input=text, capture_output=True, text=True
    ).stdout.strip()
    tokens = phonemize(text)
    print(f"Text:   {text}")
    print(f"IPA:    {ipa_raw}")
    print(f"Tokens: {tokens}")
    print(f"Count:  {len(tokens.split())}")
    print(f"\nAll phoneme tokens ({len(PHONEME_TOKENS)}):")
    print(PHONEME_TOKENS)
