"""
Extend the Orpheus tokenizer with phoneme tokens and warm-start a checkpoint.

Steps:
  1. Load base tokenizer + add 41 phoneme special tokens
  2. Save extended tokenizer
  3. Load existing LoRA checkpoint
  4. Resize embed_tokens + lm_head for new tokens
  5. Initialize new token embeddings as mean of constituent byte-token embeddings
  6. Save warm-started checkpoint

Usage:
    python scripts/extend_tokenizer.py \
        --checkpoint checkpoints/turkish-finetune-final/final \
        --output-tokenizer artifacts/tokenizer_phoneme \
        --output-checkpoint checkpoints/turkish-phoneme-warmstart
"""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from phonemize import PHONEME_TOKENS, phonemize


def mean_embedding_for_token(token_str: str, tokenizer, embeddings: torch.Tensor) -> torch.Tensor:
    """
    Initialize a new token embedding as the mean of the embeddings of its
    constituent subword tokens (using the original tokenizer).
    Falls back to random normal if the token encodes to nothing useful.
    """
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    if not ids:
        return embeddings.mean(dim=0)
    vecs = embeddings[ids]
    return vecs.mean(dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Existing LoRA adapter dir (e.g. checkpoints/turkish-finetune-final/final)")
    parser.add_argument("--base-model", default="canopylabs/orpheus-tts-0.1-pretrained")
    parser.add_argument("--tokenizer-name", default="canopylabs/orpheus-3b-0.1-pretrained")
    parser.add_argument("--output-tokenizer", required=True, help="Where to save extended tokenizer")
    parser.add_argument("--output-checkpoint", required=True, help="Where to save warm-started checkpoint")
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    output_tok_dir = Path(args.output_tokenizer)
    output_ckpt_dir = Path(args.output_checkpoint)
    output_tok_dir.mkdir(parents=True, exist_ok=True)
    output_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Extend tokenizer ───────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, token=args.hf_token)
    old_vocab_size = len(tokenizer)
    print(f"  Original vocab size: {old_vocab_size}")

    num_added = tokenizer.add_special_tokens({"additional_special_tokens": PHONEME_TOKENS})
    new_vocab_size = len(tokenizer)
    print(f"  Added {num_added} phoneme tokens → new vocab size: {new_vocab_size}")

    tokenizer.save_pretrained(str(output_tok_dir))
    print(f"  Saved extended tokenizer to {output_tok_dir}")

    # Verify a sample
    sample = phonemize("Lütfen bu numarayı kayıt edin.")
    ids = tokenizer.encode(sample, add_special_tokens=False)
    print(f"\n  Sample phoneme token count: {len(ids)} (should equal phoneme count)")
    print(f"  Sample: {sample[:80]}...")

    # ── 2. Load base model + LoRA ─────────────────────────────────────────────
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        token=args.hf_token,
        torch_dtype=torch.float32,  # float32 for safe embedding init
    )

    print(f"Loading LoRA adapter from {args.checkpoint}...")
    model = PeftModel.from_pretrained(model, args.checkpoint)
    print("  Merging LoRA weights...")
    model = model.merge_and_unload()

    # ── 3. Resize embeddings ──────────────────────────────────────────────────
    print(f"\nResizing embed_tokens + lm_head: {old_vocab_size} → {new_vocab_size}")
    old_embed = model.get_input_embeddings().weight.data.clone()  # [old_vocab, hidden]

    model.resize_token_embeddings(new_vocab_size)

    new_embed = model.get_input_embeddings().weight.data
    new_lm_head = model.get_output_embeddings().weight.data

    # ── 4. Initialize new token embeddings ───────────────────────────────────
    print("Initializing new token embeddings from constituent token means...")
    # Re-load original tokenizer for constituent lookup
    orig_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, token=args.hf_token)

    for token_str in PHONEME_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        if token_id < old_vocab_size:
            print(f"  WARNING: {token_str} mapped to existing token {token_id}, skipping init")
            continue
        init_vec = mean_embedding_for_token(token_str, orig_tokenizer, old_embed)
        new_embed[token_id] = init_vec
        new_lm_head[token_id] = init_vec  # tie weights symmetrically

    print(f"  Initialized {len(PHONEME_TOKENS)} new token embeddings")

    # ── 5. Save checkpoint ────────────────────────────────────────────────────
    print(f"\nSaving warm-started model to {output_ckpt_dir}...")
    # Ensure model_type is preserved in config (merge_and_unload can drop it)
    if not hasattr(model.config, "model_type") or not model.config.model_type:
        model.config.model_type = "llama"
    model.save_pretrained(str(output_ckpt_dir))
    tokenizer.save_pretrained(str(output_ckpt_dir))
    print("Done.")
    print(f"\nNext step: re-prepare dataset with --phonemize flag, then run training with:")
    print(f"  --model-name {output_ckpt_dir}")
    print(f"  --tokenizer-name {output_tok_dir}")


if __name__ == "__main__":
    main()
