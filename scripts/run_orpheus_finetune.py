import argparse
import json
import os
from pathlib import Path

import soundfile as sf
import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from snac import SNAC
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


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


DEFAULT_SAMPLE_TEXTS = [
    "Merhaba, bu modeli Turkce konusma kalitesi icin test ediyoruz.",
    "Bugun hava oldukca guzel, kahve alip yuruyuse cikmak istiyorum.",
    "Toplanti yarin sabah dokuzda baslayacak, lutfen hazir olun.",
]


def choose_attn_implementation():
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


def data_collator(features, pad_token):
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
    labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

    return {
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=pad_token),
        "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
        "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
    }


def deinterleave_to_layers(token_ids):
    codes_l0 = []
    codes_l1 = []
    codes_l2 = []
    for i in range(0, len(token_ids), 7):
        if i + 7 > len(token_ids):
            break
        frame = token_ids[i : i + 7]
        codes_l0.append(frame[0] - AUDIO_TOKEN_BASE)
        codes_l1.append(frame[1] - LAYER_1_OFFSET)
        codes_l2.append(frame[2] - LAYER_2_OFFSET)
        codes_l2.append(frame[3] - LAYER_2_OFFSET - 4096)
        codes_l1.append(frame[4] - LAYER_1_OFFSET - 12288)
        codes_l2.append(frame[5] - LAYER_2_OFFSET - 12288)
        codes_l2.append(frame[6] - LAYER_2_OFFSET - 16384)
    return codes_l0, codes_l1, codes_l2


def render_prompt(template, text, speaker):
    if not template:
        return text
    return template.replace("{speaker_id}", speaker).replace("{text}", text)


class SampleGenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, sample_speakers, sample_texts, output_dir, sample_rate, device, prompt_template):
        self.tokenizer = tokenizer
        self.sample_speakers = sample_speakers
        self.sample_texts = sample_texts
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.device = device
        self.prompt_template = prompt_template
        self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()

    def _prompt_ids(self, speaker, text):
        prompt = render_prompt(self.prompt_template, text, speaker)
        text_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        text_ids.append(END_OF_TEXT)
        return [START_OF_HUMAN] + text_ids + [END_OF_HUMAN] + [START_OF_AI] + [START_OF_SPEECH]

    def _generate_audio(self, model, speaker, text):
        input_ids = self._prompt_ids(speaker, text)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = model.generate(
                input_tensor,
                max_new_tokens=1400,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                eos_token_id=END_OF_AI,
                pad_token_id=PAD_TOKEN,
            )

        generated = output[0][len(input_ids) :].detach().cpu().tolist()
        if END_OF_SPEECH in generated:
            generated = generated[: generated.index(END_OF_SPEECH)]

        if len(generated) < 7:
            raise ValueError("Generated audio token sequence is too short.")

        codes_l0, codes_l1, codes_l2 = deinterleave_to_layers(generated)
        codes = [
            torch.tensor([codes_l0], dtype=torch.long, device=self.device),
            torch.tensor([codes_l1], dtype=torch.long, device=self.device),
            torch.tensor([codes_l2], dtype=torch.long, device=self.device),
        ]
        with torch.inference_mode():
            audio = self.snac.decode(codes)
        return audio.squeeze().detach().cpu().numpy()

    def _write_samples(self, model, step_label):
        sample_dir = self.output_dir / "samples" / step_label
        sample_dir.mkdir(parents=True, exist_ok=True)
        was_training = model.training
        model.eval()
        for speaker in self.sample_speakers:
            for idx, text in enumerate(self.sample_texts, start=1):
                try:
                    audio = self._generate_audio(model, speaker, text)
                    out_path = sample_dir / f"{speaker}_sample_{idx}.wav"
                    sf.write(out_path, audio, self.sample_rate)
                except Exception as exc:
                    err_path = sample_dir / f"{speaker}_sample_{idx}.error.txt"
                    err_path.write_text(str(exc), encoding="utf-8")
        if was_training:
            model.train()

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._write_samples(kwargs["model"], f"step_{state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._write_samples(kwargs["model"], "final")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model-name", default="canopylabs/orpheus-tts-0.1-pretrained")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--pad-token", type=int, default=128263)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--sample-speakers", nargs="*", default=None)
    parser.add_argument("--sample-text", action="append", default=None)
    parser.add_argument("--prompt-template", default="{speaker_id}: {text}")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--enable-samples", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)

    attn_implementation = choose_attn_implementation()
    bf16_enabled = args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16_enabled = args.fp16 and torch.cuda.is_available() and not bf16_enabled
    torch_dtype = torch.bfloat16 if bf16_enabled else (torch.float16 if fp16_enabled else None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=args.hf_token,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        modules_to_save=["lm_head", "embed_tokens"],
        bias="none",
        use_rslora=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_speakers = args.sample_speakers or sorted(set(dataset["speaker_id"]))[:2]
    sample_texts = args.sample_text or DEFAULT_SAMPLE_TEXTS

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=bf16_enabled,
        fp16=fp16_enabled,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        report_to="wandb" if "WANDB_API_KEY" in os.environ else "none",
        remove_unused_columns=True,
        dataloader_num_workers=2,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
    )

    callbacks = []
    if args.enable_samples:
        callbacks.append(
            SampleGenerationCallback(
                tokenizer=tokenizer,
                sample_speakers=sample_speakers,
                sample_texts=sample_texts,
                output_dir=output_dir,
                sample_rate=args.sample_rate,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                prompt_template=args.prompt_template,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda features: data_collator(features, args.pad_token),
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metadata = {
        "dataset_dir": args.dataset_dir,
        "model_name": args.model_name,
        "save_strategy": "epoch",
        "sample_speakers": sample_speakers,
        "sample_texts": sample_texts,
        "prompt_template": args.prompt_template,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
