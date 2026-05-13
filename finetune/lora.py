import os
from datetime import datetime
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer, TrainerCallback

class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs:
            print(logs, flush=True)

config_file = os.environ.get("ORPHEUS_FINETUNE_CONFIG", "config.yaml")

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn = config["TTS_dataset"]
model_name = config["model_name"]
tokenizer_name = config.get("tokenizer_name", model_name)
run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]
epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]
resume_from_lora = config.get("resume_from_lora", None)
gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
warmup_steps = config.get("warmup_steps", 0)
logging_steps = config.get("logging_steps", 100)

lora_rank = 32
lora_alpha = 64
lora_dropout = 0.0


def load_training_dataset(path):
    if os.path.isdir(path):
        parquet_dir = os.path.join(path, "parquet")
        if os.path.isdir(parquet_dir):
            files = sorted(
                os.path.join(parquet_dir, f)
                for f in os.listdir(parquet_dir)
                if f.endswith(".parquet")
            )
            return load_dataset("parquet", data_files=files, split="train")
    return load_dataset(path, split="train")


def data_collator(features):
    input_ids = [f["input_ids"] for f in features]
    if any("attention_mask" not in f for f in features):
        attention_mask = [[1] * len(ids) for ids in input_ids]
    else:
        attention_mask = [f["attention_mask"] for f in features]
    labels = input_ids

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(i, dtype=torch.long) for i in input_ids],
        batch_first=True, padding_value=pad_token)
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(m, dtype=torch.long) for m in attention_mask],
        batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(l, dtype=torch.long) for l in labels],
        batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2")

number_add_tokens = 7 * 4096 + 10
new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

if resume_from_lora:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading existing LoRA weights from {resume_from_lora}", flush=True)
    model = PeftModel.from_pretrained(model, resume_from_lora, is_trainable=True)
else:
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        bias="none",
        modules_to_save=["lm_head", "embed_tokens"],
        task_type="CAUSAL_LM",
        use_rslora=True,
    )
    model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

ds = load_training_dataset(dsn)

training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    logging_steps=logging_steps,
    bf16=True,
    output_dir=f"./{base_repo_id}",
    report_to="none",
    save_steps=save_steps,
    remove_unused_columns=True,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine",
    warmup_steps=warmup_steps,
    ddp_find_unused_parameters=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=data_collator,
    callbacks=[PrintLossCallback()],
)

print(f"[{datetime.now().strftime('%H:%M:%S')}] LoRA finetuning started — model={model_name} lr={learning_rate} epochs={epochs}", flush=True)
trainer.train()
print(f"[{datetime.now().strftime('%H:%M:%S')}] LoRA finetuning complete — merging", flush=True)

merged = model.merge_and_unload()
merged.save_pretrained(f"./{base_repo_id}/merged")
tokenizer.save_pretrained(f"./{base_repo_id}/merged")
print(f"[{datetime.now().strftime('%H:%M:%S')}] merged model saved to ./{base_repo_id}/merged", flush=True)
