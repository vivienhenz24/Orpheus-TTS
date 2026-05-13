import os
from datetime import datetime
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
from torch.utils.data import DataLoader, Dataset
import yaml

config_file = os.environ.get("ORPHEUS_PRETRAIN_CONFIG", "config.yaml")

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn1 = config.get("text_QA_dataset")
dsn2 = config["TTS_dataset"]

model_name = config["model_name"]
tokenizer_name = config["tokenizer_name"]

run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]

epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]
config_ratio = int(config.get("ratio", 0))




class BatchedRatioDataset(Dataset):
    def __init__(self, dataset1, dataset2, batch_total, ratio=config_ratio):
        if ratio <= 0:
            raise ValueError("BatchedRatioDataset requires ratio > 0")
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_total = batch_total
        self.ratio = ratio  

        num_cycles_ds1 = len(dataset1) // (batch_total * ratio)
        num_cycles_ds2 = len(dataset2) // batch_total
        self.num_cycles = min(num_cycles_ds1, num_cycles_ds2)

        self.length = self.num_cycles * (ratio + 1) * batch_total

    def __len__(self):
        print("accessing length", self.length)
        return int(self.length)

    def __getitem__(self, index):
        # Compute the cycle length in terms of samples.
        cycle_length = (self.ratio + 1) * self.batch_total
        cycle = index // cycle_length
        pos_in_cycle = index % cycle_length

        if pos_in_cycle < self.ratio * self.batch_total:
            batch_in_cycle = pos_in_cycle // self.batch_total
            sample_in_batch = pos_in_cycle % self.batch_total
            ds1_index = cycle * self.ratio * self.batch_total + batch_in_cycle * self.batch_total + sample_in_batch
            return self.dataset1[ds1_index]
        else:
            # We are in the dataset2 batch for this cycle.
            sample_in_batch = pos_in_cycle - self.ratio * self.batch_total
            ds2_index = cycle * self.batch_total + sample_in_batch
            return self.dataset2[ds2_index]



class FSDPTrainer(Trainer):
    def __init__(self, *args, log_ratio=config_ratio, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_ratio = log_ratio
        self.text_step  = 0
        self.audio_step = 0

    def log(self, logs, start_time=None):
        super().log(logs)
        if self.is_world_process_zero():
            ts = datetime.now().strftime("%H:%M:%S")
            if self.log_ratio <= 0:
                print(f"[{ts}] audio_step={self.audio_step} loss={logs['loss']:.4f}", flush=True)
                self.audio_step += 1
                return
            global_step = self.state.global_step
            cycle_length = self.log_ratio + 1
            if (global_step % cycle_length) + self.log_ratio - 1 < self.log_ratio:
                print(f"[{ts}] audio_step={self.audio_step} loss={logs['loss']:.4f}", flush=True)
                self.audio_step += 1
            else:
                print(f"[{ts}] text_step={self.text_step} loss={logs['loss']:.4f}", flush=True)
                self.text_step += 1

    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir
        self.model.save_pretrained(output_dir)


def data_collator(features):
    # max_length = 2656 # set a crop based on vram - ideally you have stacked all sequences to the same length
    # from 3b on 8 h100s fsdp, at bf16, 8192 works well.
    input_ids = [f["input_ids"] for f in features]

    if any("attention_mask" not in f for f in features):
        attention_mask = [[1]*len(ids) for ids in input_ids]
    else:
        attention_mask = [f["attention_mask"] for f in features]

    if any("labels" not in f for f in features):
        labels = input_ids
    else:
        labels = [f["labels"] for f in features]

    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(
        i, dtype=torch.long) for i in input_ids], batch_first=True, padding_value=pad_token)
    attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(
        m, dtype=torch.long) for m in attention_mask], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(
        l, dtype=torch.long) for l in labels], batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}



tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, attn_implementation="flash_attention_2")


number_add_tokens = 7 * 4096 + 10
new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))


def load_training_dataset(dsn):
    if os.path.isdir(dsn):
        parquet_dir = os.path.join(dsn, "parquet")
        if os.path.isdir(parquet_dir):
            parquet_files = sorted(
                os.path.join(parquet_dir, name)
                for name in os.listdir(parquet_dir)
                if name.endswith(".parquet")
            )
            if not parquet_files:
                raise ValueError(f"No parquet shards found in {parquet_dir}")
            return load_dataset("parquet", data_files=parquet_files, split="train")
        return load_from_disk(dsn)
    return load_dataset(dsn, split="train")


ds2 = load_training_dataset(dsn2)
batch_total = batch_size * number_processes
if config_ratio > 0:
    if not dsn1:
        raise ValueError("text_QA_dataset is required when ratio > 0")
    ds1 = load_training_dataset(dsn1)
    train_dataset = BatchedRatioDataset(ds1, ds2, batch_total, ratio=config_ratio)
else:
    train_dataset = ds2


training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    logging_steps=100,
    bf16=True,
    output_dir=f"./{base_repo_id}",
    report_to="none",
    save_steps=save_steps,
    remove_unused_columns=True,
    learning_rate=learning_rate,
    lr_scheduler_type="cosine", 
)


trainer = FSDPTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    log_ratio=config_ratio
)

print(f"[{datetime.now().strftime('%H:%M:%S')}] training started — model={model_name} lr={learning_rate} epochs={epochs}", flush=True)
trainer.train()
print(f"[{datetime.now().strftime('%H:%M:%S')}] training complete", flush=True)
