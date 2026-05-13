import asyncio
import json
import os
import queue
import re
import threading
import time
import uuid
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

from .decoder import tokens_decoder_sync

DEBUG_DIR = os.environ.get("ORPHEUS_DEBUG_DIR")

class OrpheusModel:
    def __init__(self, model_name, dtype=torch.bfloat16, tokenizer='canopylabs/orpheus-3b-0.1-pretrained', **engine_kwargs):
        self.model_name = self._map_model_params(model_name)
        self.dtype = dtype
        self.engine_kwargs = engine_kwargs  # vLLM engine kwargs
        self.available_voices = ["zoe", "zac","jess", "leo", "mia", "julia", "leah"]

        # Resolve the tokenizer path up front so vLLM doesn't try to infer it
        # from a checkpoint directory that only contains weights/config shards.
        self.tokenizer_path = tokenizer if tokenizer else self.model_name
        self.tokenizer = self._load_tokenizer(self.tokenizer_path)
        self.engine = self._setup_engine()

    def _load_tokenizer(self, tokenizer_path):
        """Load tokenizer from local path or HuggingFace hub"""
        try:
            # Check if tokenizer_path is a local directory
            if os.path.isdir(tokenizer_path):
                return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            else:
                return AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print(f"Falling back to default tokenizer")
            return AutoTokenizer.from_pretrained("gpt2")
    
    def _map_model_params(self, model_name):
        model_map = {
            # "nano-150m":{
            #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            # }, 
            # "micro-400m":{
            #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            # }, 
            # "small-1b":{
            #     "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            # },
            "medium-3b":{
                "repo_id": "canopylabs/orpheus-tts-0.1-finetune-prod",
            },
        }
        unsupported_models = ["nano-150m", "micro-400m", "small-1b"]
        if (model_name  in unsupported_models):
            raise ValueError(f"Model {model_name} is not supported. Only medium-3b is supported, small, micro and nano models will be released very soon")
        elif model_name in model_map:
            return model_name[model_name]["repo_id"]
        else:
            return model_name
        
    def _setup_engine(self):
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            tokenizer=self.tokenizer_path,
            dtype=self.dtype,
            **self.engine_kwargs
        )
        
        return AsyncLLMEngine.from_engine_args(engine_args)

    def _write_debug_json(self, request_id, suffix, payload):
        if not DEBUG_DIR:
            return
        debug_dir = Path(DEBUG_DIR)
        debug_dir.mkdir(parents=True, exist_ok=True)
        with (debug_dir / f"{request_id}.{suffix}.json").open("w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    
    def validate_voice(self, voice):
        if voice:
            if voice not in self.engine.available_voices:
                raise ValueError(f"Voice {voice} is not available for model {self.model_name}")
    
    def _format_prompt(self, prompt, voice="tara", model_type="larger"):
        if model_type == "smaller":
            if voice:
                return f"<custom_token_3>{prompt}[{voice}]<custom_token_4><custom_token_5>"
            else:
                return f"<custom_token_3>{prompt}<custom_token_4><custom_token_5>"
        else:
            if voice:
                adapted_prompt = f"{voice}: {prompt}"
                prompt_tokens = self.tokenizer(adapted_prompt, return_tensors="pt")
                start_token = torch.tensor([[ 128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokenizer.decode(all_input_ids[0])
                return prompt_string
            else:
                prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
                start_token = torch.tensor([[ 128259]], dtype=torch.int64)
                end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
                all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
                prompt_string = self.tokenizer.decode(all_input_ids[0])
                return prompt_string

 


    def generate_tokens_sync(self, prompt, voice=None, request_id=None, temperature=0.6, top_p=0.8, max_tokens=1200, stop_token_ids = [49158], repetition_penalty=1.3):
        if request_id is None:
            request_id = f"req-{uuid.uuid4().hex}"
        prompt_string = self._format_prompt(prompt, voice)
        debug = {
            "request_id": request_id,
            "voice": voice,
            "prompt": prompt,
            "prompt_string_preview": prompt_string[:300],
            "prompt_string_len": len(prompt_string),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop_token_ids": stop_token_ids,
            "repetition_penalty": repetition_penalty,
            "started_at": time.time(),
            "callback_count": 0,
            "stream_text_samples": [],
        }
        sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,  # Adjust max_tokens as needed.
        stop_token_ids = stop_token_ids, 
        repetition_penalty=repetition_penalty, 
        )

        token_queue = queue.Queue()

        async def async_producer():
            last_text = ""
            async for result in self.engine.generate(prompt=prompt_string, sampling_params=sampling_params, request_id=request_id):
                last_text = result.outputs[0].text
                debug["callback_count"] += 1
                if len(debug["stream_text_samples"]) < 8:
                    debug["stream_text_samples"].append({
                        "len": len(last_text),
                        "tail": last_text[-120:],
                    })
                token_queue.put(last_text)
            debug["finished_at"] = time.time()
            debug["elapsed_s"] = debug["finished_at"] - debug["started_at"]
            debug["final_text_len"] = len(last_text)
            debug["custom_token_count"] = len(re.findall(r"<custom_token_\d+>", last_text))
            debug["final_text_tail"] = last_text[-500:]
            self._write_debug_json(request_id, "engine", debug)
            token_queue.put(None)  # Sentinel to indicate completion.

        def run_async():
            asyncio.run(async_producer())

        thread = threading.Thread(target=run_async)
        thread.start()

        while True:
            token = token_queue.get()
            if token is None:
                break
            yield token

        thread.join()
    
    def generate_speech(self, **kwargs):
        request_id = kwargs.get("request_id")
        return tokens_decoder_sync(self.generate_tokens_sync(**kwargs), request_id=request_id)
