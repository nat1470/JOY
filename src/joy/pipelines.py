from typing import Optional

import torch
from bitsandbytes import BitsAndBytesConfig
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)

from .config import GenerationConfig


def resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_stt_pipeline(model_id: str, device: str):
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if device.startswith("cuda") else None,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=0 if device.startswith("cuda") else "cpu",
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=4,
        return_timestamps=False,
    )


def build_llm_pipeline(model_id: str, generation: GenerationConfig, device: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "12GiB"} if device.startswith("cuda") else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device.startswith("cuda") else -1,
        max_new_tokens=generation.max_new_tokens,
        temperature=generation.temperature,
        top_p=generation.top_p,
        repetition_penalty=generation.repetition_penalty,
        do_sample=generation.temperature > 0,
    )


def build_tts_pipeline(model_id: str, device: str):
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    return pipeline(
        "text-to-speech",
        model=model_id,
        torch_dtype=dtype,
        device=0 if device.startswith("cuda") else "cpu",
        framework="pt",
    )
