import json
from pathlib import Path
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .model import SQSMConfig, SQSMModel


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_id, config, device="cuda", dtype="bfloat16", checkpoint=None, revision=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    info = None
    if checkpoint:
        info = json.loads((Path(checkpoint) / "config.json").read_text(encoding="utf-8"))
        if info["backbone"] and info["backbone"] != model_id:
            raise ValueError("Checkpoint backbone ID does not match --model")
        config = info["sqsm"]
    backbone = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
        torch_dtype=getattr(torch, dtype), attn_implementation="eager", trust_remote_code=False).to(device)
    model = SQSMModel(backbone, SQSMConfig(**config))
    if checkpoint:
        model.load_adapter(checkpoint)
    return model, tokenizer


def eos_ids(model, tokenizer):
    ids = model.backbone.generation_config.eos_token_id
    ids = ids if isinstance(ids, list) else [ids]
    return list({i for i in ids + [tokenizer.eos_token_id] if i is not None})
