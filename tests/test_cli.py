"""End-to-end commands use a randomly initialized local toy model, never a benchmark."""
import json
from pathlib import Path
import subprocess
import sys
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


def test_train_and_evaluate_cli(tmp_path):
    model_dir = tmp_path / "toy-model"
    tokenizer_core = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "[PAD]": 2,
        "one": 3, "two": 4, "plus": 5, "?": 6, "####": 7, "2": 8, "1": 9}, unk_token="[UNK]"))
    tokenizer_core.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_core, unk_token="[UNK]", eos_token="[EOS]", pad_token="[PAD]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    tokenizer.save_pretrained(model_dir)
    config = LlamaConfig(vocab_size=10, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
                         pad_token_id=2, eos_token_id=1)
    LlamaForCausalLM(config).save_pretrained(model_dir)
    cfg = {"model": str(model_dir), "sqsm": {"slots": 2, "rank": 4},
           "training": {"learning_rate": .01, "weight_decay": .01, "epochs": 1,
             "micro_batch_size": 1, "gradient_accumulation_steps": 3, "warmup_ratio": .05,
             "max_grad_norm": 1., "max_length": 200, "max_new_tokens": 2}}
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    for split, count in [("train", 4), ("dev", 2)]:
        (tmp_path / f"{split}.jsonl").write_text("\n".join(json.dumps(dict(
            id=f"{split}:{i}", question="one plus one ?", rationale="two", answer="2", source="gsm8k")) for i in range(count)), encoding="utf-8")
    output = tmp_path / "run"
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "train.py", "--config", str(cfg_path), "--train", str(tmp_path / "train.jsonl"),
               "--dev", str(tmp_path / "dev.jsonl"), "--output", str(output), "--device", "cpu", "--dtype", "float32", "--cache-sources"]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "best" / "sqsm.safetensors").exists()
    command = [sys.executable, "evaluate.py", "--model", str(model_dir), "--checkpoint", str(output / "best"),
               "--data", str(tmp_path / "dev.jsonl"), "--output", str(output / "test.jsonl"), "--device", "cpu",
               "--dtype", "float32", "--max-new-tokens", "2"]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((output / "test.metrics.json").read_text())["count"] == 2
