"""Template-aligned question masks and weighted causal supervision."""
import json
from pathlib import Path
import torch


INSTRUCTION = "Solve the problem step by step. End with '#### ' followed by only the final answer (number, option letter, or yes/no)."


def read_records(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class PromptEncoder:
    def __init__(self, tokenizer, max_length=1536, answer_weight=2.0):
        if not tokenizer.is_fast:
            raise ValueError("A fast tokenizer is required for exact character/token alignment")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_weight = answer_weight

    def prompt(self, question):
        messages = [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": question}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        marker = "SQSMQUESTIONBOUNDARY9f718"
        if marker in question:
            raise ValueError("Reserved boundary marker occurs in question")
        marked = self.tokenizer.apply_chat_template(
            [messages[0], {"role": "user", "content": marker}], tokenize=False, add_generation_prompt=True)
        if marked.count(marker) != 1:
            raise ValueError("Chat template must preserve user question verbatim")
        prefix, suffix = marked.split(marker)
        if rendered != prefix + question + suffix:
            raise ValueError("Template transforms user content; explicit alignment is required")
        start, end = len(prefix), len(prefix) + len(question)
        encoded = self.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        # Exclude tokens straddling instruction/template boundaries.
        qmask = [b > a and a >= start and b <= end for a, b in encoded["offset_mapping"]]
        if not any(qmask):
            raise ValueError("Question has no aligned tokens")
        return {"prompt_ids": encoded["input_ids"], "question_mask": qmask}

    def training_example(self, record):
        result = self.prompt(record["question"])
        rationale = record.get("rationale", "").strip()
        rationale_prefix = rationale + "\n" if rationale else ""
        completion = rationale_prefix + "#### " + record["answer"].strip()
        target = self.tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
        ids = target["input_ids"]
        weights = [self.answer_weight if b > len(rationale_prefix) else 1.0
                   for a, b in target["offset_mapping"]]
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Tokenizer has no EOS token")
        ids = ids + [eos]
        weights = weights + [self.answer_weight]
        p = len(result["prompt_ids"])
        if p + len(ids) > self.max_length:
            return None  # whole example filtered, never truncate the answer
        result.update(input_ids=result["prompt_ids"] + ids,
                      labels=[-100] * p + ids, loss_weights=[0.0] * p + weights,
                      inject_mask=[False] * (p - 1) + [True] * (len(ids) + 1),
                      id=record["id"])
        return result


def collate(examples, pad_token_id):
    max_prompt = max(len(e["prompt_ids"]) for e in examples)
    max_total = max(len(e["input_ids"]) for e in examples)
    result = {}
    specs = {"prompt_ids": (max_prompt, pad_token_id, torch.long),
             "question_mask": (max_prompt, False, torch.bool),
             "input_ids": (max_total, pad_token_id, torch.long),
             "labels": (max_total, -100, torch.long),
             "loss_weights": (max_total, 0.0, torch.float32),
             "inject_mask": (max_total, False, torch.bool)}
    for name, (width, fill, dtype) in specs.items():
        result[name] = torch.tensor([e[name] + [fill] * (width - len(e[name])) for e in examples], dtype=dtype)
    result["prompt_mask"] = torch.tensor([[1] * len(e["prompt_ids"]) + [0] * (max_prompt - len(e["prompt_ids"])) for e in examples])
    result["attention_mask"] = torch.tensor([[1] * len(e["input_ids"]) + [0] * (max_total - len(e["input_ids"])) for e in examples])
    return result
