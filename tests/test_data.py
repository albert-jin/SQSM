import pytest
import torch
from sqsm.data import PromptEncoder, collate
from sqsm.metrics import correct


class CharacterTokenizer:
    is_fast = True
    eos_token_id = 2
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<system>" + messages[0]["content"] + "</system><user>" + messages[1]["content"] + "</user><assistant>"
    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) + 3 for c in text], "offset_mapping": [(i, i+1) for i in range(len(text))]}


def test_prompt_alignment_answer_weight_and_shift():
    encoder = PromptEncoder(CharacterTokenizer())
    e = encoder.training_example(dict(id="x", question="2 + 2?", rationale="Add two.", answer="4"))
    assert sum(e["question_mask"]) == len("2 + 2?")
    plen = len(e["prompt_ids"])
    assert e["inject_mask"][plen - 1] and not any(e["inject_mask"][:plen-1])
    assert all(v == -100 for v in e["labels"][:plen])
    assert e["loss_weights"][-1] == 2 and e["loss_weights"][plen] == 1
    shorter = encoder.training_example(dict(id="y", question="?", rationale="", answer="yes"))
    b = collate([e, shorter], 0)
    assert not b["inject_mask"][1, -1]
    assert b["labels"][1, -1] == -100 and b["loss_weights"][1, -1] == 0


def test_long_example_is_filtered():
    assert PromptEncoder(CharacterTokenizer(), max_length=10).training_example(
        dict(id="x", question="long question", rationale="rationale", answer="4")) is None


@pytest.mark.parametrize("text,gold,task,expected", [
    ("reason 3\n#### 1,200", "1200", "gsm8k", True),
    ("#### (B)", "B", "aqua", True),
    ("#### yes", "yes", "strategyqa", True),
    ("#### no", "yes", "strategyqa", False),
    ("I saw A and B", "B", "aqua", False),
    ("#### 0.5", "1/2", "asdiv", True),
])
def test_parser(text, gold, task, expected):
    assert correct(text, gold, task) is expected
