from dataclasses import replace
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
from sqsm.model import SQSMConfig, SQSMModel


torch.set_num_threads(1)


def tiny(family="llama", variant="sqsm"):
    torch.manual_seed(17)
    cls, model_cls = (LlamaConfig, LlamaForCausalLM) if family == "llama" else (Qwen2Config, Qwen2ForCausalLM)
    cfg = cls(vocab_size=101, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
              attention_dropout=0.0, pad_token_id=0, eos_token_id=2)
    cfg._attn_implementation = "eager"
    model = SQSMModel(model_cls(cfg), SQSMConfig(slots=3, rank=8, source_layer=1, read_layer=3, variant=variant))
    return model


def batch():
    ids = torch.tensor([[3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 0]])
    mask = ids != 0
    return dict(input_ids=ids, attention_mask=mask, prompt_ids=ids[:, :3], prompt_mask=mask[:, :3],
        question_mask=torch.tensor([[False, True, True], [False, True, True]]),
        inject_mask=torch.tensor([[0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 0]], dtype=torch.bool),
        labels=torch.tensor([[-100, -100, -100, 6, 7, 8], [-100, -100, -100, 12, 13, -100]]),
        loss_weights=torch.tensor([[0, 0, 0, 1, 2, 2], [0, 0, 0, 1, 2, 0]], dtype=torch.float32))


@pytest.mark.parametrize("family", ["llama", "qwen2"])
def test_zero_init_and_gradient_flow(family):
    model = tiny(family)
    b = batch()
    plain = model.backbone(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                           position_ids=model.positions(b["attention_mask"]), use_cache=False).logits
    result = model(**b)
    torch.testing.assert_close(result["logits"], plain, atol=0, rtol=0)
    result["loss"].backward()
    assert model.memory_module.output.weight.grad.abs().sum() > 0
    assert model.memory_module.source.weight.grad.abs().sum() == 0
    assert all(p.grad is None for p in model.backbone.parameters())
    optimizer = torch.optim.SGD(model.memory_module.parameters(), lr=.5)
    optimizer.step()
    optimizer.zero_grad()
    model(**b)["loss"].backward()
    for name in ["source", "query", "key", "value"]:
        assert getattr(model.memory_module, name).weight.grad.abs().sum() > 0, name
    assert model.memory_module.slots.grad.abs().sum() > 0


@pytest.mark.parametrize("family", ["llama", "qwen2"])
@pytest.mark.parametrize("variant", ["sqsm", "static", "direct"])
def test_cache_matches_full_causal_forward(family, variant):
    model = tiny(family, variant)
    torch.nn.init.normal_(model.memory_module.output.weight, std=.15)
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    prompt, question = ids[:, :3], torch.tensor([[False, True, True]])
    def memory():
        return model.build_memory(prompt, torch.ones_like(prompt), question)
    full, _ = model.conditioned_forward(ids, torch.ones_like(ids), memory(),
        torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.bool), use_cache=False)
    mem = memory()
    output, _ = model.conditioned_forward(prompt, torch.ones_like(prompt), mem,
        torch.tensor([[0, 0, 1]], dtype=torch.bool), use_cache=True)
    torch.testing.assert_close(output.logits[:, -1], full.logits[:, 2], atol=1e-6, rtol=1e-5)
    past = output.past_key_values
    for position in range(3, 6):
        output, _ = model.conditioned_forward(ids[:, position:position+1], torch.ones_like(ids[:, :position+1]),
            mem, torch.ones((1, 1), dtype=torch.bool), past_key_values=past, use_cache=True,
            position_ids=torch.tensor([[position]]))
        torch.testing.assert_close(output.logits[:, 0], full.logits[:, position], atol=1e-6, rtol=1e-5)
        past = output.past_key_values


def test_padding_question_exclusion_and_causality():
    model = tiny()
    torch.nn.init.normal_(model.memory_module.output.weight, std=.2)
    b = batch()
    original = model(**b)["logits"]
    changed = dict(b, input_ids=b["input_ids"].clone())
    changed["input_ids"][:, 4] = 99
    torch.testing.assert_close(model(**changed)["logits"][:, :4], original[:, :4], atol=0, rtol=0)
    states = torch.randn(2, 3, 32)
    mask = b["question_mask"]
    mem = model.memory_module.build(states, mask)
    states[:, 0] += 100
    mem2 = model.memory_module.build(states, mask)
    torch.testing.assert_close(mem.keys, mem2.keys)
    changed = dict(b, input_ids=b["input_ids"].clone())
    changed["input_ids"][1, -1] = 98
    torch.testing.assert_close(model(**changed)["logits"][1, :5], original[1, :5], atol=0, rtol=0)


def test_bound_and_inactive_positions():
    model = tiny()
    torch.nn.init.normal_(model.memory_module.output.weight, std=100)
    hidden = torch.randn(1, 5, 32)
    mem = model.memory_module.build(torch.randn(1, 4, 32), torch.ones(1, 4, dtype=torch.bool))
    mask = torch.tensor([[0, 0, 1, 1, 0]], dtype=torch.bool)
    updated, reg, _ = model.memory_module.inject(hidden, mem, mask)
    c = .1 * hidden.square().mean(-1, keepdim=True).sqrt() + 1e-6
    assert ((updated - hidden).abs() <= c + 2e-7).all()
    torch.testing.assert_close(updated[~mask], hidden[~mask], atol=0, rtol=0)
    assert reg >= 0


def test_per_sample_weighted_loss_and_count():
    model = tiny()
    b = batch()
    result = model(**b)
    ce = torch.nn.functional.cross_entropy(result["logits"][:, :-1].transpose(1, 2), b["labels"][:, 1:], reduction="none")
    w = b["loss_weights"][:, 1:]
    expected = ((ce * w).sum(-1) / w.sum(-1)).mean()
    torch.testing.assert_close(result["loss"], expected)
    assert model.parameter_count() == 3 * 32 * 8 + 2 * 8 * 8 + 3 * 8


def test_source_stops_early_and_hooks_are_cleaned():
    model = tiny()
    calls = []
    hook = model.backbone.model.layers[1].register_forward_hook(lambda *args: calls.append(True))
    b = batch()
    h = model.source_states(b["prompt_ids"], b["prompt_mask"])
    hook.remove()
    assert not calls and not h.requires_grad
    assert len(model.backbone.model.layers[0]._forward_hooks) == 0


def test_checkpoint_roundtrip(tmp_path):
    model = tiny()
    torch.nn.init.normal_(model.memory_module.output.weight, std=.1)
    model.save_adapter(tmp_path)
    other = tiny()
    other.load_adapter(tmp_path)
    torch.testing.assert_close(other(**batch())["logits"], model(**batch())["logits"])


def test_learns_tiny_batch():
    model = tiny()
    b = batch()
    optimizer = torch.optim.AdamW(model.memory_module.parameters(), lr=.03)
    initial = model(**b)["loss"].item()
    for _ in range(20):
        optimizer.zero_grad()
        loss = model(**b)["loss"]
        loss.backward()
        optimizer.step()
    assert model(**b)["loss"].item() < initial - .001


def test_independent_request_generation():
    model = tiny()
    torch.nn.init.normal_(model.memory_module.output.weight, std=.3)
    prompt = torch.tensor([[3, 4, 5]])
    mask = torch.ones_like(prompt)
    q = torch.tensor([[False, True, True]])
    a = model.generate(prompt, mask, q, max_new_tokens=4)["token_ids"]
    model.generate(prompt + 2, mask, q, max_new_tokens=4)
    b = model.generate(prompt, mask, q, max_new_tokens=4, use_cache=False)["token_ids"]
    assert a == b
