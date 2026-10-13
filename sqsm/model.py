"""SQSM equations and frozen-backbone integration; independent of SoftCoT's assistant.

Layer indices are one-based block outputs. Request state is explicit and hooks
are scoped to each forward; no memory is kept on a shared model attribute.
"""
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SQSMConfig:
    slots: int = 16
    rank: int = 256
    source_layer: int | None = None
    read_layer: int | None = None
    rho: float = 0.1
    eps: float = 1e-6
    answer_weight: float = 2.0
    reg_weight: float = 1e-3
    variant: str = "sqsm"  # sqsm, static, direct, unbounded, adapter


def rms_norm(x, eps=1e-6):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


@dataclass
class Memory:
    keys: torch.Tensor
    values: torch.Tensor
    mask: torch.Tensor | None = None
    static_query: torch.Tensor | None = None


class SoftMemory(nn.Module):
    def __init__(self, hidden, config):
        super().__init__()
        self.config = config
        r, k = config.rank, config.slots
        if min(r, k) < 1 or config.rho <= 0:
            raise ValueError("rank, slots and rho must be positive")
        if config.variant not in {"sqsm", "static", "direct", "unbounded", "adapter"}:
            raise ValueError(f"Unknown variant: {config.variant}")
        self.source = nn.Linear(hidden, r, bias=False)
        self.slots = nn.Parameter(torch.empty(k, r))
        self.query = nn.Linear(hidden, r, bias=False)
        self.key = nn.Linear(r, r, bias=False)
        self.value = nn.Linear(r, r, bias=False)
        self.output = nn.Linear(r, hidden, bias=False)
        for layer in (self.source, self.query, self.key, self.value):
            nn.init.xavier_uniform_(layer.weight)
        nn.init.normal_(self.slots, std=r ** -0.5)
        nn.init.zeros_(self.output.weight)
        if config.variant == "adapter":
            # Nearest bias-free bottleneck width, with exact count reported.
            target = 3 * hidden * r + 2 * r * r + k * r
            width = round(target / (2 * hidden))
            self.down = nn.Linear(hidden, width, bias=False)
            self.up = nn.Linear(width, hidden, bias=False)
            nn.init.xavier_uniform_(self.down.weight)
            nn.init.zeros_(self.up.weight)
            for p in (self.source, self.query, self.key, self.value, self.output):
                p.requires_grad_(False)
            self.slots.requires_grad_(False)

    def build(self, states, question_mask):
        if not question_mask.bool().any(-1).all():
            raise ValueError("Each sample needs at least one question token")
        u = self.source(rms_norm(states.detach(), self.config.eps))
        if self.config.variant == "direct":
            m, mask = u, question_mask.bool()
        else:
            logits = torch.einsum("kr,bnr->bkn", self.slots, u) / math.sqrt(self.config.rank)
            logits = logits.masked_fill(~question_mask[:, None, :].bool(), -torch.inf)
            m = rms_norm(self.slots[None] + logits.softmax(-1) @ u, self.config.eps)
            mask = None
        return Memory(self.key(m), self.value(m), mask)

    def inject(self, hidden, memory, inject_mask):
        h = hidden.float()
        if self.config.variant == "adapter":
            z = self.up(F.gelu(self.down(rms_norm(h, self.config.eps))))
            attention = None
        else:
            q = self.query(rms_norm(h, self.config.eps))
            if self.config.variant == "static":
                if memory.static_query is None:
                    if not inject_mask.any(-1).all():
                        raise ValueError("Static query requires last prompt position")
                    first = inject_mask.long().argmax(-1)
                    memory.static_query = q[torch.arange(q.shape[0], device=q.device), first][:, None]
                q = memory.static_query.expand(-1, hidden.shape[1], -1)
            logits = q @ memory.keys.transpose(-1, -2) / math.sqrt(self.config.rank)
            if memory.mask is not None:
                logits = logits.masked_fill(~memory.mask[:, None], -torch.inf)
            attention = logits.softmax(-1)
            z = self.output(attention @ memory.values)
        c = self.config.rho * h.square().mean(-1, keepdim=True).sqrt().detach() + self.config.eps
        delta = z if self.config.variant == "unbounded" else c * torch.tanh(z / c)
        mask = inject_mask.to(delta.dtype)
        ratio = delta.square().sum(-1) / (h.square().sum(-1).detach() + self.config.eps)
        reg = (ratio * mask).sum() / mask.sum().clamp_min(1)
        updated = hidden + (delta * mask[..., None]).to(hidden.dtype)
        return updated, reg, attention


class _SourceReady(Exception):
    pass


class SQSMModel(nn.Module):
    def __init__(self, backbone, config=None):
        super().__init__()
        self.backbone = backbone
        self.sqsm_config = config or SQSMConfig()
        if backbone.config.model_type not in {"llama", "qwen2", "mistral"}:
            raise ValueError("Supported decoder families: llama, qwen2, mistral")
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        depth = len(backbone.model.layers)
        self.source_layer = self.sqsm_config.source_layer or max(1, depth // 3)
        self.read_layer = self.sqsm_config.read_layer or max(1, 3 * depth // 4)
        if not (1 <= self.source_layer <= depth and 1 <= self.read_layer <= depth):
            raise ValueError("Layer indices must be in 1..num_hidden_layers")
        self.memory_module = SoftMemory(backbone.config.hidden_size, self.sqsm_config).to(self.device)
        self._request = ContextVar(f"sqsm_request_{id(self)}", default=None)
        self._hook = backbone.model.layers[self.read_layer - 1].register_forward_hook(self._read_hook)

    @property
    def device(self):
        return self.backbone.get_input_embeddings().weight.device

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # frozen dropout must remain disabled
        return self

    def _read_hook(self, module, args, output):
        state = self._request.get()
        if state is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        updated, state["reg"], state["attention"] = self.memory_module.inject(
            hidden, state["memory"], state["mask"])
        return (updated,) + output[1:] if isinstance(output, tuple) else updated

    @torch.no_grad()
    def source_states(self, prompt_ids, prompt_mask):
        captured = []
        def capture(module, args, output):
            captured.append((output[0] if isinstance(output, tuple) else output).detach())
            raise _SourceReady()
        handle = self.backbone.model.layers[self.source_layer - 1].register_forward_hook(capture)
        try:
            self.backbone(input_ids=prompt_ids, attention_mask=prompt_mask,
                          position_ids=self.positions(prompt_mask), use_cache=False)
        except _SourceReady:
            pass
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("Source layer did not run")
        return captured[0]

    @staticmethod
    def positions(mask):
        return (mask.long().cumsum(-1) - 1).clamp_min(0)

    def build_memory(self, prompt_ids, prompt_mask, question_mask, source_states=None):
        if self.sqsm_config.variant == "adapter":
            return None
        h = source_states if source_states is not None else self.source_states(prompt_ids, prompt_mask)
        return self.memory_module.build(h, question_mask & prompt_mask.bool())

    def conditioned_forward(self, input_ids, attention_mask, memory, inject_mask, **kwargs):
        if getattr(self.backbone, "is_gradient_checkpointing", False):
            raise ValueError("Disable backbone gradient checkpointing: scoped hooks require explicit recomputation support")
        state = {"memory": memory, "mask": inject_mask}
        token = self._request.set(state)
        try:
            output = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        finally:
            self._request.reset(token)
        return output, state["reg"]

    def forward(self, input_ids, attention_mask, prompt_ids, prompt_mask,
                question_mask, inject_mask, labels, loss_weights, source_states=None):
        memory = self.build_memory(prompt_ids, prompt_mask, question_mask, source_states)
        output, reg = self.conditioned_forward(input_ids, attention_mask, memory, inject_mask,
            position_ids=self.positions(attention_mask), use_cache=False)
        losses = F.cross_entropy(output.logits[:, :-1].float().transpose(1, 2),
                                 labels[:, 1:], reduction="none", ignore_index=-100)
        weights = loss_weights[:, 1:] * (labels[:, 1:] != -100)
        denominator = weights.sum(-1)
        if (denominator == 0).any():
            raise ValueError("Each sample must have supervised answer tokens")
        task = ((losses * weights).sum(-1) / denominator).mean()
        return {"loss": task + self.sqsm_config.reg_weight * reg,
                "task_loss": task.detach(), "reg_loss": reg.detach(), "logits": output.logits}

    @torch.inference_mode()
    def generate(self, prompt_ids, prompt_mask, question_mask, max_new_tokens=1024,
                 eos_token_ids=(), use_cache=True, baseline=False):
        if prompt_ids.shape[0] != 1 or not prompt_mask.all():
            raise ValueError("Generation accepts one unpadded request at a time")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.eval()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        memory = None if baseline else self.build_memory(prompt_ids, prompt_mask, question_mask)
        ids, mask = prompt_ids, prompt_mask
        plen = ids.shape[1]
        past, generated, first_latency = None, [], None
        for step in range(max_new_tokens):
            current = ids[:, -1:] if use_cache and step else ids
            injection = torch.zeros_like(current, dtype=torch.bool)
            if use_cache and step:
                injection[:] = True
            else:
                injection[:, plen - 1:] = True
            kwargs = dict(position_ids=self.positions(mask)[:, -current.shape[1]:],
                          use_cache=use_cache, past_key_values=past)
            if baseline:
                output = self.backbone(input_ids=current, attention_mask=mask, **kwargs)
            else:
                output, _ = self.conditioned_forward(current, mask, memory, injection, **kwargs)
            next_id = output.logits[:, -1].argmax(-1, keepdim=True)
            generated.append(next_id.item())
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            if first_latency is None:
                first_latency = time.perf_counter() - start
            if next_id.item() in eos_token_ids:
                break
            ids = torch.cat((ids, next_id), -1)
            mask = torch.cat((mask, torch.ones_like(next_id)), -1)
            past = output.past_key_values if use_cache else None
        return {"token_ids": generated, "latency_seconds": time.perf_counter() - start,
                "first_token_seconds": first_latency,
                "truncated": generated[-1] not in eos_token_ids and len(generated) == max_new_tokens}

    def save_adapter(self, directory, metadata=None):
        from safetensors.torch import save_file
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.memory_module.state_dict().items()},
                  str(directory / "sqsm.safetensors"))
        config = asdict(self.sqsm_config)
        config.update(source_layer=self.source_layer, read_layer=self.read_layer)
        info = {"sqsm": config, "backbone": self.backbone.config._name_or_path,
                "hidden_size": self.backbone.config.hidden_size, "metadata": metadata or {}}
        (directory / "config.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    def load_adapter(self, directory):
        from safetensors.torch import load_file
        self.memory_module.load_state_dict(load_file(str(Path(directory) / "sqsm.safetensors")))

    def parameter_count(self):
        return sum(p.numel() for p in self.memory_module.parameters() if p.requires_grad)
