# Validation record

Date: 2026-10-13.

## Local checks

24 tests passed on CPU with Python 3.12.14, PyTorch 2.5.1+cpu,
Transformers 4.44.2 and pytest 9.1.1. The manuscript's CUDA/PyTorch 2.3.1
environment has not been exercised here.

The tests cover:

- LLaMA and Qwen2 tiny randomly initialized backbones.
- Exact zero-initialized SQSM/backbone output identity.
- First-step output-projection gradients and subsequent builder/reader gradients.
- Frozen backbone parameters remaining gradient-free.
- Full versus cached decoding with a nonzero module for SQSM, static and direct variants.
- Future-token isolation, question masking and padding behavior.
- Residual bounds, inactive positions, label shift and per-sample weighted loss.
- Exact trainable parameter count and early stopping of pass 1.
- Adapter checkpoint save/load equivalence.
- Loss reduction on a tiny synthetic batch and independent sequential requests.
- Chat-template question alignment and strict final-answer parsing.
- End-to-end training, development generation, checkpoint saving and evaluation
  through the command-line entry points using a local toy model and tokenizer.

Release packaging audit passed: root README contains no hyperlink syntax or
external URLs, displayed figure assets exist, recognized credential patterns are
absent from staged text files, no dataset files are tracked, and split IDs are
disjoint. All evaluation gold answers are accepted by the task-specific parser.

## Prepared data

| Dataset | Fit | Development | Test |
|---|---:|---:|---:|
| GSM8K | 7,273 | 200 | 1,319 |
| ASDiv-Aug | 3,980 | 203 | 1,038 |
| AQuA | 97,247 | 200 | 254 |
| StrategyQA | 1,682 | 150 | 458 |
| Date Understanding | 0 | 0 | 250 |

Twenty AQuA training examples overlap the evaluation set under the conservative
lexical/numeric grouping rule and are excluded. ASDiv's grouped development set
has 203 examples. Raw source files, exact hashes and the union-training manifest
are stored locally in data/. None is included in the Git publication payload.

## Not yet established

No 7B/8B model training, A100 memory measurement, five-seed benchmark reproduction
or manuscript-accuracy verification was performed on this machine. The local
toy-model tests establish implementation properties and command execution only.
Manuscript result figures are illustrations copied from the paper; they are not
evidence of new runs. Exact original split IDs and paraphrase-family identifiers
remain unavailable, as described in docs/implementation.md.
