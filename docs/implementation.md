# Mapping from manuscript to code

| Manuscript component | Implementation |
|---|---|
| Detached source states after block ls | SQSMModel.source_states; early exit, no gradients, no KV reuse |
| Question-only mask | PromptEncoder.prompt; actual template and tokenizer offsets |
| U = N(H) Ws; learned E | SoftMemory.source and SoftMemory.slots |
| Slot compression and RMS normalization | SoftMemory.build |
| q = N(h) Wq; M Wk and M Wv | SoftMemory.inject and Memory |
| Zero-initialized Wo | SoftMemory.output |
| State-scaled tanh bound | SoftMemory.inject; scale detached |
| Last prompt and answer mask | PromptEncoder.training_example and SQSMModel.generate |
| Weighted task loss | SQSMModel.forward; shifted labels, per-sample normalization |
| Relative residual penalty | SoftMemory.inject; valid injection positions only |
| Frozen upper blocks transmit gradients | Only backbone parameters are frozen; pass 2 retains autograd |
| Fixed memory with exact causal KV reuse | generate reuses request-local Memory; fresh pass-2 prefill |
| Five seeds, dev-generation selection | scripts/run_experiments.py and train.py |
| DU cross-task training | prepare_data.py creates union train/dev, never DU train |

The framework figure describes the implemented three-module path. Publication
numbers and figures are retained as manuscript artifacts, never test fixtures.

## Explicit reconstruction choices

- The manuscript takes precedence over idea.txt when their defaults differ:
  five seeds, separate task models, and a four-task union model for DU.
- Seeds 0–4 and split seed 42 are chosen defaults because original identifiers
  were not supplied. Model revisions should be pinned before a reported run.
- StrategyQA follows upstream file-order 80/20 and uses supplied facts as rationale.
- ASDiv paraphrase-family IDs are unavailable; grouping currently covers lexical
  duplicates and numeric variants, not all semantic paraphrases.
- The bias-free adapter is nearest-count matched, not mislabeled as exactly
  identical capacity. Its true trainable count is saved.
- Source caching is optional and keyed by backbone identity/revision, layer,
  dtype and exact prompt tokens/masks. No cache is shared across output runs.
- Core SQSM, static query, direct reading, unbounded and no-regularizer variants
  are implemented. Specialized figure diagnostics and every appendix ablation
  have not been automatically reconstructed without original analysis artifacts.
- Generation is request-at-a-time. Training supports padded batches. Parallel
  calls on one model instance and gradient-checkpointed recomputation are not
  supported by the current scoped-hook integration.
- Exact cache equivalence means the same causal computation; normal floating
  point differences may remain between dense and cached kernels.

## Evidence needed for numerical reproduction

Run the 32-example server-side learning check, then the full training grid with
authorized model weights and sufficient GPU memory. Archive split manifests,
resolved backbone revisions, per-question predictions, filtering records and
five-seed aggregate statistics. Local CPU correctness tests do not establish
the manuscript's benchmark accuracy, timing or GPU memory claims.
