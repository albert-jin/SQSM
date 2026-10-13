"""Free-generation evaluation with per-question predictions and complete latency."""
import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm
from sqsm.data import PromptEncoder, read_records
from sqsm.metrics import correct, extract_answer
from sqsm.runtime import eos_ids, load_model, seed_everything


def evaluate_records(model, tokenizer, rows, max_new_tokens=1024, output=None, baseline=False):
    encoder = PromptEncoder(tokenizer)
    results = []
    model.eval()
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    handle = None
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        handle = Path(output).open("w", encoding="utf-8")
    try:
        for row in tqdm(rows, desc="generate"):
            prompt = encoder.prompt(row["question"])
            ids = torch.tensor([prompt["prompt_ids"]], device=model.device)
            qmask = torch.tensor([prompt["question_mask"]], device=model.device)
            result = model.generate(ids, torch.ones_like(ids), qmask,
                                    max_new_tokens=max_new_tokens, eos_token_ids=eos_ids(model, tokenizer), baseline=baseline)
            text = tokenizer.decode(result.pop("token_ids"), skip_special_tokens=True)
            result.update(id=row["id"], source=row["source"], output=text, gold=row["answer"],
                          prediction=extract_answer(text, row["source"]),
                          correct=correct(text, row["answer"], row["source"]))
            results.append(result)
            if handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
    finally:
        if handle:
            handle.close()
    n = len(results)
    if not n:
        raise ValueError("Empty evaluation split")
    return {"count": n, "accuracy": sum(r["correct"] for r in results) / n,
            "mean_latency_seconds": sum(r["latency_seconds"] for r in results) / n,
            "mean_first_token_seconds": sum(r["first_token_seconds"] for r in results) / n,
            "truncated": sum(r["truncated"] for r in results),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(model.device) if model.device.type == "cuda" else None}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--revision")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    if not args.checkpoint and not args.baseline:
        p.error("Specify --checkpoint for SQSM or --baseline for the frozen backbone")
    seed_everything(0)
    model, tokenizer = load_model(args.model, {}, args.device, args.dtype, args.checkpoint, args.revision)
    rows = read_records(args.data)
    if args.limit:
        rows = rows[:args.limit]
    metrics = evaluate_records(model, tokenizer, rows, args.max_new_tokens, args.output, args.baseline)
    metrics.update(model=args.model, checkpoint=args.checkpoint, dataset=args.data,
                   baseline=args.baseline, max_new_tokens=args.max_new_tokens)
    Path(args.output).with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
