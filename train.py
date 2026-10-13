"""Train only SQSM parameters; select checkpoints on held-out free generation."""
import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from sqsm.data import PromptEncoder, collate, read_records
from sqsm.runtime import load_model, seed_everything
from evaluate import evaluate_records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/qwen2.5-7b.json")
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int)
    p.add_argument("--limit", type=int)
    p.add_argument("--dev-limit", type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--variant", choices=["sqsm", "static", "direct", "unbounded", "adapter"])
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--no-reg", action="store_true")
    p.add_argument("--cache-sources", action="store_true")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.model:
        cfg["model"] = args.model
    if args.variant:
        cfg["sqsm"]["variant"] = args.variant
    if args.no_reg:
        cfg["sqsm"]["reg_weight"] = 0.0
    training = cfg["training"]
    if args.epochs:
        training["epochs"] = args.epochs
    if args.learning_rate:
        training["learning_rate"] = args.learning_rate
    seed_everything(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists():
        raise ValueError("Output already contains a run; choose a fresh directory")
    rows, dev = read_records(args.train), read_records(args.dev)
    if {r["id"] for r in rows} & {r["id"] for r in dev}:
        raise ValueError("Train/dev example IDs overlap")
    if args.limit:
        rows = rows[:args.limit]
    if args.dev_limit:
        dev = dev[:args.dev_limit]
    model, tokenizer = load_model(cfg["model"], cfg["sqsm"], args.device, args.dtype,
                                  revision=cfg.get("revision"))
    encoder = PromptEncoder(tokenizer, training["max_length"], model.sqsm_config.answer_weight)
    examples, filtered = [], []
    for row in tqdm(rows, desc="tokenize"):
        encoded = encoder.training_example(row)
        if encoded is None:
            filtered.append(row["id"])
        else:
            examples.append(encoded)
    if not examples:
        raise ValueError("No eligible training examples")
    metadata = {"config": cfg, "seed": args.seed, "arguments": vars(args),
                "model_revision": getattr(model.backbone.config, "_commit_hash", None),
                "train_sha256": hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
                "dev_sha256": hashlib.sha256(Path(args.dev).read_bytes()).hexdigest(),
                "trainable_parameters": model.parameter_count(), "filtered_ids": filtered,
                "retained_examples": len(examples), "torch": torch.__version__}
    (output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=training["learning_rate"], weight_decay=training["weight_decay"])
    loader = DataLoader(examples, batch_size=training["micro_batch_size"], shuffle=True,
                        collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id))
    accum = training["gradient_accumulation_steps"]
    updates = math.ceil(len(loader) / accum) * training["epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(updates * training["warmup_ratio"]), updates)
    best, step = -1.0, 0
    cache = output / "source_cache"
    if args.cache_sources:
        cache.mkdir(exist_ok=True)
    for epoch in range(training["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for index, batch in enumerate(tqdm(loader, desc=f"epoch {epoch + 1}")):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            if args.cache_sources and model.sqsm_config.variant != "adapter":
                # Safe cache key includes model revision, layer and exact token/mask bytes.
                digest = hashlib.sha256(json.dumps([cfg["model"], metadata["model_revision"], model.source_layer, args.dtype]).encode())
                digest.update(batch["prompt_ids"].cpu().numpy().tobytes())
                digest.update(batch["prompt_mask"].cpu().numpy().tobytes())
                path = cache / (digest.hexdigest() + ".pt")
                if path.exists():
                    batch["source_states"] = torch.load(path, weights_only=True, map_location=model.device)
                else:
                    h = model.source_states(batch["prompt_ids"], batch["prompt_mask"])
                    torch.save(h.cpu(), path)
                    batch["source_states"] = h
            # Normalize a short final accumulation window by its actual size.
            group_start = (index // accum) * accum
            divisor = min(accum, len(loader) - group_start)
            loss = model(**batch)["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            (loss / divisor).backward()
            total_loss += loss.item()
            if (index + 1) % accum == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(parameters, training["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
        metrics = evaluate_records(model, tokenizer, dev, training["max_new_tokens"],
                                   output / f"dev-epoch-{epoch + 1}.jsonl")
        metrics.update(epoch=epoch + 1, step=step, train_loss=total_loss / len(loader))
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
        model.save_adapter(output / "last", metadata)
        if metrics["accuracy"] > best:
            best = metrics["accuracy"]
            model.save_adapter(output / "best", dict(metadata, dev_metrics=metrics))
        print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
