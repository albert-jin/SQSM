"""Sequential paper-grid launcher. Dry-run by default; --execute runs commands."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=["configs/qwen2.5-7b.json", "configs/llama3.1-8b.json"])
    p.add_argument("--tasks", nargs="+", default=["gsm8k", "asdiv", "aqua", "strategyqa", "union"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    for config in args.configs:
        model_id = json.loads(Path(config).read_text())["model"]
        for task in args.tasks:
            for seed in args.seeds:
                run = f"runs/{Path(config).stem}/{task}/seed-{seed}"
                commands = [[sys.executable, "train.py", "--config", config, "--train", f"data/{task}/train.jsonl",
                    "--dev", f"data/{task}/dev.jsonl", "--seed", str(seed), "--output", run,
                    "--epochs", "1" if task == "aqua" else "2", "--cache-sources"],
                    [sys.executable, "evaluate.py", "--model", model_id, "--checkpoint", run + "/best",
                     "--data", f"data/{'du' if task == 'union' else task}/test.jsonl", "--output", run + "/test.jsonl"]]
                for command in commands:
                    print(shlex.join(command), flush=True)
                    if args.execute:
                        subprocess.run(command, check=True)
