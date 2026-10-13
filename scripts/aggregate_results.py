"""Aggregate actual evaluation artifacts; never fill missing seeds with paper values."""
import argparse
import json
from pathlib import Path
import statistics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    p.add_argument("--expected-seeds", type=int, default=5)
    a = p.parse_args()
    files = sorted(Path(a.directory).glob("seed-*/test.metrics.json"))
    values = [json.loads(f.read_text())["accuracy"] * 100 for f in files]
    if len(values) != a.expected_seeds:
        raise SystemExit(f"Expected {a.expected_seeds} seeds; found {len(values)}")
    print(json.dumps({"seeds": len(values), "mean_percent": statistics.mean(values),
                      "sample_std_percent": statistics.stdev(values) if len(values) > 1 else 0.0}, indent=2))
