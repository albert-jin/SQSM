"""Download public source files and prepare deterministic SQSM JSONL splits.

Raw files and generated manifests live exclusively in gitignored data/.
No remote code execution, fabricated rationales, or test-to-train reuse.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import urllib.request
import zipfile


SOURCES = {
    "asdiv_train": "https://huggingface.co/datasets/xuyige/ASDiv-Aug/resolve/main/asdiv-aug-train.jsonl",
    "asdiv_test": "https://huggingface.co/datasets/xuyige/ASDiv-Aug/resolve/main/asdiv-aug-test.jsonl",
    "aqua_train": "https://raw.githubusercontent.com/google-deepmind/AQuA/master/train.json",
    "aqua_dev": "https://raw.githubusercontent.com/google-deepmind/AQuA/master/dev.json",
    "aqua_test": "https://raw.githubusercontent.com/google-deepmind/AQuA/master/test.json",
    "strategyqa": "https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip",
    "du": "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh/date_understanding.json",
    "gsm8k_train": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train_socratic.jsonl",
    "gsm8k_test": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test_socratic.jsonl",
}


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temp = path.with_suffix(path.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "SQSM-data-preparation/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temp.replace(path)
    return path


def load_json(path):
    text = path.read_text(encoding="utf-8-sig")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.split("\n") if line.strip()]


def fingerprint(question):
    # Groups lexical duplicates AND numeric variants. Paraphrase families need
    # original family IDs; public ASDiv-Aug does not supply these.
    return re.sub(r"\s+", " ", re.sub(r"\d+(?:\.\d+)?", "<n>", question.lower())).strip()


def group_holdout(rows, target, seed):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("group_id") or fingerprint(row["question"])].append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    held, train = [], []
    for key in keys:
        (held if len(held) < target else train).extend(groups[key])
    return train, held


def record(dataset, i, question, rationale, answer, **extra):
    return dict(id=f"{dataset}:{i}", question=question.strip(), rationale=rationale.strip(),
                answer=str(answer).strip(), source=dataset, **extra)


def gsm_style(rows, dataset):
    result = []
    for i, row in enumerate(rows):
        rationale, marker, answer = row["answer"].rpartition("####")
        if not marker:
            raise ValueError(f"Missing final-answer marker in {dataset}:{i}")
        result.append(record(dataset, i, row["question"], rationale, answer))
    return result


def write_rows(path, rows, split):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row, split=split), ensure_ascii=False) + "\n")


def prepare(root, seed=42):
    root = Path(root)
    raw = root / "raw"
    files = {}
    for name, url in SOURCES.items():
        # Reuse the user's existing GSM8K rather than fetch it again.
        existing = root / "gsm8k" / ("train_socratic.jsonl" if name.endswith("train") else "test_socratic.jsonl")
        if name.startswith("gsm8k") and existing.exists():
            files[name] = existing
        else:
            files[name] = download(url, raw / (name + (".zip" if name == "strategyqa" else ".jsonl")))
        print(f"ready: {name} ({files[name].stat().st_size:,} bytes)", flush=True)
    datasets = {}
    for ds, prefix in [("gsm8k", "gsm8k"), ("asdiv", "asdiv")]:
        train = gsm_style(load_json(files[prefix + "_train"]), ds)
        test = gsm_style(load_json(files[prefix + "_test"]), ds)
        for row in test:
            row["id"] = row["id"].replace(":", ":test:", 1)
        datasets[ds] = (train, test)
    def aqua(rows, split):
        return [record("aqua", f"{split}:{i}", r["question"] + "\n" + "\n".join(r["options"]),
                       r.get("rationale", ""), r["correct"], options=r["options"])
                for i, r in enumerate(rows)]
    datasets["aqua"] = (aqua(load_json(files["aqua_train"]), "train"),
                          aqua(load_json(files["aqua_test"]), "test"))
    write_rows(root / "aqua" / "official_dev.jsonl", aqua(load_json(files["aqua_dev"]), "official_dev"), "official_dev")
    with zipfile.ZipFile(files["strategyqa"]) as archive:
        member = next(n for n in archive.namelist() if n.endswith("strategyqa_train.json"))
        original = json.loads(archive.read(member))
    strategy = [record("strategyqa", r.get("qid", i), r["question"], " ".join(r.get("facts", [])),
                       "yes" if r["answer"] else "no") for i, r in enumerate(original)]
    # Preserve the baseline's official-file-order 80/20 partition, then create
    # a separate development partition from the training portion only.
    boundary = int(len(strategy) * .8)
    datasets["strategyqa"] = (strategy[:boundary], strategy[boundary:])
    du = [record("du", i, r["input"], "", r["target"].strip("()"))
          for i, r in enumerate(load_json(files["du"])["examples"])]
    write_rows(root / "du" / "test.jsonl", du, "test")
    manifests, all_train, all_dev = {}, [], []
    for dataset, (training, test) in datasets.items():
        test_keys = {fingerprint(r["question"]) for r in test}
        eligible = [r for r in training if fingerprint(r["question"]) not in test_keys]
        fit, dev = group_holdout(eligible, 150 if dataset == "strategyqa" else 200, seed)
        for name, rows in [("train", fit), ("dev", dev), ("test", test)]:
            write_rows(root / dataset / f"{name}.jsonl", rows, name)
        manifests[dataset] = {"raw_train": len(training), "train": len(fit), "dev": len(dev),
                              "test": len(test), "excluded_train_test_overlap": len(training) - len(eligible)}
        sampled = list(fit)
        random.Random(seed).shuffle(sampled)
        all_train.extend(sampled[:7473])
        all_dev.extend(dev)
    write_rows(root / "union" / "train.jsonl", all_train, "train")
    write_rows(root / "union" / "dev.jsonl", all_dev, "dev")
    manifests["du"] = {"train": 0, "test": len(du), "training_module": "union"}
    manifests["sources"] = {name: {"url": SOURCES[name], "bytes": p.stat().st_size,
                                       "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                            for name, p in files.items()}
    manifests["split_seed"] = seed
    manifests["limitations"] = ["ASDiv-Aug public files do not include paraphrase-family IDs; only lexical/numeric variants are grouped.",
        "StrategyQA facts are used as available reference rationale, not newly generated CoT.",
        "Training/dev/test source identity manifests from the manuscript were not supplied; these are documented reconstruction splits.",
        "DU is the 250-item BBH subset, not the full BIG-bench task."]
    (root / "manifest.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifests.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.data_dir, args.seed)
