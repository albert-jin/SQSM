"""Check repository packaging before publication; no uploads."""
import json
from pathlib import Path
import re
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert not re.search(r"https?://|www\.|<a\b|(?<!!)\[[^\]]+\]\(", readme, re.I), "README contains hyperlinks"
    for name in re.findall(r'<img src="([^"]+)"', readme):
        assert (root / name).is_file(), f"Missing image: {name}"
    tracked = subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()
    assert not any(p.startswith("data/") for p in tracked), "Dataset files are tracked"
    subprocess.run(["git", "check-ignore", "data/manifest.json"], cwd=root, check=True, capture_output=True)
    forbidden = re.compile(r"(?:hf_[A-Za-z0-9]{24,}|gh[pousr]_[A-Za-z0-9]{24,}|sk-[A-Za-z0-9]{24,})")
    for name in tracked:
        path = root / name
        if path.suffix.lower() in {".py", ".json", ".md", ".sh", ".txt", ".toml"}:
            assert not forbidden.search(path.read_text(encoding="utf-8")), f"Credential pattern in {name}"
    manifest = root / "data" / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        assert [data[k]["test"] for k in ["gsm8k", "asdiv", "aqua", "strategyqa", "du"]] == [1319, 1038, 254, 458, 250]
        for task in ["gsm8k", "asdiv", "aqua", "strategyqa"]:
            subsets = []
            for split in ["train", "dev", "test"]:
                with (root / "data" / task / f"{split}.jsonl").open(encoding="utf-8") as handle:
                    rows = [json.loads(s) for s in handle if s.strip()]
                ids = {r["id"] for r in rows}
                assert len(ids) == len(rows), f"Duplicate IDs: {task}/{split}"
                subsets.append(ids)
            assert not (subsets[0] & subsets[1] or subsets[0] & subsets[2] or subsets[1] & subsets[2]), task
    print("Release audit passed: README, figures, credentials, dataset exclusion and split IDs.")


if __name__ == "__main__":
    main()
