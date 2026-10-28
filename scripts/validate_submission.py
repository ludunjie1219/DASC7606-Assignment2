"""Public draft format checker. Does not execute archived student Python or award marks."""
import argparse
import ast
import json
import math
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_model, sha256, write_json

ROLES = ("baseline", "model_experiment", "optimizer_experiment")
EVIDENCE = ("metrics.json", "config.json", "environment.json", "train.csv", "eval.jsonl",
            "samples.json", "loss_curve.png", "evaluation.json")
REQUIRED = {"submission.json", "report.pdf", "best.pt", "src/model.py", "src/train.py",
            "src/utils.py", "src/sample.py", "src/evaluate.py", "requirements.txt", "LICENSE", "README.md"}


def inspect_zip(path):
    with zipfile.ZipFile(path) as z:
        entries = z.infolist()
        names = [i.filename for i in entries]
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive paths")
        if len(names) > 2000 or sum(i.file_size for i in entries) > 200 * 1024**2:
            raise ValueError("package exceeds 200 MiB uncompressed or 2000 files")
        for i in entries:
            p = PurePosixPath(i.filename)
            if p.is_absolute() or ".." in p.parts or "\\" in i.filename or ":" in i.filename:
                raise ValueError("unsafe archive path")
            if stat.S_ISLNK(i.external_attr >> 16):
                raise ValueError("symlinks are forbidden")
            if any(part in ("data", "private", "raw", ".git", ".venv", "__pycache__") for part in p.parts):
                raise ValueError("data, private files, environments and caches are forbidden")
            if p.suffix in (".pt", ".pth") and i.filename != "best.pt":
                raise ValueError("only one best.pt checkpoint may be submitted")
        required = REQUIRED | {f"runs/{role}/{f}" for role in ROLES for f in EVIDENCE}
        if not required.issubset(names):
            raise ValueError(f"missing required files: {sorted(required-set(names))}")
        if z.read("report.pdf")[:5] != b"%PDF-":
            raise ValueError("report is not a PDF")
        tree = ast.parse(z.read("src/model.py").decode())
        if any(isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call)
               and isinstance(n.exc.func, ast.Name) and n.exc.func.id == "NotImplementedError"
               for n in ast.walk(tree)):
            raise ValueError("model still contains unimplemented TODOs")
    return names


def validate(path, data_dir="data"):
    names = inspect_zip(path)
    data_dir = Path(data_dir)
    dataset_hash = sha256(data_dir / "manifest.json")
    tokenizer_hash = sha256(data_dir / "tokenizer.json")
    with tempfile.TemporaryDirectory(prefix="a2-format-") as temp, zipfile.ZipFile(path) as z:
        info = json.loads(z.read("submission.json"))
        if info.get("format_version") != 1 or not str(info.get("student_id", "")).isdigit():
            raise ValueError("invalid submission manifest or student ID")
        if info["selected_run"] not in ROLES:
            raise ValueError("invalid selected run")
        z.extract("best.pt", temp)
        checkpoint_path = Path(temp) / "best.pt"
        model, ckpt = load_model(checkpoint_path)
        if sha256(checkpoint_path) != info["checkpoint_sha256"]:
            raise ValueError("checkpoint checksum mismatch")
        if ckpt["tokenizer_sha256"] != tokenizer_hash or ckpt["data_fingerprint"] != dataset_hash:
            raise ValueError("checkpoint/data/tokenizer mismatch")
        if model.config.vocab_size != 8192 or model.config.block_size != 256:
            raise ValueError("fixed vocabulary/context mismatch")
        if ckpt["step"] <= 0 or not math.isfinite(ckpt["val_loss"]):
            raise ValueError("untrained or invalid checkpoint")
        if model.parameter_count() != ckpt["parameter_count"]:
            raise ValueError("parameter metadata mismatch")
        budgets = []
        for role in ROLES:
            cfg = json.loads(z.read(f"runs/{role}/config.json"))
            m = json.loads(z.read(f"runs/{role}/metrics.json"))
            t = cfg["training"]
            per_step = t["batch_size"] * t["gradient_accumulation"] * cfg["model"]["block_size"]
            budget = t["max_steps"] * per_step
            if m["status"] != "complete" or m["step"] != t["max_steps"] or m["train_tokens"] != budget:
                raise ValueError("incomplete run or incorrect token accounting")
            if m["data_fingerprint"] != dataset_hash or not math.isfinite(m["best_val_loss"]):
                raise ValueError("run data or loss mismatch")
            budgets.append(budget)
            if role == info["selected_run"]:
                if cfg["model"] != ckpt["model_config"] or ckpt["train_tokens"] != ckpt["step"] * per_step:
                    raise ValueError("selected checkpoint/config mismatch")
                if ckpt["step"] > t["max_steps"]:
                    raise ValueError("checkpoint exceeds training budget")
                if abs(ckpt["val_loss"] - m["best_val_loss"]) > 1e-6:
                    raise ValueError("selected checkpoint is not the recorded best checkpoint")
        if len(set(budgets)) != 1 or budgets[0] != 10_240_000:
            raise ValueError("draft protocol requires 10,240,000 tokens for each run")
    return {"status": "format_pass", "archive_sha256": sha256(path), "files": len(names),
            "parameters": model.parameter_count(), "tokens_per_run": budgets[0],
            "note": "Not a grade, report review, correctness proof, or training-provenance proof."}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out")
    args = p.parse_args()
    result = validate(args.archive, args.data_dir)
    if args.out:
        write_json(args.out, result)
    print(json.dumps(result, indent=2))
