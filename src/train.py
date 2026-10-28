"""Offline single-GPU trainer with exact token accounting and resumable state."""
import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.model import GPT, GPTConfig
from src.utils import (VERSION, TokenData, amp_context, estimate_loss, learning_rate,
                       load_config, save_checkpoint, sha256, write_json)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-after-steps", type=int, help="testing interruption; does not change LR schedule")
    args = p.parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["training"]
    device = torch.device(args.device)
    torch.set_num_threads(4)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for tiny tests")
    if device.type == "cuda" and tcfg["precision"] == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support bfloat16; choose float16 and revalidate")
    torch.manual_seed(tcfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(tcfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = TokenData(args.data_dir, cfg["model"]["block_size"], cfg["model"]["vocab_size"])
    model = GPT(GPTConfig(**cfg["model"])).to(device)
    if model.parameter_count() > 20_000_000:
        raise ValueError("model exceeds 20M parameters")
    params = list(model.parameters())
    optimizer = torch.optim.AdamW([
        {"params": [p for p in params if p.ndim >= 2], "weight_decay": tcfg["weight_decay"]},
        {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}],
        lr=tcfg["learning_rate"], betas=(0.9, 0.95), fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and tcfg["precision"] == "float16")
    rng = torch.Generator().manual_seed(tcfg["seed"] + 1)
    out = args.out_dir
    if not args.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite run {out}; choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    step, best_loss, prior_seconds, train_seconds = 0, float("inf"), 0.0, 0.0
    prior_peak = 0.0
    if args.resume:
        ckpt = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        if ckpt["config"] != cfg or ckpt["data_fingerprint"] != data.fingerprint:
            raise ValueError("resume configuration/data mismatch")
        if ckpt["device_type"] != device.type:
            raise ValueError("resume on the same device type to retain RNG semantics")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        rng.set_state(ckpt["batch_rng"])
        torch.set_rng_state(ckpt["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        step, best_loss = ckpt["step"], ckpt["best_val_loss"]
        prior_seconds, train_seconds = ckpt["runtime_seconds"], ckpt["train_seconds"]
        prior_peak = ckpt["peak_gpu_memory_mb"]
    tokens_per_step = tcfg["batch_size"] * tcfg["gradient_accumulation"] * cfg["model"]["block_size"]
    if args.stop_after_steps is not None and not step < args.stop_after_steps <= tcfg["max_steps"]:
        raise ValueError("stop-after-steps must be greater than resumed step and <= max_steps")
    stop_step = args.stop_after_steps or tcfg["max_steps"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    write_json(out / "config.json", cfg)
    write_json(out / "environment.json", {"python": platform.python_version(),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "source_sha256": {str(p.relative_to(Path(__file__).parent.parent)): sha256(p)
                          for p in sorted(Path(__file__).parent.glob("*.py"))}})

    def checkpoint(losses):
        nonlocal best_loss
        runtime = prior_seconds + time.perf_counter() - started
        peak = max(prior_peak, torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0)
        basic = {"format_version": 1, "repository_version": VERSION, "model_config": cfg["model"],
            "model": model.state_dict(), "step": step, "train_tokens": step * tokens_per_step,
            "data_fingerprint": data.fingerprint, "tokenizer_sha256": data.manifest["files"]["tokenizer.json"]["sha256"],
            "val_loss": losses["val"], "seed": tcfg["seed"], "parameter_count": model.parameter_count()}
        # best.pt deliberately excludes optimizer and RNG state.
        if losses["val"] < best_loss:
            best_loss = losses["val"]
            save_checkpoint(out / "best.pt", basic)
        save_checkpoint(out / "last.pt", {**basic, "config": cfg, "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "batch_rng": rng.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "device_type": device.type, "best_val_loss": best_loss, "runtime_seconds": runtime,
            "train_seconds": train_seconds, "peak_gpu_memory_mb": peak})
        metrics = {"repository_version": VERSION, "status": "complete" if step == tcfg["max_steps"] else "incomplete",
            "seed": tcfg["seed"], "step": step, "max_steps": tcfg["max_steps"],
            "parameter_count": model.parameter_count(), "train_tokens": step * tokens_per_step,
            "tokens_per_step": tokens_per_step, "runtime_seconds": runtime, "train_seconds": train_seconds,
            "peak_gpu_memory_mb": peak, "final_train_loss": losses["train"], "final_val_loss": losses["val"],
            "final_val_perplexity": math.exp(losses["val"]), "best_val_loss": best_loss,
            "data_fingerprint": data.fingerprint, "precision": tcfg["precision"]}
        write_json(out / "metrics.json", metrics)
        with open(out / "eval.jsonl", "a") as f:
            f.write(json.dumps({"step": step, "train_tokens": step * tokens_per_step,
                "train_loss": losses["train"], "val_loss": losses["val"], "runtime_seconds": runtime}) + "\n")
        print(json.dumps(metrics), flush=True)

    if step == 0:
        checkpoint(estimate_loss(model, data, tcfg["batch_size"], tcfg["eval_batches"], device, tcfg["precision"]))
    model.train()
    with open(out / "train.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "tokens", "loss", "lr", "seconds"])
        if f.tell() == 0:
            writer.writeheader()
        while step < stop_step:
            if device.type == "cuda":
                torch.cuda.synchronize()
            begin = time.perf_counter()
            lr = learning_rate(step, tcfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(tcfg["gradient_accumulation"]):
                x, y = data.batch("train", tcfg["batch_size"], rng, device)
                with amp_context(device, tcfg["precision"]):
                    _, loss = model(x, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at step {step}")
                loss_sum += loss.detach().item() / tcfg["gradient_accumulation"]
                scaler.scale(loss / tcfg["gradient_accumulation"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            train_seconds += elapsed
            step += 1
            if step % tcfg["log_interval"] == 0 or step == stop_step:
                row = {"step": step, "tokens": step * tokens_per_step, "loss": loss_sum, "lr": lr, "seconds": elapsed}
                writer.writerow(row)
                f.flush()
                print(json.dumps(row), flush=True)
            if step % tcfg["eval_interval"] == 0 or step == stop_step:
                checkpoint(estimate_loss(model, data, tcfg["batch_size"], tcfg["eval_batches"], device, tcfg["precision"]))


if __name__ == "__main__":
    main()
