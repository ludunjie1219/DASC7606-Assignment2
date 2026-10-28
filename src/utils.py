import hashlib
import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

from .model import GPT, GPTConfig

VERSION = "assignment2-rc1"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    model = GPTConfig(**config["model"])
    train = config["training"]
    for field in ("batch_size", "gradient_accumulation", "max_steps", "eval_interval", "eval_batches", "log_interval"):
        if type(train[field]) is not int or train[field] <= 0:
            raise ValueError(f"invalid {field}")
    if not 0 <= train["warmup_steps"] < train["max_steps"]:
        raise ValueError("warmup must be smaller than max_steps")
    if not 0 <= train["min_lr"] <= train["learning_rate"]:
        raise ValueError("invalid learning rate range")
    for field in ("min_lr", "learning_rate", "weight_decay", "grad_clip"):
        if not math.isfinite(train[field]) or train[field] < 0:
            raise ValueError(f"invalid {field}")
    if train["learning_rate"] == 0 or train["grad_clip"] == 0:
        raise ValueError("learning rate and gradient clip must be positive")
    if train["precision"] not in ("float32", "bfloat16", "float16"):
        raise ValueError("invalid precision")
    if model.vocab_size > 65536:
        raise ValueError("uint16 data requires vocab_size <= 65536")
    return config


def amp_context(device, precision):
    if device.type == "cpu" or precision == "float32":
        return nullcontext()
    return torch.autocast("cuda", dtype=getattr(torch, precision))


class TokenData:
    def __init__(self, directory, block_size, vocab_size, verify=True):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.fingerprint = sha256(self.directory / "manifest.json")
        if self.manifest["vocab_size"] != vocab_size:
            raise ValueError("tokenizer vocabulary does not match model")
        self.block_size = block_size
        self.arrays = {}
        for name in ("train", "val"):
            path = self.directory / f"{name}.bin"
            if verify and sha256(path) != self.manifest["files"][path.name]["sha256"]:
                raise ValueError(f"data checksum mismatch: {path.name}")
            a = np.memmap(path, dtype="<u2", mode="r")
            if len(a) <= block_size or int(a.max()) >= vocab_size:
                raise ValueError(f"invalid token data: {name}")
            self.arrays[name] = a
        if verify and sha256(self.directory / "tokenizer.json") != self.manifest["files"]["tokenizer.json"]["sha256"]:
            raise ValueError("tokenizer checksum mismatch")

    def batch(self, split, batch_size, generator, device):
        a = self.arrays[split]
        starts = torch.randint(len(a) - self.block_size, (batch_size,), generator=generator).tolist()
        batch = np.stack([a[i:i + self.block_size + 1] for i in starts]).astype(np.int64)
        t = torch.from_numpy(batch).to(device)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()


@torch.no_grad()
def estimate_loss(model, data, batch_size, batches, device, precision):
    was_training = model.training
    model.eval()
    losses = {}
    try:
        for split in ("train", "val"):
            rng = torch.Generator().manual_seed(2026)
            total = 0.0
            for _ in range(batches):
                x, y = data.batch(split, batch_size, rng, device)
                with amp_context(device, precision):
                    _, loss = model(x, y)
                total += loss.item()
            losses[split] = total / batches
    finally:
        model.train(was_training)
    return losses


def learning_rate(step, cfg):
    # step is zero based; the first update uses a nonzero learning rate.
    if step < cfg["warmup_steps"]:
        return cfg["learning_rate"] * (step + 1) / max(1, cfg["warmup_steps"])
    progress = (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"] - 1)
    return cfg["min_lr"] + 0.5 * (1 + math.cos(math.pi * min(1, progress))) * (cfg["learning_rate"] - cfg["min_lr"])


def save_checkpoint(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def load_model(path, device="cpu"):
    # No pickle globals or submitted Python are executed by this loader.
    if Path(path).stat().st_size > 150 * 1024**2:
        raise ValueError("checkpoint exceeds 150 MiB loader limit")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] != 1:
        raise ValueError("unsupported checkpoint version")
    cfg = GPTConfig(**checkpoint["model_config"])
    c = cfg.n_embd
    expected_parameters = cfg.vocab_size * c + cfg.block_size * c + cfg.n_layer * (12*c*c + 13*c) + 2*c
    if expected_parameters > 20_000_000 or cfg.block_size > 1024:
        raise ValueError("model exceeds the 20M parameter limit")
    model = GPT(cfg)
    for value in checkpoint["model"].values():
        if not torch.isfinite(value).all():
            raise ValueError("checkpoint contains non-finite values")
    if not torch.equal(checkpoint["model"]["transformer.wte.weight"], checkpoint["model"]["lm_head.weight"]):
        raise ValueError("tied embedding weights disagree")
    model.load_state_dict(checkpoint["model"], strict=True)
    for value in model.state_dict().values():
        if not torch.isfinite(value).all():
            raise ValueError("checkpoint contains non-finite values")
    return model.to(device).eval(), checkpoint
