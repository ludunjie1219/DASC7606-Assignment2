"""Checkpoint-only evaluation; never imports submitted Python or retrains a model."""
import argparse
import math
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.utils import load_model, sha256, write_json


@torch.no_grad()
def evaluate(model, tokens, batch_size=16, max_tokens=262144, device="cpu"):
    # Disjoint target windows: every evaluated target is scored exactly once.
    length = model.config.block_size
    usable = min(len(tokens) - 1, max_tokens) if max_tokens else len(tokens) - 1
    if usable < 1 or batch_size < 1:
        raise ValueError("evaluation requires nonempty token data and positive batch size")
    total_loss, count = 0.0, 0
    model.eval()
    full = usable // length
    for start in range(0, full, batch_size):
        rows = [np.asarray(tokens[i * length:i * length + length + 1], dtype=np.int64)
                for i in range(start, min(full, start + batch_size))]
        t = torch.from_numpy(np.stack(rows)).to(device)
        _, loss = model(t[:, :-1], t[:, 1:])
        n = t.shape[0] * length
        total_loss += loss.item() * n
        count += n
    remainder = usable - count
    if remainder:
        t = torch.tensor(np.array(tokens[count:count + remainder + 1], dtype=np.int64), device=device)[None]
        _, loss = model(t[:, :-1], t[:, 1:])
        total_loss += loss.item() * remainder
        count += remainder
    ce = total_loss / count
    if not math.isfinite(ce):
        raise ValueError("non-finite evaluation loss")
    return {"cross_entropy": ce, "perplexity": math.exp(ce), "evaluated_tokens": count,
            "context_length": length, "precision": "float32",
            "protocol": "non-overlapping next-token windows with exact weighted remainder"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=262144, help="0 evaluates every target")
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.max_tokens < 0:
        raise ValueError("max-tokens cannot be negative")
    model, checkpoint = load_model(args.checkpoint, args.device)
    if sha256(args.tokenizer) != checkpoint["tokenizer_sha256"]:
        raise ValueError("checkpoint/tokenizer mismatch")
    tokens = np.memmap(args.tokens, dtype="<u2", mode="r")
    if len(tokens) < 2 or int(tokens.max()) >= model.config.vocab_size:
        raise ValueError("invalid evaluation tokens")
    result = evaluate(model, tokens, args.batch_size, args.max_tokens, args.device)
    result.update(checkpoint_sha256=sha256(args.checkpoint), tokens_sha256=sha256(args.tokens),
                  model_parameters=model.parameter_count(), checkpoint_step=checkpoint["step"])
    write_json(args.out, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
