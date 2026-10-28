import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("run", type=Path)
args = p.parse_args()
rows = [json.loads(line) for line in (args.run / "eval.jsonl").read_text().splitlines()]
fig, ax = plt.subplots(figsize=(7, 4))
for key in ("train_loss", "val_loss"):
    ax.plot([r["train_tokens"] / 1e6 for r in rows], [r[key] for r in rows], label=key)
ax.set(xlabel="Processed training tokens (millions)", ylabel="Cross-entropy (nats)")
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(args.run / "loss_curve.png", dpi=160)
