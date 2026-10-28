"""Verify the fixed data and dependencies without running unfinished model methods."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tokenizers import Tokenizer
from src.model import GPT, GPTConfig
from src.utils import TokenData, load_config

if __name__ == "__main__":
    torch.set_num_threads(2)
    cfg = load_config("configs/baseline.yaml")
    data = TokenData("data", cfg["model"]["block_size"], cfg["model"]["vocab_size"])
    tok = Tokenizer.from_file("data/tokenizer.json")
    assert tok.get_vocab_size() == cfg["model"]["vocab_size"]
    assert tok.token_to_id("<|endoftext|>") == data.manifest["eos_id"]
    print(json.dumps({"status": "setup_pass", "torch": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(), "parameters": GPT(GPTConfig(**cfg["model"])).parameter_count(),
        "data_fingerprint": data.fingerprint, "note": "TODO correctness has not been tested"}, indent=2))
