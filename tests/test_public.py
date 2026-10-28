"""Public behavioral checks, not the complete grading suite."""
import math
from dataclasses import replace
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from src.model import GPT, GPTConfig, CausalSelfAttention, Block
from src.evaluate import evaluate
from src.utils import learning_rate, load_model, save_checkpoint
torch.set_num_threads(2)

@pytest.fixture
def cfg():
    return GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0)

def test_shapes_gradients_and_weight_tying(cfg):
    model = GPT(cfg)
    x = torch.randint(32, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 32)
    assert model.lm_head.weight is model.transformer.wte.weight
    loss.backward()
    assert all((p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

def test_no_future_token_leakage(cfg):
    model = GPT(cfg).eval()
    x = torch.randint(32, (2, 12))
    altered = x.clone()
    altered[:, 7:] = (altered[:, 7:] + 1) % 32
    torch.testing.assert_close(model(x)[0][:, :7], model(altered)[0][:, :7], atol=1e-07, rtol=1e-06)

def test_next_token_loss(cfg):
    model = GPT(cfg)
    x, y = (torch.randint(32, (2, 8)), torch.randint(32, (2, 8)))
    logits, loss = model(x, y)
    torch.testing.assert_close(loss, F.cross_entropy(logits.flatten(0, 1), y.flatten()))

def test_generation_crops_context_and_top1_is_greedy(cfg):
    model = GPT(cfg).train()
    x = torch.randint(32, (1, 20))
    model.eval()
    expected = model(x[:, -16:])[0][:, -1].argmax(-1)
    model.train()
    out = model.generate(x, 1, top_k=1)
    assert model.training and out.shape == (1, 21)
    assert torch.equal(out[:, -1], expected)

def test_generation_seed_and_eos(cfg):
    model = GPT(cfg)
    x = torch.tensor([[1]])
    a = model.generate(x, 5, generator=torch.Generator().manual_seed(7))
    b = model.generate(x, 5, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    model.eval()
    eos = int(model(x)[0][0, -1].argmax())
    assert model.generate(x, 8, top_k=1, eos_id=eos).shape[1] == 2

@pytest.mark.parametrize('kwargs', [{'temperature': 0}, {'temperature': float('nan')}, {'top_k': 0}])
def test_bad_generation_rejected(cfg, kwargs):
    with pytest.raises(ValueError):
        GPT(cfg).generate(torch.tensor([[1]]), 2, **kwargs)

def test_tiny_batch_overfit(cfg):
    torch.manual_seed(5)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    x = torch.tensor([[1, 2, 3, 4]] * 4)
    y = torch.tensor([[2, 3, 4, 5]] * 4)
    initial = model(x, y)[1].item()
    for _ in range(50):
        opt.zero_grad()
        loss = model(x, y)[1]
        loss.backward()
        opt.step()
    assert loss.item() < 0.15 and loss.item() < initial / 10

def test_evaluation_counts_remainder(cfg):
    model = GPT(cfg)
    for p in model.parameters():
        p.data.zero_()
    result = evaluate(model, np.arange(71) % 32, batch_size=3, max_tokens=67)
    assert result['evaluated_tokens'] == 67
    assert abs(result['cross_entropy'] - math.log(32)) < 1e-06

def test_schedule_endpoints():
    cfg = {'warmup_steps': 2, 'max_steps': 10, 'learning_rate': 0.001, 'min_lr': 0.0001}
    assert learning_rate(0, cfg) == 0.0005
    assert learning_rate(1, cfg) == 0.001
    assert abs(learning_rate(9, cfg) - 0.0001) < 1e-12

def test_checkpoint_round_trip_and_nan_rejection(cfg, tmp_path):
    from dataclasses import asdict
    model = GPT(cfg).eval()
    path = tmp_path / 'model.pt'
    checkpoint = {'format_version': 1, 'model_config': asdict(cfg), 'model': model.state_dict()}
    save_checkpoint(path, checkpoint)
    restored, _ = load_model(path)
    x = torch.tensor([[1, 2]])
    torch.testing.assert_close(model(x)[0], restored(x)[0])
    checkpoint['model']['transformer.wte.weight'][0, 0] = float('nan')
    save_checkpoint(path, checkpoint)
    with pytest.raises(ValueError, match='non-finite'):
        load_model(path)

@pytest.mark.parametrize('kwargs', [{'n_embd': 17}, {'n_layer': 0}, {'dropout': 1.0}])
def test_invalid_model_config(cfg, kwargs):
    with pytest.raises(ValueError):
        replace(cfg, **kwargs)
