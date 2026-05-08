"""
Minimal test to diagnose backward() failure on CUDA.
Run on VM: python scripts/test_backward.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import scripts.ssl_fix
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

import torch
import torch.nn.functional as F
from comet import download_model, load_from_checkpoint

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
local_ckpt = "/tmp/cometkiwi22/checkpoints/model.ckpt"
if os.path.exists(local_ckpt):
    model_path = local_ckpt
else:
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model = load_from_checkpoint(model_path)
model = model.to(device)

# Freeze encoder
for name, param in model.named_parameters():
    if "encoder" in name or "layernorm_embedding" in name or "embed_tokens" in name:
        param.requires_grad = False

trainable = sum(1 for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable}")

# ---- TEST 1: Minimal forward+backward ----
print("\n=== TEST 1: Minimal forward+backward ===")
model.train()
samples = [{"src": "Hello", "mt": "Hallo"}]
batch = model.prepare_sample(samples, stage="predict")
inp = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch[0].items()}
pred = model.forward(**inp)
loss = pred.score.mean()
print(f"score={pred.score}, requires_grad={pred.score.requires_grad}, device={pred.score.device}")
print(f"grad_fn={pred.score.grad_fn}")
try:
    loss.backward()
    print("TEST 1 PASSED")
except RuntimeError as e:
    print(f"TEST 1 FAILED: {e}")
model.zero_grad()

# ---- TEST 2: Two samples ----
print("\n=== TEST 2: Two samples ===")
model.train()
samples = [{"src": "Hello world", "mt": "Hallo Welt"}, {"src": "Good day", "mt": "Guten Tag"}]
batch = model.prepare_sample(samples, stage="predict")
inp = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch[0].items()}
pred = model.forward(**inp)
loss = pred.score.mean()
try:
    loss.backward()
    print("TEST 2 PASSED")
except RuntimeError as e:
    print(f"TEST 2 FAILED: {e}")
model.zero_grad()

# ---- TEST 3: Two separate forward passes + combined loss ----
print("\n=== TEST 3: Two forward passes + combined loss ===")
model.train()
s1 = [{"src": "Hello world", "mt": "Hallo Welt"}]
s2 = [{"src": "Hello world", "mt": "Hallo schlecht"}]
b1 = model.prepare_sample(s1, stage="predict")
b2 = model.prepare_sample(s2, stage="predict")
i1 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b1[0].items()}
i2 = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b2[0].items()}
p1 = model.forward(**i1).score
p2 = model.forward(**i2).score
loss = torch.clamp(0.1 - (p1 - p2), min=0).mean()
print(f"p1={p1.item():.4f}, p2={p2.item():.4f}, loss={loss.item():.6f}")
try:
    loss.backward()
    print("TEST 3 PASSED")
except RuntimeError as e:
    print(f"TEST 3 FAILED: {e}")
model.zero_grad()

# ---- TEST 4: Batch of 32 (real-sized) ----
print("\n=== TEST 4: Batch of 32 ===")
model.train()
samples = [{"src": f"Source text number {i}", "mt": f"Translation number {i}"} for i in range(32)]
batch = model.prepare_sample(samples, stage="predict")
inp = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch[0].items()}
pred = model.forward(**inp)
gold = torch.rand(32, device=device)
loss = F.mse_loss(pred.score, gold)
try:
    loss.backward()
    print("TEST 4 PASSED")
except RuntimeError as e:
    print(f"TEST 4 FAILED: {e}")
model.zero_grad()

# ---- TEST 5: eval(no_grad) then train+backward ----
print("\n=== TEST 5: eval->no_grad->train->backward ===")
model.eval()
with torch.no_grad():
    samples = [{"src": "Test", "mt": "Test"}]
    batch = model.prepare_sample(samples, stage="predict")
    inp = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch[0].items()}
    _ = model.forward(**inp)
model.train()
samples = [{"src": "Hello", "mt": "Hallo"}]
batch = model.prepare_sample(samples, stage="predict")
inp = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch[0].items()}
pred = model.forward(**inp)
loss = pred.score.mean()
try:
    loss.backward()
    print("TEST 5 PASSED")
except RuntimeError as e:
    print(f"TEST 5 FAILED: {e}")
model.zero_grad()

# ---- TEST 6: With real data ----
print("\n=== TEST 6: With real training data ===")
import pandas as pd
train = pd.read_parquet("outputs/train_text.parquet")
train_data = train[(train["src_lang"] == "en") & (train["tgt_lang"].isin(["de", "zh"]))].head(100)
model.train()

# Pick two real samples
row1 = train_data.iloc[0]
row2 = train_data.iloc[1]
s_better = [{"src": str(row1["src_text"]), "mt": str(row1["tgt_text"])}]
s_worse = [{"src": str(row2["src_text"]), "mt": str(row2["tgt_text"])}]

b_better = model.prepare_sample(s_better, stage="predict")
b_worse = model.prepare_sample(s_worse, stage="predict")
i_better = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b_better[0].items()}
i_worse = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b_worse[0].items()}
p_better = model.forward(**i_better).score
p_worse = model.forward(**i_worse).score
gold = torch.tensor([0.8], dtype=torch.float32, device=device)
loss = torch.clamp(0.1 - (p_better - p_worse), min=0).mean() + F.mse_loss(p_better, gold)
print(f"p_better={p_better.item():.4f}, p_worse={p_worse.item():.4f}, loss={loss.item():.6f}")
try:
    loss.backward()
    print("TEST 6 PASSED")
except RuntimeError as e:
    print(f"TEST 6 FAILED: {e}")

# ---- TEST 7: With optimizer step ----
print("\n=== TEST 7: Optimizer step ===")
model.zero_grad()
model.train()
head_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW([{"params": head_params, "lr": 1e-5}])

samples = [{"src": str(row1["src_text"]), "mt": str(row1["tgt_text"])}] * 4
batch = model.prepare_sample(samples, stage="predict")
inp = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch[0].items()}
pred = model.forward(**inp)
loss = F.mse_loss(pred.score, torch.tensor([0.5]*4, dtype=torch.float32, device=device))
optimizer.zero_grad()
try:
    loss.backward()
    optimizer.step()
    print("TEST 7 PASSED")
except RuntimeError as e:
    print(f"TEST 7 FAILED: {e}")

print("\n=== ALL TESTS COMPLETE ===")
