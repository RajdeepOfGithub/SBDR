# SBDR Project — ROCm Environment Briefing for Claude Code

## Machine Overview
- **Device:** ASUS ROG Flow Z13 (GZ302EA)
- **CPU:** AMD Ryzen AI Max+ 395
- **GPU:** AMD Radeon 8060S (gfx1100 / RDNA 3 architecture)
- **Unified Memory:** 64GB total, ~32GB usable by GPU
- **OS:** Fedora 43 (kernel 6.19.7-200.fc43.x86_64)
- **No NVIDIA GPU — No CUDA cores available**

---

## Critical: This Machine Uses ROCm, Not CUDA

This is an AMD GPU system. All GPU acceleration runs through **AMD ROCm**, not NVIDIA CUDA.
PyTorch is installed with ROCm support — `torch.cuda.is_available()` returns `True` on this
machine because PyTorch maps its CUDA API to ROCm on AMD hardware transparently.

**Never assume CUDA = NVIDIA here. CUDA API calls route to ROCm automatically.**

---

## Environment State (as of 2026-03-15)

### ROCm
- **Version:** 6.4 (Fedora native packages)
- **Previously broken:** Mixed ROCm 6.3 (RHEL9 repo) + ROCm 6.4 (Fedora) conflict
- **Now fixed:** Old RHEL9 ROCm repo removed, clean Fedora-native ROCm 6.4 stack
- **GPU Architecture:** gfx1100 (RDNA 3) — confirmed working via `rocminfo`

### Python Environment
- **Python:** 3.11.9 via pyenv (`~/.pyenv/versions/3.11.9/`)
- **PyTorch:** 2.9.1+rocm6.4 (upgraded from broken rocm6.3)
- **torchvision:** 0.24.1+rocm6.4
- **torchaudio:** 2.9.1+rocm6.4

### Verification Commands
```bash
# Verify GPU is detected
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('ROCm available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')
"
# Expected output:
# PyTorch: 2.9.1+rocm6.4
# ROCm available: True
# GPU: AMD Radeon 8060S
# VRAM: 32.0 GB

# Verify ROCm stack
rocminfo | /usr/bin/grep -E "Name:|Marketing"
```

---

## Why FinBERT Was Failing Before

There were 3 compounding issues:

### Issue 1 — PyTorch had no ROCm support (Primary)
Standard `pip install torch` installs the CUDA build which has zero AMD GPU support.
PyTorch must be installed explicitly with:
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
```
Without this, `torch.cuda.is_available()` returned `False` and everything silently
fell back to CPU — making FinBERT training extremely slow (hours instead of minutes).

### Issue 2 — ROCm version conflict (Secondary)
An old AMD RHEL9 repo (`/etc/yum.repos.d/rocm.repo`) was installed providing ROCm 6.3,
conflicting with Fedora's native ROCm 6.4 packages. This caused:
- `hipblas` version mismatch errors
- Ollama GPU inference failures
- Unstable GPU context for PyTorch

**Fix applied:** Removed `/etc/yum.repos.d/rocm.repo`, updated all ROCm packages to
clean Fedora-native 6.4, force-reinstalled PyTorch with rocm6.4 build.

### Issue 3 — HuggingFace defaulting to CPU
HuggingFace Transformers (used by FinBERT) defaults to CPU if `device` is not explicitly
set. Even with ROCm working, the model would run on CPU unless you explicitly call `.to(device)`.

---

## How to Run FinBERT on GPU — Correct Pattern

Always use this pattern in SBDR code:

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ALWAYS do this first — confirms GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Should print: Using device: cuda

# Load FinBERT and move to GPU
tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
model = model.to(device)  # CRITICAL — moves 440MB model to 32GB GPU

# Move input tensors to GPU too
def get_finbert_sentiment(text: str) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    # Move ALL input tensors to same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    labels = ["positive", "negative", "neutral"]
    scores = {label: prob.item() for label, prob in zip(labels, probs[0])}

    # Distress score: negative sentiment maps to financial distress
    distress_score = scores["negative"]
    return {"scores": scores, "distress_score": distress_score}
```

---

## Batch Processing FinBERT on GPU (For 30K Customer Dataset)

For SBDR's 30,000 customer dataset, use batched inference for maximum GPU utilization:

```python
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def batch_finbert_inference(texts: list, batch_size: int = 32) -> list:
    """
    Run FinBERT on a list of texts in batches.
    batch_size=32 works well for 32GB GPU memory.
    Increase to 64 if memory allows, decrease to 16 if OOM errors occur.
    """
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model = model.to(device)
    model.eval()

    dataset = TextDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_distress_scores = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="FinBERT inference"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)
            # Index 1 = negative = financial distress
            distress_scores = probs[:, 1].cpu().numpy()
            all_distress_scores.extend(distress_scores.tolist())

    return all_distress_scores


# Usage for SBDR 30K customer dataset:
# customer_texts = df["communication_text"].tolist()  # your chat logs column
# distress_scores = batch_finbert_inference(customer_texts, batch_size=32)
# df["finbert_distress_score"] = distress_scores
```

---

## BiLSTM on GPU

```python
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Initialize and move to GPU
model = BiLSTMModel(
    input_size=88,      # your 88 features
    hidden_size=128,
    num_layers=2,
    output_size=4       # 4 recovery tiers
).to(device)

# Training loop — always move batches to device
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)  # CRITICAL
        y_batch = y_batch.to(device)  # CRITICAL

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
```

---

## Known Warnings (Safe to Ignore)

```
/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory
```
This appears on every PyTorch import. It's a missing cosmetic file listing GPU marketing
names. Has zero impact on GPU functionality. Ignore it.

```
Deleting nftables IPv4/IPv6 rules  error="exit status 1"
```
Docker daemon startup warning. Harmless, Docker falls back to firewalld successfully.

---

## SBDR Venv — Important Note

The SBDR project has its own virtual environment. Before running SBDR code, check
whether the venv has its own PyTorch or inherits from pyenv global:

```bash
# Activate SBDR venv first
source /path/to/sbdr/venv/bin/activate

# Check PyTorch version inside venv
pip show torch

# If it shows rocm6.3 or no ROCm, upgrade it:
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.4 \
    --force-reinstall
```

---

## Quick GPU Memory Reference

| Model | Size | GPU Memory Used | Batch Size Recommendation |
|-------|------|-----------------|--------------------------|
| FinBERT | ~440MB | ~2GB at batch 32 | 32–64 |
| BiLSTM (SBDR) | ~50MB | ~1GB | 128–256 |
| XGBoost | CPU only | N/A | N/A |
| All together | ~3GB | Well within 32GB | No OOM risk |

---

## Summary for Claude Code

When working on the SBDR project:
1. **Always set** `device = torch.device("cuda")` — this maps to AMD ROCm on this machine
2. **Always call** `.to(device)` on both the model AND input tensors
3. **FinBERT is now GPU-ready** — use batch size 32 for the 30K dataset
4. **BiLSTM is GPU-ready** — use batch size 128-256 for sequence data
5. **XGBoost runs on CPU** — this is fine, XGBoost doesn't benefit much from GPU
6. **ROCm 6.4 + PyTorch 2.9.1+rocm6.4** is the confirmed working stack as of 2026-03-15

## Venv Status (updated 2026-03-15)
- Location: /home/roy/Main_Dir/projects/SBDR/.venv
- PyTorch: 2.9.1+rocm6.4 (NVIDIA packages fully removed)
- GPU verified: AMD Radeon 8060S, 32GB, torch.cuda.is_available() = True

