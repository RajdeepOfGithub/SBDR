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

---

## Project Phase Status (updated 2026-05-10)

### Phase A — COMPLETE ✅
All 5 notebooks done. Final output: `data/processed/sbdr_final_dataset.csv` (30K × 88)

### Phase B — COMPLETE ✅
- B1 FinBERT: zero-shot (ROCm GPU segfaults prevented fine-tuning) → `data/processed/07_with_distress_scores.csv` (30K × 94)
- B2 BiLSTM: trained 100 epochs, anomaly_rate=5% → `data/processed/06_with_stress_vectors.csv` (30K × 122)
- B3 XGBoost: accuracy=93.4%, AUC-ROC=0.990 → `data/processed/08_with_recovery_tiers.csv` (30K × 135)
- B3.5 Audit Layer: 485 Tier 5 flagged (+121%) → `data/processed/09_with_audit_tiers.csv` (30K × 142)
- B4 SHAP: captured in Notebook 08 (LIME out of scope)

### Phase C — COMPLETE ✅
- C1 Dashboard (`dashboard.py`): COMPLETE ✅ — see details below
- C2 Fairness Audit: COMPLETE ✅ — in dashboard, SEX/AGE/EDUCATION sections
- C3 Final Testing + Documentation: COMPLETE ✅ — `tests/test_pipeline.py` (38 tests, all passing)

### Phase E — Dashboard Deployment — COMPLETE ✅ (2026-05-13)
- **Live URL:** https://sbdr-dashboard.streamlit.app/
- **Platform:** Streamlit Community Cloud (free tier, public repo)
- **Data hosting:** `09_with_audit_tiers.csv` (60MB) attached to GitHub Release `v1.0-data`
  — URL: `https://github.com/RajdeepOfGithub/SBDR/releases/download/v1.0-data/09_with_audit_tiers.csv`
  — NOT committed to git (correct practice — data files stay out of repo history)
- **Deployment deps:** `requirements.txt` slimmed to 3 packages (streamlit, pandas, plotly)
  — Full ML deps preserved in `requirements_deploy.txt` for reference
- **Data loading logic in `dashboard.py`:**
  1. Local file (`data/processed/09_with_audit_tiers.csv`) — used in local dev
  2. Env var `SBDR_DATA_URL` — optional override
  3. Streamlit secret `SBDR_DATA_URL` or `data_url` — optional override
  4. Hardcoded `DATA_URL_FALLBACK` (GitHub Release URL) — production fallback, zero config needed
- **`runtime.txt`:** `python-3.11` (already present)

---

## Dashboard (`dashboard.py`) — Current State

**Streamlit version:** 1.55.0
**Live (cloud):** https://sbdr-dashboard.streamlit.app/
**Local start command:**
```bash
source .venv/bin/activate && nohup streamlit run dashboard.py --server.port 8501 --server.headless true > /tmp/streamlit.log 2>&1 &
```
**Local access:** http://localhost:8501

### Design
- Pure black background `#000000`, cards `#080808`
- Fonts: **Inter** (body, via config.toml + st.html) + **Orbitron** (KPI values) + **DM Mono** (monospace)
- CSS injected via `st.html()` (not `st.markdown`) — required for reliable `@import` and font inheritance
- `.streamlit/config.toml` sets Inter as theme font (protects Material Symbols sidebar toggle)
- JS-animated KPI counter cards via `st.components.v1.html()`
- Plotly charts: transparent bg, `rgba(255,255,255,0.04)` gridlines

### Font fix (sidebar toggle)
- Root cause: `* { font-family: Inter !important }` broke Material Symbols Rounded ligature on sidebar toggle icon, rendering raw text "keyboard_double_arrow_right"
- Fix: use `st.html()` (not `st.markdown`), scope CSS via `:where()` and `html/body/.stApp` inheritance, explicitly protect `[data-testid="collapsedControl"] *` with Material Symbols Rounded font

### Data source
`data/processed/09_with_audit_tiers.csv` — 30,000 rows × 142 columns

### Tabs (7 total)
1. **Overview** — Full-width tier distribution bar chart + 5 tier summary pills (count, %, exposure $M)
2. **Portfolio** — Credit exposure by tier ($M) + Risk Map scatter (distress vs delay) + distress histogram + customer roster (top 200 by distress, searchable)
3. **Customer Insight** — Per-customer deep dive: pick any of 30K customers, see FinBERT turns, BiLSTM payment timeline, XGBoost tier probs, Sparkov signals, audit flags
4. **Audit & Risk** — 4 KPI cards + B3.5 spotlight case + Tier 5 strategic default roster (485 accounts, searchable, rule breakdown)
5. **Fairness** — Stacked 100% bar charts by gender/age/education + fairness verdict summary table
6. **What-If** — Interactive simulator: 5 sliders/toggles (pay delay, distress, fraud rate, anomaly, default) → live tier prediction + NBA + audit rules triggered
7. **Model** — SHAP feature importance chart + branch contribution donut + 5 performance metrics (large coloured numbers) + 4 artifact images (confusion matrix, SHAP beeswarm, audit lift, BiLSTM loss curve)

### Sidebar
- Shield SVG brand icon + "Debt Recovery Intelligence" label
- Filters: Recovery Tier (multiselect), Distress Score (range slider), Anomaly Only (checkbox)
- Active filter count badge, Reset Filters button
- Live pulse dot footer

### Known CSS fixes applied
- Sidebar toggle: `st.html()` + scoped CSS + Material Symbols protection
- All near-invisible dark text colors (`#1e1e1e`, `#1a1a1a`, `#2d2d2d`, `#222222`, `#0f0f0f`) replaced with `#475569` / `#64748b`
- Inactive tab color: `#3a3a3a` → `#475569`; hover: `#6b6b6b` → `#94a3b8`
- Chart textfont, tickfont, hline annotation colors all updated to `#475569`

### Supporting docs
- `LIMITATIONS.md` — L1–L5 known limitations with root causes, measured impact, mitigations
- `README.md` — all phases complete
- `tests/test_pipeline.py` — 38 unit tests (data shapes, model artifacts, pipeline integrity, dashboard syntax)

---

## Final Pipeline Column Map (09_with_audit_tiers.csv — 142 cols)
- UCI raw: 24 cols (LIMIT_BAL, SEX, AGE, EDUCATION, MARRIAGE, PAY_0–6, BILL_AMT1–6, PAY_AMT1–6, default payment next month)
- UCI derived: 38 cols (pay ratios, util ratios, trend features, interaction terms)
- LC: 12 cols (lc_loan_amnt_mean, lc_annual_inc_mean, etc.)
- Sparkov: 7 cols (sp_total_spend, sp_fraud_rate, etc.)
- Chat text: 3 cols (chat_turn_1/2/3)
- FinBERT: 6 cols (distress_turn1/2/3, distress_avg, distress_max, distress_shift)
- BiLSTM: 34 cols (bilstm_dim_0–31, bilstm_recon_error, bilstm_anomaly_flag)
- XGBoost: 7 cols (recovery_tier, tier_prob_1–5)
- Audit: 7 cols (recovery_tier_final, audit_rule, audit_escalated, audit_deescalated, etc.)



---

## Phase D — Explainer Video (`sbdr_explainer.html`) — COMPLETE ✅

### File locations
- **HTML player:** `sbdr_explainer.html` (self-contained, ~1,200 lines)
- **Audio files:** `sbdr_video_assets/audio/s-*.mp3` (16 ElevenLabs files + legacy files)
- **Narration script:** `sbdr_narration_script.md`
- **Asset manifest:** `sbdr_video_assets.md`

### Video design
- **18 scenes** across 4 acts (Intro / Act 1 Architecture / Act 2 Five Characters / Act 3 Results + Outro)
- **5 character stories** covering all recovery tiers: Sofia Chen (T1), Priya Sharma (T2), Maria Rodriguez (T3), Robert Thompson (T4), James Mitchell (T5)
- **Story scene structure:** bad outcome (without SBDR) → film-reel REWIND overlay → good outcome (with SBDR)
- **Fallback timer:** every scene has a `dur`-second duration timer; speech completion fires a shorter 1.5s advance that wins first
- **Start paused:** `running = false` on init; first ▶ Play click is the required browser user gesture for speech API

### ElevenLabs audio (16 files)
- **Voice:** Roger (`voice_id: CwhRBWXzGAHq8TQ4Fs17`) — Laid-Back, Casual, Resonant
- **Model:** `eleven_multilingual_v2`
- **Settings:** stability=0.55, similarity_boost=0.75, style=0.15, speed=0.92, mp3_44100_128
- **Files generated:**
  - `s-title.mp3`, `s-problem.mp3`, `s-arch.mp3`
  - `s-sofia-story-bad.mp3`, `s-sofia-story-good.mp3`
  - `s-priya-story-bad.mp3`, `s-priya-story-good.mp3`
  - `s-maria-story-bad.mp3`, `s-maria-story-good.mp3`
  - `s-robert-story-bad.mp3`, `s-robert-story-good.mp3`
  - `s-james-story-bad.mp3`, `s-james-story-good.mp3`
  - `s-shap.mp3`, `s-numbers.mp3`, `s-close.mp3`
- **7 tech/analytics scenes** (`s-sofia-tech`, `s-priya-tech`, `s-maria-tech`, `s-robert-tech`, `s-james-tech`, `s-metrics`, `s-fairness`) use Web Speech API TTS fallback — no ElevenLabs file

### Audio playback architecture (in HTML JS)
```javascript
// AUDIO_MAP maps scene IDs to MP3 paths
const AUDIO_BASE = 'sbdr_video_assets/audio/';
const AUDIO_MAP = { 's-title': AUDIO_BASE + 's-title.mp3', ... };

// speak() tries MP3 first, falls back to Web Speech API
function speak(text, onEnd, audioSrc) { ... }
function speakTTS(text, onEnd) { ... }  // Web Speech API
function stopSpeech() { synth.cancel(); /* also stops currentAudio */ }
```
- `startScene(idx)` passes `AUDIO_MAP[s.id]`
- `startStoryScene(idx)` passes `AUDIO_MAP[s.id + '-bad']` and `AUDIO_MAP[s.id + '-good']`

### Known bugs fixed during development
1. **No sound on page load** — browser blocks speech without user gesture; fixed by `running = false` init
2. **Fast-forward race** — `u.onerror` was calling `onEnd()` → 1.5s advance every scene; fixed by removing `onEnd` from onerror
3. **Missing fallback timer** — normal scenes had no duration timer if voice failed; fixed by always calling `scheduleAdvance(idx, s.dur * 1000)` first
4. **Bad story panel invisible** — `.story-panel.bad` had `opacity:0` with no reveal rule; fixed with CSS `.scene.active .story-panel.bad { animation: slideInLeft ... }`
5. **Mini SHAP bars invisible** — `.ms-row` needed `.scene.active .ms-row` CSS animation rule with nth-child delays

### Rewind overlay
- `#rewind-overlay` div with CSS film strip animation (`@keyframes stripScroll`)
- Orbitron font "⟳ REWIND" label, fades in → holds 2.6s → fades out
- `showRewind(onDone)` callback fires after overlay fades out + 400ms

### ElevenLabs free tier note
- Budget at time of generation: ~6,453 chars remaining of 10,000/month
- 16 files consumed ~6,271 chars total

---

### Phase D Update — Sound Effects + Stick Figure Animations (2026-05-12)

#### Layer 1 — Sound Effects (Web Audio API)
- **ElevenLabs `text_to_sound_effects` blocked** — requires `sound_generation` permission (paid Creator tier+). Free tier only covers TTS.
- **Solution:** Synthesized sound effects using browser-native Web Audio API (zero cost, works offline)
- Three functions added to `sbdr_explainer.html` JS:
  - `sfxRewind()` — bandpass-filtered white noise burst, pitch sweeps 2800→600Hz (~0.42s). Fires at top of `showRewind()`.
  - `sfxSuccess()` — ascending C-E-G major chime (523→659→783 Hz, sine waves, 0.13s stagger). Fires when good outcome panel slides in.
  - `sfxAlert()` — triple square-wave buzz (200/250/200 Hz, 0.19s stagger). Fires instead of success for `s-james-story` only (legal escalation outcome).
- Web Audio context (`_aC`) created lazily on first use — avoids browser autoplay restrictions.

#### Layer 2 — Stick Figure SVG Animations
- Every story panel (10 total — bad + good × 5 characters) now has an animated SVG stick figure
- Layout: `.panel-inner` flex row — figure (70px wide) on left, text on right (`.panel-text`)
- SVG spec: `viewBox="0 0 72 90"`, 64×80px rendered, `stroke-linecap:round`
- CSS animations:
  - Bad figures: `sfShake` (±4px + ±5deg) at 2s after scene activates, 4 cycles
  - Good figures: `sfBounce` (−8px) + `sfPop` (scale 1.14) at 3.6s after good panel slides in, 3 cycles

| Character | Bad pose | Bad props | Good pose | Good props |
|---|---|---|---|---|
| Sofia (T1) | Arms raised in shock | ✉️ ❌ | Arms in victory V | 📱 ✓ |
| Priya (T2) | Arms out stressed | 📞 ⚡ | Arms relaxed/calm | ⏸ 📅 |
| Maria (T3) | Arms drooping overwhelmed | 📬 😰 | Handshake extended | 🤝 📋 |
| Robert (T4) | Body hunched | 🏥 ⚖️ | Open welcoming arms | 🫂 ❄️ |
| James (T5) | Casual smug walk | 💳 🏃 | Stop-hand raised | 🚫 ⚖️ |

#### How to serve the video locally
```bash
cd /home/roy/Main_Dir/projects/SBDR
python3 -m http.server 8502 --bind 127.0.0.1 &
# Open: http://127.0.0.1:8502/sbdr_explainer.html
```
**Must use HTTP server** (not `file://`) — MP3 audio files load via relative paths which require a proper HTTP origin.
