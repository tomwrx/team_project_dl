# Privacy Masker — Team Project

A deep learning pipeline for high-precision PII (Personally Identifiable Information) detection and masking in mobile UI screenshots, with a Slack integration for safely sharing screenshots in workplace channels. Built on the [RICO-ScreenQA](https://huggingface.co/datasets/rootsautomation/RICO-ScreenQA) dataset.

**Team:** Aurimas Bžėskis (Data Engineer) · Tomas Stankevicius (ML Engineer) · Aida Katkauskaitė (Demo Engineer)

---

## What it does

Mobile applications routinely expose sensitive personal information (emails, balances, names, addresses) that is unintentionally leaked when users share screenshots. Privacy Masker automatically detects PII regions in screenshots and returns bounding boxes for downstream redaction. The system ships as both a Gradio demo and a production-grade Slack bot that intercepts images sent privately via DM, scans them, and only posts the blurred version to the intended channel — the unblurred image never appears publicly.

---

## Repository Structure

```
project/
├── notebooks/
│   ├── 01_data_prep_final.ipynb         # Raw RICO-ScreenQA → curated PII JSONL  (M1)
│   ├── 02_train_two_stage_stg_1.ipynb   # PaddleOCR layout extraction            (M2)
│   ├── 02_train_two_stage_stg_2.ipynb   # Gemma-4 LoRA classifier training       (M2)
│   ├── 03_evaluate.ipynb                # Per-class P/R/F1, JSON validity        (M2)
│   └── 02_train_paligemma.ipynb         # Single-stage VLM baseline (reference)  (M2)
│
├── app/                                  # Demo + Slack bot                       (M3)
│   ├── app.py                            # FastAPI backend (inference + blur)
│   ├── slack_bot.py                      # Slack Bolt bot (DM flow)
│   └── addin/editor.html                 # Web-based box editor
│
├── SCHEMA.md                             # Dataset contract
├── REPORT.md                             # Final project report
└── README.md                             # This file
```

---

## Datasets

All datasets are derived from `rootsautomation/RICO-ScreenQA` (HuggingFace). Each `.jsonl` line follows the same schema:

```json
{
  "image":        "images/<screen_id>.png",
  "image_width":  1080,
  "image_height": 1920,
  "screen_id":    "12345",
  "objects": [
    {
      "bbox":            [x1, y1, x2, y2],
      "label":           "email_address",
      "source_question": "What is the support email address?",
      "source_answer":   "support@example.com"
    }
  ]
}
```

Screens with no PII have `"objects": []` and serve as hard negative examples.

### Version History

| Version | Screens | Splits | Notes | Download |
|---------|--------:|--------|-------|----------|
| `pii_v1` | 3,645 | train only (baseline) | Broad keyword regex (`email`, `phone`, `name`...). 80/10/10 split added later. ~15% negatives. | [Drive](https://drive.google.com/drive/folders/1uycS4idphBz_cZfWB8aq6zIWZTM0Trv5?usp=share_link) |
| `pii_v2` | 4,549 | 80/10/10 | Same keyword map as v1; added val + test splits. ~15% negatives. | [Drive](https://drive.google.com/drive/folders/1hGHE_ald1wqY_1g5CTNDwGaovNmWA5VD?usp=share_link) |
| `pii_v3` | 10,645 | — | Expanded keyword lists; **not uploaded** — too many false positives. | — |
| **`pii_v5`** | **9,989** | 80/10/10 stratified | **Final dataset.** Precision-first regex engine, rarest-label stratified split, pruned ambiguous labels, biometrics added. ~12% negatives. | [Drive](https://drive.google.com/drive/folders/1DnbkCIIex6pEyrxMQenOcWaxxL8l7PW-?usp=share_link) |

### Final Dataset (`pii_v5`) Label Distribution

| Label | Train | Val | Test | Priority |
|-------|------:|----:|-----:|:--------:|
| `address` | 3,135 | 373 | 417 | P1 |
| `transaction_amount` | 2,316 | 323 | 279 | P2 |
| `email_address` | 1,975 | 240 | 229 | P1 |
| `other_sensitive` | 1,260 | 183 | 138 | P2 |
| `date_of_birth` | 884 | 115 | 121 | P1 |
| `phone_number` | 816 | 109 | 92 | P1 |
| `username` | 570 | 66 | 76 | P2 |
| `full_name` | 201 | 27 | 25 | P1 |
| `account_balance` | 153 | 18 | 19 | P2 |
| **Total screens** | **7,912** | **989** | **997** | |

**P1** — core PII (identity + contact). **P2** — extended PII (financial, credentials, biometrics).

### PII Categories & Regex Rules (`pii_v5`)

QA pairs are classified using compiled regex patterns matched against the question text. The keywords below are the *primary signals* from n-gram analysis; the regex intentionally matches a wider surface (e.g. `email` also catches "enter your mail").

| Label | Primary Keywords | Regex |
|-------|------------------|-------|
| `email_address` | email, mail, gmail | `\b(?:email\|mail\|gmail)(?:\s+(?:address\|id))?\b` |
| `phone_number` | phone, contact, helpline | `\b(?:phone\|contact\|helpline)(?:\s+number)?\b` |
| `full_name` | first/last/full/display name | `\b(?:first\|last\|full\|display)\s+name\b` |
| `address` | address, street, zip, postal, location, city, country, destination | `\b(?:address\|street\|zip\|postal\|location\|city\|country\|destination)\b` |
| `date_of_birth` | dob, birthday, born, age | `\b(?:date\s+of\s+birth\|dob\|birthday\|born\|age)\b` |
| `account_balance` | balance, account balance | `\b(?:account\s+)?balance\b` |
| `transaction_amount` | charge, price, cost, fee, rent + bigrams | `\b(?:charge\|price\|cost\|fee\|rent)\b\|\b(?:transaction\|payment\|total\|purchase\|monthly)\s+(?:amount\|price\|cost\|payment)\b` |
| `username` | username, login name, user id | `\b(?:username\|login\s+name\|user\s+id)\b` |
| `other_sensitive` | gender, password, passcode, pin, weight, height | `\b(?:gender\|password\|passcode\|pin\|weight\|height)\b` |

### V5 Improvements Over V1/V2

1. **Stratified multi-label splitting — "rarest-label heuristic":** each screen's primary label is the globally rarest PII label it contains; screens are then split 80/10/10 within each primary-label group. Guarantees rare classes (`full_name`, `account_balance`) appear in val/test.
2. **Precision-first regex engine:** high-noise keywords like bare `name` and `date` were removed (caused >40% false positives in V1/V2). Only structurally unambiguous terms and bigrams retained.
3. **Biometric compliance:** `weight` and `height` added to `other_sensitive` for fitness/health app coverage.
4. **Label pruning:** `account_number` and `id_number` deprecated — no reliable regex signature in QA phrasing, high FP rate.

---

## Model Architecture

We evaluated **two architectures**; final production uses the two-stage pipeline.

### Two-Stage Pipeline (Final)

| Stage | Component | Purpose |
|-------|-----------|---------|
| 1 | **PaddleOCR** (angle-corrected, English) | Extract every text region as `{bbox, text, confidence}`. No fine-tuning. |
| 2 | **Gemma-4-E4B + LoRA** (4-bit NF4, r=32, α=64) | Classify each OCR region as PII or not via layout-aware prompting. 69.8M trainable params (0.87%). |

**Stage 2 prompt format** — each OCR region rendered as `[index@x,y] "text"` with coordinates normalized to a 1000×1000 grid so the model sees spatial layout as well as content. Model returns a JSON map from tag to PII class.

**Why two stages:** Decouples localization (deterministic, OCR) from classification (semantic, LLM). Eliminates the autoregressive 1-box bias of single-stage VLMs and lets the model label dense screens with 4+ PII fields correctly.

### Single-Stage VLM Baseline (Reference)

`google/paligemma2-3b-pt-448` in 4-bit NF4 with LoRA on language-model projections only (47.5M trainable params). Detection framed as autoregressive sequence generation (`<loc####>` tokens). Kept as a reference baseline — see `02_train_paligemma.ipynb`.

### Final Results (Two-Stage on `pii_v5` test)

- **Micro F1: 0.586** · **Macro F1: 0.536** · **JSON validity: 100%**
- Per-class F1: `email_address` 0.84 · `phone_number` 0.59 · `address` 0.56 · `full_name` 0.56 · `account_balance` 0.56 · `transaction_amount` 0.55 · `username` 0.47 · `other_sensitive` 0.42 · `date_of_birth` 0.28

Full iteration history, per-class breakdowns, and diagnostic findings in [REPORT.md](./REPORT.md).

---

## Google Drive Layout

All notebooks read from / write to the same shared folder so that team members can hand off artifacts cleanly:

```
MyDrive/VU_DL_Team_Project/
├── data/
│   ├── raw/
│   │   └── unique_uis.tar.gz             # RICO tarball — upload ONCE
│   ├── pii_v1/ pii_v2/ pii_v5/           # curated datasets (M1 drops here)
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   ├── test.jsonl
│   │   └── images.zip
│   └── ocr_cache/                        # PaddleOCR output (reusable across runs)
│       └── {train,val,test}_ocr.jsonl
└── outputs/
    ├── checkpoints/                      # Lightning checkpoints (auto-resume)
    ├── lora_adapters/final/              # what M3 loads in the demo
    └── eval_report_*.json                # one per experiment for the report table
```

**Why this layout:** survives Colab disconnects, gives the team one source of truth, and lets you switch datasets by changing a single constant in the training/eval notebooks:

```python
DATA_SUBDIR = 'pii_v5'   # was 'pii_v1' or 'pii_v2'
```

Images are stored as a single `.zip` and unpacked to Colab's local disk at training time — reading hundreds of small files directly from Drive is ~50× slower.

---

## One-Time Setup

1. Create folder `MyDrive/VU_DL_Team_Project/` on Google Drive.
2. Have M1 register for RICO at [interactionmining.org/rico.html](http://www.interactionmining.org/rico.html) and upload `unique_uis.tar.gz` to `data/raw/`. **Start this early — email approval can take hours.**
3. Accept the Gemma license at [huggingface.co/google/gemma-4-E4B](https://huggingface.co/google/gemma-4-E4B). (And the PaliGemma 2 license if running the baseline.)
4. Create an HF token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). In Colab:

   ```python
   import os
   os.environ['HF_TOKEN'] = '...'   # or use Colab Secrets
   ```

5. Run `01_data_prep_final.ipynb` end-to-end to produce `pii_v5/`.

---

## End-to-End Workflow

```
┌──────────────────────────────────────────────────────────────┐
│ 1. Data prep (M1)                                            │
│    RICO-ScreenQA → regex matching → bbox extraction          │
│    → pii_v5/{train,val,test}.jsonl + images.zip              │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. Stage 1 — OCR layout extraction (M2)                      │
│    PaddleOCR on every screen → {bbox, text, confidence}      │
│    → ocr_cache/{train,val,test}_ocr.jsonl                    │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. Stage 2 — LLM classifier training (M2)                    │
│    Layout-aware prompt + Gemma-4 LoRA                        │
│    → outputs/lora_adapters/final/                            │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. Evaluation (M2)                                           │
│    Per-class P/R/F1, JSON validity, error analysis           │
│    → outputs/eval_report_*.json                              │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. Demo + Slack bot (M3)                                     │
│    FastAPI backend + Slack Bolt + web editor                 │
└──────────────────────────────────────────────────────────────┘
```

---

## Slack Integration

A production-style Slack bot built around a **private DM workflow**: users send images to the Privacy Masker bot via DM, the bot processes them, shows a blurred preview, and waits for explicit approval before posting the blurred version to the intended channel. The unblurred image never appears publicly.

### Components

| Component | Tech | Purpose |
|-----------|------|---------|
| **Slack Bot** | Slack Bolt (Python) | Listens for DM file uploads; handles channel selection, edit/approve/discard; posts final blurred image |
| **FastAPI Backend** | FastAPI on Colab GPU + ngrok tunnel | Receives base64 images, runs inference, applies Gaussian blur (PIL), manages editor sessions |
| **Web Editor** | HTML5 canvas served by backend | Lets user draw additional blur regions beyond what the model detected |

### User Flow

```
USER sends image to Privacy Masker via DM
        │
        ▼
BOT downloads image → POST /process-image → model inference → blur
        │
        ▼
BOT replies in DM with blurred preview + detected regions list + buttons:
   ┌─────────────────┬──────────────┬──────────────┐
   │ Post to channel │ ✏️ Edit boxes │  ✕ Discard   │
   └────────┬────────┴───────┬──────┴──────────────┘
            │                │
            │                ▼
            │    Opens web editor → user draws extra boxes
            │    → POST /confirm-edit → bot polls /edit-result/{id}
            │                │
            └────────────────┘
                     │
                     ▼
        BOT posts blurred image to chosen channel
        "Shared by @username\n<original message>"
```

### Required Slack Scopes

| Scope | Purpose |
|-------|---------|
| `chat:write` | Send messages |
| `files:read` / `files:write` | Download / upload images |
| `im:history` / `im:read` / `im:write` | Read & write DMs |
| `channels:read` / `groups:read` / `mpim:read` | List channels for the selector |
| `users:read` | Get sender's real name for attribution |

### Security

- All backend calls require an `x-api-key` secret header — requests without it return 403.
- The backend URL is a private ngrok tunnel that rotates every session; never published.
- Unblurred images are only ever seen by the sending user in their private DM.
- Images are processed in memory — nothing persists on disk.
- The bot ignores images uploaded directly to channels; it only processes DMs.
- If deployed on company-internal servers, user-corrected boxes could be stored for fine-tuning on company-specific PII patterns.

---

## Hand-off Snippet for Inference

Drop-in code for loading the final two-stage model:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from paddleocr import PaddleOCR

DRIVE_ROOT = '/content/drive/MyDrive/VU_DL_Team_Project'
ADAPTER    = f'{DRIVE_ROOT}/outputs/lora_adapters/final'
GEMMA_ID   = 'google/gemma-4-E4B'

# Stage 1
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en',
                       use_gpu=torch.cuda.is_available(), show_log=False)

# Stage 2
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained(GEMMA_ID, quantization_config=bnb,
                                            device_map='auto')
classifier = PeftModel.from_pretrained(base, ADAPTER).eval()
tokenizer  = AutoTokenizer.from_pretrained(GEMMA_ID)
```

See `app/app.py` for the full inference + masking pipeline with prompt construction and JSON parsing.

---

## Final Report

See [REPORT.md](./REPORT.md) for the full writeup — architecture comparison, three-iteration result table on both datasets, per-class diagnostics, the `date_of_birth` regression analysis, and limitations / future work.
