# Privacy Masker — ML Engineer's Working Repo

## Repository Structure
| File | Description |
|------|-------------|
| `01_data_prep_final.ipynb` | Data preparation notebook (Member 1) |
| `SCHEMA.md` | Dataset schema contract (Member 2) |
| `README.md` | This file |

## Dataset
pii_v1 dataset is available on Google Drive:

[Download pii_v1 dataset](https://drive.google.com/drive/folders/1uycS4idphBz_cZfWB8aq6zIWZTM0Trv5?usp=share_link)

### Dataset Structure
| Path | Description |
|------|-------------|
| `train.jsonl` | 2,916 screens |
| `val.jsonl` | 364 screens |
| `test.jsonl` | 365 screens |
| `images.zip` | 1,471 MB — all PNG files zipped |
| `images/` | Raw PNG files |
| `eda/` | EDA plots |

### Stats
- **Total screens:** 3,645
- **Split:** 80% train / 10% val / 10% test
- **PII labels:** email_address, phone_number, full_name, username, account_balance, transaction_amount, address, date_of_birth
- **Negative examples:** ~15% empty screens
- **Validator:** OK


## Google Drive layout

All three notebooks read from / write to the same shared folder:

```
MyDrive/VU_DL_Team_Project/
├── data/
│   ├── raw/
│   │   └── unique_uis.tar.gz        # RICO tarball — upload ONCE, shared by team
│   ├── screenqa_raw/                # notebook 01 output (your day-1 baseline)
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   ├── test.jsonl
│   │   ├── images/                  # raw PNG/JPG files (for inspection)
│   │   └── images.zip               # used by training (fast local-disk unpacking)
│   └── pii_v1/                      # M1 drops curated dataset here, same layout
│       ├── train.jsonl
│       ├── val.jsonl
│       ├── test.jsonl
│       └── images.zip
└── outputs/
    ├── checkpoints/                 # Lightning checkpoints (auto-resume on disconnect)
    ├── lora_adapters/final/         # what M3 loads in the demo
    └── eval_report_*.json           # one per dataset for the report comparison table
```

## Why this layout

- **Survives Colab disconnects.** Checkpoints and adapters live on Drive.
- **Team-shared source of truth.** M1 drops `pii_v1/` next to `screenqa_raw/`; you switch by changing one constant. M3 grabs the adapter folder.
- **Fast training.** Images are stored on Drive as a single `.zip` and unpacked to Colab's local disk at training time. Reading hundreds of small files directly from Drive is ~50× slower.
- **Two-line resume.** If a session dies mid-training, the next run auto-picks-up from `outputs/checkpoints/last.ckpt`.

## One-line config switch

When M1 delivers his curated data, change exactly one line in `02_train.ipynb` and `03_evaluate.ipynb`:

```python
DATA_SUBDIR = 'pii_v1'   # was 'screenqa_raw'
```

## Files

| File | Purpose |
|---|---|
| `SCHEMA.md` | Dataset contract. Send to M1 today. |
| `01_data_prep.ipynb` | Raw ScreenQA → canonical JSONL on Drive. Has synthetic-data fallback. |
| `02_train.ipynb` | PaliGemma 2 + LoRA + Lightning. Drive-backed checkpoints, auto-resume. |
| `03_evaluate.ipynb` | IoU, P/R/F1, mAP proxy. Writes report JSON to Drive. |

## One-time setup before Day 1

1. Create folder `MyDrive/VU_DL_Team_Project/` (you've done this).
2. Have M1 sign up for RICO at http://www.interactionmining.org/rico.html and upload `unique_uis.tar.gz` to `data/raw/` on Drive. **Start this now — RICO email approval can take a few hours.**
3. Accept the PaliGemma 2 license at https://huggingface.co/google/paligemma2-3b-pt-448.
4. Get an HF token (`huggingface.co/settings/tokens`). In Colab: `import os; os.environ['HF_TOKEN'] = '...'` or use Colab Secrets.
5. Run `01_data_prep.ipynb` with synthetic mode first to validate the pipeline (no RICO needed).

## Hand-off snippet for M3 (Demo Engineer)

```python
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
import torch, re

DRIVE_ROOT = '/content/drive/MyDrive/VU_DL_Team_Project'
ADAPTER = f'{DRIVE_ROOT}/outputs/lora_adapters/final'
PROMPT = 'detect sensitive\n'

base = PaliGemmaForConditionalGeneration.from_pretrained(
    'google/paligemma2-3b-pt-448', torch_dtype=torch.bfloat16, device_map='auto')
model = PeftModel.from_pretrained(base, ADAPTER).eval()
processor = PaliGemmaProcessor.from_pretrained(ADAPTER)

def detect(image):
    inputs = processor(text=PROMPT, images=image, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    text = processor.batch_decode(out, skip_special_tokens=False)[0]
    pattern = re.compile(r'<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*(\w+)?')
    h, w = image.height, image.width
    return [
        {'bbox': [int(int(nx1)/1024*w), int(int(ny1)/1024*h),
                  int(int(nx2)/1024*w), int(int(ny2)/1024*h)],
         'label': label or 'sensitive'}
        for ny1, nx1, ny2, nx2, label in pattern.findall(text)
    ]
```
