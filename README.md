# Privacy Masker — ML Engineer's Working Repo

Tomas's working copy for the deep-learning team project. Goal: fine-tune a small vision-language model to detect sensitive UI elements in mobile screenshots so they can be blurred.

## Files

| File | Purpose |
|---|---|
| `SCHEMA.md` | **Send this to Member 1 (Data) on Day 1.** Locks the dataset format so the ML pipeline doesn't depend on the curation work finishing. |
| `01_data_prep.ipynb` | Converts raw ScreenQA/RICO into the canonical JSONL schema (no PII filtering). Lets you start training while Member 1 builds the curated dataset. Falls back to synthetic data if RICO isn't downloaded yet. |
| `02_train.ipynb` | PaliGemma 2 (3B, 448px) + 4-bit + LoRA, wrapped in PyTorch Lightning. Outputs LoRA adapters in `outputs/lora_adapters/final/` for Member 3's demo app. Includes Plan-B notes for Gemma 4. |
| `03_evaluate.ipynb` | IoU, P/R/F1 @ IoU=0.5 and 0.75, mAP proxy, side-by-side visualizations. Saves `outputs/eval_report.json`. |

## Model choice (recommendation)

**`google/paligemma2-3b-pt-448`** — picked for native location-token bbox output, screen-understanding pretraining, and mature LoRA recipe.

A newer option exists: **Gemma 4** (April 2026 release, E2B/E4B vision variants). HuggingFace's launch blog explicitly tested it on GUI element detection with native JSON bbox output, and TRL supports fine-tuning out of the box. The notebook contains a Plan-B section for swapping to Gemma 4 if you finish PaliGemma 2 early — running both gives you a "model comparison" section in the report, which scores points with the grader.

## Workflow for the next 3 days (while Member 1 preps data)

1. **Day 1**
   - Send `SCHEMA.md` to Member 1, get sign-off on bbox format and label taxonomy.
   - Set up Colab Pro / EC2 with a T4 or better. Get `huggingface-cli login` working and accept PaliGemma license on the HF model page.
   - Run `01_data_prep.ipynb` with `USE_SYNTHETIC=True` to generate 50 dummy samples and verify the pipeline.
2. **Day 2**
   - Download ScreenQA (HF mirror or GitHub) + RICO screenshots into `data/raw/rico/combined/`.
   - Re-run `01_data_prep.ipynb` end-to-end. Verify the visual sanity check shows boxes in the right places.
   - Run `02_train.ipynb` for a few hundred steps to confirm loss decreases.
3. **Day 3**
   - Run a full 1-epoch baseline on the raw ScreenQA data. Save `outputs/eval_report.json` as your *baseline*.
   - Have a complete end-to-end pipeline ready for when Member 1 delivers `data/pii_v1/`.

When Member 1 delivers their curated dataset, you change exactly one line in `02_train.ipynb`:
```python
DATA_DIR = Path('./data/pii_v1')   # was ./data/screenqa_raw
```

## Hand-off to Member 3 (Demo Engineer)

Member 3 needs:
- The LoRA adapter directory: `outputs/lora_adapters/final/` (contains adapter weights + processor).
- The two parsing helpers from `02_train.ipynb`: `pixels_to_loc_tokens` and `loc_tokens_to_pixels` (the latter is what their app needs to turn model output into OpenCV blur coordinates).
- The prompt string: `"detect sensitive\n"`.

A 20-line inference snippet they can drop into Gradio:
```python
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
import torch, re

PROMPT = 'detect sensitive\n'
base = PaliGemmaForConditionalGeneration.from_pretrained(
    'google/paligemma2-3b-pt-448', torch_dtype=torch.bfloat16, device_map='auto')
model = PeftModel.from_pretrained(base, 'outputs/lora_adapters/final').eval()
processor = PaliGemmaProcessor.from_pretrained('outputs/lora_adapters/final')

def detect(image):
    inputs = processor(text=PROMPT, images=image, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    text = processor.batch_decode(out, skip_special_tokens=False)[0]
    # parse <locYYYY><locXXXX><locYYYY><locXXXX> label ; ...
    pattern = re.compile(r'<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*(\w+)?')
    boxes = []
    for m in pattern.finditer(text):
        ny1, nx1, ny2, nx2, label = m.groups()
        h, w = image.height, image.width
        boxes.append({
            'bbox': [int(int(nx1)/1024*w), int(int(ny1)/1024*h),
                     int(int(nx2)/1024*w), int(int(ny2)/1024*h)],
            'label': label or 'sensitive',
        })
    return boxes
```
