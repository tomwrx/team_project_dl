# Dataset Schema — Privacy Masker Project

**Audience:** Member 1 (Data Engineer)
**Author:** Member 2 (ML Engineer)
**Status:** v1.0 — please flag any concerns before you start producing data on Day 2.

This document is the single source of truth for how the curated PII dataset must be formatted. Following it exactly means I can drop your file into the training pipeline with a one-line config change.

---

## 1. File format

- **One JSONL file per split**: `train.jsonl`, `val.jsonl`, `test.jsonl`.
- One JSON object per line. No trailing commas. UTF-8 encoded.
- Place all three files in a single folder, e.g. `data/pii_v1/`.
- Images live in a sibling folder, e.g. `data/pii_v1/images/`.

Recommended split ratio: **80 / 10 / 10** by *screen ID* (not by example). This prevents the same screenshot leaking between train and val.

---

## 2. Per-example schema (canonical, model-agnostic)

Every line in the JSONL must be a JSON object with **exactly these keys**:

```json
{
  "image": "images/12345.png",
  "image_width": 1440,
  "image_height": 2560,
  "screen_id": "12345",
  "objects": [
    {
      "bbox": [120, 340, 980, 410],
      "label": "email_address",
      "source_question": "What is the email address shown?",
      "source_answer": "tomas@example.com"
    },
    {
      "bbox": [120, 1200, 980, 1280],
      "label": "account_balance",
      "source_question": "What is the total balance?",
      "source_answer": "$2,453.12"
    }
  ]
}
```

### Field-by-field contract

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | string | ✅ | Path **relative to the JSONL file's directory**. Use forward slashes. PNG or JPG. |
| `image_width` | int | ✅ | Original pixel width of the image on disk. Do not pre-resize images. |
| `image_height` | int | ✅ | Original pixel height of the image on disk. |
| `screen_id` | string | ✅ | The RICO screen ID. Used to guarantee no leakage across splits. |
| `objects` | list | ✅ | Zero or more sensitive elements. **An empty list is valid and useful** — it teaches the model "this screen has nothing to mask." Aim for ~10–20% empty examples. |
| `objects[].bbox` | list[int] | ✅ | **See section 3.** |
| `objects[].label` | string | ✅ | **See section 4.** |
| `objects[].source_question` | string | optional | The original ScreenQA question. Helpful for debugging. |
| `objects[].source_answer` | string | optional | The original ScreenQA answer text. Helpful for debugging. |

---

## 3. Bounding box format — READ CAREFULLY

This is the #1 place where things go wrong. Get it right and we save a day of debugging.

### The contract

```
bbox = [x_min, y_min, x_max, y_max]
```

- **Order**: `x_min, y_min, x_max, y_max` (a.k.a. XYXY / Pascal VOC style).
- **Units**: absolute pixels of the original image (not normalized, not 0–1, not 0–1000).
- **Origin**: top-left of the image is `(0, 0)`. X increases to the right, Y increases downward.
- **Type**: integers (round, don't truncate). Cast with `int(round(v))`.
- **Constraints** that must hold:
  - `0 <= x_min < x_max <= image_width`
  - `0 <= y_min < y_max <= image_height`
  - Width (`x_max - x_min`) and height (`y_max - y_min`) must each be `>= 4` pixels. Drop anything smaller — these are noise.

### What ScreenQA actually gives you

ScreenQA's `answers_and_bboxes` already provides `[x_min, y_min, x_max, y_max]` in absolute pixels of the original RICO screenshot. So in most cases **you can pass it through unchanged**.

> ⚠️ Sanity-check this on Day 1 with 5–10 examples by drawing the box on the image. RICO has had reports of mismatched aspect ratios in some splits.

### Why this format and not `[x, y, w, h]`?

PaliGemma and Gemma 4 both internally consume XYXY-style coordinates. Storing canonical XYXY means I never have to convert in the training loop — I just normalize. Conversion to `[x, y, w, h]` for the OpenCV blur step (Member 3) is trivial and happens on his side.

### I will handle (you do NOT need to do)

- Normalizing to 0–1024 PaliGemma location tokens.
- Converting to Gemma 4 JSON `box_2d` format with `[y1, x1, y2, x2]` ordering.
- Image resizing.
- Tokenization.

You stay in pixel space. I handle everything model-specific.

---

## 4. Label taxonomy

Use one of these exact strings as the `label`. If you find a sensitive element that doesn't fit, ping me and we'll add a new category together — don't invent new strings ad hoc.

| Label | Examples |
|---|---|
| `email_address` | "user@example.com" shown anywhere |
| `phone_number` | Personal phone numbers |
| `full_name` | Person's first + last name as displayed |
| `username` | Account handles, login names |
| `account_balance` | "Total Balance", "Available", "Net worth" |
| `transaction_amount` | Individual transaction values, payment amounts |
| `account_number` | Bank account, card numbers (partial or full) |
| `address` | Physical/mailing addresses |
| `date_of_birth` | DOB fields |
| `profile_photo` | Avatar / profile image regions |
| `id_number` | SSN, passport, national ID |
| `other_sensitive` | Use sparingly — flag for review |

**Phase 1 priority**: focus on the first 6 labels. They're the most common and impactful. Add the rest only if time allows.

---

## 5. Filtering rules (ScreenQA → PII subset)

Suggested workflow for Day 2:

1. Load every QA pair from ScreenQA.
2. Apply keyword + regex filters on the **question** text (e.g. `email`, `balance`, `total`, `account`, `phone`, `address`, `name`). Maintain a mapping from matched keyword → label.
3. For each matched QA pair, take the `answer_bboxes` and emit one `object` entry per bbox.
4. Group all objects belonging to the same `screen_id` into a single JSONL line.
5. Add ~10–20% empty-objects examples (random RICO screens with no sensitive matches) so the model learns to output "nothing here."
6. Run the validator script (section 8) before handing the data over.

Target dataset size for our 14-day timeline:
- **train**: 800–1,500 examples
- **val**: 100–200 examples
- **test**: 100–200 examples

More is not better here — quality > quantity. A clean 1K beats a noisy 10K for LoRA fine-tuning.

---

## 6. Example: a complete `train.jsonl` (3 lines)

```
{"image":"images/53412.png","image_width":1440,"image_height":2560,"screen_id":"53412","objects":[{"bbox":[88,412,742,488],"label":"email_address","source_question":"What is the user's email?","source_answer":"jane@example.com"}]}
{"image":"images/61203.png","image_width":1440,"image_height":2560,"screen_id":"61203","objects":[{"bbox":[120,1180,1320,1290],"label":"account_balance","source_question":"What is the total balance?","source_answer":"$2,453.12"},{"bbox":[120,1350,1320,1430],"label":"transaction_amount","source_question":"How much was the last payment?","source_answer":"$45.00"}]}
{"image":"images/77891.png","image_width":1080,"image_height":1920,"screen_id":"77891","objects":[]}
```

---

## 7. Folder layout I expect to receive

```
data/pii_v1/
├── train.jsonl
├── val.jsonl
├── test.jsonl
├── README.md            # short notes: filter rules used, label counts, known issues
└── images/
    ├── 53412.png
    ├── 61203.png
    └── ...
```

Zip the whole `pii_v1/` folder and drop it in our shared drive. Don't change the folder name.

---

## 8. Validator script

I'll provide a `validate_dataset.py` in the repo that you can run before handing over:

```bash
python validate_dataset.py --data-dir data/pii_v1/
```

It will check: all required fields present, bbox bounds valid, image files exist and match declared dimensions, labels in taxonomy, no screen_id leakage between splits, and print per-label counts. **Please do not hand over data that fails validation.**

---

## 9. Versioning

If the schema changes, we bump the folder name: `pii_v1/` → `pii_v2/`. Never edit a delivered dataset in place — make a new version. This way I can run regressions comparing model performance across data versions for the final report.

---

## 10. Open questions for Member 1

Please confirm or push back on Day 1:

1. ☐ You're OK with XYXY pixel format (not XYWH, not normalized)?
2. ☐ You can produce the validator-passing dataset by end of Day 3?
3. ☐ The label taxonomy in section 4 covers what ScreenQA actually contains, or do we need to add categories?
4. ☐ Any RICO screens that should be excluded (corrupted images, etc.)?

Reply in our team channel or directly on this doc. Once confirmed, we lock the schema for v1.
