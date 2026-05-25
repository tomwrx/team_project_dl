# Dataset Schema — Privacy Masker Project

**Audience:** Team (Data Engineer, ML Engineer, Demo Engineer)
**Authors:** Aurimas Bžėskis (Data Engineer) · Tomas Stankevičius (ML Engineer) · Aida Katkauskaitė (Demo Engineer)
**Status:** v2.0 — final, reflects the delivered `pii_v5` dataset and the two-stage pipeline.

This document is the single source of truth for the curated PII dataset format. Following it exactly means any future dataset version drops into the training pipeline with a one-line config change. The contract has been stable since `pii_v2` and is unchanged in `pii_v5` — only the regex engine and split strategy have evolved.

---

## 1. File format

- **One JSONL file per split**: `train.jsonl`, `val.jsonl`, `test.jsonl`.
- One JSON object per line. No trailing commas. UTF-8 encoded.
- All three files live in a single dataset folder, e.g. `data/pii_v5/`.
- Images are distributed as `images.zip` alongside the JSONL files. Unpack to `images/` for inspection or training.

**Split ratio**: 80 / 10 / 10 by **screen ID** (not by example), so the same screenshot never leaks between train and val/test. As of `pii_v5`, splits are also **stratified** — see §5.

---

## 2. Per-example schema (canonical, model-agnostic)

Every line in the JSONL is a JSON object with **exactly these keys**:

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
| `image` | string | ✅ | Path **relative to the JSONL file's directory**. Forward slashes. PNG or JPG. |
| `image_width` | int | ✅ | Original pixel width on disk. Do not pre-resize images. |
| `image_height` | int | ✅ | Original pixel height on disk. |
| `screen_id` | string | ✅ | The RICO screen ID. Guarantees no leakage across splits. |
| `objects` | list | ✅ | Zero or more sensitive elements. **An empty list is valid and useful** — it teaches the model to abstain on PII-free screens. `pii_v5` ships with ~12% empty examples per split. |
| `objects[].bbox` | list[int] | ✅ | XYXY pixel coordinates. See §3. |
| `objects[].label` | string | ✅ | One of the taxonomy strings in §4. |
| `objects[].source_question` | string | optional | Original ScreenQA question. Kept for debugging and audit. |
| `objects[].source_answer` | string | optional | Original ScreenQA answer text. Kept for debugging and audit. |

---

## 3. Bounding box format — READ CAREFULLY

This is the #1 place things go wrong. Getting it right saved us a day of debugging.

### The contract

```
bbox = [x_min, y_min, x_max, y_max]
```

- **Order**: `x_min, y_min, x_max, y_max` (XYXY / Pascal VOC style).
- **Units**: absolute pixels of the original image (not normalized, not 0–1, not 0–1000).
- **Origin**: top-left of the image is `(0, 0)`. X increases right, Y increases down.
- **Type**: integers (round, don't truncate). Cast with `int(round(v))`.
- **Constraints** that must hold:
  - `0 <= x_min < x_max <= image_width`
  - `0 <= y_min < y_max <= image_height`
  - Width and height must each be `>= 4` pixels. Drop anything smaller — noise.

### Where the boxes come from

ScreenQA's `answers_and_bboxes` provides `[x_min, y_min, x_max, y_max]` in absolute pixels of the original RICO screenshot. In most cases this is passed through unchanged. Polygons from RICO are converted to their axis-aligned bounding box.

### Why XYXY and not `[x, y, w, h]`?

Both PaliGemma and Gemma internally consume XYXY-style coordinates. Storing canonical XYXY means the training pipeline never converts in the hot path — it just normalizes. Conversion to `[x, y, w, h]` for the OpenCV blur step happens in the demo app, not in the dataset.

### What the dataset producer does NOT need to do

The ML pipeline handles all model-specific transformations:

- Normalizing to 0–1024 PaliGemma `<loc>` tokens.
- Building Gemma layout-aware prompts (`[idx@nx,ny]` tags on a 1000×1000 grid).
- Image resizing.
- Tokenization.

The dataset stays in raw pixel space.

---

## 4. Label taxonomy

Use one of these exact strings as the `label`. The taxonomy has been stable since `pii_v2`. Two labels were deprecated in `pii_v5` due to unreliable regex signatures — see §5.

### Active labels (`pii_v5`)

| Label | Priority | Examples |
|---|:---:|---|
| `email_address` | P1 | "<user@example.com>" displayed anywhere |
| `phone_number` | P1 | Personal phone numbers, contact numbers, helplines |
| `full_name` | P1 | First/last/full/display name fields |
| `address` | P1 | Physical/mailing addresses, street, zip, city, country, destination |
| `date_of_birth` | P1 | DOB, birthday, born, age |
| `username` | P2 | Account handles, login names, user IDs |
| `account_balance` | P2 | "Total Balance", "Available", "Net worth" |
| `transaction_amount` | P2 | Individual transaction values, payment amounts, prices, fees |
| `other_sensitive` | P2 | Gender, password, passcode, PIN, weight, height (biometrics) |

**P1** — core PII (identity + contact). **P2** — extended PII (financial, credentials, biometrics).

### Deprecated labels (do not use)

| Label | Why deprecated |
|---|---|
| `account_number` | No reliable regex signature in ScreenQA question phrasing. High false-positive rate from generic "account" mentions. |
| `id_number` | Same issue — "id" is used generically throughout app UI text (user id, item id, order id). |
| `profile_photo` | Not implementable from QA text alone; would require image-level annotation. |

If a sensitive element doesn't fit any active label, classify as `other_sensitive` and flag it in the team channel — do not invent new label strings.

---

## 5. Filtering pipeline (ScreenQA → PII subset)

The dataset is built by classifying each ScreenQA QA pair via compiled regex matched against the **question** text, then taking the answer's bbox.

### Regex engine (`pii_v5` — precision-first)

| Label | Primary keywords | Regex |
|---|---|---|
| `email_address` | email, mail, gmail | `\b(?:email\|mail\|gmail)(?:\s+(?:address\|id))?\b` |
| `phone_number` | phone, contact, helpline | `\b(?:phone\|contact\|helpline)(?:\s+number)?\b` |
| `full_name` | first/last/full/display name | `\b(?:first\|last\|full\|display)\s+name\b` |
| `address` | address, street, zip, postal, location, city, country, destination | `\b(?:address\|street\|zip\|postal\|location\|city\|country\|destination)\b` |
| `date_of_birth` | dob, birthday, born, age | `\b(?:date\s+of\s+birth\|dob\|birthday\|born\|age)\b` |
| `account_balance` | balance, account balance | `\b(?:account\s+)?balance\b` |
| `transaction_amount` | charge, price, cost, fee, rent + bigrams | `\b(?:charge\|price\|cost\|fee\|rent)\b\|\b(?:transaction\|payment\|total\|purchase\|monthly)\s+(?:amount\|price\|cost\|payment)\b` |
| `username` | username, login name, user id | `\b(?:username\|login\s+name\|user\s+id)\b` |
| `other_sensitive` | gender, password, passcode, pin, weight, height | `\b(?:gender\|password\|passcode\|pin\|weight\|height)\b` |

### Pipeline

1. Load every QA pair from `rootsautomation/RICO-ScreenQA`.
2. Apply the regex above to each question. Maintain a mapping from matched regex → label.
3. For each matched QA pair, emit one `object` entry per `answer_bbox`.
4. Group all objects belonging to the same `screen_id` into a single JSONL line.
5. Add hard negative examples (~12% per split) — random RICO screens whose questions matched **no** PII regex. These teach the model to abstain.
6. **Stratified split** by primary label (see below), 80/10/10.
7. Run the validator (§8).

### Stratified splitting — "rarest-label heuristic"

Each screen's *primary label* is the globally rarest PII label it contains. Screens are grouped by primary label and split 80/10/10 within each group. This guarantees rare classes (`full_name`, `account_balance`) appear proportionally in val and test instead of being concentrated in train.

---

## 6. Example: a complete `train.jsonl` (3 lines)

```
{"image":"images/53412.png","image_width":1440,"image_height":2560,"screen_id":"53412","objects":[{"bbox":[88,412,742,488],"label":"email_address","source_question":"What is the user's email?","source_answer":"jane@example.com"}]}
{"image":"images/61203.png","image_width":1440,"image_height":2560,"screen_id":"61203","objects":[{"bbox":[120,1180,1320,1290],"label":"account_balance","source_question":"What is the total balance?","source_answer":"$2,453.12"},{"bbox":[120,1350,1320,1430],"label":"transaction_amount","source_question":"How much was the last payment?","source_answer":"$45.00"}]}
{"image":"images/77891.png","image_width":1080,"image_height":1920,"screen_id":"77891","objects":[]}
```

---

## 7. Folder layout

```
data/pii_v5/
├── train.jsonl
├── val.jsonl
├── test.jsonl
├── README.md            # short notes: regex used, label counts, known issues
├── images.zip           # all PNG files zipped (for transport / training)
└── images/              # unpacked PNG files (for inspection)
    ├── 53412.png
    ├── 61203.png
    └── ...
```

Zip the whole `pii_v5/` folder and drop it in shared Drive. Do not change the folder name.

---

## 8. Validator

Run before handing over any dataset version:

```bash
python validate_dataset.py --data-dir data/pii_v5/
```

It checks:

- All required fields present
- Bbox bounds valid (XYXY, integers, within image dimensions, ≥4px in each axis)
- Image files exist and match declared `image_width` / `image_height`
- All labels are in the active taxonomy (§4)
- No `screen_id` leakage between splits
- Negative-example ratio (10–15% per split)
- Per-label counts (logged for the report table)

**Do not hand over data that fails validation.**

---

## 9. Versioning

Schema changes bump the folder name: `pii_v1/` → `pii_v2/` → `pii_v5/`. Never edit a delivered dataset in place — always create a new version. This lets us run regressions comparing model performance across data versions for the final report.

### Version history

| Version | Screens | Notes |
|---------|--------:|-------|
| `pii_v1` | 3,645 | Baseline regex, train-only (no val/test split). Used for §4 (PaliGemma single-stage) and §5 v1–v2 (two-stage) in the report. |
| `pii_v2` | 4,549 | Same regex as v1; added 80/10/10 val + test splits. |
| `pii_v3` | 10,645 | Expanded keyword lists — **not uploaded**, too many false positives (>40% on `name`, `date`). |
| **`pii_v5`** | **9,989** | **Final dataset.** Precision-first regex (§5), stratified split, pruned ambiguous labels, biometrics added. Used for §5 v3 in the report. |

### Key changes in `pii_v5`

| Label | Change | Rationale |
|---|---|---|
| `full_name` | Removed bare `"name"` keyword | Matched product names, field labels, etc. Kept only `first/last/full/display name` bigrams. |
| `date_of_birth` | Strict patterns only | `"date"` alone matched app version dates, news article dates. Retained only `dob`, `birthday`, `born`, `age`. |
| `account_balance` | Added contextual bigrams | `"account balance"` reduces collision with non-financial uses of `"balance"`. |
| `other_sensitive` | Expanded to biometrics | Added `weight`, `height` alongside `gender`, `password`, `passcode`, `pin`. |
| `account_number` | **Deprecated** | No reliable regex signature in QA question phrasing. |
| `id_number` | **Deprecated** | `"id"` used generically throughout app UI text. |

---

## 10. Known label-quality issues (from `pii_v5` evaluation)

Two issues were identified during model evaluation and documented for future dataset iterations:

1. **`date_of_birth` conflates dates and ages.** The current regex matches both "date of birth" (date strings like `03/15/1990`) and "age" (integer values like `25`) under the same label. Future versions should either split these into `date_of_birth` and `age`, or restrict to date-shaped values only.

2. **`other_sensitive` is structurally heterogeneous.** Mixing biometrics (weight/height), credentials (password/pin), and demographics (gender) under one label gives the model no consistent visual or textual signal. Splitting into `credential`, `biometric`, and `demographic` buckets would likely improve per-class F1.

3. **Personal vs. business entities are not distinguished.** `address` and `phone_number` ground truth labels apply equally to personal data and business contact info (e.g. a casino's address). The model correctly tends to ignore business entities but is penalized for it. Re-annotating with subject ownership would raise measured recall on both classes.

---

## 11. Downstream pipeline reminder

The curated dataset is consumed by a **two-stage pipeline**:

1. **Stage 1** — PaddleOCR runs on each image to extract every text region as `{bbox, text, confidence}`. The dataset's GT boxes are matched to OCR regions via IoU ≥ 0.3 with a containment fallback (≥70% of OCR box inside GT) to handle multi-line entities.
2. **Stage 2** — Gemma-4 + LoRA classifier reads a layout-aware prompt and labels each OCR region as PII or not.

The dataset schema is identical regardless of which model architecture consumes it. Pixel-space XYXY boxes work for both the single-stage PaliGemma baseline and the two-stage final pipeline.
