# Privacy Masker - Final Project Report

**Team Project, VU Deep Learning**
**Members:** Aurimas Bžėskis (Data Engineer) · Tomas Stankevicius (ML Engineer) · Aida Katkauskaitė (Demo Engineer)

---

## 1. Project Idea & Real-World Relevance

Mobile applications routinely expose sensitive personal information (PII) - emails, balances, names, addresses - that is unintentionally leaked when users share screenshots in chat or social channels. **Privacy Masker** automatically detects PII regions in mobile UI screenshots and produces bounding boxes for downstream redaction. The system ships as a Slack bot: users send images privately via DM, the bot blurs the sensitive regions, and only the redacted version is posted to the intended public channel.

Detecting PII visually requires three things at once - **layout understanding** (a string is PII only in some positions on a form), **semantic context** (distinguishing a name from a heading), and **pixel-precise localization**. Pure regex on OCR text loses position; pure object detection loses meaning. We therefore explored deep-learning approaches that combine both.

---

## 2. Data Engineering

All datasets are derived from `rootsautomation/RICO-ScreenQA` (HuggingFace) - real Android UI screenshots with pixel-precise ScreenQA bounding-box answers. The data engineer ran **five iterations** before arriving at the final dataset, each fixing a specific failure mode found by ML evaluation on the previous version.

### Iteration history

| Version | Screens | Splits | What changed | Verdict |
|---------|--------:|--------|--------------|---------|
| `pii_v1` | 3,645 | train only (added split later) | Broad keyword regex (`email`, `phone`, `name`, `address`, …). ~15% negatives. | Baseline. Used by ML for 4 and 5 sections v1-v2. |
| `pii_v2` | 4,549 | 80/10/10 random | Same keyword map as v1; first proper val + test splits. | Random split underweighted rare classes in val/test. |
| `pii_v3` | 10,645 | - | Expanded keyword lists. | **Rejected - never uploaded:** >40% false positives on `name`/`date` from over-generic keywords. |
| `pii_v4` | - | - | Internal cleanup iteration on regex anchoring. | Superseded before release. |
| **`pii_v5`** | **9,989** | **80/10/10 stratified** | Precision-first regex engine, rarest-label stratified split, label pruning, biometric coverage added. ~12% negatives. | **Final dataset.** Used in §5 v3 and as the basis for production. |

### v5 regex engine (precision-first)

Aurimas redesigned the labeling pipeline from broad keyword matching to a **precision-first regex engine**. The construction process was data-driven:

1. **Top-50 n-gram analysis** on `RICO-ScreenQA` question text (filtering English stopwords) to find which words *actually* signal each PII class in the corpus, rather than guessing.
2. **Bigram / trigram passes restricted to `"what is the/your ..."` questions** - these phrasings carry the strongest direct-elicitation signal - to discover compound triggers like `account balance`, `transaction amount`, `helpline number`.
3. **Per-keyword sample inspection** - for every candidate keyword (including noisy ones like `date`, `id`, `total`), 10 random matched QA pairs were printed and manually reviewed before promotion or rejection. Keywords with high false-positive rates (`name` alone, `date` alone) were dropped from broad context and only kept inside anchored bigrams (`first name`, `date of birth`).

The resulting compiled regex table:

| Label | Pattern |
|-------|---------|
| `email_address` | `\b(?:email\|mail\|gmail)(?:\s+(?:address\|id))?\b` |
| `phone_number` | `\b(?:phone\|contact\|helpline)(?:\s+number)?\b` |
| `account_balance` | `\b(?:account\s+)?balance\b` |
| `transaction_amount` | `\b(?:charge\|price\|cost\|fee\|rent)\b \| \b(?:transaction\|payment\|total\|purchase\|monthly)\s+(?:amount\|price\|cost\|payment)\b` |
| `full_name` | `\b(?:first\|last\|full\|display)\s+name\b` |
| `username` | `\b(?:username\|login\s+name\|user\s+id)\b` |
| `address` | `\b(?:address\|street\|zip\|postal\|location\|city\|country\|destination)\b` |
| `date_of_birth` | `\b(?:date\s+of\s+birth\|dob\|birthday\|born\|age)\b` |
| `other_sensitive` | `\b(?:gender\|password\|passcode\|pin\|weight\|height)\b` |

### Stratified multi-label splitting ("rarest-label heuristic")

Random splitting on a long-tailed multi-label dataset under-represents rare classes in val/test (the v2 problem). The v5 split assigns each screen a **primary label = the globally rarest PII label it contains**, then performs an 80/10/10 split *within each primary-label group*. This guarantees that rare classes (`full_name`, `account_balance`) appear in val/test in proportion to train.

### Label pruning

Three labels from v1/v2 were deprecated in v5:

- **`account_number`, `id_number`** - no reliable keyword signature in QA phrasing; FP rate too high to rescue.
- **`profile_photo`** - visual class, not a text-region task; not learnable by the OCR-classifier architecture.

**`other_sensitive`** was introduced as a catch-all for biometrics + credentials + demographics (`weight`, `height`, `password`, `pin`, `gender`).

### Final `pii_v5` label distribution

| Label | Train | Val | Test |
|-------|------:|----:|-----:|
| `address` | 3,135 | 373 | 417 |
| `transaction_amount` | 2,316 | 323 | 279 |
| `email_address` | 1,975 | 240 | 229 |
| `other_sensitive` | 1,260 | 183 | 138 |
| `date_of_birth` | 884 | 115 | 121 |
| `phone_number` | 816 | 109 | 92 |
| `username` | 570 | 66 | 76 |
| `full_name` | 201 | 27 | 25 |
| `account_balance` | 153 | 18 | 19 |
| **Total screens** | **7,912** (913 neg) | **989** (114 neg) | **997** (115 neg) |

### Data pipeline outputs

Each `.jsonl` line follows a single schema (`image`, `image_width`, `image_height`, `screen_id`, `objects[].bbox`, `objects[].label`) so ML and demo notebooks consume identical inputs. Images are zipped into a single `images.zip` per version (reading hundreds of small files directly from Drive is ~50× slower than unpacking once locally). Negative-example screens (~12%) ship with `"objects": []` to teach abstention.

---

## 3. ML Approach - Single-Stage VLM (Baseline)

Initial architecture: a Vision-Language Model that ingests the full screenshot and **autoregressively** emits a sequence of bounding boxes with labels.

- **Task formulation:** detection-as-sequence-generation. The model outputs token sequences like `<loc0143><loc0512>...<loc0789> sensitive_info`.
- **Model:** `google/paligemma2-3b-pt-448` in 4-bit NF4 quantization.
- **LoRA setup:** vision tower and embeddings strictly frozen; LoRA (r=32, α=64) on the LM attention and MLP projections only. **47.5M trainable parameters (1.54%)**. An early bug that leaked LoRA into `vision_tower` was identified by a checkpoint-loading error and patched by restricting `target_modules` to language-model layers.
- **Metrics:** IoU + Hungarian matching, F1@0.5.

### Experiments & results

We ran three iterations targeting a structural **"1-box prediction bias"** - the autoregressive decoder rarely emits more than one box because the training distribution was dominated by single-PII screens (66% of `pii_v1` screens have exactly 1 PII object).

| Setup | Mean IoU | Precision | Recall | **F1@0.5** |
|---|---:|---:|---:|---:|
| **v1 (baseline):** no oversampling, num_beams=3 | **0.570** | **0.619** | 0.353 | **0.449** |
| **v2 (aggressive oversampling):** 8× multi-box sampling | 0.482 | 0.305 | **0.399** | 0.346 |
| **v3 (mild oversampling):** max 3× sampling, cosine LR | 0.491 | 0.463 | 0.343 | 0.394 |

**Key takeaway:** data rebalancing alone cannot break the VLM's autoregressive bias. Aggressive oversampling improved multi-box recall but destroyed precision (hallucinated boxes on negative screens). **v1 remained the best single-stage configuration** and is kept as a reference baseline.

---

## 4. ML Approach - Two-Stage Pipeline (Final)

Motivated by the 1-box bias, we re-architected the task by **decoupling localization from classification**. Stage 1 uses a deterministic OCR engine to extract every text region; Stage 2 uses a fine-tuned LLM to classify each region independently. The model no longer has to *generate* coordinates - only consume them - which eliminates the autoregressive bottleneck entirely.

### Architecture

| Stage | Component | Purpose |
|-------|-----------|---------|
| 1 | **PaddleOCR** (`use_angle_cls=True`, `lang='en'`) | Extract every text region as `{bbox, text, confidence}`. No fine-tuning. |
| 2 | **`google/gemma-4-E4B` + LoRA** (4-bit NF4, r=32, α=64, LM projections only) | Classify each OCR region by PII class. **69.8M trainable params (0.87%).** |

### Layout-aware prompting

Each OCR region is rendered into the prompt as `[index@x,y] "text"` with coordinates normalized to a 1000×1000 grid, so the LLM sees **spatial layout as well as content**. The model emits a JSON map from tag → PII class (or `null`).

### Training data construction

OCR regions are matched to ground-truth PII boxes using:

- **IoU ≥ 0.3**, OR
- **containment ≥ 70%** (OCR box inside GT) - rescues multi-line entities where OCR splits a single GT region into several lines.
- **Content-based override:** strings matching strict email / phone regex are forced to their true class, even when the host UI field is labeled differently.

### Metrics

Token-level set Precision / Recall / F1 (per-class and micro-averaged), plus JSON formatting validity.

### Experiments & results

Three iterations across two datasets:

| Setup | Train | Test | Precision | Recall | **Micro F1** | Macro F1 |
|---|---|---|---:|---:|---:|---:|
| **v1 (flat string list):** OCR text as plain list, no layout | pii_v1 (2.9k) | pii_v1 | 0.327 | 0.410 | 0.364 | 0.230 |
| **v2 (layout-aware):** normalized (x,y) coords + containment matching + content override | pii_v1 (2.9k) | pii_v1 | 0.767 | 0.615 | **0.683** | 0.517 |
| **v3 (pii_v5 retrain):** same v2 setup, trained on the 3× larger stratified dataset | pii_v5 (7.9k) | pii_v5 | 0.636 | 0.544 | 0.586 | **0.536** |

**Per-class F1 (v2 on pii_v1):** `email_address` 0.86 · `date_of_birth` 0.74 · `full_name` 0.62 · `username` 0.58 · `phone_number` 0.55 · `address` 0.50 · `transaction_amount` 0.48 · `account_balance` 0.33.

**Per-class F1 (v3 on pii_v5):** `email_address` 0.84 · `phone_number` 0.59 · `address` 0.56 · `full_name` 0.56 · `account_balance` 0.56 · `transaction_amount` 0.55 · `username` 0.47 · `other_sensitive` 0.42 · `date_of_birth` 0.28.

### Key takeaways

1. **Decoupling eliminates the 1-box bias entirely.** The model labels every OCR region independently, so dense screens with 4+ PII fields are no longer penalized.
2. **Layout matters as much as content.** Injecting normalized coordinates into the prompt nearly doubled micro F1 (0.36 → 0.68) on the same data, model, and training budget. Stripping coordinates discards the visual prior.
3. **v3 on pii_v5 looks worse on micro F1 but is actually a tougher exam.** The pii_v5 test set has 6.4× more `address` and 7.6× more `transaction_amount` instances than pii_v1 - the two hardest classes now dominate. Macro F1 in fact *improved* (0.517 → 0.536), and per-class precision rose on `account_balance` (+0.21), `full_name` (+0.09), and `phone_number` (+0.07).
4. **`date_of_birth` regressed sharply (F1 0.74 → 0.28).** The v5 regex tightened DOB to `dob`/`birthday`/`born`/`age` - mixing date strings (`"03/15/1990"`) with age integers (`"25"`) under one label. The model has no consistent signal to pick the right OCR token. Splitting `date_of_birth` from `age` in v6 would likely fix this with no model changes.
5. **`other_sensitive` is structurally weak (F1 0.42)** - biometrics (weight/height), credentials (password/pin), and demographics (gender) share no visual or textual structure.
6. **Annotation policy mismatch caps `address` and `phone_number` recall on both datasets.** Manual error analysis showed the model consistently declines to label *business* addresses, phone numbers, and corporate emails - semantically correct behavior penalized by ground truth. Reported recall is a lower bound on real-world performance.
7. **Greedy decoding outperformed beam search.** Beam search with mild length penalty pushed the model to emit more entities but predominantly wrong ones (micro F1 dropped to 0.607).

---

## 5. Deployment - Slack Bot Demo

The deliverable is a production-style **Slack bot built around a private-DM workflow**: users send images to the Privacy Masker bot via direct message, the bot processes them on a Colab-hosted backend, shows a blurred preview, and waits for explicit approval before posting the redacted version to the intended channel. The unblurred image never appears publicly.

### Components

| Component | Tech | Purpose |
|-----------|------|---------|
| **Slack bot** | Slack Bolt (Python), Socket Mode | Listens for DM file uploads, runs the channel-selector / edit / approve / discard flow, posts the final blurred image. |
| **FastAPI backend** | FastAPI on Colab GPU + ngrok tunnel | Receives base64-encoded images, runs Stage 1 + Stage 2 inference, applies PIL Gaussian blur, manages editor sessions. |
| **Web editor** | HTML5 canvas (vanilla JS) served by the backend | Lets the user draw additional blur boxes that the model missed, then re-blurs with the combined region set. |

### User flow

1. User DMs a screenshot to the bot.
2. Bot replies `🔍 Scanning for sensitive information...` and POSTs the image to `/process-image` on the backend.
3. Backend runs OCR + classifier, applies Gaussian blur (`radius=20`) to predicted PII boxes, returns the blurred image + box list.
4. Bot uploads the blurred preview to the DM with three buttons:
   - **Post to channel** → channel selector (built dynamically from `conversations.list`) → `files_upload_v2` to the chosen channel, with attribution `Shared by @username\n<original message>`.
   - **✏️ Edit boxes** → opens the web editor with the original image and current boxes; user draws additional regions; on confirm, backend re-blurs with the combined box set and the bot posts.
   - **✕ Discard** → session dropped, nothing posted.
5. The bot **rejects images uploaded directly to channels** - it only processes DMs.

### Security

- All backend calls require an `x-api-key` secret header - requests without it return 403.
- The backend URL is a **private ngrok tunnel** that rotates every Colab session; it is never published.
- **Images are processed in memory only** - nothing persists on disk, no logs of image bytes.
- The unblurred image is only ever seen by the sending user in their own private DM.
- Edit sessions live in an in-memory dict on the backend and expire when the Colab session ends.

### Required Slack scopes

`chat:write` · `files:read` · `files:write` · `im:history` · `im:read` · `im:write` · `channels:read` · `groups:read` · `mpim:read` · `users:read`.

### Earlier prototype

An earlier interface iteration (`TeamProjectInterfacev1.ipynb`) wired the same FastAPI backend into an **Office add-in** (taskpane HTML + manifest.xml), targeting Outlook/Word as the host. The Slack DM workflow superseded it because Slack's file-share + interactive-button model fits the "preview-before-post" pattern more naturally than an add-in.

---

## 6. Limitations & Future Work

1. **OCR ceiling.** Pipeline recall is bounded above by Stage 1. Text PaddleOCR misses (low-contrast UI, icons, non-Latin scripts) is invisible to Stage 2.
2. **Label-noise dominates the remaining error budget.** Two concrete issues identified in v3: (a) `date_of_birth` conflates dates and ages; (b) `other_sensitive` mixes class-incoherent buckets. Splitting these in a v6 would likely raise micro F1 by 5-10 points with no model changes.
3. **Personal vs. business ambiguity** on `address` and `phone_number` is a ground-truth policy issue, not a model failure. Re-annotating to mark subject ownership would raise measured recall on both datasets.
4. **Missing OCR + regex baseline** to quantify how much the LLM contributes beyond simple pattern matching on OCR output.
5. **Single-stage VLM constraints.** PaliGemma's autoregressive 1-box bias (section 3) is a structural limit. Higher input resolution (`pt-896`) or grounded-decoding losses might mitigate it, but the two-stage approach sidesteps the problem entirely.
6. **Demo deployment is Colab + ngrok.** Adequate for the project demo but not production: the tunnel rotates each session, the Colab GPU is non-persistent, and there is no horizontal scaling. A containerized inference service (vLLM + a persistent reverse proxy) would be the next step.

---

## 7. Summary

Two architectures were explored for PII detection on mobile screenshots:

- **Single-stage VLM (PaliGemma 2 + LoRA)** - F1 = 0.449 on `pii_v1`, capped by autoregressive 1-box bias.
- **Two-stage pipeline (PaddleOCR + Gemma-4 LoRA classifier)** - F1 = 0.683 on `pii_v1` and F1 = 0.586 on the 3× larger `pii_v5` (with higher macro F1 = 0.536 due to a harder class distribution).

For small-data, multi-entity detection on structured screens, **task decomposition with layout-aware prompting outperforms end-to-end VLM fine-tuning**. The v3 evaluation on `pii_v5` further isolates label noise - rather than model capacity - as the dominant remaining error source, pointing directly to the next data-engineering iteration.

The deliverable ships as a **Slack DM bot** wrapping the two-stage pipeline, with a private-by-default workflow and a web editor for human-in-the-loop correction - a pattern that matches the real-world privacy guarantee the project is trying to provide.
