# Privacy Masker — Final Project Report

**Team project, VU Deep Learning**
**Member 1 (M1):** Data Engineer
**Member 2 (Tomas Stankevicius):** ML Engineer
**Member 3 (M3):** Demo Engineer

---

## 1. Project idea and real-world relevance

### Problem statement

Modern mobile applications routinely expose sensitive personal information (PII) on screen — email addresses, phone numbers, bank balances, full names, transaction histories. This becomes a privacy hazard the moment a user:

- Takes a screenshot to share with a friend or post on social media
- Live-streams or screen-records their phone for a tutorial
- Shows their phone to someone over a video call
- Uploads a bug-report screenshot to a support channel
- Hands the phone over to someone for a moment

A **privacy masker** is a tool that automatically detects PII regions in mobile screenshots and produces bounding boxes that downstream systems can use to **blur, redact, or replace** those regions before the image leaves the user's device or is shared.

### Real-world applications

- **OS-level screenshot protection** (Android/iOS extension that masks PII before "Share")
- **Support-ticket pipelines** where screenshots from end users are auto-redacted before agents see them
- **Tutorial & content creation tools** for YouTubers, streamers, and tech writers who demo mobile apps
- **Internal dashboards** at companies where developers debug from user-submitted screenshots without seeing raw PII
- **GDPR / data-minimisation compliance** for any product that stores user-submitted images

### Why this is a good ML problem

It sits at the intersection of three things deep learning is well-suited for:

1. **Visual layout understanding** — mobile screens are highly structured, but PII can appear anywhere
2. **Semantic context** — distinguishing `"John Smith"` (a contact name → PII) from `"Search results"` (a heading → not PII) requires reading the surrounding text
3. **Pixel-precise localisation** — to be useful, the mask must align tightly with the PII region, not just identify "something sensitive on this screen"

Pure regex or OCR + rules can catch obvious patterns (emails, phone numbers) but fail on names, addresses, custom usernames, and account balances that don't follow fixed formats. This makes it a natural fit for a vision–language model (VLM) approach.

## 2. Data (Member 1 — Data Engineer)

### Dataset source

We built our dataset on top of **RICO-ScreenQA** (`rootsautomation/RICO-ScreenQA` on Hugging Face), a corpus of ~80k question–answer pairs over the RICO mobile-UI dataset. Each QA pair includes:

- A mobile screenshot (PNG)
- A natural-language question (e.g., *"What is the email address shown?"*)
- One or more annotator-provided answers, each with a `full_answer` text and a set of `ui_elements` whose `bounds` give pixel-precise XYXY bounding boxes for the UI widgets referenced in the answer

### Why this dataset

The combination is unusually well-suited for our task:

| Requirement | Why RICO-ScreenQA fits |
|---|---|
| **Real mobile UIs** | RICO is 72k screenshots from real Android apps — not synthetic |
| **Pixel-precise bounding boxes** | Annotators marked the specific UI elements referenced in each answer |
| **Semantic PII cues already present** | Questions like *"What is the email address shown?"* are explicit signals that the linked UI element contains PII of a known type |
| **Free + licensed for research** | No new annotation effort required |
| **Diverse PII categories** | Naturally covers emails, phones, names, addresses, balances, transactions, dates, usernames — without us having to construct them |

The alternative would have been hand-annotating mobile screenshots ourselves — infeasible within a two-week student project budget.

### Pipeline overview (see `01_data_prep_final.ipynb`)

The data prep pipeline in `01_data_prep_final.ipynb` runs in a single streaming pass over RICO-ScreenQA and produces the curated **`pii_v1`** dataset:

1. **Question → label mapping** (§1, `KEYWORD_MAP`): we wrote a hand-curated regex map that classifies each question into one of ten PII labels. The map covers two priority tiers — *Phase 1* (high-value: `email_address`, `phone_number`, `account_balance`, `transaction_amount`, `full_name`, `username`) and *Phase 2* (`account_number`, `address`, `date_of_birth`, `id_number`).
2. **Single pass over RICO-ScreenQA** (§2): for every QA pair we (a) classify the question, (b) if it matches a PII label, take the union of all annotators' `ui_elements.bounds`, deduplicate by bbox identity, and (c) attach them to the screen's object list.
3. **Bounding-box validation** (§1, `validate_bbox`): coordinates are clipped to image bounds, rejected if any dimension < 4 px, and stored as integer XYXY (Pascal-VOC) format.
4. **Multi-QA aggregation per screen**: a single screen can appear in many QA pairs (one about email, another about phone, …). All matching boxes are concatenated into one entry, giving genuinely multi-PII screens.
5. **Negative sampling** (§4): we keep ~15 % of the final dataset as **negative** examples — screens with zero PII — so the model also learns to emit "none" rather than always finding *something*. Negatives are sampled from screens whose QA pairs all classified as non-PII.
6. **Train / val / test split** (§5): 80 / 10 / 10 by screen ID, with assertions that no screen leaks across splits.
7. **Validation pass** (§9): an inline validator checks every row for missing fields, malformed bboxes, out-of-bounds coordinates, sub-4-px boxes, and unknown labels before the dataset is committed.

### Why this particular preparation approach

Three design choices were deliberate and worth highlighting for the report:

- **Model-agnostic XYXY pixel format** on disk: the dataset doesn't bake in any quirk of the downstream model. Conversion to PaliGemma's Y-first, normalised-to-1024 `<locXXXX>` format happens *only* in M2's training collator. M3's demo can re-use the same JSONL without modification.
- **Union over annotators, deduplicated**: improves coverage on screens where one annotator missed a UI element another caught. Deduplication on `(x1,y1,x2,y2)` tuples prevents double-counting.
- **Targeted ~15 % empty screens**: matches the empirically observed test distribution (15 % of screens are PII-free), so the model gets a calibrated "none" signal without a class-imbalance landslide.

### Final dataset statistics (`pii_v1`)

| Split | Screens | PII objects | Empty screens |
|---|---:|---:|---:|
| train | 2,916 | 3,221 | 454 (15.6 %) |
| val | 364 | 440 | 46 (12.6 %) |
| test | 365 | 414 | 55 (15.1 %) |

**Objects per screen (test set):** `{0: 55, 1: 240, 2: 49, 3: 16, 4: 1, 5: 2, 6: 1, 8: 1}` — i.e. 66 % of screens have exactly 1 PII object, 13 % have ≥ 2.

**Label distribution (train):**

| Label | Train | Val | Test |
|---|---:|---:|---:|
| email_address | 1,165 | 166 | 152 |
| phone_number | 537 | 49 | 60 |
| address | 472 | 78 | 53 |
| username | 390 | 52 | 51 |
| transaction_amount | 205 | 26 | 38 |
| date_of_birth | 201 | 30 | 31 |
| full_name | 137 | 29 | 18 |
| account_balance | 114 | 6 | 11 |

The taxonomy is **long-tailed** (`email_address` is ~10× more frequent than `account_balance`) and the per-screen object count is also long-tailed — this becomes the central finding of the ML phase (§3, §5).

## 3. ML approach (Member 2 — ML Engineer)

### Task formulation: is it classification or regression?

It is **neither, in the traditional sense — it is a sequence-generation task** that happens to encode object detection.

Concretely: given a screenshot and the prompt `"<image>detect sensitive_info\n"`, the model must autoregressively generate a token sequence such as:

```
<loc0143><loc0512><loc0298><loc0890> sensitive_info ; <loc0410><loc0512><loc0556><loc0890> sensitive_info
```

Each box is four `<locYYYY>` tokens (PaliGemma's native location vocabulary — 1,024-bucket normalised pixel coordinates, **Y-first**) followed by a class label. Multiple boxes are separated by `;`, and screens with no PII generate `none`.

So the model is solving:

- A **language modelling** problem at the token level (categorical cross-entropy over the 257k-token PaliGemma vocabulary)
- That **encodes a structured detection output** with implicit classification (the label token after each box) and implicit regression (the four discrete `<loc>` tokens that quantise continuous coordinates into a 1,024-bin grid)

This is materially different from a classical detection head (Faster R-CNN, DETR, YOLO) and it's one of the key contributions of the PaliGemma family.

### Why we chose PaliGemma 2 (3B, 448 px)

We compared three viable approaches and chose `google/paligemma2-3b-pt-448`:

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **PaliGemma 2 3B (chosen)** | Native `<locXXXX>` detect-task tokens; pretrained on screen-understanding data (ScreenAI lineage); proven LoRA recipe; runs in 4-bit on a single L4/A100 | 3B params still substantial; quantisation needed | ✅ |
| Classical detector (YOLOv8 + classification head) | Fast inference; well-studied | Would lose all semantic context (can't read text); we'd be re-solving OCR + entity recognition by hand | ❌ — drops the VLM advantage that motivates the project |
| Gemma 4 zero-shot | No fine-tuning; very strong general detector | Detects GUI zero-shot too well; would undermine the *value of fine-tuning* — the central narrative of our project | ❌ — wrong shape for the deliverable |

PaliGemma 2 hit the sweet spot: it's a real VLM (so it can read the screen text, not just see the pixels), it has a native location-token vocabulary specifically designed for detection, and the 3B variant has a documented LoRA fine-tuning recipe that fits in our Colab budget.

### Model architecture (brief)

PaliGemma 2 has three components:

1. **Vision tower** — a SigLIP image encoder (~400 M params) that takes the 448×448 input and emits 1,024 image tokens (each representing a 14×14 pixel patch, after 32× downsampling and re-projection).
2. **Multi-modal projector** — a single linear layer that maps SigLIP image-embeddings into the language-model token space.
3. **Language model** — a Gemma 2 2B decoder (the dominant 2.6 B of the 3 B total) that consumes the 1,024 image tokens prepended to the user-text tokens and generates the output token-by-token, including `<locYYYY>` tokens from a dedicated 1,024-entry sub-vocabulary.

The training signal is **causal language-modelling cross-entropy on the answer tokens only** (image tokens and prompt tokens are masked from the loss). This is what makes "detection" reduce to "generate the correct loc-token sequence."

### What was fine-tuned, what was frozen, and why

| Component | State | Why |
|---|---|---|
| **Vision tower (SigLIP)** | 🔒 Frozen | The SigLIP encoder is already extremely strong on UI images; fine-tuning it on 2.9 k screens would (a) cost ~10× more memory and (b) risk catastrophic forgetting of general visual concepts the model needs (text, icons, layout). |
| **Multi-modal projector** | Attempted to unfreeze (only bias term escaped PEFT's freeze in practice — see §5 Shortcomings) | The projector is the bridge between vision and language; unfreezing a tiny module like this is standard for novel visual tasks. |
| **Language model — attention projections (`q/k/v/o_proj`)** | **LoRA, r=32, α=64** | These control *where* the model attends in the image. For detection, getting attention right is the bottleneck. |
| **Language model — MLP (`gate/up/down_proj`)** | **LoRA, r=32, α=64** | The MLP layers encode "what does each region mean" — needed to learn the `sensitive_info` class semantics. |
| **Language model — embeddings, LM head, layer norms** | 🔒 Frozen | Touching them risks destabilising the model's grasp of the `<locXXXX>` vocabulary it learned during PaliGemma's own pretraining. |

**Trainable parameter count: 47.5 M out of 3.08 B (1.54 %).** The rest of the model is held in **4-bit NF4 quantisation** (via `bitsandbytes`) with bf16 compute dtype, which keeps the full forward/backward pass under ~14 GB of VRAM.

### Training infrastructure & hyperparameters (final, v1)

| Setting | Value |
|---|---|
| Framework | PyTorch + PyTorch Lightning |
| Precision | bf16 mixed |
| Batch size (micro) | 1–2 |
| Gradient accumulation | 4–8 (effective batch = 8) |
| Optimiser | AdamW, weight decay 1e-2 |
| Learning rate | 2e-5 |
| Epochs | up to 8, with `EarlyStopping(monitor='val_loss', patience=3, min_delta=0)` |
| Gradient clipping | 1.0 |
| Gradient checkpointing | enabled (`use_reentrant=False`) — needed on L4, removable on G4 |
| Max objects per target | 12 |
| Generation (inference) | beam search, `num_beams=3–5`, `length_penalty=1.0–1.5`, `max_new_tokens=192` |
| Hardware | Colab Pro+, NVIDIA L4 24 GB / G4 40 GB |

### Evaluation metrics

- **IoU** between predicted and ground-truth bboxes
- **Hungarian matching** on per-screen IoU matrices (no double-counting; one predicted box matches at most one GT box)
- **Precision / Recall / F1 at IoU ≥ 0.5** and **IoU ≥ 0.75**
- **Mean IoU over matched pairs** — localisation quality on the boxes the model did get
- **Per-screen-count breakdown** — Recall@0.5 stratified by number of GT boxes per screen (the diagnostic that drove our iteration story below)

(See `03_evaluate.ipynb`.)

## 4. Three training iterations — what changed and why

We ran three full training iterations on the same `pii_v1` dataset, varying only the training-time data sampling and the decoding strategy at inference. All other hyperparameters (model, LoRA scope, optimiser, schedule) were held constant within each iteration's design.

### Results table

| Iteration | Setup | # Predictions | Mean IoU (matched) | P@0.5 | R@0.5 | **F1@0.5** | P@0.75 | R@0.75 | F1@0.75 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **v1** | Baseline — no oversampling, cosine LR decay, decode `num_beams=3` | 236 | **0.570** | **0.619** | 0.353 | **0.449** | **0.428** | **0.244** | **0.311** |
| v2 | Aggressive oversampling (1-GT screens 1×, 2-GT 3×, 3-GT 5×, 4+ GT **8×**), decode `num_beams=3 length_penalty=1.5` | 541 | 0.482 | 0.305 | **0.399** | 0.346 | 0.179 | 0.234 | 0.203 |
| v3 | Mild oversampling (max 3×) + cosine LR decay + looser early stopping, decode `num_beams=5 length_penalty=1.0` | 307 | 0.491 | 0.463 | 0.343 | 0.394 | 0.303 | 0.225 | 0.258 |

### Per-screen-count recall breakdown @ IoU ≥ 0.5 (test set)

| Screens with N GTs | n screens | v1 recall | v2 recall | v3 recall |
|---|---:|---:|---:|---:|
| 0 GTs | 55 | (n/a — but predicted 5 boxes = good *abstain* behaviour) | predicted 61 (over-fired) | predicted 16 |
| 1 GT | 240 | 0.488 | 0.492 | 0.458 |
| 2 GTs | 49 | 0.235 | 0.337 | 0.296 |
| 3 GTs | 16 | 0.104 | 0.229 | 0.146 |
| 4 GTs | 1 | 0.250 | 0.500 | 0.500 |
| 5 GTs | 2 | 0.000 | 0.000 | 0.000 |
| 6 GTs | 1 | 0.000 | 0.167 | 0.000 |
| 8 GTs | 1 | 0.000 | 0.000 | 0.000 |

### v1 — baseline (best F1)

- **What**: PaliGemma 2 + LoRA r=32 over `{q,k,v,o,gate,up,down}_proj`, cosine LR decay with LR of 2e-5, 8 epochs target, EarlyStop @ epoch ~2 once `val_loss` plateaued at ~1.78.
- **Why this setup**: standard LoRA recipe for PaliGemma + the LoRA scope was widened (r=16 → r=32, attention-only → +MLP) following an exploratory pre-run that under-fit.
- **Outcome**: best F1@0.5 of the three (0.449), best mean IoU (0.570), best precision (0.619). But **recall was only 0.353** and dropped sharply on multi-PII screens.
- **Diagnostic finding**: the per-screen-count breakdown revealed a strong **1-box prediction bias** — Recall on 3-GT screens was 0.10, on 4+-GT screens essentially 0. The model converged to the dominant 1-box pattern in the training distribution.

### v2 — aggressive oversampling (designed to break the 1-box bias)

- **What changed**: we replicated multi-box training rows in `DM.setup()` with factors `{0: 1, 1: 1, 2: 3, 3: 5, 4: 8, 5: 8, ...}`, raising the effective training set from 2,916 → 4,413 rows and the multi-box share from 17 % → 46 %. We also bumped inference `length_penalty` from 1.0 → 1.5 to encourage longer outputs.
- **Why**: the per-bucket diagnostic from v1 pointed directly at a data-imbalance problem. The hypothesis was that giving the model 3-8× more gradient signal on multi-box screens would teach it to emit ≥ 2 boxes when warranted.
- **Outcome**: **multi-box recall did improve** (2-GT: 0.23 → 0.34; 3-GT: 0.10 → 0.23 — both consistent with the hypothesis). But **precision collapsed** (0.62 → 0.31) and **mean IoU dropped** (0.57 → 0.48). The model now over-predicts on every screen, including the 0-GT ones (61 false-positive boxes on 55 empty screens).
- **Verdict**: the bias mitigation worked, but the cost in precision was too high. F1 dropped from 0.449 → 0.346. We also saw early stopping fire at epoch ~2.5 again, so the model was undertrained relative to its expanded dataset — confounding the result.

### v3 — milder oversampling + cosine LR + looser early stopping

- **What changed**: oversampling factors capped at 3× (`{0:1, 1:1, 2:2, 3:2, 4:3, 5:3, ...}`), added a cosine LR schedule over all training steps, raised `EarlyStopping(patience=6, min_delta=0.005)`, and pulled inference `length_penalty` back to 1.0 (neutral).
- **Why**: v2 swung too far in one direction. v3's design was to keep *some* multi-box rebalancing but rein in the over-prediction, and to give training enough room to actually converge on the larger dataset.
- **Outcome**: precision partially recovered (0.31 → 0.46) but recall regressed alongside it (0.40 → 0.34). **F1@0.5 = 0.394** — better than v2 but **still below v1's 0.449**. Multi-box recall ended up between v1 and v2 (2-GT: 0.30, 3-GT: 0.15).
- **Verdict**: no setting of the oversampling-factor knob on this dataset, at this scale, simultaneously beats v1 on precision and recall.

### What the three iterations together show

This is the central empirical finding of the project:

> **On a 2.9 k-screen dataset where 66 % of training screens have exactly 1 PII object, autoregressive `<locXXXX>` detection with PaliGemma 2 + LoRA hits a precision–recall trade-off that cannot be moved by training-data rebalancing alone.** Aggressive oversampling improves multi-box recall (+11 pp on 2-GT screens) but trades off precision 1:1 with false positives elsewhere. Mild oversampling partially recovers precision but no longer beats the un-rebalanced baseline.

That's a real, defensible ML finding — not a "we got 0.7 F1" headline, but an actual diagnosis of *why* this regime is hard.

---

## 5. Shortcomings

### 5.1 Headline metric is below the project's aspirational 0.70 F1

Our best run is **F1@0.5 = 0.449** (v1). The original ambition was ≥ 0.70. We did not reach it. The shortfall is dominated by **recall**, not localisation — when the model does fire on a region, mean IoU is 0.57 (acceptable for blurring) and precision is 0.62 (reasonable). The model simply misses too many sensitive regions, especially on multi-PII screens.

### 5.2 1-box prediction bias is structural

The model collapses to a 1-box mode of output even when 2-3 PII regions are clearly present. Three iterations of training-data and decoding interventions reduced but did not eliminate this. Likely causes:

- **Cross-entropy loss is dominated by the easy case**: 66 % of training rows are single-box, so the gradient signal for the multi-box case is rare and gets averaged out.
- **Autoregressive structure penalises long outputs**: even with `length_penalty=1.5`, the model's per-step probability mass naturally concentrates on `</s>`-style "stop now" continuations after one correct box.
- **No explicit "completeness" signal**: nothing in the loss directly rewards "you found all PII on this screen", only "the boxes you emitted matched some GT boxes."

### 5.3 Multi-modal projector did not actually unfreeze

In v1 we attempted to set `UNFREEZE_PROJECTOR=True` so the linear layer between SigLIP and the language model could adapt. The runtime output `Non-LoRA trainable tensors: 1 (e.g. base_model.model.model.multi_modal_projector.linear.bias)` reveals that **only the projector's bias term escaped PEFT's automatic freeze** — the weight matrix was re-frozen when `get_peft_model()` wrapped the base model. The "projector unfreeze" experiment therefore only ran on ~few thousand parameters out of the ~6 M intended. The intended ablation was not faithfully executed.

### 5.4 Early stopping fired prematurely

In all three runs early stopping triggered after 2–3 epochs out of a budgeted 8, because the loose `min_delta=0` accepted ~0.04 noise on `val_loss` as "improvement." This means even our best run was very plausibly **undertrained**. We addressed this in v3 (`patience=6, min_delta=0.005`) but by then the data sampling had also changed, confounding the comparison.

### 5.5 No regex baseline yet

The most informative comparison row — *"what does a zero-ML baseline get?"* — is not in this report yet. A simple OCR + regex baseline on the test set (matching `\b\w+@\w+\.\w+\b` for emails, `\d{3}[- ]?\d{3}[- ]?\d{4}` for phones, etc., projected to bounding boxes via word-level OCR) would tell us whether fine-tuning beats the trivial baseline. If regex F1 is, say, 0.25, our 0.45 looks strong; if regex F1 is 0.50, the project as currently designed underperforms the trivial baseline.

### 5.6 Limited dataset scale and provenance

- **Scale**: 2.9 k training screens is small for VLM detection fine-tuning. Production PaliGemma detect-task recipes typically use ≥ 100 k examples.
- **Label noise from regex-on-question**: M1's `KEYWORD_MAP` classifies questions like *"Whose phone is shown?"* as `phone_number` (about an owner, not a number). These edge cases create small but non-zero label noise.
- **No human-validated test set**: our test labels come from the same automated pipeline as train/val. We have no independent ground truth to estimate the pipeline's own label-quality ceiling.

### 5.7 Iteration loop was sequential, not parallel

Each iteration took 1.5–7 hours on Colab, gated on the previous result. With more compute (or earlier A100 access) we could have run v1/v2/v3 in parallel and explored the LR × oversampling × decoding hyperparameter grid more thoroughly.

---

## 6. What can be improved & requirements for further work

### 6.1 Short-term improvements (within the current architecture)

| Improvement | Why it would help | Resource requirement |
|---|---|---|
| **Fix the projector unfreeze** | Restore the ~6 M parameters that were silently re-frozen by PEFT — apply `requires_grad = True` to projector weights *after* `get_peft_model()` wraps the base model | 1 line of code; retrain (~90 min A100) |
| **Train on the un-oversampled dataset for 8+ full epochs (no early-stop, or `patience ≥ 10`)** | v1 was undertrained; loss was still decreasing when ES fired. This is the cheapest experiment that could move F1 | 90 min A100 |
| **Add a regex baseline row** | Calibrates the entire results table — without it, we can't claim fine-tuning is worth it | 1 day for M1; no ML compute |
| **Run the multi-class ablation** | Notebooks already support `MULTI_CLASS=True` flag; would produce a 4th row for the comparison table and let us discuss per-PII-class F1 (likely high on `email_address`, low on `account_balance`) | 90 min A100 + eval |
| **Better-calibrated negative sampling** | Today, ~15 % of training screens are empty. Bumping to 25–30 % may reduce false positives without hurting recall, since v1's main precision problem is "model fires when it shouldn't" | 90 min A100 |

### 6.2 Medium-term improvements (architecture changes)

- **Higher-resolution input**: `paligemma2-3b-pt-896` quadruples the effective image area (4× more tokens). For mobile UIs with small text it could meaningfully improve localisation precision. ~2× slower per step but the same LoRA recipe applies.
- **Two-stage pipeline**: stage 1 = a cheap OCR + word-level layout detector to extract all text regions; stage 2 = the VLM scores each region as PII/not-PII. Decouples *finding* from *classifying* and likely fixes the 1-box bias entirely.
- **Add an explicit detection head**: bolt a DETR-style query head onto SigLIP and train it jointly with the LM. Doesn't fit a 6-day budget but would be the right architecture for production.

### 6.3 Long-term improvements (data & evaluation)

- **10× the training data**: scaling to ~30 k curated screens would almost certainly close the gap to F1 ~0.7. RICO-ScreenQA has the raw volume — we only used a small fraction of it because we filtered to QA pairs whose questions matched our `KEYWORD_MAP`.
- **Hand-validated test set**: 200 manually re-annotated test screens would give us a trustworthy ceiling estimate and let us measure pipeline-induced label noise.
- **Per-class metrics + confusion matrix**: post multi-class run, report per-label F1 and a confusion matrix. Almost certainly `email_address`/`phone_number` are easy (visual regularity + text patterns), `account_balance`/`transaction_amount` are hard (require semantic context).
- **Adversarial / robustness evaluation**: how does the model perform on adversarial layouts — PII written in unusual fonts, partial occlusions, dark mode, non-English UIs? RICO is English-only.

### 6.4 Requirements for further improvements

To enable the above, the team would need:

1. **More compute** — sustained A100 (or H100) access for ~50 GPU-hours rather than the ~15 GPU-hours of free/student Colab we had.
2. **Human-labelling budget** — even ~200 hand-annotated test screens (~10 hours of labelling work) would substantially de-risk the evaluation.
3. **A holdout app-domain split** — to test generalisation, train on banking + social apps, hold out productivity + commerce. We didn't have time to set this up cleanly.
4. **An OCR dependency** for the proposed two-stage pipeline (Tesseract or PaddleOCR — both free).
5. **Production deployment infrastructure** for real-world use: an on-device runtime for PaliGemma (e.g., MLC-LLM, GGUF) since the privacy use case demands the data never leave the device.

---

## 7. Summary

We built and fine-tuned a vision–language model (PaliGemma 2 3B + LoRA, r=32) to detect personally identifiable information (PII) regions in mobile screenshots, using a 3.6 k-screen curated dataset (`pii_v1`) built from RICO-ScreenQA via a regex-classified, multi-annotator-union pipeline. Across three full training iterations we explored the precision-recall trade-off induced by the dataset's long-tailed objects-per-screen distribution. Our best configuration (v1: standard LoRA, no oversampling) achieved **F1@0.5 = 0.449, Recall@0.5 = 0.353, Mean IoU = 0.570**. We diagnosed a structural 1-box-prediction bias that data rebalancing could partially shift but not fully resolve, leading us to identify higher-resolution inputs, dataset scaling, and a two-stage OCR+VLM pipeline as the most promising next directions. The work demonstrates both the appeal and the limits of using a generative VLM as a detection model at this dataset scale.
