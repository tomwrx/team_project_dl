# Privacy Masker — Final Project Report

**Team Project, VU Deep Learning**
**Members:** M1 (Data Engineer), Tomas Stankevicius (ML Engineer), M3 (Demo Engineer)

---

## 1. Project Idea & Real-World Relevance

Mobile applications routinely expose sensitive personal information (PII) like emails, balances, and names. This creates a privacy hazard when users share screenshots. Our **Privacy Masker** automatically detects PII regions and produces bounding boxes for downstream redaction.

We approached this using a Vision-Language Model (VLM) because PII detection requires **visual layout understanding**, **semantic context** (distinguishing a name from a heading), and **pixel-precise localization**. Pure regex or OCR fails on unstructured PII, making VLMs the ideal solution.

## 2. Data Engineering

We built two datasets from the `RICO-ScreenQA` corpus (real Android UIs with pixel-precise bounds):

* **`pii_v1`** — 3,645 screens, broad keyword regex, no stratified split. Used for §4 (PaliGemma) and §5 v1–v2 (two-stage).
* **`pii_v5`** — 9,989 screens, precision-first regex engine, stratified multi-label split (rarest-label heuristic), pruned ambiguous labels, biometrics added to `other_sensitive`. Used for §5 v3.

Both follow the same per-screen schema (`image`, `objects[].bbox`, `objects[].label`). Negative examples (~12–15%) are included to teach abstention.

**Key finding (v1):** The dataset is highly long-tailed — **66% of screens contain exactly 1 PII object**, which heavily influenced the model's prediction bias in §4.

## 3. ML Approach

* **Task Formulation:** We framed detection as autoregressive sequence generation. The model outputs token sequences like `<loc0143><loc0512>... sensitive_info`.
* **Model:** `google/paligemma2-3b-pt-448` in 4-bit NF4 quantization.
* **Fine-Tuning (LoRA):** The vision tower and embeddings were strictly frozen. We applied LoRA (r=32, α=64) to the LM's attention and MLP layers. Trainable parameters: 47.5M (1.54%).
* **Metrics:** Evaluated using Intersection over Union (IoU), Hungarian matching, and F1@0.5.

## 4. Experiments & Results

We ran three iterations to address a structural **"1-box prediction bias"** (the model rarely predicted >1 box due to the training distribution).

| Setup | Mean IoU | Precision | Recall | **F1@0.5** |
|---|---:|---:|---:|---:|
| **v1 (Baseline):** No oversampling, num_beams=3 | **0.570** | **0.619** | 0.353 | **0.449** |
| **v2 (Aggressive Oversampling):** 8x multi-box sampling | 0.482 | 0.305 | **0.399** | 0.346 |
| **v3 (Mild Oversampling):** Max 3x sampling, Cosine LR | 0.491 | 0.463 | 0.343 | 0.394 |

**Key Takeaway:** Data rebalancing alone cannot break the VLM's autoregressive bias. Aggressive oversampling improved multi-box recall but destroyed precision (hallucinating boxes on empty screens). v1 remains the most robust configuration.

## 5. Two-Stage Pipeline (OCR + LLM Classifier)

Motivated by the 1-box bias finding in §4, we re-architected the task by decoupling **localization** from **classification**. Stage 1 uses an off-the-shelf OCR engine to extract every text region; stage 2 uses a fine-tuned LLM to label each region as PII or not. This removes the autoregressive bottleneck entirely — the model no longer has to generate coordinates, only classify them.

* **Stage 1 — OCR Layout Extraction:** `PaddleOCR` (angle-corrected, English) is run on each screenshot to produce a list of `{bbox, text, confidence}` tuples. No fine-tuning required.
* **Stage 2 — LLM Classifier:** `google/gemma-4-E4B` in 4-bit NF4 with LoRA (r=32, α=64) applied to the language-model projections only. Trainable parameters: 69.8M (0.87%).
* **Prompt Format:** Each OCR region is rendered as `[index@x,y] "text"` with coordinates normalized to a 1000×1000 grid, so the model sees spatial layout as well as content. The model emits a JSON map from tag to PII class.
* **Label Assignment (training data):** OCR regions are matched to ground-truth PII boxes via IoU ≥ 0.3 with a containment fallback (≥70% of OCR box inside GT) to rescue multi-line entities. A content-based override forces email- and phone-shaped strings to their true class, even when the host UI field is labeled differently.
* **Metrics:** Token-level set Precision / Recall / F1 (per-class and micro-averaged) plus JSON formatting validity.

### Experiments & Results

Three iterations across two datasets:

| Setup | Train | Test | Precision | Recall | **Micro F1** | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| **v1 (Flat string list):** OCR text as plain list, no layout | pii_v1 (2.9k) | pii_v1 | 0.327 | 0.410 | 0.364 | 0.230 |
| **v2 (Layout-aware):** Normalized (x,y) coords + containment matching + content override | pii_v1 (2.9k) | pii_v1 | 0.767 | 0.615 | **0.683** | 0.517 |
| **v3 (pii_v5 retrain):** Same v2 setup, trained on the 3×-larger stratified dataset | pii_v5 (7.9k) | pii_v5 | 0.636 | 0.544 | 0.586 | **0.536** |

**Per-class F1 (v2 on pii_v1):** `email_address` 0.86 · `date_of_birth` 0.74 · `full_name` 0.62 · `username` 0.58 · `phone_number` 0.55 · `address` 0.50 · `transaction_amount` 0.48 · `account_balance` 0.33.

**Per-class F1 (v3 on pii_v5):** `email_address` 0.84 · `phone_number` 0.59 · `address` 0.56 · `full_name` 0.56 · `account_balance` 0.56 · `transaction_amount` 0.55 · `username` 0.47 · `other_sensitive` 0.42 · `date_of_birth` 0.28.

**Key Takeaways:**
1. **Decoupling localization from classification eliminates the 1-box bias entirely** — the model labels every OCR region independently, so dense screens with 4+ PII fields are no longer penalized.
2. **Layout matters as much as content.** Injecting normalized coordinates into the prompt nearly doubled micro F1 (0.36 → 0.68) on the same data, model, and training budget. Stripping coordinates discards the visual prior.
3. **v3 on pii_v5 looks worse on micro F1 but is actually a tougher exam.** The pii_v5 test set has 6.4× more `address` and 7.6× more `transaction_amount` instances than pii_v1 — the two hardest classes now dominate. Macro F1 in fact *improved* (0.517 → 0.536), and per-class precision rose on `account_balance` (+0.21), `full_name` (+0.09), and `phone_number` (+0.07).
4. **`date_of_birth` regressed sharply (F1 0.74 → 0.28) due to a label-noise issue in pii_v5.** The v5 regex tightened DOB to `dob`/`birthday`/`born`/`age` — mixing date strings ("03/15/1990") with age integers ("25") under one label. The model has no consistent signal to pick the right OCR token.
5. **`other_sensitive` is structurally weak (F1 0.42)** — mixing biometrics (weight/height), credentials (password/pin), and demographics (gender) under one label provides no shared visual or textual structure.
6. **Annotation policy mismatch caps `address` and `phone_number` recall on both datasets.** Manual error analysis showed the model consistently declines to label *business* addresses, phone numbers, and corporate emails as PII — semantically correct behavior penalized by ground truth. Reported recall on these classes is a lower bound on real-world performance.
7. **Greedy decoding outperformed beam search.** Beam search with mild length penalty pushed the model to emit more entities but predominantly wrong ones (micro F1 dropped to 0.607).

## 6. Limitations & Future Work

1. **OCR ceiling:** Pipeline recall is bounded above by Stage 1. Text PaddleOCR misses (low-contrast UI, icons, non-Latin scripts) is invisible to Stage 2.
2. **Label noise dominates remaining error budget.** Two concrete issues identified in v3: (a) `date_of_birth` conflates dates and ages; (b) `other_sensitive` mixes class-incoherent buckets. Splitting these would likely raise micro F1 by 5–10 points with no model changes.
3. **Personal vs. business ambiguity** on `address` and `phone_number` is a ground-truth policy issue, not a model failure. Re-annotating to mark subject ownership would raise measured recall on both datasets.
4. **Missing OCR + Regex baseline** to quantify how much the LLM contributes beyond simple pattern matching.
5. **Single-stage VLM constraints:** PaliGemma's autoregressive 1-box bias (§4) is a structural limit. Higher input resolution (`pt-896`) or grounded-decoding losses might mitigate it, but the two-stage approach sidesteps the problem entirely.

## 7. Summary

Two architectures explored for PII detection on mobile screenshots:

- **Single-stage VLM (PaliGemma 2 + LoRA)** — F1 = 0.449 on pii_v1, capped by autoregressive 1-box bias.
- **Two-stage pipeline (PaddleOCR + Gemma-4 LoRA classifier)** — F1 = 0.683 on pii_v1 and 0.586 on the 3×-larger pii_v5 (which has higher macro F1 = 0.536 due to harder class distribution).

For small-data, multi-entity detection on structured screens, **task decomposition with layout-aware prompting outperforms end-to-end VLM fine-tuning**. The v3 evaluation on pii_v5 further isolates label noise (rather than model capacity) as the dominant remaining error source — a finding that points directly to next steps for the data engineering side.
