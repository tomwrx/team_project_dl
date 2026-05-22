# Privacy Masker — Final Project Report

**Team Project, VU Deep Learning**
**Members:** M1 (Data Engineer), Tomas Stankevicius (ML Engineer), M3 (Demo Engineer)

---

## 1. Project Idea & Real-World Relevance

Mobile applications routinely expose sensitive personal information (PII) like emails, balances, and names. This creates a privacy hazard when users share screenshots. Our **Privacy Masker** automatically detects PII regions and produces bounding boxes for downstream redaction.

We approached this using a Vision-Language Model (VLM) because PII detection requires **visual layout understanding**, **semantic context** (distinguishing a name from a heading), and **pixel-precise localization**. Pure regex or OCR fails on unstructured PII, making VLMs the ideal solution.

## 2. Data Engineering

We built the **`pii_v1`** dataset from the `RICO-ScreenQA` corpus (real Android UIs with pixel-precise bounds).

* **Pipeline:** We mapped QA pairs to 10 PII categories using regex, deduplicated bounding boxes, and aggregated them per screen to capture multi-PII instances.
* **Negative Sampling:** 15% of screens were intentionally left empty (no PII) to teach the model to abstain.
* **Statistics:** 3,645 total screens (80/10/10 split).
* **Key Finding:** The dataset is highly long-tailed. **66% of screens contain exactly 1 PII object**, which heavily influenced the model's prediction bias.

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

We ran two iterations on the same training data (~2.9k screens):

| Setup | JSON Valid | Precision | Recall | **Micro F1** | Macro F1 |
|---|---:|---:|---:|---:|---:|
| **v1 (Flat string list):** OCR text concatenated as plain Python list, no layout | 100% | 0.327 | 0.410 | 0.364 | 0.230 |
| **v2 (Layout-aware):** Normalized (x,y) coords in prompt + containment matching + content override | 100% | **0.767** | **0.615** | **0.683** | **0.517** |

**Per-class F1 (v2):** `email_address` 0.86 · `date_of_birth` 0.74 · `full_name` 0.62 · `username` 0.58 · `phone_number` 0.55 · `address` 0.50 · `transaction_amount` 0.48 · `account_balance` 0.33.

**Key Takeaways:**
1. **Decoupling localization from classification eliminates the 1-box bias entirely** — the model labels every OCR region independently, so dense screens with 4+ PII fields are no longer penalized.
2. **Layout matters as much as content.** Injecting normalized coordinates into the prompt nearly doubled micro F1 (0.36 → 0.68) on the same data, the same model, and the same training budget. Stripping coordinates is equivalent to discarding the visual prior.
3. **Annotation policy mismatch caps `address` and `phone_number` recall.** Manual error analysis revealed the model consistently declines to label *business* addresses, phone numbers, and corporate emails as PII — semantically correct behavior penalized by ground truth, which labels any address-shaped or phone-shaped string regardless of subject. Reported recall on these classes is therefore a lower bound on real-world performance.
4. **Greedy decoding outperformed beam search.** Beam search with mild length penalty pushed the model to emit more entities but predominantly wrong ones (micro F1 dropped to 0.607), confirming the conservative greedy policy as optimal.

## 6. Limitations & Future Work

The two-stage pipeline reached micro F1 = 0.683 (macro 0.517) — a meaningful step up from the single-stage VLM ceiling at 0.449, but still short of production thresholds.

1. **OCR ceiling:** The pipeline is bounded above by Stage 1 recall. Any text PaddleOCR misses (low-contrast UI, icons containing text, non-Latin scripts) is invisible to Stage 2.
2. **Annotation noise:** The original `pii_v1` labels do not distinguish personal vs. business entities, suppressing measured F1 on `address` and `phone_number`. A re-annotation pass with this distinction would likely raise micro F1 by 3–5 points without retraining.
3. **Missing OCR + Regex baseline:** We still lack a regex-only baseline to quantify how much the LLM contributes beyond simple pattern matching on OCR output.
4. **Scale:** ~2.9k training screens is small. The 0.36 → 0.68 jump was driven by representation, not data — future gains likely require expanding to 10k+ screens with cleaner annotations.
5. **Single-stage VLM constraints:** PaliGemma's autoregressive 1-box bias (§4) is a structural limit, not a training artifact. Higher input resolution (`pt-896`) or grounded-decoding losses might mitigate it, but the two-stage approach sidesteps the problem entirely.

## 7. Summary

We explored two architectures for PII detection on mobile screenshots:

- **Single-stage VLM (PaliGemma 2 + LoRA)** reached **F1 = 0.449** with strong localization (Mean IoU 0.570) but a structural 1-box generation bias that capped recall on dense screens.
- **Two-stage pipeline (PaddleOCR + Gemma-4 LoRA classifier)** reached **F1 = 0.683** by decoupling localization from classification and injecting normalized layout coordinates into the prompt — nearly doubling performance on the same training data.

The project demonstrates that for small-data, multi-entity detection on structured screens, **task decomposition with layout-aware prompting outperforms end-to-end VLM fine-tuning**. The semantic power of VLMs is real, but their autoregressive constraints make them a poor fit for object detection at this data scale — a cheap OCR + LLM-classifier architecture wins decisively.
