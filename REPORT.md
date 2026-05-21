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

## 5. Limitations & Future Work

Our best F1 of 0.449 fell short of the 0.70 aspiration, heavily bottlenecked by recall.

1. **Undertraining & Bugs:** Early stopping fired prematurely (epoch 2).
2. **Missing Baseline:** We lack an OCR + Regex baseline to prove fine-tuning significantly outperforms a trivial approach.
3. **Structural Limits:** The causal LM structure penalizes long outputs. As a fix  can be to move to a two-stage pipeline (OCR layout detection → VLM classification) or increase input resolution (`pt-896`).
4. **Scale:** 2.9k training screens is too small; production VLMs use 100k+. As a dataset was quite tiny most improvements can be made here - expanding the dataset and allocating budget for a human-validated test set.

## 6. Summary

We successfully fine-tuned PaliGemma 2 to detect PII in mobile screenshots. While we achieved a solid localization foundation (Mean IoU 0.570), we identified a structural 1-box generation bias that limits recall on dense screens. This highlights both the semantic power and the autoregressive constraints of using VLMs for pure object detection at a **small data scale.**
