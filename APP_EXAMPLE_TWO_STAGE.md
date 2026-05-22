# 🚀 Privacy-First UI Masker: Handoff Plan for Member 3

## Step 1: The "Handoff Package" (What you need to give Member 3)

Member 3 does not need to understand 4-bit quantization or LoRA, but they **must** have these three things from you:

1. **The LoRA Adapter:** The extracted `lora_adapter` folder from your Google Drive (contains `adapter_config.json` and `adapter_model.safetensors`).
2. **The Parser Function:** Your exact `loc_tokens_to_pixels` and regular expression (`LOC_RE`) functions to translate the model's text output back into math.
3. **The Prompt Rule:** Remind them of our hard-learned lesson: The prompt must be exactly `"<image>detect sensitive_info"` (or your multi-class string) with **NO trailing `\n`**.

## Step 2: The UI/UX Design (The "Demo Day" Flow)

For a presentation, the app needs to be foolproof. Member 3 should build a UI with:

* **Header:** "Privacy-First UI Masker" with a short description.
* **Split Screen:** Input Image (Left) ➔ Masked Image (Right).
* **Style Toggle:** A radio button letting the user choose between "Solid Redaction" (black boxes) or "Gaussian Blur" (looks highly professional).
* **Pre-loaded Examples:** This is critical. Live demos always fail when you try to upload a file under pressure. M3 must pre-load 3-4 distinct UI screenshots at the bottom of the app that the judges can just click to run.

Shared project link: <https://drive.google.com/drive/folders/1ttyaB7asqXWYA19vKnIcCFJYDv58cXEr?usp=sharing>
---

## 1. The Strategy: Google Colab + Gradio

To ensure our live Demo Day presentation is flawless, we cannot run this model locally on a standard laptop. A 3-Billion parameter Vision-Language Model requires a dedicated GPU to process images in real-time.

We will use **Option 1: Google Colab**.

* **Why:** It gives us free access to an NVIDIA T4 GPU.
* **How:** You will run the Gradio app inside a Colab notebook. By setting `share=True`, Gradio will generate a temporary public URL (e.g., `https://1234abcd.gradio.live`). We will open this link in the browser on the presentation screen, giving us a beautiful, full-screen SaaS UI backed by cloud GPU compute.

---

## 2. Step-by-Step Execution Guide

### Step A: Colab Setup

1. Open a new notebook in Google Colab.
2. Go to **Runtime > Change runtime type** and select **T4 GPU**.
3. Create a code cell and install our required dependencies by running:

   ```bash
   # Note: pytorch-lightning is no longer needed!
   !pip install -q gradio transformers peft bitsandbytes opencv-python-headless
    ```

Example code for app.py:

```python
import os
import re
import json
import logging
from pathlib import Path

import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image

from paddleocr import PaddleOCR
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ==========================================
# 1. CONFIGURATION
# ==========================================
GEMMA_MODEL_ID = "google/gemma-4-E4B"
# ⚠️ M3: UPDATE THIS PATH to wherever the adapter sits in shared Drive
ADAPTER_DIR = "/content/drive/MyDrive/VU_DL_Team_Project/outputs/gemma4_hf_adapter"

ALLOWED_LABELS = {
    "email_address", "phone_number", "full_name", "username", "address",
    "date_of_birth", "account_balance", "transaction_amount",
    "profile_photo", "other_sensitive",
}

logging.getLogger("ppocr").setLevel(logging.ERROR)


# ==========================================
# 2. STAGE 1 — OCR ENGINE
# ==========================================
print("⏳ Initializing PaddleOCR...")
ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    use_gpu=torch.cuda.is_available(),
    show_log=False,
)
print("✅ OCR ready")


def extract_text_elements(pil_image):
    """Run PaddleOCR on a PIL image. Returns list of {bbox, text, confidence}."""
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    results = ocr_engine.ocr(bgr, cls=True)
    if not results or not results[0]:
        return []

    elements = []
    for line in results[0]:
        polygon, (text, conf) = line
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        elements.append({
            "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
            "text": text,
            "confidence": round(float(conf), 3)
        })
    return elements


# ==========================================
# 3. STAGE 2 — LLM CLASSIFIER
# ==========================================
print(f"⏳ Loading Gemma base model in 4-bit: {GEMMA_MODEL_ID}")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    GEMMA_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

print(f"⏳ Loading LoRA adapter from {ADAPTER_DIR}")
classifier = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
classifier.eval()

tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_ID)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("✅ Classifier ready")


# ==========================================
# 4. PROMPT BUILDER (must match training format exactly)
# ==========================================
def build_prompt(ocr_elements, img_w, img_h):
    """Build the same prompt format used during training."""
    if not ocr_elements:
        return None, []

    row_bucket = max(1, img_h // 50)
    sorted_elements = sorted(
        ocr_elements,
        key=lambda e: (e["bbox"][1] // row_bucket, e["bbox"][0]),
    )

    lines = []
    tag_to_box = {}  # we need this to map predictions back to pixel coords for masking
    for i, ocr in enumerate(sorted_elements):
        x1, y1, x2, y2 = ocr["bbox"]
        nx = min(999, max(0, int(1000 * (x1 + x2) / 2 / max(1, img_w))))
        ny = min(999, max(0, int(1000 * (y1 + y2) / 2 / max(1, img_h))))
        tag = f"[{i}@{nx},{ny}]"
        lines.append(f'{tag} "{ocr["text"]}"')
        tag_to_box[tag] = ocr["bbox"]

    elements_str = "\n".join(lines)
    prompt = (
        "You are a privacy auditor. Below are text elements from a mobile screenshot. "
        "Each line is formatted as [index@x,y] \"text\" where x,y is the element's center "
        "on a 1000x1000 normalized grid (top-left = 0,0).\n\n"
        "Classify each element that contains personally identifiable information (PII) "
        "into one of: email_address, phone_number, full_name, username, address, "
        "date_of_birth, account_balance, transaction_amount, profile_photo, other_sensitive.\n\n"
        "Return ONLY a JSON object mapping the [index@x,y] tag to its label. "
        "Omit non-PII elements. Example: {\"[3@500,200]\": \"email_address\"}.\n\n"
        "ELEMENTS:\n" + elements_str + "\n\nJSON:"
    )
    return prompt, tag_to_box


# ==========================================
# 5. RESPONSE PARSER
# ==========================================
def extract_json_from_output(text):
    """Pull the first {...} block from generated text."""
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except json.JSONDecodeError:
        pass
    return {}


# ==========================================
# 6. MASKING
# ==========================================
def mask_image(pil_image, boxes_with_labels, style, show_labels=False):
    """Apply blur or solid-box masking. Optionally annotate with labels."""
    cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    for (x1, y1, x2, y2), label in boxes_with_labels:
        # Clamp to image bounds
        h, w = cv_img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        if style == "Solid Black Box":
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 0, 0), -1)
        elif style == "Gaussian Blur":
            roi = cv_img[y1:y2, x1:x2]
            # Kernel size must be odd and not larger than ROI
            k = max(3, min(51, (min(roi.shape[:2]) // 2) * 2 + 1))
            cv_img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)

        if show_labels:
            cv2.putText(
                cv_img, label, (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )

    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


# ==========================================
# 7. END-TO-END PIPELINE
# ==========================================
def process_screenshot(image, style, show_labels):
    if image is None:
        return None, "No image uploaded."

    img_w, img_h = image.size

    # Stage 1: OCR
    ocr_elements = extract_text_elements(image)
    if not ocr_elements:
        return image, "⚠️ OCR found no text on this screenshot."

    # Build training-format prompt
    prompt, tag_to_box = build_prompt(ocr_elements, img_w, img_h)

    # Stage 2: classify
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048,
    ).to(classifier.device)

    with torch.no_grad():
        gen_ids = classifier.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    raw_output = tokenizer.decode(gen_ids[0][input_len:], skip_special_tokens=True)

    predictions = extract_json_from_output(raw_output)
    # Filter to allowed labels only
    predictions = {k: v for k, v in predictions.items() if v in ALLOWED_LABELS}

    # Map tag -> pixel box -> label
    boxes_with_labels = []
    for tag, label in predictions.items():
        if tag in tag_to_box:
            boxes_with_labels.append((tag_to_box[tag], label))

    masked = mask_image(image, boxes_with_labels, style, show_labels=show_labels)

    # Summary text for the right-hand panel
    if not boxes_with_labels:
        summary = "✅ No PII detected."
    else:
        from collections import Counter
        counts = Counter(label for _, label in boxes_with_labels)
        summary_lines = [f"Detected {len(boxes_with_labels)} PII element(s):"]
        for label, n in counts.most_common():
            summary_lines.append(f"  • {label}: {n}")
        summary = "\n".join(summary_lines)

    return masked, summary


# ==========================================
# 8. GRADIO UI
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div style="text-align: center; max-width: 900px; margin: 0 auto;">
            <h1>🛡️ Privacy-First UI Masker</h1>
            <p>Two-stage pipeline: <b>PaddleOCR</b> finds text regions, then a fine-tuned <b>Gemma-4 + LoRA</b> classifier
            labels each region as PII or not. Detected PII is blurred or blacked out before display.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Original Screenshot")
            mask_style = gr.Radio(
                ["Gaussian Blur", "Solid Black Box"],
                value="Gaussian Blur",
                label="Masking Style",
            )
            show_labels = gr.Checkbox(
                value=False,
                label="Show predicted labels on output (debug)",
            )
            submit_btn = gr.Button("Protect Privacy", variant="primary")

        with gr.Column():
            output_img = gr.Image(type="pil", label="Masked Output", interactive=False)
            detection_summary = gr.Textbox(
                label="Detection Summary",
                lines=6,
                interactive=False,
            )

    submit_btn.click(
        fn=process_screenshot,
        inputs=[input_img, mask_style, show_labels],
        outputs=[output_img, detection_summary],
    )

    gr.Markdown(
        """
        ---
        **How it works:** Stage 1 extracts every text region with bounding boxes via OCR.
        Stage 2 feeds the spatial layout (text + normalized x,y coordinates) to a fine-tuned
        Gemma-4 LLM, which returns a JSON mapping of which regions contain PII. Detected regions
        are masked client-side. The model was trained on ~3k mobile screenshots from the
        ScreenQA dataset with PII annotations across 8 categories.
        """
    )


demo.launch(share=True, debug=True)
