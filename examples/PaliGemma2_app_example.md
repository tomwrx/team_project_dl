# 🚀 Privacy-First UI Masker: Handoff Plan for Member 3

## Step 1: The "Handoff Package" (What you need to give Member 3)
Member 3 does not need to understand 4-bit quantization or LoRA, but they **must** have these three things from you:
1.  **The LoRA Adapter:** The extracted `lora_adapter` folder from your Google Drive (contains `adapter_config.json` and `adapter_model.safetensors`).
2.  **The Parser Function:** Your exact `loc_tokens_to_pixels` and regular expression (`LOC_RE`) functions to translate the model's text output back into math.
3.  **The Prompt Rule:** Remind them of our hard-learned lesson: The prompt must be exactly `"<image>detect sensitive_info"` (or your multi-class string) with **NO trailing `\n`**. 

## Step 2: The UI/UX Design (The "Demo Day" Flow)
For a presentation, the app needs to be foolproof. Member 3 should build a UI with:
* **Header:** "Privacy-First UI Masker" with a short description.
* **Split Screen:** Input Image (Left) ➔ Masked Image (Right).
* **Style Toggle:** A radio button letting the user choose between "Solid Redaction" (black boxes) or "Gaussian Blur" (looks highly professional).
* **Pre-loaded Examples:** This is critical. Live demos always fail when you try to upload a file under pressure. M3 must pre-load 3-4 distinct UI screenshots at the bottom of the app that the judges can just click to run.

Shared project link: https://drive.google.com/drive/folders/1ttyaB7asqXWYA19vKnIcCFJYDv58cXEr?usp=sharing
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
import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
import re
from transformers import (
    PaliGemmaProcessor, 
    PaliGemmaForConditionalGeneration, 
    BitsAndBytesConfig
)
from peft import PeftModel

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
MODEL_ID = 'google/paligemma2-3b-pt-448'
# ⚠️ M3: UPDATE THIS PATH TO OUR SHARED DRIVE LOCATION!
ADAPTER_DIR = '/content/drive/MyDrive/VU_DL_Team_Project/outputs/lora_adapter'

# ==========================================
# 2. INITIALIZATION (Clean PEFT Loading)
# ==========================================
print("⏳ Loading Processor...")
processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)

print("⏳ Loading Frozen Base Model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base_model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
)

print("⏳ Applying LoRA Adapter from Drive...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()
if torch.cuda.is_available():
    model.to('cuda')
print("✅ Model Ready!")

# ==========================================
# 3. PARSER & CV LOGIC
# ==========================================
LOC_RE = re.compile(r'<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*([A-Za-z_][\w_]*)?')

def parse_boxes(text, img_w, img_h):
    boxes = []
    for m in LOC_RE.finditer(text):
        ny1, nx1, ny2, nx2, label = m.groups()
        y1, x1 = int(ny1)/1024*img_h, int(nx1)/1024*img_w
        y2, x2 = int(ny2)/1024*img_h, int(nx2)/1024*img_w
        if x2 > x1 and y2 > y1:
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes

def mask_image(image, boxes, style):
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for (x1, y1, x2, y2) in boxes:
        if style == "Solid Black Box":
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 0, 0), -1)
        elif style == "Gaussian Blur":
            roi = cv_img[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
            cv_img[y1:y2, x1:x2] = blurred_roi
            
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

# ==========================================
# 4. INFERENCE PIPELINE
# ==========================================
def process_screenshot(image, style):
    if image is None:
        return None
    
    img_w, img_h = image.size
    prompt = "<image>detect sensitive_info" 
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=192)
    
    raw_text = processor.decode(outputs[0], skip_special_tokens=True)
    boxes = parse_boxes(raw_text, img_w, img_h)
    
    masked_img = mask_image(image, boxes, style)
    return masked_img

# ==========================================
# 5. GRADIO FRONTEND
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div style="text-align: center; max-width: 800px; margin: 0 auto;">
            <h1>🛡️ Privacy-First UI Masker</h1>
            <p>Upload a screenshot. Our fine-tuned PaliGemma 2 model will semantically detect and mask Sensitive PII (Emails, Balances, Usernames) on the fly.</p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Original Screenshot")
            mask_style = gr.Radio(
                ["Gaussian Blur", "Solid Black Box"], 
                value="Gaussian Blur", 
                label="Masking Style"
            )
            submit_btn = gr.Button("Protect Privacy", variant="primary")
        
        with gr.Column():
            output_img = gr.Image(type="pil", label="Masked Output", interactive=False)
            
    submit_btn.click(
        fn=process_screenshot, 
        inputs=[input_img, mask_style], 
        outputs=output_img
    )

# Launch with share=True to generate the public Demo Day link!
demo.launch(share=True, debug=True)