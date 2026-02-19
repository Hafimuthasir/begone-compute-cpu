"""
Begone Compute Server
Runs InSPyReNet via transparent-background to remove image backgrounds.
"""
import os
import io
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Header
from fastapi.responses import Response
from PIL import Image
from huggingface_hub import hf_hub_download

app = FastAPI(title="Begone Compute")

API_KEY = os.getenv("API_KEY", "")
DEFAULT_MODE = os.getenv("DEFAULT_MODE", "fast")  # Mode parameter (all modes use same ONNX model)

# Lazy-loaded model
_remover_plus_ultra = None

class ONNXRemover:
    def __init__(self, provider=['CPUExecutionProvider']):
        self.session = None
        self.provider = provider
        self.load_model()

    def load_model(self):
        repo_id = "OS-Software/InSPyReNet-SwinB-Plus-Ultra-ONNX"
        filename = "onnx/model.onnx"
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        
        try:
            # Try specified providers (e.g., CoreML)
            self.session = ort.InferenceSession(model_path, providers=self.provider)
        except Exception as e:
            print(f"Warning: Could not load with {self.provider}. Fallback to CPU. Error: {e}")
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def process(self, img, type="rgba"):
        # Preprocess
        # Using 768x768 to reduce memory usage and prevent OOM on servers with limited RAM
        # This still provides high quality results while being more memory efficient
        input_size = (768, 768) 
        
        img_pil = img.convert('RGB')
        original_size = img_pil.size
        
        img_input = img_pil.resize(input_size, Image.BILINEAR)
        img_input = np.array(img_input).astype(np.float32) / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_input = (img_input - mean) / std
        
        img_input = img_input.transpose(2, 0, 1)
        img_input = np.expand_dims(img_input, axis=0)
        
        # Inference
        pred = self.session.run([self.output_name], {self.input_name: img_input})[0]
        pred = pred.squeeze()
        
        # Post-process
        # Resize back to original
        # We need cv2 for resize, or use PIL
        pred_img = Image.fromarray((pred * 255).astype(np.uint8), mode='L')
        pred_img = pred_img.resize(original_size, Image.BILINEAR)
        
        if type == "map":
             return pred_img
        
        # RGBA
        img_pil.putalpha(pred_img)
        return img_pil


def get_remover(mode: str):
    global _remover_fast, _remover_base, _remover_plus_ultra

    # All modes now use ONNX InSPyReNet model (no transparent_background dependency)
    if _remover_plus_ultra is None:
        providers = ['CPUExecutionProvider']
        _remover_plus_ultra = ONNXRemover(provider=providers)
    return _remover_plus_ultra


@app.get("/health")
def health():
    return {"status": "ok", "model": "inspyrenet"}


@app.post("/remove-background")
async def remove_background(
    file: UploadFile = File(...),
    model: str = Query("inspyrenet_cpu"),
    x_api_key: str = Header(default="")
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Determine mode
    mode = DEFAULT_MODE
    if "fast" in model.lower():
        mode = "fast"
    elif "base" in model.lower():
        mode = "base"
    elif "plus_ultra" in model.lower() or "plus" in model.lower():
        mode = "plus_ultra"

    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    remover = get_remover(mode)
    result = remover.process(img, type="rgba")  # PIL RGBA image

    output = io.BytesIO()
    result.save(output, format="PNG")
    return Response(content=output.getvalue(), media_type="image/png")
