"""
Begone Compute Server
InSPyReNet ONNX background removal using local model file.
"""
import os
import io
import numpy as np
import onnxruntime as ort
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Header
from fastapi.responses import Response
from PIL import Image

app = FastAPI(title="Begone Compute")

API_KEY = os.getenv("API_KEY", "")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "25000000"))

MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "inspyrenet_plus_ultra.onnx"
INPUT_SIZE = (1024, 1024)

_remover = None


def _build_session_options() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.enable_mem_pattern = False
    opts.enable_mem_reuse = True
    return opts


class ONNXRemover:
    def __init__(self, model_path, input_size=(768, 768)):
        self.input_size = input_size
        opts = _build_session_options()
        try:
            self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=['CPUExecutionProvider'])
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_path}: {e}")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"Loaded model from {model_path}")

    def process(self, img, output_type="rgba"):
        img_pil = img.convert('RGB')
        original_size = img_pil.size

        img_input = img_pil.resize(self.input_size, Image.BILINEAR)
        img_input = np.array(img_input).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_input = (img_input - mean) / std

        img_input = img_input.transpose(2, 0, 1)
        img_input = np.expand_dims(img_input, axis=0)

        pred = self.session.run([self.output_name], {self.input_name: img_input})[0]
        del img_input
        pred = pred.squeeze()

        if len(pred.shape) == 3:
            pred = pred[0]

        pred_img = Image.fromarray((pred * 255).astype(np.uint8), mode='L')
        del pred
        pred_img = pred_img.resize(original_size, Image.BILINEAR)

        if output_type == "map":
            return pred_img

        img_pil.putalpha(pred_img)
        return img_pil


def get_remover() -> ONNXRemover:
    global _remover
    if _remover is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _remover = ONNXRemover(MODEL_PATH, input_size=INPUT_SIZE)
    return _remover


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "inspyrenet_plus_ultra",
        "model_loaded": _remover is not None,
    }


@app.post("/remove-background")
async def remove_background(
    file: UploadFile = File(...),
    model: str = Query("inspyrenet_cpu"),
    x_api_key: str = Header(default=""),
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_FILE_SIZE_MB} MB)")

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    if img.width * img.height > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=413, detail=f"Image too large (max {MAX_IMAGE_PIXELS // 1_000_000} MP)")

    try:
        remover = get_remover()
        result = remover.process(img, output_type="rgba")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    output = io.BytesIO()
    result.save(output, format="PNG")
    data = output.getvalue()
    output.close()
    return Response(content=data, media_type="image/png")
