"""
Begone Compute Server
InSPyReNet ONNX background removal with multi-model support and LRU memory management.
"""
import os
import io
import threading
from collections import OrderedDict
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
MEMORY_LIMIT_MB = int(os.getenv("MEMORY_LIMIT_MB", "1200"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "base_512")

MODEL_CONFIGS = {
    "fast": {
        "file": "inspyrenet_fast.onnx",
        "input_size": (384, 384),
        "memory_mb": 355,
    },
    "base_384": {
        "file": "inspyrenet_base_384.onnx",
        "input_size": (384, 384),
        "memory_mb": 355,
    },
    "base_512": {
        "file": "inspyrenet_base_512.onnx",
        "input_size": (512, 512),
        "memory_mb": 355,
    },
    "base_640": {
        "file": "inspyrenet_base_640.onnx",
        "input_size": (640, 640),
        "memory_mb": 355,
    },
    "base_768": {
        "file": "inspyrenet_base_768.onnx",
        "input_size": (768, 768),
        "memory_mb": 355,
    },
    "plus_ultra": {
        "file": "inspyrenet_plus_ultra.onnx",
        "input_size": (1024, 1024),
        "memory_mb": 377,
    },
}

_loaded_models: OrderedDict = OrderedDict()  # name → ONNXRemover (MRU at end)
_model_lock = threading.Lock()


def _build_session_options() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.enable_mem_pattern = False
    opts.enable_mem_reuse = True
    return opts


def _current_memory_mb() -> int:
    """Calculate total memory used by loaded models (estimated)."""
    return sum(
        MODEL_CONFIGS[name]["memory_mb"] for name in _loaded_models.keys()
    )


class ONNXRemover:
    def __init__(self, model_path, input_size=(768, 768)):
        self.input_size = input_size
        opts = _build_session_options()
        try:
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_path}: {e}")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"[load] loaded model from {model_path}")

    def process(self, img, output_type="rgba"):
        img_pil = img.convert("RGB")
        original_size = img_pil.size

        img_input = img_pil.resize(self.input_size, Image.BILINEAR)
        img_input = np.array(img_input).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_input = (img_input - mean) / std

        img_input = img_input.transpose(2, 0, 1)
        img_input = np.expand_dims(img_input, axis=0)

        pred = self.session.run([self.output_name], {self.input_name: img_input})[
            0
        ]
        del img_input
        pred = pred.squeeze()

        if len(pred.shape) == 3:
            pred = pred[0]

        pred_img = Image.fromarray((pred * 255).astype(np.uint8), mode="L")
        del pred
        pred_img = pred_img.resize(original_size, Image.BILINEAR)

        if output_type == "map":
            return pred_img

        img_pil.putalpha(pred_img)
        return img_pil


def get_model(name: str) -> ONNXRemover:
    """Load model by name, with LRU eviction when memory limit is exceeded."""
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{name}'")

    # Fast path — check if already loaded (without lock)
    if name in _loaded_models:
        with _model_lock:
            if name in _loaded_models:  # double-check inside lock
                _loaded_models.move_to_end(name)
                return _loaded_models[name]

    # Slow path — load with eviction logic
    with _model_lock:
        if name in _loaded_models:  # another thread loaded it
            _loaded_models.move_to_end(name)
            return _loaded_models[name]

        required_mb = MODEL_CONFIGS[name]["memory_mb"]

        # Evict LRU models until budget allows new model
        while _loaded_models and (
            _current_memory_mb() + required_mb > MEMORY_LIMIT_MB
        ):
            evicted_name, evicted = _loaded_models.popitem(last=False)
            evicted.session = None  # release ONNX C++ session
            print(f"[lru] evicted '{evicted_name}'")

        # Load the model
        model_path = MODELS_DIR / MODEL_CONFIGS[name]["file"]
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        remover = ONNXRemover(model_path, input_size=MODEL_CONFIGS[name]["input_size"])
        _loaded_models[name] = remover
        return remover


@app.get("/health")
def health():
    """Return health status with detailed model information."""
    loaded_names = list(_loaded_models.keys())

    models_info = {}
    for api_name, config in MODEL_CONFIGS.items():
        model_path = MODELS_DIR / config["file"]
        models_info[api_name] = {
            "loaded": api_name in loaded_names,
            "memory_mb": config["memory_mb"],
            "input_size": f"{config['input_size'][0]}x{config['input_size'][1]}",
            "file_exists": model_path.exists(),
        }

    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL,
        "memory_used_mb": _current_memory_mb(),
        "memory_limit_mb": MEMORY_LIMIT_MB,
        "loaded_models": loaded_names,
        "models": models_info,
    }


@app.post("/remove-background")
async def remove_background(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_MODEL),
    x_api_key: str = Header(default=""),
):
    """Remove background from image using specified model."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Validate model name early
    if model not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {list(MODEL_CONFIGS.keys())}",
        )

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"File too large (max {MAX_FILE_SIZE_MB} MB)"
        )

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    if img.width * img.height > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large (max {MAX_IMAGE_PIXELS // 1_000_000} MP)",
        )

    try:
        remover = get_model(model)
        result = remover.process(img, output_type="rgba")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    output = io.BytesIO()
    result.save(output, format="PNG")
    data = output.getvalue()
    output.close()
    return Response(content=data, media_type="image/png")
