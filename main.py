"""
Begone Compute Server
Runs InSPyReNet via transparent-background to remove image backgrounds.
"""
import os
import io
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Header
from fastapi.responses import Response
from PIL import Image

app = FastAPI(title="Begone Compute")

API_KEY = os.getenv("API_KEY", "")
DEFAULT_MODE = os.getenv("DEFAULT_MODE", "fast")  # "fast" or "base"

# Lazy-loaded models
_remover_fast = None
_remover_base = None


def get_remover(fast: bool):
    global _remover_fast, _remover_base
    from transparent_background import Remover
    if fast:
        if _remover_fast is None:
            _remover_fast = Remover(mode="fast", device="cpu")
        return _remover_fast
    else:
        if _remover_base is None:
            _remover_base = Remover(mode="base", device="cpu")
        return _remover_base


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

    # Determine mode from model name or default
    if "fast" in model.lower():
        fast_mode = True
    elif "base" in model.lower():
        fast_mode = False
    else:
        fast_mode = (DEFAULT_MODE == "fast")

    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    remover = get_remover(fast_mode)
    result = remover.process(img, type="rgba")  # PIL RGBA image

    output = io.BytesIO()
    result.save(output, format="PNG")
    return Response(content=output.getvalue(), media_type="image/png")
