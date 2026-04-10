from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
from app.predict import predict_mask
from app.config import API_KEY

app = FastAPI(title="House Segmentation API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None)
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    content = await file.read()
    image = Image.open(io.BytesIO(content))
    mask = predict_mask(image)

    white_pixels = int((mask > 0).sum())
    return JSONResponse({
        "message": "prediction_done",
        "mask_shape": list(mask.shape),
        "house_pixels": white_pixels
    })
