import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from app.config import MODEL_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

_model = None

def get_model():
    global _model
    if _model is None:
        _model = build_model()
    return _model

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def predict_mask(image: Image.Image):
    model = get_model()
    x = preprocess_image(image)
    with torch.no_grad():
        y = model(x)
        y = torch.sigmoid(y)
        y = (y > 0.5).float()
    mask = y.squeeze().cpu().numpy().astype(np.uint8) * 255
    return mask
