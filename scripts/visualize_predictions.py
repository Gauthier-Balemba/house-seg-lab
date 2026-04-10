from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

img_dir = Path("data/processed/test/images")
mask_dir = Path("data/processed/test/masks")

files = sorted([p.name for p in img_dir.iterdir()])[:3]

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)
model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

plt.figure(figsize=(12, 12))

for i, name in enumerate(files):
    img = cv2.imread(str(img_dir / name))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0

    mask = cv2.imread(str(mask_dir / name), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

    x = np.transpose(img_resized, (2, 0, 1))
    x = torch.tensor(x).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(x)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float().squeeze().cpu().numpy() * 255

    plt.subplot(len(files), 3, i*3 + 1)
    plt.imshow(img_rgb)
    plt.title("Image")
    plt.axis("off")

    plt.subplot(len(files), 3, i*3 + 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(len(files), 3, i*3 + 3)
    plt.imshow(pred, cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

plt.tight_layout()
plt.savefig("prediction_examples.png")
print("Saved prediction_examples.png")
