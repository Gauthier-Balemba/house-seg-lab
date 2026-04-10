from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
TEST_IMG = Path("data/processed/test/images")
TEST_MASK = Path("data/processed/test/masks")

class TestDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = sorted([p.name for p in img_dir.iterdir()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img = cv2.imread(str(self.img_dir / name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
        img = np.transpose(img, (2,0,1))

        mask = cv2.imread(str(self.mask_dir / name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = (mask > 127).astype("float32")
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(img), torch.tensor(mask)

def dice_score(preds, targets, eps=1e-7):
    preds = (torch.sigmoid(preds) > 0.5).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    return ((2.0 * intersection + eps) / (union + eps)).mean()

def iou_score(preds, targets, eps=1e-7):
    preds = (torch.sigmoid(preds) > 0.5).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) - intersection
    return ((intersection + eps) / (union + eps)).mean()

def main():
    ds = TestDataset(TEST_IMG, TEST_MASK)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )

    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    dices = []
    ious = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            dices.append(dice_score(out, y).item())
            ious.append(iou_score(out, y).item())

    print("Test Dice:", sum(dices)/len(dices))
    print("Test IoU :", sum(ious)/len(ious))

if __name__ == "__main__":
    main()
