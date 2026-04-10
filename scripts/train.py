import os
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-3

TRAIN_IMG = Path("data/processed/train/images")
TRAIN_MASK = Path("data/processed/train/masks")
VAL_IMG = Path("data/processed/val/images")
VAL_MASK = Path("data/processed/val/masks")

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = sorted([p.name for p in img_dir.iterdir()])
        self.tf = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img = cv2.imread(str(self.img_dir / name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.mask_dir / name), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype("float32")

        aug = self.tf(image=img, mask=mask)
        img = aug["image"].astype("float32") / 255.0
        mask = aug["mask"].astype("float32")

        img = np.transpose(img, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

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

def run_epoch(loader, model, loss_fn, optimizer=None):
    training = optimizer is not None
    total_loss, total_dice, total_iou = 0, 0, 0

    if training:
        model.train()
    else:
        model.eval()

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.set_grad_enabled(training):
            out = model(x)
            loss = loss_fn(out, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(out, y).item()
        total_iou += iou_score(out, y).item()

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n

def main():
    os.makedirs("models", exist_ok=True)

    train_ds = SegDataset(TRAIN_IMG, TRAIN_MASK)
    val_ds = SegDataset(VAL_IMG, VAL_MASK)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(DEVICE)

    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss = torch.nn.BCEWithLogitsLoss()

    def loss_fn(preds, targets):
        return dice_loss(preds, targets) + bce_loss(preds, targets)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_dice = -1.0

    for epoch in range(EPOCHS):
        train_loss, train_dice, train_iou = run_epoch(train_loader, model, loss_fn, optimizer)
        val_loss, val_dice, val_iou = run_epoch(val_loader, model, loss_fn)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} train_iou={train_iou:.4f} | "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "models/best_model.pth")
            print("Saved best model.")

    torch.save(model.state_dict(), "models/last_model.pth")
    print("Saved last model.")

if __name__ == "__main__":
    main()
