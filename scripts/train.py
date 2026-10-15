"""
scripts/train.py
Offline training/evaluation entry point. Reuses the exact same
model/transform/loss/metric code that the API uses at inference time,
instead of duplicating it (as the notebook did inline).

Run from the project root:
    python -m scripts.train
"""
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from app.services.vision.dataset import BurnSegmentationDataset
from app.services.vision.transforms import get_train_transforms, get_transforms
from app.services.vision.model import FusionDeepLab
from app.services.vision.losses import MulticlassDiceLoss
from app.services.vision.metrics import evaluate

# --- Paths (override via env vars in real deployments) ---
IMAGE_DIR = os.environ.get("BURN_IMAGE_DIR", "/data/burn_dataset/skin/image_total")
MASK_DIR = os.environ.get("BURN_MASK_DIR", "/data/burn_dataset/skin/label_total")
CHECKPOINT_OUT = os.environ.get("MODEL_CHECKPOINT_PATH", "weights/fusion_deeplab.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 40
BATCH_SIZE = 4
LR = 1e-4


def main():
    all_images = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")])
    np.random.seed(42)
    np.random.shuffle(all_images)
    test_images = all_images[-15:]
    train_images = all_images[:-15]

    train_dataset = BurnSegmentationDataset(IMAGE_DIR, MASK_DIR, train_images, transform=get_train_transforms())
    test_dataset = BurnSegmentationDataset(IMAGE_DIR, MASK_DIR, test_images, transform=get_transforms())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = FusionDeepLab(encoder_name="resnet34", num_classes=3).to(DEVICE)
    criterion = MulticlassDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for images, masks, _ in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {total_loss / len(train_loader):.4f}")

    test_metrics = evaluate(model, test_loader, device=DEVICE)
    print("\nFinal Test Set Evaluation:")
    for i in range(3):
        print(
            f"  Class {i}: IoU={test_metrics['iou'][i]:.3f}, Dice={test_metrics['dice'][i]:.3f}, "
            f"Prec={test_metrics['precision'][i]:.3f}, Recall={test_metrics['recall'][i]:.3f}, "
            f"HD95={test_metrics['hd95'][i]:.2f}, BFScore={test_metrics['bfscore'][i]:.3f}"
        )

    os.makedirs(os.path.dirname(CHECKPOINT_OUT), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_OUT)
    print(f"\nSaved checkpoint to {CHECKPOINT_OUT}")


if __name__ == "__main__":
    main()
