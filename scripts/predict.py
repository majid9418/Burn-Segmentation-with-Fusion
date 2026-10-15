"""
scripts/predict.py
Standalone local test: load the trained checkpoint, run it on a single
image from disk, and save/show the predicted mask + an overlay.

Usage:
    python -m scripts.predict --image path/to/image.png --checkpoint path/to/best_fusion_deeplab.pt

Optional:
    --output out_dir          (default: "predictions")
    --encoder resnet34        (must match what the checkpoint was trained with)
    --num-classes 3
    --alpha 0.5
    --input-size 768
    --device cuda|cpu         (default: auto-detect)
"""
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from app.services.vision.model import FusionDeepLab
from app.services.vision.transforms import get_transforms


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(3, 1, 1)
    img = (image_tensor * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).cpu().numpy()


def load_model(checkpoint_path: str, encoder_name: str, num_classes: int, alpha: float, device: torch.device) -> FusionDeepLab:
    model = FusionDeepLab(encoder_name=encoder_name, num_classes=num_classes, alpha=alpha, encoder_weights=None)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support both a plain state_dict and the richer {"model_state_dict": ...} format
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def predict_single_image(model: FusionDeepLab, image_path: str, input_size: int, device: torch.device):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = rgb.shape[:2]

    transform = get_transforms(input_size=input_size)
    transformed = transform(image=rgb)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Resize prediction back to the original image size
    # Undo LongestMaxSize + PadIfNeeded before resizing back to the original size
    scale = input_size / max(original_h, original_w)
    resized_h, resized_w = round(original_h * scale), round(original_w * scale)
    pad_top = (input_size - resized_h) // 2
    pad_left = (input_size - resized_w) // 2
    pred_mask = pred_mask[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w]

    pred_mask = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    input_vis = denormalize(input_tensor.squeeze(0))

    return rgb, pred_mask, input_vis


def save_and_show(rgb: np.ndarray, pred_mask: np.ndarray, image_name: str, output_dir: str, num_classes: int):
    os.makedirs(output_dir, exist_ok=True)

    mask_path = os.path.join(output_dir, f"{image_name}_mask.png")
    cv2.imwrite(mask_path, pred_mask)
    print(f"Saved raw mask (class-index PNG) to: {mask_path}")

    # Class pixel stats
    total = pred_mask.size
    for cls in range(num_classes):
        count = int((pred_mask == cls).sum())
        pct = 100.0 * count / total if total else 0.0
        print(f"  Class {cls}: {count} px ({pct:.2f}%)")

    fig_path = os.path.join(output_dir, f"{image_name}_preview.png")
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(rgb)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, cmap="gray", vmin=0, vmax=max(num_classes - 1, 1))
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"Saved side-by-side preview to: {fig_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run local inference on a single image.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained .pt checkpoint")
    parser.add_argument("--output", default="predictions", help="Output directory")
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--input-size", type=int, default=768)
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, args.encoder, args.num_classes, args.alpha, device)
    rgb, pred_mask, _ = predict_single_image(model, args.image, args.input_size, device)

    image_name = os.path.splitext(os.path.basename(args.image))[0]
    save_and_show(rgb, pred_mask, image_name, args.output, args.num_classes)


if __name__ == "__main__":
    main()
