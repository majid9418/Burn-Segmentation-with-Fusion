"""
services/vision/metrics.py
Segmentation evaluation metrics (IoU, Dice, precision/recall, HD95,
boundary F-score) used for offline evaluation in scripts/train.py.
"""
import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def calculate_metrics(pred: torch.Tensor, true: torch.Tensor, num_classes: int = 3, edge_tolerance: int = 2):
    pred = pred.cpu().numpy()
    true = true.cpu().numpy()

    metrics = {
        "iou": [],
        "dice": [],
        "precision": [],
        "recall": [],
        "hd95": [],
        "bfscore": [],
    }

    for cls in range(num_classes):
        pred_c = (pred == cls).astype(np.uint8)
        true_c = (true == cls).astype(np.uint8)

        # ---- Core region metrics ----
        intersection = np.logical_and(pred_c, true_c).sum()
        union = np.logical_or(pred_c, true_c).sum()
        tp = intersection
        fp = pred_c.sum() - tp
        fn = true_c.sum() - tp

        iou = tp / union if union != 0 else 0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0

        # ---- Hausdorff Distance (HD95) ----
        pred_pts = np.argwhere(pred_c > 0)
        true_pts = np.argwhere(true_c > 0)

        if len(pred_pts) > 0 and len(true_pts) > 0:
            dt_pred = distance_transform_edt(1 - pred_c)
            dt_true = distance_transform_edt(1 - true_c)
            dist_pred_to_true = dt_true[pred_c == 1]
            dist_true_to_pred = dt_pred[true_c == 1]
            hd95 = np.percentile(np.hstack([dist_pred_to_true, dist_true_to_pred]), 95)
        else:
            hd95 = np.nan

        # ---- Boundary F1-score ----
        kernel = np.ones((3, 3), np.uint8)
        pred_edge = cv2.morphologyEx(pred_c, cv2.MORPH_GRADIENT, kernel)
        true_edge = cv2.morphologyEx(true_c, cv2.MORPH_GRADIENT, kernel)

        true_dil = cv2.dilate(true_edge, np.ones((edge_tolerance * 2 + 1, edge_tolerance * 2 + 1), np.uint8))
        pred_dil = cv2.dilate(pred_edge, np.ones((edge_tolerance * 2 + 1, edge_tolerance * 2 + 1), np.uint8))

        tp_b = np.logical_and(pred_edge, true_dil).sum()
        fp_b = np.logical_and(pred_edge, np.logical_not(true_dil)).sum()
        fn_b = np.logical_and(true_edge, np.logical_not(pred_dil)).sum()

        prec_b = tp_b / (tp_b + fp_b + 1e-6)
        rec_b = tp_b / (tp_b + fn_b + 1e-6)
        bfscore = (2 * prec_b * rec_b) / (prec_b + rec_b + 1e-6)

        metrics["iou"].append(iou)
        metrics["dice"].append(dice)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["hd95"].append(hd95)
        metrics["bfscore"].append(bfscore)

    return metrics


def evaluate(model, loader, device: str = "cuda"):
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            for p, t in zip(preds, masks):
                all_metrics.append(calculate_metrics(p, t))

    avg = {}
    for key in all_metrics[0]:
        avg[key] = np.mean([m[key] for m in all_metrics], axis=0)
    return avg
