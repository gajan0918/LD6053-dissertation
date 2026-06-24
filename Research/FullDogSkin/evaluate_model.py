#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import datasets
from PIL import Image

from data_utils import build_eval_transform
from inference_utils import TTA_MODES, build_prediction_tensor_batch, normalize_tta_mode
from model import build_model


BASE_DIR = Path(__file__).resolve().parent


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint_state(model_instance, model_path, device):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}

    model_instance.load_state_dict(checkpoint)


def build_loader(dataset_root, split, image_size, batch_size):
    split_dir = dataset_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")

    dataset = datasets.ImageFolder(split_dir, transform=build_eval_transform(image_size))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def collect_logits(model_instance, loader, device):
    all_logits = []
    all_labels = []

    model_instance.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model_instance(images).detach().cpu()
            all_logits.append(logits)
            all_labels.append(labels.detach().cpu())

    return torch.cat(all_logits), torch.cat(all_labels)


def collect_logits_tta(model_instance, dataset, image_size, batch_size, device, tta_mode):
    all_logits = []
    all_labels = []
    batch_views = []
    batch_labels = []
    view_count = None

    model_instance.eval()
    with torch.no_grad():
        for image_path, label in dataset.samples:
            with Image.open(image_path) as image:
                views = build_prediction_tensor_batch(image, image_size, tta_mode)

            if view_count is None:
                view_count = views.size(0)
            batch_views.append(views)
            batch_labels.append(label)

            if len(batch_views) >= batch_size:
                logits, labels = predict_tta_batch(model_instance, batch_views, batch_labels, view_count, device)
                all_logits.append(logits)
                all_labels.append(labels)
                batch_views = []
                batch_labels = []

        if batch_views:
            logits, labels = predict_tta_batch(model_instance, batch_views, batch_labels, view_count, device)
            all_logits.append(logits)
            all_labels.append(labels)

    return torch.cat(all_logits), torch.cat(all_labels)


def predict_tta_batch(model_instance, batch_views, batch_labels, view_count, device):
    stacked = torch.cat(batch_views, dim=0).to(device)
    logits = model_instance(stacked)
    logits = logits.reshape(len(batch_views), view_count, -1).mean(dim=1).detach().cpu()
    labels = torch.tensor(batch_labels, dtype=torch.long)
    return logits, labels


def fit_temperature(logits, labels, device):
    logits = logits.to(device)
    labels = labels.to(device)
    log_temperature = nn.Parameter(torch.zeros(1, device=device))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=100)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(min=1e-6)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().clamp(min=1e-6).item())


def expected_calibration_error(confidences, predictions, labels, n_bins=10):
    confidences = confidences.numpy()
    correctness = predictions.eq(labels).numpy().astype(np.float64)
    total = len(confidences)
    ece = 0.0

    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        if bin_index == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        if not mask.any():
            continue

        bin_accuracy = correctness[mask].mean()
        bin_confidence = confidences[mask].mean()
        ece += (mask.sum() / total) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def build_report(
    logits,
    labels,
    class_names,
    temperature,
    split,
    tta_mode,
    guard_logits=None,
    guard_temperature=1.0,
    none_guard_mode="off",
    none_rejection_threshold=0.5,
):
    probabilities = torch.softmax(logits / temperature, dim=1)
    none_idx = class_names.index("None") if "None" in class_names else None
    none_guard_overrides = 0

    if guard_logits is not None and none_idx is not None:
        guard_probs = torch.softmax(guard_logits / guard_temperature, dim=1)
        guard_confidences, guard_predictions = guard_probs.max(dim=1)
        guard_none_probs = guard_probs[:, none_idx]
        none_mask = (guard_predictions == none_idx) | (guard_none_probs >= none_rejection_threshold)
        none_guard_overrides = int(none_mask.sum().item())
        probabilities = probabilities.clone()
        probabilities[none_mask] = 0.0
        probabilities[none_mask, none_idx] = torch.maximum(
            guard_confidences[none_mask],
            guard_none_probs[none_mask],
        )

    confidences, predictions = probabilities.max(dim=1)
    num_classes = len(class_names)

    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, predicted_label in zip(labels.tolist(), predictions.tolist()):
        confusion[true_label, predicted_label] += 1

    per_class = []
    supports = Counter(labels.tolist())
    for index, name in enumerate(class_names):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        support = int(supports.get(index, 0))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({
            "class": name,
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

    accuracy = float(predictions.eq(labels).float().mean().item())
    top3 = float(
        probabilities.topk(min(3, num_classes), dim=1).indices.eq(labels.unsqueeze(1)).any(dim=1).float().mean().item()
    )

    macro_precision = float(np.mean([item["precision"] for item in per_class]))
    macro_recall = float(np.mean([item["recall"] for item in per_class]))
    macro_f1 = float(np.mean([item["f1"] for item in per_class]))

    return {
        "split": split,
        "num_samples": int(labels.numel()),
        "num_classes": num_classes,
        "temperature": round(float(temperature), 6),
        "tta_mode": tta_mode,
        "none_guard_mode": none_guard_mode,
        "none_guard_temperature": round(float(guard_temperature), 6),
        "none_guard_overrides": none_guard_overrides,
        "accuracy": round(accuracy, 4),
        "top3_accuracy": round(top3, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "expected_calibration_error": round(
            expected_calibration_error(confidences, predictions, labels),
            4,
        ),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": class_names,
            "matrix": confusion.tolist(),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the dog skin classifier.")
    parser.add_argument("--dataset-dir", default="Dataset")
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH", "best_efficientnet_b0_model.pth"))
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=int(os.getenv("IMAGE_SIZE", "224")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("SOFTMAX_TEMPERATURE", "1.0")))
    parser.add_argument("--fit-temperature", action="store_true")
    parser.add_argument("--tta-mode", default=os.getenv("PREDICTION_TTA_MODE", "none"), choices=sorted(TTA_MODES))
    parser.add_argument("--none-guard-mode", default=os.getenv("NONE_GUARD_TTA_MODE", "off"), choices=["off", *sorted(TTA_MODES)])
    parser.add_argument("--none-guard-temperature", type=float, default=float(os.getenv("NONE_GUARD_TEMPERATURE", "0.775777")))
    parser.add_argument("--none-rejection-threshold", type=float, default=float(os.getenv("NONE_REJECTION_THRESHOLD", "0.5")))
    parser.add_argument("--output", default="evaluation_report.json")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = resolve_path(args.dataset_dir)
    model_path = resolve_path(args.model_path)
    output_path = resolve_path(args.output)
    device = get_best_device()
    tta_mode = normalize_tta_mode(args.tta_mode)
    none_guard_mode = "off" if args.none_guard_mode == "off" else normalize_tta_mode(args.none_guard_mode)

    train_dataset, _ = build_loader(dataset_root, "train", args.image_size, args.batch_size)
    model_instance = build_model(len(train_dataset.classes), pretrained=False, fine_tune_blocks=0).to(device)
    load_checkpoint_state(model_instance, model_path, device)

    temperature = max(args.temperature, 1e-6)
    if args.fit_temperature:
        valid_dataset, valid_loader = build_loader(dataset_root, "valid", args.image_size, args.batch_size)
        if valid_dataset.classes != train_dataset.classes:
            raise ValueError("Validation classes do not match training classes.")
        if tta_mode == "none":
            valid_logits, valid_labels = collect_logits(model_instance, valid_loader, device)
        else:
            valid_logits, valid_labels = collect_logits_tta(
                model_instance,
                valid_dataset,
                args.image_size,
                args.batch_size,
                device,
                tta_mode,
            )
        temperature = fit_temperature(valid_logits, valid_labels, device)

    eval_dataset, eval_loader = build_loader(dataset_root, args.split, args.image_size, args.batch_size)
    if eval_dataset.classes != train_dataset.classes:
        raise ValueError(f"{args.split} classes do not match training classes.")

    if tta_mode == "none":
        logits, labels = collect_logits(model_instance, eval_loader, device)
    else:
        logits, labels = collect_logits_tta(
            model_instance,
            eval_dataset,
            args.image_size,
            args.batch_size,
            device,
            tta_mode,
        )
    guard_logits = None
    if none_guard_mode != "off":
        guard_logits, guard_labels = collect_logits_tta(
            model_instance,
            eval_dataset,
            args.image_size,
            args.batch_size,
            device,
            none_guard_mode,
        )
        if not torch.equal(guard_labels, labels):
            raise ValueError("None-guard labels do not match primary evaluation labels.")

    report = build_report(
        logits,
        labels,
        train_dataset.classes,
        temperature,
        args.split,
        tta_mode,
        guard_logits=guard_logits,
        guard_temperature=max(args.none_guard_temperature, 1e-6),
        none_guard_mode=none_guard_mode,
        none_rejection_threshold=args.none_rejection_threshold,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Evaluation report saved to: {output_path}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"TTA mode: {report['tta_mode']}")
    print(f"None guard mode: {report['none_guard_mode']}")
    print(f"Macro F1: {report['macro_f1']:.4f}")
    print(f"Expected calibration error: {report['expected_calibration_error']:.4f}")
    print(
        "Use these API settings for calibrated confidence: "
        f"PREDICTION_TTA_MODE={tta_mode} "
        f"SOFTMAX_TEMPERATURE={temperature:.6f}"
    )


if __name__ == "__main__":
    main()
