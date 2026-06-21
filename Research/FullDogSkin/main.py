import os
import random
import time

import numpy as np
import torch

from config import *
from data_utils import get_data_loaders
from federated_utils import average_params, get_params, set_params
from model import build_model
from train_utils import evaluate_model, train_local


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_class_distribution(class_names, class_counts):
    print("Training class distribution:")
    for class_idx, class_name in enumerate(class_names):
        print(f"  - {class_name}: {class_counts.get(class_idx, 0)}")


def log_metrics(label, metrics):
    print(
        f"{label} loss={metrics['loss']:.4f} "
        f"acc={metrics['accuracy']*100:.2f}% "
        f"macro_acc={metrics['macro_accuracy']*100:.2f}%"
    )
    worst_classes = sorted(
        metrics["per_class_accuracy"].items(),
        key=lambda item: item[1]
    )[:3]
    if worst_classes:
        summary = ", ".join(
            f"{name}={score*100:.1f}%"
            for name, score in worst_classes
        )
        print(f"Worst class accuracy: {summary}")


seed_everything(SEED)

train_loader, client_loaders, valid_loader, test_loader, metadata = get_data_loaders(
    dataset_dir=DATASET_DIR,
    num_clients=NUM_CLIENTS,
    batch_size=BATCH_SIZE,
    seed=SEED,
    num_workers=NUM_WORKERS,
    image_size=IMAGE_SIZE,
    class_weight_power=CLASS_WEIGHT_POWER,
    use_weighted_sampler=USE_WEIGHTED_SAMPLER,
)

num_classes = metadata["num_classes"]
class_names = metadata["class_names"]
class_counts = metadata["class_counts"]
class_weights = metadata["class_weights"]

print("Classes:", class_names)
print(f"Device: {DEVICE}")
print(f"Image size: {IMAGE_SIZE}")
print(f"DataLoader workers: {NUM_WORKERS}")
print(f"Federated rounds: {ROUNDS}")
print(f"Local epochs per client: {LOCAL_EPOCHS}")
print(f"Central fine-tune epochs: {CENTRAL_FINE_TUNE_EPOCHS}")
print(f"Head LR: {LR_HEAD}")
print(f"Backbone LR: {LR_BACKBONE}")
print(f"Weighted sampler: {USE_WEIGHTED_SAMPLER}")
print(f"Fine-tuned EfficientNet-B0 blocks: {FINE_TUNE_BLOCKS}")
print_class_distribution(class_names, class_counts)

global_model = build_model(
    num_classes,
    pretrained=True,
    fine_tune_blocks=FINE_TUNE_BLOCKS,
).to(DEVICE)

best_score = float("-inf")
best_model_path = BEST_MODEL_PATH
rounds_without_improvement = 0

print("\nFederated Training Started\n")

for round_idx in range(1, ROUNDS + 1):
    print(f"\n--- Round {round_idx}/{ROUNDS} ---")
    round_start = time.time()
    client_params = []
    client_sizes = []
    client_losses = []
    client_accuracies = []

    for cid, client_loader in enumerate(client_loaders, start=1):
        print(f"Training client {cid}/{NUM_CLIENTS}...")
        client_model = build_model(
            num_classes,
            pretrained=False,
            fine_tune_blocks=FINE_TUNE_BLOCKS,
        ).to(DEVICE)
        set_params(client_model, get_params(global_model), DEVICE)

        train_stats = train_local(
            client_model,
            client_loader,
            LOCAL_EPOCHS,
            LR_HEAD,
            DEVICE,
            client_id=cid,
            class_weights=class_weights,
            backbone_lr=LR_BACKBONE,
            weight_decay=WEIGHT_DECAY,
            label_smoothing=LABEL_SMOOTHING,
            grad_clip_norm=GRAD_CLIP_NORM,
        )

        client_params.append(get_params(client_model))
        client_sizes.append(len(client_loader.dataset))
        client_losses.append(train_stats["loss"])
        client_accuracies.append(train_stats["accuracy"])

    total_samples = sum(client_sizes)
    weights = [client_size / total_samples for client_size in client_sizes]
    new_params = average_params(client_params, weights)
    set_params(global_model, new_params, DEVICE)

    avg_client_loss = sum(client_losses) / len(client_losses)
    avg_client_acc = sum(client_accuracies) / len(client_accuracies)
    print(
        f"Round {round_idx} client summary: "
        f"loss={avg_client_loss:.4f} "
        f"acc={avg_client_acc*100:.2f}%"
    )

    val_metrics = evaluate_model(global_model, valid_loader, DEVICE, num_classes, class_names)
    log_metrics("Validation", val_metrics)
    print(f"Round {round_idx} completed in {time.time() - round_start:.1f}s")

    if val_metrics["macro_accuracy"] > best_score:
        best_score = val_metrics["macro_accuracy"]
        rounds_without_improvement = 0
        torch.save(global_model.state_dict(), best_model_path)
        print(
            f"New best model saved to {best_model_path} "
            f"with macro accuracy {best_score*100:.2f}%"
        )
    else:
        rounds_without_improvement += 1
        print(
            f"No validation improvement for {rounds_without_improvement} "
            f"round(s)"
        )
        if rounds_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

if os.path.exists(best_model_path):
    global_model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

if CENTRAL_FINE_TUNE_EPOCHS > 0:
    print("\nStarting centralized fine-tuning on the full training set...\n")
    train_local(
        global_model,
        train_loader,
        CENTRAL_FINE_TUNE_EPOCHS,
        LR_HEAD * 0.5,
        DEVICE,
        client_id="Central",
        class_weights=class_weights,
        backbone_lr=LR_BACKBONE * 0.5,
        weight_decay=WEIGHT_DECAY,
        label_smoothing=LABEL_SMOOTHING,
        grad_clip_norm=GRAD_CLIP_NORM,
    )
    val_metrics = evaluate_model(global_model, valid_loader, DEVICE, num_classes, class_names)
    log_metrics("Post fine-tune validation", val_metrics)

    if val_metrics["macro_accuracy"] > best_score:
        best_score = val_metrics["macro_accuracy"]
        torch.save(global_model.state_dict(), best_model_path)
        print(
            f"Fine-tuned model improved validation macro accuracy to "
            f"{best_score*100:.2f}% and was saved."
        )

if os.path.exists(best_model_path):
    global_model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

test_metrics = evaluate_model(global_model, test_loader, DEVICE, num_classes, class_names)
print("\nFinal evaluation on the held-out test set:")
log_metrics("Test", test_metrics)
print(f"Best model saved to: {best_model_path}")
