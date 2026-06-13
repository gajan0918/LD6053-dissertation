import torch
import torch.nn as nn
import torch.optim as optim
import time


def create_optimizer(model, head_lr, backbone_lr, weight_decay):
    classifier_params = [param for param in model.classifier.parameters() if param.requires_grad]
    backbone_params = [
        param for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith("classifier.")
    ]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if classifier_params:
        param_groups.append({"params": classifier_params, "lr": head_lr})

    return optim.AdamW(param_groups, weight_decay=weight_decay)


def train_local(
    model,
    dataloader,
    local_epochs,
    lr,
    device,
    client_id=None,
    class_weights=None,
    backbone_lr=None,
    weight_decay=0.0,
    label_smoothing=0.0,
    grad_clip_norm=0.0,
):
    model.train()
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
    optimizer = create_optimizer(
        model,
        head_lr=lr,
        backbone_lr=backbone_lr if backbone_lr is not None else lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(local_epochs, 1),
        eta_min=min((backbone_lr if backbone_lr is not None else lr), lr) * 0.1,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epoch_stats = []
    for epoch_idx in range(local_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        sample_count = 0
        batch_count = 0
        start_time = time.time()
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(xb)
                loss = criterion(out, yb)

            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            epoch_loss += loss.item()
            epoch_correct += (out.argmax(1) == yb).sum().item()
            sample_count += yb.size(0)
            batch_count += 1
        scheduler.step()

        average_loss = epoch_loss / batch_count if batch_count else 0.0
        average_acc = epoch_correct / sample_count if sample_count else 0.0
        epoch_stats.append({"loss": average_loss, "accuracy": average_acc})
        client_label = f"Client {client_id}" if client_id is not None else "Client"
        print(
            f"{client_label} epoch {epoch_idx + 1}/{local_epochs} "
            f"loss={average_loss:.4f} "
            f"acc={average_acc*100:.2f}% "
            f"time={time.time() - start_time:.1f}s"
        )

    return epoch_stats[-1] if epoch_stats else {"loss": 0.0, "accuracy": 0.0}


def evaluate_model(model, dataloader, device, num_classes, class_names=None):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total = 0, 0
    total_loss = 0.0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += criterion(logits, yb).item()
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

            for target, prediction in zip(yb, pred):
                target_idx = int(target.item())
                class_total[target_idx] += 1
                class_correct[target_idx] += int(prediction.item() == target_idx)

    per_class_accuracy = {}
    class_names = class_names or [str(index) for index in range(num_classes)]
    for class_idx, class_name in enumerate(class_names):
        per_class_accuracy[class_name] = (
            class_correct[class_idx] / class_total[class_idx]
            if class_total[class_idx] > 0 else 0.0
        )

    macro_accuracy = (
        sum(per_class_accuracy.values()) / len(per_class_accuracy)
        if per_class_accuracy else 0.0
    )

    return {
        "loss": total_loss / len(dataloader) if len(dataloader) > 0 else 0.0,
        "accuracy": correct / total if total > 0 else 0.0,
        "macro_accuracy": macro_accuracy,
        "per_class_accuracy": per_class_accuracy,
    }


def test_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total if total > 0 else 0.0
