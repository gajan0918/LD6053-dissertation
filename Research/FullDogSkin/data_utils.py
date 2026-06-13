import os
import random
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(image_size):
    resize_size = int(image_size * 1.15)
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def split_indices_stratified(targets, num_clients, seed):
    rng = random.Random(seed)
    indices_by_class = defaultdict(list)
    for index, target in enumerate(targets):
        indices_by_class[target].append(index)

    client_indices = [[] for _ in range(num_clients)]
    for class_indices in indices_by_class.values():
        rng.shuffle(class_indices)
        for offset, index in enumerate(class_indices):
            client_indices[offset % num_clients].append(index)

    for indices in client_indices:
        rng.shuffle(indices)

    return client_indices


def compute_class_weights(targets, num_classes, power):
    counts = Counter(targets)
    max_count = max(counts.values())
    weights = []
    for class_idx in range(num_classes):
        count = counts.get(class_idx, 1)
        weights.append((max_count / count) ** power)

    mean_weight = sum(weights) / len(weights)
    return [weight / mean_weight for weight in weights]


def create_subset_sampler(dataset, indices, class_weights):
    sample_weights = [class_weights[dataset.targets[index]] for index in indices]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(indices),
        replacement=True,
    )


def build_loader(dataset, batch_size, num_workers, sampler=None, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_data_loaders(
    dataset_dir,
    num_clients,
    batch_size,
    seed=42,
    num_workers=0,
    image_size=224,
    class_weight_power=0.5,
    use_weighted_sampler=True,
):
    train_transform = build_train_transform(image_size)
    eval_transform = build_eval_transform(image_size)

    train_dataset = datasets.ImageFolder(os.path.join(dataset_dir, "train"), train_transform)
    valid_dataset = datasets.ImageFolder(os.path.join(dataset_dir, "valid"), eval_transform)
    test_dataset = datasets.ImageFolder(os.path.join(dataset_dir, "test"), eval_transform)

    num_classes = len(train_dataset.classes)
    class_weights = compute_class_weights(train_dataset.targets, num_classes, class_weight_power)
    client_indices = split_indices_stratified(train_dataset.targets, num_clients, seed)

    client_loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        sampler = create_subset_sampler(train_dataset, indices, class_weights) if use_weighted_sampler else None
        client_loaders.append(
            build_loader(
                subset,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=sampler,
                shuffle=not use_weighted_sampler,
            )
        )

    full_train_sampler = (
        create_subset_sampler(train_dataset, list(range(len(train_dataset))), class_weights)
        if use_weighted_sampler else None
    )
    train_loader = build_loader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=full_train_sampler,
        shuffle=not use_weighted_sampler,
    )
    valid_loader = build_loader(valid_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = build_loader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    metadata = {
        "num_classes": num_classes,
        "class_names": train_dataset.classes,
        "class_counts": dict(Counter(train_dataset.targets)),
        "class_weights": class_weights,
    }

    return train_loader, client_loaders, valid_loader, test_loader, metadata
