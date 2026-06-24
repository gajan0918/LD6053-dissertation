#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def image_files(directory):
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def source_files(directory, prefix):
    originals = [
        path
        for path in image_files(directory)
        if not path.name.startswith(prefix)
    ]
    return originals or image_files(directory)


def average_edge_color(image):
    small = image.resize((1, 1), Image.Resampling.BILINEAR)
    return tuple(small.getpixel((0, 0)))


def random_resized_crop(image, rng):
    width, height = image.size
    scale = rng.uniform(0.82, 1.0)
    crop_width = max(1, int(width * scale))
    crop_height = max(1, int(height * scale))
    left = rng.randint(0, max(width - crop_width, 0))
    top = rng.randint(0, max(height - crop_height, 0))
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), Image.Resampling.BILINEAR)


def augment_image(image, rng):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = random_resized_crop(image, rng)

    if rng.random() < 0.5:
        image = ImageOps.mirror(image)

    angle = rng.uniform(-18, 18)
    image = image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        fillcolor=average_edge_color(image),
    )

    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.82, 1.18))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.2))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.85, 1.15))
    image = ImageEnhance.Sharpness(image).enhance(rng.uniform(0.85, 1.25))

    if rng.random() < 0.18:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.7)))

    return image


def next_output_path(class_dir, prefix, index):
    return class_dir / f"{prefix}_{index:05d}.jpg"


def first_available_index(class_dir, prefix):
    index = 1
    while next_output_path(class_dir, prefix, index).exists():
        index += 1
    return index


def balance_class(class_dir, target_count, prefix, rng):
    existing = image_files(class_dir)
    if len(existing) >= target_count:
        return 0, len(existing)

    sources = source_files(class_dir, prefix)
    if not sources:
        raise FileNotFoundError(f"No source images found in {class_dir}")

    missing = target_count - len(existing)
    output_index = first_available_index(class_dir, prefix)

    for _ in range(missing):
        source = rng.choice(sources)
        with Image.open(source) as image:
            augmented = augment_image(image, rng)

        output_path = next_output_path(class_dir, prefix, output_index)
        augmented.save(output_path, format="JPEG", quality=92, optimize=True)
        output_index += 1

    return missing, target_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Balance each training class by creating augmented images."
    )
    parser.add_argument("--dataset-dir", default="Dataset/train")
    parser.add_argument("--target-count", type=int, default=1000)
    parser.add_argument("--prefix", default="aug_balanced")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories found in {dataset_dir}")

    total_created = 0
    for class_dir in class_dirs:
        class_rng = random.Random(f"{args.seed}:{class_dir.name}")
        created, final_count = balance_class(
            class_dir,
            args.target_count,
            args.prefix,
            class_rng,
        )
        total_created += created
        print(f"{class_dir.name}: created {created}, final count {final_count}")

    print(f"Total augmented images created: {total_created}")


if __name__ == "__main__":
    main()
