from PIL import Image, ImageOps
import torch
from torchvision.transforms import functional as TF

from data_utils import IMAGENET_MEAN, IMAGENET_STD


TTA_MODES = {"none", "hflip", "five_crop", "ten_crop"}


def normalize_tta_mode(mode):
    normalized = (mode or "none").strip().lower()
    if normalized not in TTA_MODES:
        allowed = ", ".join(sorted(TTA_MODES))
        raise ValueError(f"Unsupported TTA mode '{mode}'. Use one of: {allowed}.")
    return normalized


def build_prediction_tensor_batch(image_pil, image_size, tta_mode="none"):
    images = build_tta_images(image_pil, image_size, normalize_tta_mode(tta_mode))
    return torch.stack([preprocess_pil_image(image) for image in images])


def build_tta_images(image_pil, image_size, tta_mode):
    base = image_pil.convert("RGB")

    if tta_mode == "none":
        return [resize_square(base, image_size)]

    if tta_mode == "hflip":
        return [
            resize_square(base, image_size),
            resize_square(ImageOps.mirror(base), image_size),
        ]

    crops = five_crops(base, image_size)
    if tta_mode == "five_crop":
        return crops

    if tta_mode == "ten_crop":
        return crops + [ImageOps.mirror(crop) for crop in crops]

    return [resize_square(base, image_size)]


def resize_square(image_pil, image_size):
    return image_pil.resize((image_size, image_size), Image.Resampling.BILINEAR)


def five_crops(image_pil, image_size):
    resized_size = max(int(image_size * 1.15), image_size)
    resized = resize_square(image_pil, resized_size)
    width, height = resized.size
    left = width - image_size
    top = height - image_size
    center_left = left // 2
    center_top = top // 2

    boxes = [
        (0, 0, image_size, image_size),
        (left, 0, width, image_size),
        (0, top, image_size, height),
        (left, top, width, height),
        (center_left, center_top, center_left + image_size, center_top + image_size),
    ]
    return [resized.crop(box) for box in boxes]


def preprocess_pil_image(image_pil):
    tensor = TF.to_tensor(image_pil)
    return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
