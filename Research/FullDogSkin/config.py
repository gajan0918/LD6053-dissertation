import os
import torch


def _get_env_int(name, default):
    value = os.getenv(name)
    return int(value) if value is not None else default


def _get_env_float(name, default):
    value = os.getenv(name)
    return float(value) if value is not None else default


def _get_env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DATASET_DIR = os.getenv("DATASET_DIR", "Dataset")
NUM_CLIENTS = _get_env_int("NUM_CLIENTS", 4)
LOCAL_EPOCHS = _get_env_int("LOCAL_EPOCHS", 3)
ROUNDS = _get_env_int("ROUNDS", 12)
BATCH_SIZE = _get_env_int("BATCH_SIZE", 32)
IMAGE_SIZE = _get_env_int("IMAGE_SIZE", 224)
LR_HEAD = _get_env_float("LR_HEAD", 1e-3)
LR_BACKBONE = _get_env_float("LR_BACKBONE", 1e-4)
LR = LR_HEAD
WEIGHT_DECAY = _get_env_float("WEIGHT_DECAY", 1e-4)
LABEL_SMOOTHING = _get_env_float("LABEL_SMOOTHING", 0.05)
CLASS_WEIGHT_POWER = _get_env_float("CLASS_WEIGHT_POWER", 0.5)
FOCUS_CLASS_NAME = os.getenv("FOCUS_CLASS_NAME", "")
FOCUS_CLASS_WEIGHT_MULTIPLIER = _get_env_float("FOCUS_CLASS_WEIGHT_MULTIPLIER", 1.0)
PRETRAINED = _get_env_bool("PRETRAINED", True)
FINE_TUNE_BLOCKS = _get_env_int("FINE_TUNE_BLOCKS", 6)
CENTRAL_FINE_TUNE_EPOCHS = _get_env_int("CENTRAL_FINE_TUNE_EPOCHS", 4)
EARLY_STOPPING_PATIENCE = _get_env_int("EARLY_STOPPING_PATIENCE", 5)
GRAD_CLIP_NORM = _get_env_float("GRAD_CLIP_NORM", 1.0)
USE_WEIGHTED_SAMPLER = _get_env_bool("USE_WEIGHTED_SAMPLER", True)
SEED = _get_env_int("SEED", 42)
NUM_WORKERS = _get_env_int("NUM_WORKERS", 0)
BEST_MODEL_PATH = os.getenv("BEST_MODEL_PATH", "best_efficientnet_b0_model.pth")
RESUME_MODEL_PATH = os.getenv("RESUME_MODEL_PATH", "")
DEVICE = get_best_device()
