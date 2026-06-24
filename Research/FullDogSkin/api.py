import base64
import io
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms
from werkzeug.exceptions import RequestEntityTooLarge

from inference_utils import build_prediction_tensor_batch, normalize_tta_mode
from loadJson import load_disease_data
from model import build_model as build_classifier_model


BASE_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())


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


def _resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# CONFIG
DATASET_DIR = _resolve_path(os.getenv("API_DATASET_DIR", "Dataset/train"))
MODEL_PATH = _resolve_path(os.getenv("MODEL_PATH", os.getenv("BEST_MODEL_PATH", "best_efficientnet_b0_model.pth")))
DEVICE = get_best_device()
IMAGE_SIZE = _get_env_int("IMAGE_SIZE", 224)
MAX_UPLOAD_MB = _get_env_int("MAX_UPLOAD_MB", 12)
MIN_IMAGE_DIMENSION = _get_env_int("MIN_IMAGE_DIMENSION", 64)
MAX_IMAGE_DIMENSION = _get_env_int("MAX_IMAGE_DIMENSION", 10000)
MAX_IMAGE_PIXELS = _get_env_int("MAX_IMAGE_PIXELS", 60_000_000)
LOW_CONFIDENCE_THRESHOLD = _get_env_float("LOW_CONFIDENCE_THRESHOLD", 0.75)
SOFTMAX_TEMPERATURE = max(_get_env_float("SOFTMAX_TEMPERATURE", 0.772956), 1e-6)
PREDICTION_TTA_MODE = normalize_tta_mode(os.getenv("PREDICTION_TTA_MODE", "five_crop"))
ENABLE_NONE_GUARD = _get_env_bool("ENABLE_NONE_GUARD", True)
NONE_GUARD_TTA_MODE = normalize_tta_mode(os.getenv("NONE_GUARD_TTA_MODE", "none"))
NONE_GUARD_TEMPERATURE = max(_get_env_float("NONE_GUARD_TEMPERATURE", 0.775777), 1e-6)
NONE_CLASS_NAME = os.getenv("NONE_CLASS_NAME", "None")
NONE_REJECTION_THRESHOLD = _get_env_float("NONE_REJECTION_THRESHOLD", 0.5)
STRICT_INVALID_NONE_THRESHOLD = _get_env_float("STRICT_INVALID_NONE_THRESHOLD", 0.35)
SECONDARY_REJECTION_THRESHOLD = _get_env_float("SECONDARY_REJECTION_THRESHOLD", 0.2)
AUX_DOG_RESCUE_THRESHOLD = _get_env_float("AUX_DOG_RESCUE_THRESHOLD", 0.9)
CONTENT_VALIDATOR_MODE = os.getenv("ENABLE_CONTENT_VALIDATOR", "auto").strip().lower()
ENABLE_EXPLAINABILITY = _get_env_bool("ENABLE_EXPLAINABILITY", True)
EXPLANATION_MAX_SIDE = _get_env_int("EXPLANATION_MAX_SIDE", 640)
EXPLANATION_ALPHA = min(max(_get_env_float("EXPLANATION_ALPHA", 0.45), 0.0), 1.0)
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DOG_BREED_CLASS_INDICES = list(range(151, 269))
AUX_NON_DOG_LABELS = {
    "sports car", "tabby", "sorrel", "lighter", "television", "seashore",
    "axolotl", "African chameleon", "indigo bunting", "ox", "ram", "ibex",
    "monastery", "palace", "mobile home", "suit", "jersey", "brassiere",
    "cicada", "long-horned beetle", "bee", "ballpoint", "plunger",
    "hand-held computer", "cleaver", "mountain tent", "golf ball",
    "Windsor tie", "lab coat", "academic gown", "trench coat", "sweatshirt",
    "lipstick", "hair slide", "neck brace", "bikini", "maillot",
    "maillot tank suit"
}

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
cors_origins = os.getenv("CORS_ORIGINS", "*")
CORS(
    app,
    resources={r"/*": {"origins": cors_origins if cors_origins == "*" else cors_origins.split(",")}},
)


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = []
disease_data = {"dog_skin_diseases": []}
none_class_idx = None
model = None
startup_error = None
content_validator_model = None
content_validator_transform = None
content_validator_categories = None


def load_class_names(dataset_path):
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    classes = sorted(entry.name for entry in dataset_path.iterdir() if entry.is_dir())
    if not classes:
        raise ValueError(f"No class folders found in dataset path: {dataset_path}")
    return classes


def build_model(num_classes):
    return build_classifier_model(num_classes, pretrained=False, fine_tune_blocks=0)


def load_checkpoint_state(model_instance, model_path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}

    model_instance.load_state_dict(checkpoint)


def load_content_validator():
    if CONTENT_VALIDATOR_MODE in {"0", "false", "no", "off", "disabled"}:
        LOGGER.info("Content validator disabled by ENABLE_CONTENT_VALIDATOR=0")
        return None, None, None

    try:
        # Separate ImageNet model used only to reject obviously unrelated uploads.
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        if CONTENT_VALIDATOR_MODE == "auto" and not weights_cached(weights):
            LOGGER.info(
                "Content validator weights are not cached; set ENABLE_CONTENT_VALIDATOR=1 to download them."
            )
            return None, None, None

        validator_model = models.efficientnet_b0(weights=weights).to(DEVICE)
        validator_model.eval()
        return validator_model, weights.transforms(), weights.meta["categories"]
    except Exception as exc:
        LOGGER.warning("Content validator unavailable: %s", exc)
        return None, None, None


def initialize_app_state():
    global class_names
    global disease_data
    global none_class_idx
    global model
    global startup_error
    global content_validator_model
    global content_validator_transform
    global content_validator_categories

    try:
        class_names = load_class_names(DATASET_DIR)
        disease_data = load_disease_data()
        none_class_idx = class_names.index(NONE_CLASS_NAME) if NONE_CLASS_NAME in class_names else None

        loaded_model = build_model(len(class_names)).to(DEVICE)
        load_checkpoint_state(loaded_model, MODEL_PATH)
        loaded_model.eval()
        model = loaded_model
        startup_error = None
        LOGGER.info("Loaded model with %d classes on %s", len(class_names), DEVICE)
    except Exception as exc:
        model = None
        startup_error = str(exc)
        LOGGER.exception("API startup is degraded: %s", exc)

    (
        content_validator_model,
        content_validator_transform,
        content_validator_categories,
    ) = load_content_validator()


def service_ready():
    return model is not None and bool(class_names)


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_):
    return jsonify({
        "status": "file_too_large",
        "error": f"Image is too large. Upload a file smaller than {MAX_UPLOAD_MB} MB.",
    }), 413


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "frontend.html")


@app.route("/health", methods=["GET"])
def health():
    ready = service_ready()
    response = {
        "status": "healthy" if ready else "degraded",
        "ready": ready,
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "dataset_loaded": DATASET_DIR.exists(),
        "dataset_path": str(DATASET_DIR),
        "num_classes": len(class_names),
        "device": str(DEVICE),
        "content_validator_loaded": content_validator_model is not None,
        "content_validator_mode": CONTENT_VALIDATOR_MODE,
        "softmax_temperature": SOFTMAX_TEMPERATURE,
        "confidence_calibrated": SOFTMAX_TEMPERATURE != 1.0,
        "prediction_tta_mode": PREDICTION_TTA_MODE,
        "none_guard_enabled": ENABLE_NONE_GUARD,
        "none_guard_tta_mode": NONE_GUARD_TTA_MODE,
        "none_guard_temperature": NONE_GUARD_TEMPERATURE,
        "strict_invalid_none_threshold": STRICT_INVALID_NONE_THRESHOLD,
        "explainability_enabled": ENABLE_EXPLAINABILITY,
    }
    if startup_error:
        response["startup_error"] = startup_error
    return jsonify(response), 200 if ready else 503


@app.route("/predict", methods=["POST"])
def predict():
    if not service_ready():
        return jsonify({
            "status": "service_unavailable",
            "error": "Prediction service is not ready.",
            "details": startup_error or "Model or class metadata failed to load.",
        }), 503

    if "image" not in request.files:
        return jsonify({"status": "bad_request", "error": "No image file uploaded."}), 400

    try:
        img = load_request_image(request.files["image"])
        pred, conf, probabilities, pred_idx = predict_image(img, model, class_names)
        top_predictions = get_top_predictions(probabilities, class_names)
        none_prob = float(probabilities[none_class_idx].item()) if none_class_idx is not None else 0.0
        none_guard = get_none_guard_result(img)
        if none_guard:
            none_prob = max(none_prob, none_guard["none_prob"])
            if none_guard["is_none"]:
                pred = NONE_CLASS_NAME
                conf = none_guard["confidence"]
                probabilities = none_guard["probabilities"]
                top_predictions = get_top_predictions(probabilities, class_names)
        dog_score, aux_label, aux_conf = get_aux_content_signals(img)

        if pred == NONE_CLASS_NAME or none_prob >= NONE_REJECTION_THRESHOLD:
            if dog_score is not None and dog_score >= AUX_DOG_RESCUE_THRESHOLD:
                return jsonify(build_uncertain_dog_response(
                    none_prob,
                    dog_score,
                    aux_label,
                    aux_conf,
                    top_predictions,
                )), 422
            return jsonify(build_invalid_content_response(
                none_prob,
                dog_score,
                aux_label,
                aux_conf,
                top_predictions,
            )), 422

        if none_prob >= STRICT_INVALID_NONE_THRESHOLD:
            if dog_score is not None and dog_score >= AUX_DOG_RESCUE_THRESHOLD:
                return jsonify(build_uncertain_dog_response(
                    none_prob,
                    dog_score,
                    aux_label,
                    aux_conf,
                    top_predictions,
                )), 422
            return jsonify(build_invalid_content_response(
                none_prob,
                dog_score,
                aux_label,
                aux_conf,
                top_predictions,
            )), 422

        if (
            none_prob >= SECONDARY_REJECTION_THRESHOLD
            and aux_label in AUX_NON_DOG_LABELS
            and (dog_score is None or dog_score < AUX_DOG_RESCUE_THRESHOLD)
        ):
            return jsonify(build_invalid_content_response(
                none_prob,
                dog_score,
                aux_label,
                aux_conf,
                top_predictions,
            )), 422

        if conf < LOW_CONFIDENCE_THRESHOLD:
            return jsonify(build_low_confidence_response(pred, conf, top_predictions)), 422

        info = get_disease_info(pred, disease_data)
        response = build_success_response(pred, conf, info, top_predictions)
        explanation = build_explanation_response(img, pred_idx)
        if explanation:
            response["explanation"] = explanation
        return jsonify(response)

    except ValueError as exc:
        return jsonify({"status": "bad_request", "error": str(exc)}), 400
    except Exception as exc:
        LOGGER.exception("Prediction failed: %s", exc)
        return jsonify({
            "status": "server_error",
            "error": "Prediction failed unexpectedly. Please try another image.",
        }), 500


def load_request_image(file_storage):
    if not file_storage or not file_storage.filename:
        raise ValueError("No file selected.")

    extension = Path(file_storage.filename).suffix.lower()
    if extension and extension not in ALLOWED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_IMAGE_EXTENSIONS))
        raise ValueError(f"Unsupported file type. Please upload one of: {allowed}.")

    image_bytes = file_storage.read()
    if not image_bytes:
        raise ValueError("Uploaded image is empty.")
    if len(image_bytes) > app.config["MAX_CONTENT_LENGTH"]:
        raise RequestEntityTooLarge()

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(io.BytesIO(image_bytes)) as image:
            img = image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded file is not a readable image.") from exc

    width, height = img.size
    if (
        width < MIN_IMAGE_DIMENSION
        or height < MIN_IMAGE_DIMENSION
        or width > MAX_IMAGE_DIMENSION
        or height > MAX_IMAGE_DIMENSION
    ):
        raise ValueError(
            "Image dimensions are invalid. Upload a clear dog photo between "
            f"{MIN_IMAGE_DIMENSION}px and {MAX_IMAGE_DIMENSION}px per side."
        )

    return img


def get_aux_content_signals(image_pil):
    if (
        content_validator_model is None
        or content_validator_transform is None
        or content_validator_categories is None
    ):
        return None, None, None

    tensor = content_validator_transform(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(content_validator_model(tensor), dim=1)[0]

    dog_score = float(probs[DOG_BREED_CLASS_INDICES].sum().item())
    top_conf, top_idx = probs.max(dim=0)
    top_label = content_validator_categories[int(top_idx.item())]
    return dog_score, top_label, float(top_conf.item())


def weights_cached(weights):
    filename = Path(urlparse(weights.url).path).name
    return (Path(torch.hub.get_dir()) / "checkpoints" / filename).exists()


def build_invalid_content_response(none_prob, dog_score, aux_label=None, aux_conf=None, top_predictions=None):
    response = {
        "status": "invalid_content",
        "error": "Image does not appear to be a supported dog skin photo.",
        "suggestion": "Upload a clear close-up photo of a dog's skin, not a human, another animal, or an unrelated object.",
        "none_confidence": round(none_prob * 100, 2),
        "top_predictions": top_predictions or [],
    }
    add_confidence_metadata(response)
    add_validator_fields(response, dog_score, aux_label, aux_conf)
    return response


def build_uncertain_dog_response(none_prob, dog_score, aux_label=None, aux_conf=None, top_predictions=None):
    response = {
        "status": "low_confidence",
        "error": "The image looks like a dog, but no supported skin condition was confidently identified.",
        "suggestion": "Try a closer photo that focuses on the affected skin area with good lighting and less background.",
        "none_confidence": round(none_prob * 100, 2),
        "top_predictions": top_predictions or [],
    }
    add_confidence_metadata(response)
    add_validator_fields(response, dog_score, aux_label, aux_conf)
    return response


def build_low_confidence_response(prediction, confidence, top_predictions):
    response = {
        "status": "low_confidence",
        "error": "Prediction confidence is too low to show a reliable result.",
        "suggestion": "Use a sharper, closer, well-lit image of the affected skin area.",
        "prediction": prediction,
        "predicted_disease": prediction,
        "confidence": round(confidence * 100, 2),
        "required_confidence": round(LOW_CONFIDENCE_THRESHOLD * 100, 2),
        "top_predictions": top_predictions,
    }
    add_confidence_metadata(response)
    return response


def add_validator_fields(response, dog_score, aux_label, aux_conf):
    if dog_score is not None:
        response["dog_score"] = round(dog_score * 100, 2)
    if aux_label is not None:
        response["validator_label"] = aux_label
    if aux_conf is not None:
        response["validator_confidence"] = round(aux_conf * 100, 2)


def add_confidence_metadata(response):
    response["confidence_method"] = "softmax"
    response["softmax_temperature"] = round(SOFTMAX_TEMPERATURE, 6)
    response["confidence_calibrated"] = SOFTMAX_TEMPERATURE != 1.0
    response["prediction_tta_mode"] = PREDICTION_TTA_MODE
    response["none_guard_enabled"] = ENABLE_NONE_GUARD
    if ENABLE_NONE_GUARD:
        response["none_guard_tta_mode"] = NONE_GUARD_TTA_MODE
        response["none_guard_temperature"] = round(NONE_GUARD_TEMPERATURE, 6)
        response["strict_invalid_none_threshold"] = round(STRICT_INVALID_NONE_THRESHOLD, 6)


def build_success_response(prediction, confidence, info, top_predictions):
    response = {
        "status": "success",
        "prediction": prediction,
        "predicted_disease": prediction,
        "confidence": round(confidence * 100, 2),
        "description": info.get("description", "") if info else "No information found.",
        "symptoms": info.get("symptoms", []) if info else [],
        "causes": info.get("causes", []) if info else [],
        "treatment": info.get("treatment", []) if info else [],
        "when_to_see_vet": info.get("when_to_see_vet", "") if info else "",
        "top_predictions": top_predictions,
    }
    add_confidence_metadata(response)
    return response


def predict_image(image_pil, model_instance, labels):
    return predict_with_tta(
        image_pil,
        model_instance,
        labels,
        PREDICTION_TTA_MODE,
        SOFTMAX_TEMPERATURE,
    )


def get_none_guard_result(image_pil):
    if not ENABLE_NONE_GUARD or none_class_idx is None:
        return None

    pred, conf, probabilities, _pred_idx = predict_with_tta(
        image_pil,
        model,
        class_names,
        NONE_GUARD_TTA_MODE,
        NONE_GUARD_TEMPERATURE,
    )
    none_prob = float(probabilities[none_class_idx].item())
    return {
        "is_none": pred == NONE_CLASS_NAME or none_prob >= NONE_REJECTION_THRESHOLD,
        "prediction": pred,
        "confidence": conf,
        "none_prob": none_prob,
        "probabilities": probabilities,
    }


def predict_with_tta(image_pil, model_instance, labels, tta_mode, temperature):
    tensor_batch = build_prediction_tensor_batch(
        image_pil,
        IMAGE_SIZE,
        tta_mode,
    ).to(DEVICE)

    model_instance.eval()
    with torch.no_grad():
        output = model_instance(tensor_batch).mean(dim=0, keepdim=True)
        probs = torch.softmax(output / temperature, dim=1)
        conf, pred_class = probs.max(1)

    predicted_index = int(pred_class.item())
    predicted_name = labels[predicted_index]
    confidence_value = float(conf.item())
    probabilities = probs[0].detach().cpu()
    return predicted_name, confidence_value, probabilities, predicted_index


def build_explanation_response(image_pil, target_index):
    if not ENABLE_EXPLAINABILITY:
        return None

    try:
        overlay = build_grad_cam_overlay(image_pil, model, target_index)
        return {
            "type": "grad_cam",
            "target_class": class_names[target_index],
            "image_mime": "image/png",
            "image_base64": encode_png_base64(overlay),
        }
    except Exception as exc:
        LOGGER.warning("Grad-CAM explanation unavailable: %s", exc)
        return None


def build_grad_cam_overlay(image_pil, model_instance, target_index):
    target_layer = get_grad_cam_target_layer(model_instance)
    activations = {}
    gradients = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
        tensor.requires_grad_(True)
        model_instance.zero_grad(set_to_none=True)

        with torch.enable_grad():
            output = model_instance(tensor)
            score = output[0, target_index]
            score.backward()

        if "value" not in activations or "value" not in gradients:
            raise RuntimeError("Unable to capture model activations for explanation.")

        activation = activations["value"].detach()
        gradient = gradients["value"].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=(image_pil.height, image_pil.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        cam = cam - cam.min()
        max_value = cam.max()
        if max_value <= 0:
            raise RuntimeError("Explanation heatmap was empty.")
        heatmap = (cam / max_value).detach().cpu().numpy()
        return overlay_heatmap(image_pil, heatmap)
    finally:
        forward_handle.remove()
        backward_handle.remove()


def get_grad_cam_target_layer(model_instance):
    features = getattr(model_instance, "features", None)
    if features is None or len(features) == 0:
        raise RuntimeError("Model does not expose EfficientNet feature layers.")
    return features[-1]


def overlay_heatmap(image_pil, heatmap):
    base = image_pil.convert("RGBA")
    heatmap = np.clip(heatmap, 0.0, 1.0)

    colored = np.zeros((*heatmap.shape, 4), dtype=np.uint8)
    colored[..., 0] = 255
    colored[..., 1] = (heatmap * 180).astype(np.uint8)
    colored[..., 3] = (heatmap * 255 * EXPLANATION_ALPHA).astype(np.uint8)

    heatmap_image = Image.fromarray(colored, mode="RGBA")
    return Image.alpha_composite(base, heatmap_image).convert("RGB")


def encode_png_base64(image_pil):
    preview = image_pil.copy()
    preview.thumbnail((EXPLANATION_MAX_SIDE, EXPLANATION_MAX_SIDE), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    preview.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def get_top_predictions(probabilities, labels, limit=3):
    limit = min(limit, len(labels))
    if limit <= 0:
        return []

    top_probs, top_indices = torch.topk(probabilities, k=limit)
    return [
        {
            "label": labels[int(index.item())],
            "confidence": round(float(prob.item()) * 100, 2),
        }
        for prob, index in zip(top_probs, top_indices)
    ]


def get_disease_info(disease_name, data):
    for disease in data.get("dog_skin_diseases", []):
        if disease.get("name", "").lower() == disease_name.lower():
            return disease
    return None


def find_smoke_test_image():
    test_dir = DATASET_DIR.parent / "test"
    if not test_dir.exists():
        return None

    for path in test_dir.rglob("*"):
        if path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
            return path
    return None


def run_smoke_test():
    with app.test_client() as client:
        health_resp = client.get("/health")
        sample_image_path = find_smoke_test_image()
        if sample_image_path is None:
            raise FileNotFoundError(f"No smoke test image found under: {DATASET_DIR.parent / 'test'}")

        with sample_image_path.open("rb") as image_file:
            buf = io.BytesIO(image_file.read())

        pred_resp = client.post(
            "/predict",
            data={"image": (buf, sample_image_path.name)},
            content_type="multipart/form-data",
        )
        result = pred_resp.get_json(silent=True) or {}
        print(
            "Smoke test completed:",
            f"/health={health_resp.status_code}",
            f"/predict={pred_resp.status_code}",
            f"status={result.get('status', 'unknown')}",
            f"prediction={result.get('prediction', result.get('predicted_disease', 'n/a'))}",
            f"confidence={result.get('confidence', 'n/a')}",
        )


initialize_app_state()


if __name__ == "__main__":
    if os.getenv("SKIP_SERVER_BIND") == "1":
        run_smoke_test()
        raise SystemExit(0)

    host = os.getenv("API_HOST", "0.0.0.0")
    port = _get_env_int("PORT", 5001)
    debug = _get_env_bool("FLASK_DEBUG", False)
    app.run(host=host, port=port, debug=debug)
