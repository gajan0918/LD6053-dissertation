from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import tempfile
import traceback
from loadJson import load_disease_data

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ==========================
# CONFIG
# ==========================
DATASET_DIR = "Dataset/train"
MODEL_PATH = "best_global_model.pth"
UPLOAD_FOLDER = "temp_uploads"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ==========================
# LOAD CLASS NAMES
# ==========================
def load_class_names(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    temp_dataset = ImageFolder(dataset_path)
    return temp_dataset.classes

# ==========================
# BUILD MODEL
# ==========================
def build_model(num_classes):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# ==========================
# IMAGE TRANSFORM
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================
# LOAD MODEL SAFELY
# ==========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

class_names = load_class_names(DATASET_DIR)
disease_data = load_disease_data()

# Build model for current dataset classes
model = build_model(len(class_names)).to(DEVICE)

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Filter out classifier weights if number of classes mismatches
filtered_dict = {k: v for k, v in checkpoint.items() if "classifier" not in k}

# Load feature extractor weights
model.load_state_dict(filtered_dict, strict=False)

# Classifier will remain initialized for current number of classes
model.eval()
print("Model and disease data loaded successfully!")

# ==========================
# PREDICTION
# ==========================
def predict_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred_class = probs.max(1)

        predicted_label = class_names[pred_class.item()]
        confidence = float(conf.item() * 100)

        return predicted_label, confidence

    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return None, 0.0

# ==========================
# GET DISEASE INFO
# ==========================
def get_disease_info(disease_name):
    disease_name_lower = disease_name.lower()
    for disease in disease_data["dog_skin_diseases"]:
        if disease["name"].lower() == disease_name_lower:
            return disease
    return None

# ==========================
# HEALTH CHECK ENDPOINT
# ==========================
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": os.path.exists(MODEL_PATH),
        "dataset_loaded": os.path.exists(DATASET_DIR),
        "endpoint_type": "direct_image_upload"
    })

# ==========================
# PREDICT ENDPOINT
# ==========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save temp file
        file_ext = os.path.splitext(image_file.filename)[1] or ".jpg"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, dir=UPLOAD_FOLDER)
        image_file.save(temp_file.name)
        temp_file.close()

        # Predict
        predicted_label, confidence = predict_image(temp_file.name)

        # Delete temp file
        try:
            os.remove(temp_file.name)
        except:
            pass

        if predicted_label is None:
            return jsonify({"error": "Prediction failed"}), 500

        # Get disease info
        disease_info = get_disease_info(predicted_label)

        if disease_info:
            result = {
                "predicted_disease": predicted_label,
                "confidence": round(confidence, 2),
                "description": disease_info["description"],
                "symptoms": disease_info["symptoms"],
                "causes": disease_info["causes"],
                "treatment": disease_info["treatment"],
                "when_to_see_vet": disease_info["when_to_see_vet"],
                "status": "success"
            }
        else:
            result = {
                "predicted_disease": predicted_label,
                "confidence": round(confidence, 2),
                "description": "No disease information found.",
                "status": "success"
            }

        return jsonify(result)

    except Exception as e:
        print(f"Server error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ==========================
# RUN APP
# ==========================
if __name__ == "__main__":
    print("Starting Dog Skin Disease Prediction API...")
    app.run(host="0.0.0.0", port=5000, debug=True)