from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import tempfile
import traceback
import json
import re
import sys

# Load environment variables (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed, using environment variables directly")

# Handle both old and new OpenAI library versions
try:
    # For newer versions (v1.0.0+)
    from openai import OpenAI

    NEW_OPENAI = True
except ImportError:
    # For older versions
    import openai

    NEW_OPENAI = False

# ==========================
# CONFIG
# ==========================
DATASET_DIR = "Dataset/train"
MODEL_PATH = "best_global_model.pth"
UPLOAD_FOLDER = "temp_uploads"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OpenAI API key - load from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

# Initialize OpenAI client based on version (only if API key is available)
if OPENAI_API_KEY:
    if NEW_OPENAI:
        client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        openai.api_key = OPENAI_API_KEY
else:
    print("WARNING: OPENAI_API_KEY not found. LLM features will be disabled.")
    client = None

# ==========================
# CREATE UPLOAD FOLDER
# ==========================
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
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes
    )
    return model


# ==========================
# IMAGE TRANSFORM
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ==========================
# LOAD MODEL - FIXED
# ==========================
print("Loading model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

# Load class names
class_names = load_class_names(DATASET_DIR)
print(f"Found {len(class_names)} classes: {class_names}")

# Build and load model
model = build_model(len(class_names)).to(DEVICE)

try:
    # Load the entire state dict without filtering
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # If checkpoint is a dict with 'state_dict' key (common in some training scripts)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded successfully!")

except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    raise


# ==========================
# PREDICT IMAGE
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
        print(f"Prediction: {predicted_label} with confidence {confidence:.2f}%")

        return predicted_label, confidence

    except Exception as e:
        print("Prediction error:", e)
        traceback.print_exc()
        return None, 0.0


# ==========================
# SAFE JSON EXTRACT
# ==========================
def extract_json(text):
    try:
        # Try to find JSON object in the text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())

        # If no JSON found, try to parse the entire text as JSON
        return json.loads(text)
    except json.JSONDecodeError:
        print("Failed to parse JSON from response")
        return None
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return None


# ==========================
# DEFAULT INFO
# ==========================
def default_disease_info():
    return {
        "description": "Information currently unavailable.",
        "symptoms": "Information not available",
        "causes": "Information not available",
        "treatment": "Please consult with a veterinarian",
        "when_to_see_vet": "Consult a veterinarian if symptoms persist or worsen"
    }


# ==========================
# GET DISEASE INFO FROM OPENAI
# ==========================
def get_disease_info_openai(disease_name):
    try:
        prompt = f"""
        Provide veterinary information about the dog skin disease "{disease_name}".

        Respond ONLY in valid JSON format with the following structure:
        {{
            "description": "A brief description of the disease",
            "symptoms": "Common symptoms (as a bulleted list with - at start of each line)",
            "causes": "Common causes (as a bulleted list with - at start of each line)",
            "treatment": "Typical treatment approaches (as a bulleted list with - at start of each line)",
            "when_to_see_vet": "Guidance on when to seek veterinary care"
        }}

        Make sure all fields are filled with relevant information.
        """

        if NEW_OPENAI:
            # New OpenAI version (v1.0.0+)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You are a veterinary expert specializing in canine dermatology. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            answer_text = response.choices[0].message.content.strip()
        else:
            # Old OpenAI version
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You are a veterinary expert specializing in canine dermatology. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            answer_text = response["choices"][0]["message"]["content"].strip()

        print(f"OpenAI response: {answer_text}")

        disease_info = extract_json(answer_text)

        if disease_info and all(
                key in disease_info for key in ["description", "symptoms", "causes", "treatment", "when_to_see_vet"]):
            return disease_info
        else:
            print("Invalid or incomplete JSON from OpenAI")
            return default_disease_info()

    except Exception as e:
        print("OpenAI error:", e)
        traceback.print_exc()
        return default_disease_info()


# ==========================
# FLASK APP
# ==========================
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": os.path.exists(MODEL_PATH),
        "dataset_loaded": os.path.exists(DATASET_DIR),
        "num_classes": len(class_names),
        "classes": class_names,
        "device": str(DEVICE),
        "openai_version": "new" if NEW_OPENAI else "old"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({
                "error": "No image file provided",
                "status": "error"
            }), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({
                "error": "No file selected",
                "status": "error"
            }), 400

        # Save uploaded file temporarily
        file_ext = os.path.splitext(image_file.filename)[1] or ".jpg"
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
            dir=UPLOAD_FOLDER
        )
        image_file.save(temp_file.name)
        temp_file.close()

        # Make prediction
        predicted_label, confidence = predict_image(temp_file.name)

        # Clean up temp file
        try:
            os.remove(temp_file.name)
        except Exception as e:
            print(f"Error removing temp file: {e}")

        if predicted_label is None:
            return jsonify({
                "error": "Prediction failed",
                "status": "error"
            }), 500

        # Get disease information from OpenAI
        disease_info = get_disease_info_openai(predicted_label)

        # Format response
        result = {
            "predicted_disease": predicted_label,
            "confidence": round(confidence, 2),
            "description": disease_info.get("description", ""),
            "symptoms": disease_info.get("symptoms", ""),
            "causes": disease_info.get("causes", ""),
            "treatment": disease_info.get("treatment", ""),
            "when_to_see_vet": disease_info.get("when_to_see_vet", ""),
            "status": "success"
        }

        print(f"Sending response for {predicted_label}")
        return jsonify(result)

    except Exception as e:
        print("Server error:", e)
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


# ==========================
# RUN SERVER
# ==========================
if __name__ == "__main__":
    print("=" * 50)
    print("Starting Dog Skin Disease Prediction API...")
    print(f"Device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Dataset path: {DATASET_DIR}")
    print(f"Number of classes: {len(class_names)}")
    print("=" * 50)

    app.run(host="0.0.0.0", port=5000, debug=True)