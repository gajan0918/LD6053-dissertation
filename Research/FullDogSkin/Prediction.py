import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import os
from loadJson import load_disease_data
from model import build_model as build_classifier_model

# CONFIG
DATASET_DIR = "Dataset/train"
MODEL_PATH = "best_efficientnet_b0_model.pth"
IMAGE_PATH = "img_1.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#  CLASS NAMES
def load_class_names(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    temp_dataset = ImageFolder(dataset_path)
    return temp_dataset.classes



# BUILD MODEL
def build_model(num_classes):
    return build_classifier_model(num_classes, pretrained=False, fine_tune_blocks=0)



# PREPROCESS
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])



# PREDICT
def predict_single_image(image_path, model, class_names):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred_class = probs.max(1)

    return class_names[pred_class.item()], float(conf.item())



# FIND DISEASE INFO FROM JSON
def get_disease_info(disease_name, data):
    disease_name_lower = disease_name.lower()

    for disease in data["dog_skin_diseases"]:
        if disease["name"].lower() == disease_name_lower:
            return disease

    return None




if __name__ == "__main__":
    class_names = load_class_names(DATASET_DIR)
    disease_data = load_disease_data()

    # Setup model
    model = build_model(len(class_names)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Predict on single image
    pred, conf = predict_single_image(IMAGE_PATH, model, class_names)


    print("Image:", IMAGE_PATH)
    print("Prediction:", pred)
    print(f"Confidence: {conf*100:.2f}%")
    info = get_disease_info(pred, disease_data)

    if info:
        print("\n---- Disease Details ----")
        print("Name:", info["name"])
        print("Description:", info["description"])
        print("\nSymptoms:")
        for s in info["symptoms"]:
            print("-", s)

        print("\nCauses:")
        for c in info["causes"]:
            print("-", c)

        print("\nTreatment:")
        for t in info["treatment"]:
            print("-", t)

        print("\nWhen to See Vet:", info["when_to_see_vet"])

    else:
        print("\nNo disease information found in JSON for:", pred)

