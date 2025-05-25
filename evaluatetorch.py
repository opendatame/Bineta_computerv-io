import os
import sys
import torch
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Ajouter le dossier parent pour accéder aux modules locaux
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.cnnbypytorch import ResNet50FineTune

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def load_model(model_path, device='cpu'):
    from torchvision.models import resnet50
    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, data_dir, device='cpu'):
    y_true = []
    y_pred = []

    for label_idx, label in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, label)
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(class_dir, fname)
            img_tensor = preprocess_image(img_path).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                predicted = torch.argmax(output, 1).item()

            y_true.append(label_idx)
            y_pred.append(predicted)

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    print("\n🧩 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Évaluation du modèle PyTorch sur les données de test")
    parser.add_argument('--model', required=True, help="chemin vers le modèle .torch")
    parser.add_argument('--data', required=True, help="dossier contenant les sous-dossiers par classe (ex: testing/)")
    parser.add_argument('--device', default='cpu', help="cpu ou cuda")

    args = parser.parse_args()
    device = torch.device(args.device)

    model = load_model(args.model, device)
    evaluate_model(model, args.data, device)
