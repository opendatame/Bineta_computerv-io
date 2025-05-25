import sys
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import tensorflow as tf  # Import TensorFlow classique
import os


# Ajouter le dossier parent pour accéder aux modules locaux
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import du modèle PyTorch (ton fichier models/cnnbypytorch.py doit contenir cette classe)
from models.cnnbypytorch import ResNet50FineTune

# Liste des classes (doit correspondre à l'ordre de tes labels dans train.py)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def preprocess_image_pytorch(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # même taille que pour l'entraînement
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # mêmes stats que ImageNet
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Ajouter la dimension batch

def predict_pytorch(model_path, image_path, device='cpu'):
    device = torch.device(device)

    from torchvision.models import resnet50
    model = resnet50(weights=None)  # Pas de poids préchargés ici
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, len(CLASS_NAMES))  # 4 classes comme en entraînement

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image_tensor = preprocess_image_pytorch(image_path).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    return CLASS_NAMES[predicted.item()]

def preprocess_image_tensorflow(image_path):
    from tensorflow.keras.preprocessing import image as kimage
    img = kimage.load_img(image_path, target_size=(224, 224))
    x = kimage.img_to_array(img)
    x = x / 255.0  # Normalisation [0,1]
    x = np.expand_dims(x, axis=0)
    return x

def predict_tensorflow(model_path, image_path):
    model = tf.keras.models.load_model(model_path)
    img = preprocess_image_tensorflow(image_path)
    preds = model.predict(img)
    pred_idx = np.argmax(preds, axis=1)[0]
    return CLASS_NAMES[pred_idx]

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prédiction d'image avec modèle PyTorch ou TensorFlow")
    parser.add_argument('--backend', choices=['pytorch', 'tensorflow'], required=True, help="backend à utiliser")
    parser.add_argument('--model', required=True, help="chemin vers le modèle sauvegardé")
    parser.add_argument('--image', required=True, help="chemin vers l'image à classifier")
    parser.add_argument('--device', default='cpu', help="device pour PyTorch (ex: cpu ou cuda)")

    args = parser.parse_args()

    if args.backend == 'pytorch':
        pred_class = predict_pytorch(args.model, args.image, device=args.device)
    else:
        pred_class = predict_tensorflow(args.model, args.image)

    print(f"Classe prédite : {pred_class}")

if __name__ == "__main__":
    main()