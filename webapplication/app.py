import os
from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50
import tensorflow as tf
import numpy as np

# --- Constantes ---
UPLOAD_FOLDER = 'webapplication/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MIN_WIDTH = 100
MIN_HEIGHT = 100
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- Flask app ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'mon_secret_key'

# --- Préparation du modèle PyTorch ---
device = torch.device('cpu')  # changer en 'cuda' si GPU dispo

print("[DEBUG] Chargement modèle PyTorch...")
model_path_pytorch = os.path.join(os.path.dirname(__file__), '..', 'bineta_model.torch')
model_pytorch = resnet50(weights=None)
num_features = model_pytorch.fc.in_features
model_pytorch.fc = torch.nn.Linear(num_features, len(CLASS_NAMES))
model_pytorch.load_state_dict(torch.load(model_path_pytorch, map_location=device))
model_pytorch.to(device)
model_pytorch.eval()
print("[DEBUG] Modèle PyTorch chargé.")

# --- Préparation du modèle TensorFlow ---
print("[DEBUG] Chargement modèle TensorFlow...")
model_path_tf = os.path.join(os.path.dirname(__file__), '..', 'bineta_model.tensorflow')
model_tf = tf.keras.models.load_model(model_path_tf)
print("[DEBUG] Modèle TensorFlow chargé.")

# --- Fonctions utiles ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_pytorch(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def preprocess_image_tensorflow(image_path):
    from tensorflow.keras.preprocessing import image as kimage
    img = kimage.load_img(image_path, target_size=(224, 224))
    x = kimage.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def predict_pytorch_loaded(image_path):
    image_tensor = preprocess_image_pytorch(image_path).to(device)
    with torch.no_grad():
        outputs = model_pytorch(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return CLASS_NAMES[predicted.item()]

def predict_tensorflow_loaded(image_path):
    img = preprocess_image_tensorflow(image_path)
    preds = model_tf.predict(img)
    pred_idx = np.argmax(preds, axis=1)[0]
    return CLASS_NAMES[pred_idx]

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/', methods=['POST'])
def predict():
    backend = request.form.get('backend')
    print(f"[DEBUG] Backend choisi : {backend}")

    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        flash("Extension de fichier non autorisée ou fichier manquant.", "error")
        return render_template('index.html', prediction=None)

    filename = secure_filename(file.filename)
    upload_folder = app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)
    print(f"[DEBUG] Fichier sauvegardé : {filepath}")

    try:
        with Image.open(filepath) as img:
            width, height = img.size
            print(f"[DEBUG] Image size: {width}x{height}")
            if width < MIN_WIDTH or height < MIN_HEIGHT:
                os.remove(filepath)
                flash(f"L'image est trop petite (min {MIN_WIDTH}x{MIN_HEIGHT}).", "error")
                return render_template('index.html', prediction=None)

        if backend == 'pytorch':
            prediction = predict_pytorch_loaded(filepath)
        elif backend == 'tensorflow':
            prediction = predict_tensorflow_loaded(filepath)
        else:
            flash("Backend non supporté.", "error")
            return render_template('index.html', prediction=None)

        print(f"[DEBUG] Prédiction : {prediction}")
        return render_template('index.html', prediction=prediction, image_file=filename)

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        flash(f"Erreur lors du traitement : {str(e)}", "error")
        return render_template('index.html', prediction=None)

@app.route('/health')
def health():
    return "OK"

# --- Main ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[DEBUG] Lancement sur le port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
