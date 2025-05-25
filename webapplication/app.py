import os
import sys

# Ajouter le dossier parent au sys.path AVANT import de predict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename
from PIL import Image  # Pour ouvrir et inspecter l'image
from predict import predict_pytorch, predict_tensorflow

# Paramètres
UPLOAD_FOLDER = 'webapplication/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MIN_WIDTH = 100
MIN_HEIGHT = 100

# Initialisation de Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'mon_secret_key'  # À changer en production

# Vérifie si le fichier a une extension autorisée
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route principale
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        backend = request.form.get('backend')
        device = request.form.get('device', 'cpu')

        # Chemin absolu des modèles
        BASE_DIR = os.path.abspath(os.path.dirname(__file__))
        model_path = None
        if backend == 'pytorch':
            model_path = os.path.join(BASE_DIR, '..', 'bineta_model.torch')
        elif backend == 'tensorflow':
            model_path = os.path.join(BASE_DIR, '..', 'bineta_model.tensorflow')
        else:
            flash("Backend non supporté.", "error")
            return render_template('index.html', prediction=None)

        # Gestion du fichier
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_folder = app.config['UPLOAD_FOLDER']
            os.makedirs(upload_folder, exist_ok=True)
            filepath = os.path.join(upload_folder, filename)

            try:
                file.save(filepath)
                print(f"[DEBUG] Fichier sauvegardé : {filepath}")

                # Vérification des dimensions de l'image
                with Image.open(filepath) as img:
                    width, height = img.size
                    print(f"[DEBUG] Image size: {width}x{height}, mode: {img.mode}")
                    if width < MIN_WIDTH or height < MIN_HEIGHT:
                        os.remove(filepath)
                        flash(f"L'image est trop petite (min {MIN_WIDTH}x{MIN_HEIGHT} pixels requis).", "error")
                        return render_template('index.html', prediction=None)

                # Appel du bon backend
                print(f"[DEBUG] Backend: {backend}, Device: {device}, Model path: {model_path}")
                if backend == 'pytorch':
                    prediction = predict_pytorch(model_path, filepath, device=device)
                else:
                    prediction = predict_tensorflow(model_path, filepath)

                print(f"[DEBUG] Prediction brute: '{prediction}'")

                # Nettoyage de la prédiction
                cleaned_pred = prediction.strip().lower()

                # Optionnel : vérifier que la classe est valide
                # valid_classes = ['glioma', 'meningioma', 'pituitary', 'notumor']
                # if cleaned_pred not in valid_classes:
                #     os.remove(filepath)
                #     flash("Image invalide ou non médicale. Veuillez charger une IRM cérébrale.", "error")
                #     return render_template('index.html', prediction=None)

                return render_template('index.html', prediction=prediction, image_file=filename)

            except Exception as e:
                if os.path.exists(filepath):
                    os.remove(filepath)
                flash(f"Erreur lors du traitement de l'image : {str(e)}", "error")
                return render_template('index.html', prediction=None)

        else:
            flash("Extension de fichier non autorisée. Veuillez uploader un fichier  JPG ", "error")

    return render_template('index.html', prediction=None)

# Lancement local de l'application
if __name__ == '__main__':
    app.run(debug=True)
