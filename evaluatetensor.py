import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def preprocess_image_tf(image_path, target_size=(224, 224)):
    img = keras_image.load_img(image_path, target_size=target_size)
    img_array = keras_image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def load_model_tf(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle TensorFlow introuvable : {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model

def evaluate_model_tf(model, data_dir):
    y_true = []
    y_pred = []

    for label_idx, label in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, label)
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Dossier de classe introuvable : {class_dir}")
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(class_dir, fname)
            img_tensor = preprocess_image_tf(img_path)

            predictions = model.predict(img_tensor)
            predicted = np.argmax(predictions, axis=1)[0]

            y_true.append(label_idx)
            y_pred.append(predicted)

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    print("\n🧩 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation modèle TensorFlow sur les données de test")
    parser.add_argument('--model', required=True, help="chemin vers le modèle TensorFlow (dossier)")
    parser.add_argument('--data', required=True, help="dossier des images de test, organisé par classe")
    args = parser.parse_args()

    model = load_model_tf(args.model)
    evaluate_model_tf(model, args.data)
