import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import os

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

def preprocess_image_tf(image_path, target_size=(224, 224)):
    """Charge et prétraite une image pour TensorFlow/Keras."""
    img = keras_image.load_img(image_path, target_size=target_size)
    img_array = keras_image.img_to_array(img)
    img_array = img_array / 255.0  # Normalisation [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter dimension batch
    return img_array

def load_model_tf(model_path):
    """Charge un modèle TensorFlow/Keras depuis un chemin."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle TensorFlow introuvable : {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model

def predict_tensorflow(model_path, image_path):
    """Prédit la classe à partir d'une image et d'un modèle TensorFlow."""
    model = load_model_tf(model_path)
    img = preprocess_image_tf(image_path)

    predictions = model.predict(img)
    pred_index = np.argmax(predictions, axis=1)[0]
    predicted_class = CLASS_NAMES[pred_index]
    
    # Optionnel : afficher la probabilité
    confidence = predictions[0][pred_index]
    print(f"[DEBUG TensorFlow] Classe prédite: {predicted_class} avec confiance {confidence:.4f}")

    return predicted_class
