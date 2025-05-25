import os
import sys
import argparse
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from utils.prep import get_train_transforms, get_test_transforms  


def train_pytorch(epochs=10, patience=3):
    print("==> Entraînement avec PyTorch (ResNet)...")

    train_dir = "breast_cancer/breast_cancer/training"
    test_dir = "breast_cancer/breast_cancer/testing"
    
    train_dataset = datasets.ImageFolder(train_dir, transform=get_train_transforms())
    test_dataset = datasets.ImageFolder(test_dir, transform=get_test_transforms())

    print(f"Nombre d'images d'entraînement : {len(train_dataset)}")
    print(f"Nombre de classes : {len(train_dataset.classes)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("Début de la boucle d'entraînement...")
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        print(f"Epoch {epoch + 1}/{epochs}")

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = total_loss / total
        train_acc = 100 * correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        end_time = time.time()
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
              f"Time: {end_time - start_time:.2f} sec\n")

        # Early stopping manuel simple
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Sauvegarder le meilleur modèle (avec nom conforme au projet)
            torch.save(model.state_dict(), "bineta_model.torch")
            print("Meilleur modèle sauvegardé sous 'bineta_model.torch'.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping activé après {patience} époques sans amélioration.")
                break

    # Affichage des courbes
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs_range, val_losses, 'r-', label='Val Loss')
    plt.title('Loss par epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, 'b-', label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies, 'r-', label='Val Accuracy')
    plt.title('Accuracy par epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("Entraînement PyTorch terminé.")


def train_tensorflow():
    print("==> Entraînement avec TensorFlow (ResNet)...")

    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import layers, models, optimizers, callbacks

    train_dir = "breast_cancer/breast_cancer/training"
    test_dir = "breast_cancer/breast_cancer/testing"

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=15
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    base_model = tf.keras.applications.ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(4, activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callback EarlyStopping 
    es_callback = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=test_generator,
        callbacks=[es_callback]
    )

    
    model.save("bineta_model.tensorflow")

    loss, accuracy = model.evaluate(test_generator)
    print(f"Accuracy sur le set de test : {accuracy * 100:.2f}%")

    print("Modèle TensorFlow ResNet sauvegardé sous : bineta_model.tensorflow")

    # Affichage des courbes 
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss par epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy par epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['pytorch', 'tensorflow'], required=True,
                        help="Choisir le backend à utiliser : pytorch ou tensorflow")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Nombre d'époques d'entraînement (défaut 10)")
    args = parser.parse_args()

    if args.backend == 'pytorch':
        train_pytorch(epochs=args.epochs)
    elif args.backend == 'tensorflow':
        train_tensorflow()


if __name__ == "__main__":
    main()
