import sys
import os

# Ajouter le dossier models au chemin pour pouvoir importer train.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from train import main as train_main  # importe la fonction main de train.py

if __name__ == "__main__":
    train_main()


  