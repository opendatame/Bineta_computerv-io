import os
from PIL import Image

# Correction du chemin vers le dossier contenant training et testing
base_dir = 'breast_cancer/breast_cancer/'
print(f"Vérification du dossier de base : {base_dir}")

sub_dirs = ['training', 'testing']
for sub_dir in sub_dirs:
    train_dir = os.path.join(base_dir, sub_dir)
    print(f"-> Exploration de {train_dir}")
    
    if not os.path.exists(train_dir):
        print(f"Erreur : le dossier {train_dir} n'existe pas.")
        continue
    
    for subfolder in os.listdir(train_dir):
        subfolder_path = os.path.join(train_dir, subfolder)
        print(f"  Sous-dossier trouvé : {subfolder}")
        
        if os.path.isdir(subfolder_path):
            print(f"    Classe : {subfolder}")
            found_images = False
            for img_file in os.listdir(subfolder_path):
                if img_file.lower().endswith(('jpg', 'jpeg', 'png')):
                    found_images = True
                    img_path = os.path.join(subfolder_path, img_file)
                    img = Image.open(img_path)
                    print(f"      Image : {img_file} - Taille : {img.size}")
            if not found_images:
                print(f"      Aucun fichier image dans {subfolder}")

print("Vérification terminée.")
