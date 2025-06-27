import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
import os

def load_amhcd_dataset(data_dir, img_size=(32, 32)):
    """
    Charge les images de la base AMHCD depuis un répertoire organisé par classes.
    
    Args:
        data_dir (str): Chemin vers le dossier contenant les sous-dossiers de classes.
        img_size (tuple): Taille cible pour redimensionner les images.
        
    Returns:
        X (np.ndarray): Tableau d'images (N, H, W, 1).
        y (np.ndarray): Tableau de labels (N,).
    """
    data = []
    labels = []

    class_names = sorted(os.listdir(data_dir))
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            try:
                img = Image.open(img_path).convert('L')  # Convertir en niveaux de gris
                img = img.resize(img_size)              # Redimensionner
                img_array = np.array(img) / 255.0       # Normaliser [0, 1]
                data.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Erreur lors du chargement de {img_path}: {e}")

    X = np.array(data).reshape(-1, img_size[0], img_size[1], 1)
    y = np.array(labels)

    return X, y


def prepare_dataset(data_dir, img_size=(32, 32), test_size=0.2, val_size=0.1, random_state=42):
    """
    Charge et divise les données en ensembles d'entraînement, validation et test.
    
    Args:
        data_dir (str): Chemin vers les données.
        img_size (tuple): Taille des images.
        test_size (float): Proportion pour le test.
        val_size (float): Proportion pour la validation.
        random_state (int): Graine aléatoire.
        
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    X, y = load_amhcd_dataset(data_dir, img_size)

    # Diviser en entraînement + validation vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Diviser entre entraînement et validation
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, stratify=y_train_val, random_state=random_state
    )

    return X_train, y_train, X_val, y_val, X_test, y_test