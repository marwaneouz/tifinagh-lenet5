import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def compute_accuracy(y_true, y_pred):
    """
    Calcule l'accuracy.
    
    Args:
        y_true (np.ndarray): Labels vrais (classes entières).
        y_pred (np.ndarray): Probabilités ou logits prédits.
        
    Returns:
        float: Accuracy.
    """
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return accuracy_score(y_true, y_pred)


def compute_confusion_matrix(y_true, y_pred, num_classes=33):
    """
    Calcule la matrice de confusion.
    
    Args:
        y_true (np.ndarray): Labels vrais.
        y_pred (np.ndarray): Prédiction du modèle.
        num_classes (int): Nombre total de classes.
        
    Returns:
        np.ndarray: Matrice de confusion (num_classes x num_classes).
    """
    return confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))


def classification_report_str(y_true, y_pred, target_names=None):
    """
    Génère un rapport de classification.
    
    Args:
        y_true (np.ndarray): Labels vrais.
        y_pred (np.ndarray): Prédiction du modèle.
        target_names (list): Noms des classes.
        
    Returns:
        str: Rapport texte.
    """
    return classification_report(y_true, y_pred, target_names=target_names)