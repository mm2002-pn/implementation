"""
Metriques d'evaluation pour les modeles de reconnaissance emotionnelle.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)


# =============================================================================
# METRIQUES CLASSIFICATION SINGLE-LABEL
# =============================================================================

def compute_accuracy(y_true, y_pred):
    """Calcule l'accuracy."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    return accuracy_score(y_true, y_pred)


def compute_f1(y_true, y_pred, average='weighted'):
    """
    Calcule le F1-score.

    Args:
        y_true: labels reels
        y_pred: labels predits
        average: 'weighted', 'macro', 'micro', ou None pour par classe

    Returns:
        f1 score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    return f1_score(y_true, y_pred, average=average, zero_division=0)


def compute_precision_recall(y_true, y_pred, average='weighted'):
    """Calcule precision et recall."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)

    return precision, recall


def compute_confusion_matrix(y_true, y_pred, labels=None):
    """
    Calcule la matrice de confusion.

    Returns:
        confusion matrix (n_classes, n_classes)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    return confusion_matrix(y_true, y_pred, labels=labels)


def get_classification_report(y_true, y_pred, target_names=None):
    """
    Genere un rapport de classification complet.

    Returns:
        str: rapport formate
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


def compute_all_metrics(y_true, y_pred, target_names=None):
    """
    Calcule toutes les metriques pour classification single-label.

    Returns:
        dict avec accuracy, f1, precision, recall, confusion_matrix
    """
    metrics = {
        'accuracy': compute_accuracy(y_true, y_pred),
        'f1_weighted': compute_f1(y_true, y_pred, average='weighted'),
        'f1_macro': compute_f1(y_true, y_pred, average='macro'),
        'precision': compute_precision_recall(y_true, y_pred)[0],
        'recall': compute_precision_recall(y_true, y_pred)[1],
        'confusion_matrix': compute_confusion_matrix(y_true, y_pred)
    }

    return metrics


# =============================================================================
# METRIQUES CLASSIFICATION MULTI-LABEL (GoEmotions)
# =============================================================================

def compute_multilabel_accuracy(y_true, y_pred, threshold=0.5):
    """
    Calcule l'accuracy pour multi-label.
    Exact match ratio: proportion d'exemples parfaitement predits.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Binariser predictions si probas
    if y_pred.max() <= 1.0 and y_pred.min() >= 0.0:
        y_pred_binary = (y_pred >= threshold).astype(int)
    else:
        y_pred_binary = y_pred

    # Exact match
    exact_match = np.all(y_true == y_pred_binary, axis=1).mean()

    return exact_match


def compute_multilabel_f1(y_true, y_pred, threshold=0.5, average='micro'):
    """
    Calcule le F1-score pour multi-label.

    Args:
        average: 'micro', 'macro', 'weighted', ou 'samples'
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Binariser
    if y_pred.max() <= 1.0 and y_pred.min() >= 0.0:
        y_pred_binary = (y_pred >= threshold).astype(int)
    else:
        y_pred_binary = y_pred

    return f1_score(y_true, y_pred_binary, average=average, zero_division=0)


def compute_multilabel_precision_recall(y_true, y_pred, threshold=0.5, average='micro'):
    """Calcule precision et recall pour multi-label."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Binariser
    if y_pred.max() <= 1.0 and y_pred.min() >= 0.0:
        y_pred_binary = (y_pred >= threshold).astype(int)
    else:
        y_pred_binary = y_pred

    precision = precision_score(y_true, y_pred_binary, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, average=average, zero_division=0)

    return precision, recall


def compute_roc_auc(y_true, y_proba, average='macro'):
    """
    Calcule ROC-AUC pour multi-label.

    Args:
        y_true: labels reels (n, n_classes)
        y_proba: probabilites (n, n_classes)
        average: 'macro', 'micro', 'weighted'
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_proba, torch.Tensor):
        y_proba = y_proba.cpu().numpy()

    try:
        return roc_auc_score(y_true, y_proba, average=average)
    except ValueError:
        # Peut echouer si une classe n'a qu'une seule valeur
        return 0.0


def compute_average_precision(y_true, y_proba, average='macro'):
    """Calcule Average Precision pour multi-label."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_proba, torch.Tensor):
        y_proba = y_proba.cpu().numpy()

    try:
        return average_precision_score(y_true, y_proba, average=average)
    except ValueError:
        return 0.0


def compute_all_multilabel_metrics(y_true, y_proba, threshold=0.5):
    """
    Calcule toutes les metriques pour classification multi-label.

    Args:
        y_true: labels reels (n, n_classes)
        y_proba: probabilites (n, n_classes)
        threshold: seuil pour binarisation

    Returns:
        dict avec toutes les metriques
    """
    metrics = {
        'exact_match': compute_multilabel_accuracy(y_true, y_proba, threshold),
        'f1_micro': compute_multilabel_f1(y_true, y_proba, threshold, 'micro'),
        'f1_macro': compute_multilabel_f1(y_true, y_proba, threshold, 'macro'),
        'f1_samples': compute_multilabel_f1(y_true, y_proba, threshold, 'samples'),
        'precision_micro': compute_multilabel_precision_recall(y_true, y_proba, threshold, 'micro')[0],
        'recall_micro': compute_multilabel_precision_recall(y_true, y_proba, threshold, 'micro')[1],
        'roc_auc': compute_roc_auc(y_true, y_proba),
        'average_precision': compute_average_precision(y_true, y_proba)
    }

    return metrics


# =============================================================================
# UTILITAIRES
# =============================================================================

def print_metrics(metrics, title="Metrics"):
    """Affiche les metriques de maniere formatee."""
    print("="*50)
    print(title)
    print("="*50)

    for key, value in metrics.items():
        if key == 'confusion_matrix':
            print(f"\n{key}:")
            print(value)
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("="*50)


class MetricsTracker:
    """
    Tracker pour suivre les metriques pendant l'entrainement.
    """

    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def update(self, train_loss, val_loss, train_acc=None, val_acc=None,
               train_f1=None, val_f1=None, epoch=0):
        """Met a jour les metriques."""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)

        if train_acc is not None:
            self.history['train_acc'].append(train_acc)
        if val_acc is not None:
            self.history['val_acc'].append(val_acc)
        if train_f1 is not None:
            self.history['train_f1'].append(train_f1)
        if val_f1 is not None:
            self.history['val_f1'].append(val_f1)

        # Tracker best
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

        if val_acc is not None and val_acc > self.best_val_acc:
            self.best_val_acc = val_acc

    def get_best(self):
        """Retourne les meilleures metriques."""
        return {
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }

    def print_epoch(self, epoch, n_epochs):
        """Affiche les metriques de l'epoque courante."""
        train_loss = self.history['train_loss'][-1]
        val_loss = self.history['val_loss'][-1]

        msg = f"Epoch {epoch+1}/{n_epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}"

        if self.history['train_acc']:
            train_acc = self.history['train_acc'][-1]
            val_acc = self.history['val_acc'][-1]
            msg += f", train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}"

        print(msg)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Test des metriques...\n")

    # Test single-label
    print("="*50)
    print("Test Single-Label")
    print("="*50)

    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 1])

    metrics = compute_all_metrics(y_true, y_pred, target_names=['angry', 'sad', 'happy'])
    print_metrics(metrics, "Single-Label Metrics")

    print("\nClassification Report:")
    print(get_classification_report(y_true, y_pred, target_names=['angry', 'sad', 'happy']))

    # Test multi-label
    print("\n" + "="*50)
    print("Test Multi-Label")
    print("="*50)

    y_true_ml = np.array([
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])

    y_proba_ml = np.array([
        [0.9, 0.2, 0.8, 0.1],
        [0.1, 0.9, 0.7, 0.2],
        [0.8, 0.7, 0.3, 0.1],
        [0.2, 0.1, 0.6, 0.8]
    ])

    metrics_ml = compute_all_multilabel_metrics(y_true_ml, y_proba_ml)
    print_metrics(metrics_ml, "Multi-Label Metrics")

    print("\nTest reussi!")
