"""
Fusion Multimodale pour la reconnaissance emotionnelle.

Combine les predictions de 3 modalites:
- Biometrique (WESAD): CNN-LSTM -> 3 classes
- Visuel (FER2013): CNN -> 7 classes
- Texte (GoEmotions): BERT -> 28 classes multi-label

Strategies de fusion implementees:
1. Moyenne simple
2. Moyenne ponderee
3. Vote majoritaire
4. MLP Fusion (apprentissage)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import FUSION_CONFIG, EMOTION_MAPPING


# =============================================================================
# MAPPING DES EMOTIONS
# =============================================================================

def map_to_common_emotions(predictions, source, emotion_mapping=None):
    """
    Mappe les predictions d'un modele vers les emotions communes.

    Args:
        predictions: array de probabilites (batch, n_classes_source)
        source: 'wesad', 'fer2013', ou 'goemotions'
        emotion_mapping: dictionnaire de mapping (defaut: EMOTION_MAPPING)

    Returns:
        array de probabilites (batch, n_common_emotions)
    """
    if emotion_mapping is None:
        emotion_mapping = EMOTION_MAPPING

    common_emotions = list(emotion_mapping.keys())
    n_common = len(common_emotions)
    batch_size = len(predictions)

    # Labels par source
    source_labels = {
        'wesad': ['baseline', 'stress', 'amusement'],
        'fer2013': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
        'goemotions': [
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness", "surprise",
            "neutral"
        ]
    }

    labels = source_labels[source]
    mapped = np.zeros((batch_size, n_common))

    for i, common_emotion in enumerate(common_emotions):
        source_emotions = emotion_mapping[common_emotion].get(source, [])

        for source_emotion in source_emotions:
            if source_emotion in labels:
                idx = labels.index(source_emotion)
                if idx < predictions.shape[1]:
                    mapped[:, i] += predictions[:, idx]

        # Normaliser si plusieurs emotions mappees
        if len(source_emotions) > 1:
            mapped[:, i] /= len(source_emotions)

    return mapped


# =============================================================================
# STRATEGIES DE FUSION
# =============================================================================

def average_fusion(predictions_list, weights=None):
    """
    Fusion par moyenne (simple ou ponderee).

    Args:
        predictions_list: liste de arrays (batch, n_classes) pour chaque modalite
        weights: liste de poids (optionnel)

    Returns:
        array fusionne (batch, n_classes)
    """
    if weights is None:
        weights = [1.0 / len(predictions_list)] * len(predictions_list)

    fused = np.zeros_like(predictions_list[0])

    for pred, weight in zip(predictions_list, weights):
        fused += weight * pred

    # Normaliser
    fused /= np.sum(weights)

    return fused


def weighted_average_fusion(predictions_list, weights=None):
    """
    Fusion par moyenne ponderee selon la confiance.
    Les modalites avec predictions plus confiantes ont plus de poids.
    """
    if weights is None:
        # Calculer poids selon entropie (moins d'entropie = plus confiant)
        weights = []
        for pred in predictions_list:
            entropy = -np.sum(pred * np.log(pred + 1e-10), axis=1, keepdims=True)
            confidence = 1.0 / (entropy + 1e-10)
            weights.append(confidence.mean())

        weights = np.array(weights)
        weights = weights / weights.sum()

    return average_fusion(predictions_list, weights)


def voting_fusion(predictions_list, weights=None):
    """
    Fusion par vote majoritaire.
    Chaque modalite vote pour la classe avec la plus haute probabilite.
    """
    batch_size = predictions_list[0].shape[0]
    n_classes = predictions_list[0].shape[1]

    # Votes
    votes = np.zeros((batch_size, n_classes))

    for pred in predictions_list:
        predicted_class = np.argmax(pred, axis=1)
        for i, cls in enumerate(predicted_class):
            votes[i, cls] += 1

    # Normaliser en probabilites
    votes = votes / len(predictions_list)

    return votes


def max_fusion(predictions_list):
    """
    Fusion par maximum: prend la prediction la plus confiante.
    """
    stacked = np.stack(predictions_list, axis=0)  # (n_modalities, batch, n_classes)
    return np.max(stacked, axis=0)


# =============================================================================
# MLP FUSION (APPRENTISSAGE)
# =============================================================================

class MLPFusion(nn.Module):
    """
    Reseau de neurones pour apprendre la fusion optimale.

    Concatene les predictions des modalites et apprend a les combiner.
    """

    def __init__(self, input_sizes, n_output_classes, hidden_dim=64, dropout=0.3):
        """
        Args:
            input_sizes: liste des tailles d'entree pour chaque modalite
            n_output_classes: nombre de classes de sortie
            hidden_dim: dimension de la couche cachee
            dropout: taux de dropout
        """
        super(MLPFusion, self).__init__()

        total_input = sum(input_sizes)

        self.fusion = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_output_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, *predictions):
        """
        Args:
            predictions: tuple de tensors, un par modalite

        Returns:
            logits (batch, n_output_classes)
        """
        # Concatener toutes les predictions
        x = torch.cat(predictions, dim=1)
        return self.fusion(x)

    def predict(self, *predictions):
        """Retourne les probabilites."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(*predictions)
            return F.softmax(logits, dim=1)


class AttentionFusion(nn.Module):
    """
    Fusion par attention: apprend a ponderer dynamiquement les modalites.
    """

    def __init__(self, input_sizes, n_output_classes, hidden_dim=64):
        super(AttentionFusion, self).__init__()

        self.n_modalities = len(input_sizes)

        # Projeter chaque modalite dans le meme espace
        self.projections = nn.ModuleList([
            nn.Linear(size, hidden_dim) for size in input_sizes
        ])

        # Attention weights
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * self.n_modalities, self.n_modalities),
            nn.Softmax(dim=1)
        )

        # Classifieur final
        self.classifier = nn.Linear(hidden_dim, n_output_classes)

    def forward(self, *predictions):
        # Projeter
        projected = [proj(pred) for proj, pred in zip(self.projections, predictions)]
        projected = torch.stack(projected, dim=1)  # (batch, n_modalities, hidden)

        # Attention
        concat = projected.view(projected.size(0), -1)  # (batch, n_modalities * hidden)
        attention_weights = self.attention(concat)  # (batch, n_modalities)

        # Weighted sum
        weighted = projected * attention_weights.unsqueeze(-1)  # (batch, n_modalities, hidden)
        fused = weighted.sum(dim=1)  # (batch, hidden)

        return self.classifier(fused)


# =============================================================================
# CLASSE PRINCIPALE DE FUSION
# =============================================================================

class MultimodalFusion:
    """
    Classe principale pour la fusion multimodale.
    """

    def __init__(self, method='weighted_average', weights=None):
        """
        Args:
            method: 'average', 'weighted_average', 'voting', 'max', 'mlp', 'attention'
            weights: poids pour les modalites (optionnel)
        """
        self.method = method
        self.weights = weights
        self.model = None  # Pour MLP/Attention

    def fuse(self, bio_pred=None, visual_pred=None, text_pred=None,
             map_to_common=True):
        """
        Fusionne les predictions des 3 modalites.

        Args:
            bio_pred: predictions biometriques (batch, 3)
            visual_pred: predictions visuelles (batch, 7)
            text_pred: predictions texte (batch, 28)
            map_to_common: mapper vers emotions communes avant fusion

        Returns:
            predictions fusionnees
        """
        predictions_list = []

        if bio_pred is not None:
            if map_to_common:
                bio_pred = map_to_common_emotions(bio_pred, 'wesad')
            predictions_list.append(bio_pred)

        if visual_pred is not None:
            if map_to_common:
                visual_pred = map_to_common_emotions(visual_pred, 'fer2013')
            predictions_list.append(visual_pred)

        if text_pred is not None:
            if map_to_common:
                text_pred = map_to_common_emotions(text_pred, 'goemotions')
            predictions_list.append(text_pred)

        if len(predictions_list) == 0:
            raise ValueError("Au moins une modalite doit etre fournie")

        if len(predictions_list) == 1:
            return predictions_list[0]

        # Appliquer la strategie de fusion
        if self.method == 'average':
            return average_fusion(predictions_list)
        elif self.method == 'weighted_average':
            return weighted_average_fusion(predictions_list, self.weights)
        elif self.method == 'voting':
            return voting_fusion(predictions_list)
        elif self.method == 'max':
            return max_fusion(predictions_list)
        else:
            raise ValueError(f"Methode inconnue: {self.method}")

    def get_final_prediction(self, fused_probs):
        """
        Retourne la classe predite et sa probabilite.
        """
        predicted_class = np.argmax(fused_probs, axis=1)
        confidence = np.max(fused_probs, axis=1)

        common_emotions = list(EMOTION_MAPPING.keys())
        predicted_emotions = [common_emotions[i] for i in predicted_class]

        return predicted_emotions, confidence


# =============================================================================
# TEST
# =============================================================================

def test_fusion():
    """Test des strategies de fusion."""
    print("Test des strategies de fusion...\n")

    # Simuler des predictions
    batch_size = 4

    # Bio: 3 classes (baseline, stress, amusement)
    bio_pred = np.random.softmax(np.random.randn(batch_size, 3), axis=1)

    # Visual: 7 classes
    visual_pred = np.random.softmax(np.random.randn(batch_size, 7), axis=1)

    # Text: 28 classes
    text_pred = np.random.softmax(np.random.randn(batch_size, 28), axis=1)

    print("Predictions originales:")
    print(f"  Bio shape: {bio_pred.shape}")
    print(f"  Visual shape: {visual_pred.shape}")
    print(f"  Text shape: {text_pred.shape}")

    # Mapper vers emotions communes
    bio_mapped = map_to_common_emotions(bio_pred, 'wesad')
    visual_mapped = map_to_common_emotions(visual_pred, 'fer2013')
    text_mapped = map_to_common_emotions(text_pred, 'goemotions')

    print(f"\nApres mapping vers emotions communes:")
    print(f"  Bio mapped shape: {bio_mapped.shape}")
    print(f"  Visual mapped shape: {visual_mapped.shape}")
    print(f"  Text mapped shape: {text_mapped.shape}")

    # Test des strategies
    for method in ['average', 'weighted_average', 'voting', 'max']:
        fusion = MultimodalFusion(method=method)
        fused = fusion.fuse(bio_pred, visual_pred, text_pred, map_to_common=True)
        emotions, conf = fusion.get_final_prediction(fused)

        print(f"\n{method.upper()} fusion:")
        print(f"  Predictions: {emotions}")
        print(f"  Confidence: {conf}")

    print("\nTest reussi!")


if __name__ == "__main__":
    # Fix pour softmax numpy
    def softmax(x, axis=None):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)
    np.random.softmax = softmax

    test_fusion()
