"""
BERT/DistilBERT Model pour la classification multi-label des émotions (GoEmotions)

Architecture:
    Input (texte brut)
        ↓
    Tokenization (DistilBERT tokenizer)
        ↓
    DistilBERT Encoder (pré-entraîné)
        ↓
    [CLS] token embedding
        ↓
    Classification Head
        ↓
    Output (28 émotions, multi-label avec sigmoid)

Utilise DistilBERT (40% plus léger que BERT) pour 8GB RAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import GOEMOTIONS_CONFIG

# Flag pour vérifier si transformers est installé
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import (
        DistilBertModel,
        DistilBertTokenizer,
        DistilBertForSequenceClassification,
        AutoModel,
        AutoTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers not installed. Install with: pip install transformers")


class BertEmotionClassifier(nn.Module):
    """
    Classificateur d'émotions basé sur DistilBERT.

    Utilise DistilBERT pré-entraîné avec une tête de classification personnalisée
    pour la classification multi-label des émotions.

    Args:
        model_name: Nom du modèle HuggingFace (défaut: distilbert-base-uncased)
        num_labels: Nombre d'émotions (défaut: 28 pour GoEmotions)
        dropout: Taux de dropout
        freeze_bert: Si True, gèle les poids de BERT (fine-tuning partiel)
    """

    def __init__(
        self,
        model_name=None,
        num_labels=28,
        dropout=0.3,
        freeze_bert=False
    ):
        super(BertEmotionClassifier, self).__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required. Install with: pip install transformers")

        # Configuration
        if model_name is None:
            model_name = GOEMOTIONS_CONFIG.get('model_name', 'distilbert-base-uncased')

        self.model_name = model_name
        self.num_labels = num_labels

        # =====================================================================
        # Charger DistilBERT pré-entraîné
        # =====================================================================
        print(f"Loading {model_name}...")
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

        # Taille du hidden state de BERT
        self.hidden_size = self.bert.config.hidden_size  # 768 pour distilbert-base

        # Geler les poids de BERT si demandé (économise mémoire GPU)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("BERT weights frozen")

        # =====================================================================
        # Tête de classification personnalisée (améliorée)
        # =====================================================================
        self.dropout = nn.Dropout(dropout)

        # Couche intermédiaire pour plus de capacité
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout / 2)  # Dropout supplémentaire
        self.fc2 = nn.Linear(self.hidden_size // 2, self.hidden_size // 4)  # Couche supplémentaire
        self.dropout3 = nn.Dropout(dropout / 2)
        self.classifier = nn.Linear(self.hidden_size // 4, num_labels)
        self.bn1 = nn.LayerNorm(self.hidden_size // 2)

        # Couche de sortie (multi-label)
        self.classifier = nn.Linear(self.hidden_size // 2, num_labels)

        # Initialisation des nouvelles couches
        self._init_weights()

    def _init_weights(self):
        """Initialisation des couches ajoutées."""
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass.

        Args:
            input_ids: Tensor (batch, seq_len) - IDs des tokens
            attention_mask: Tensor (batch, seq_len) - Masque d'attention

        Returns:
            logits: Tensor (batch, num_labels) - Scores avant sigmoid
        """
        # BERT forward
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Prendre le [CLS] token (premier token)
        # last_hidden_state: (batch, seq_len, hidden_size)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)

        # Classification head
        x = self.dropout(cls_embedding)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.gelu(x)  # GELU comme dans BERT original
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits

    def predict(self, texts, threshold=0.5, device='cpu'):
        """
        Prédit les émotions pour une liste de textes.

        Args:
            texts: Liste de strings ou string unique
            threshold: Seuil pour la classification multi-label
            device: 'cpu' ou 'cuda'

        Returns:
            predictions: Liste de listes d'émotions prédites
            probabilities: Array (n_texts, num_labels)
        """
        if isinstance(texts, str):
            texts = [texts]

        self.eval()
        self.to(device)

        # Tokenization
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=GOEMOTIONS_CONFIG.get('max_length', 128),
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # Prédiction
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probabilities = torch.sigmoid(logits)

        probabilities = probabilities.cpu().numpy()

        # Convertir en labels
        emotion_names = GOEMOTIONS_CONFIG.get('emotions', [])
        predictions = []

        for probs in probabilities:
            predicted_emotions = []
            for i, prob in enumerate(probs):
                if prob >= threshold:
                    if i < len(emotion_names):
                        predicted_emotions.append(emotion_names[i])
                    else:
                        predicted_emotions.append(f"emotion_{i}")
            predictions.append(predicted_emotions)

        return predictions, probabilities

    def tokenize(self, texts, max_length=None):
        """
        Tokenize une liste de textes.

        Args:
            texts: Liste de strings
            max_length: Longueur maximale (défaut depuis config)

        Returns:
            input_ids: Tensor (batch, seq_len)
            attention_mask: Tensor (batch, seq_len)
        """
        if max_length is None:
            max_length = GOEMOTIONS_CONFIG.get('max_length', 128)

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        return encoded['input_ids'], encoded['attention_mask']

    def count_parameters(self, trainable_only=True):
        """Compte le nombre de paramètres."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def summary(self):
        """Affiche un résumé du modèle."""
        print("="*60)
        print("BERT Emotion Classifier Summary")
        print("="*60)
        print(f"Base model: {self.model_name}")
        print(f"Hidden size: {self.hidden_size}")
        print(f"Num labels: {self.num_labels}")
        print(f"Total parameters: {self.count_parameters(trainable_only=False):,}")
        print(f"Trainable parameters: {self.count_parameters(trainable_only=True):,}")
        print("="*60)


class BertEmotionClassifierSimple(nn.Module):
    """
    Version simplifiée utilisant directement HuggingFace.
    Plus facile à utiliser mais moins personnalisable.
    """

    def __init__(self, model_name=None, num_labels=28):
        super(BertEmotionClassifierSimple, self).__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers required")

        if model_name is None:
            model_name = GOEMOTIONS_CONFIG.get('model_name', 'distilbert-base-uncased')

        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def predict(self, texts, threshold=0.5):
        if isinstance(texts, str):
            texts = [texts]

        self.eval()
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = self.model(**encoded)
            probs = torch.sigmoid(outputs.logits)

        return probs.numpy()


class TextClassifierWithoutBert(nn.Module):
    """
    Classificateur de texte simple sans BERT.
    Utilise embeddings + LSTM. Plus léger mais moins performant.
    À utiliser si BERT est trop lourd.
    """

    def __init__(self, vocab_size=30000, embed_dim=128, hidden_dim=64, num_labels=28):
        super(TextClassifierWithoutBert, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch, seq_len)
        x = self.embedding(input_ids)  # (batch, seq_len, embed_dim)

        if attention_mask is not None:
            # Masquer les positions de padding
            x = x * attention_mask.unsqueeze(-1)

        _, (h_n, _) = self.lstm(x)

        # Concaténer forward et backward
        x = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden_dim * 2)

        x = self.dropout(x)
        logits = self.fc(x)

        return logits


def create_model(use_bert=True, num_labels=28, freeze_bert=False):
    """
    Factory function pour créer le modèle.

    Args:
        use_bert: Si True, utilise BERT. Sinon, LSTM simple.
        num_labels: Nombre d'émotions
        freeze_bert: Si True, gèle les poids de BERT

    Returns:
        model: Instance du modèle
    """
    if use_bert and TRANSFORMERS_AVAILABLE:
        return BertEmotionClassifier(num_labels=num_labels, freeze_bert=freeze_bert)
    else:
        print("Using simple LSTM classifier (BERT not available or disabled)")
        return TextClassifierWithoutBert(num_labels=num_labels)


def test_model():
    """Test rapide du modèle."""
    print("Test du modèle BERT Emotion Classifier...")

    if not TRANSFORMERS_AVAILABLE:
        print("transformers not installed. Testing simple LSTM model...")
        model = TextClassifierWithoutBert(num_labels=28)
        x = torch.randint(0, 1000, (4, 64))  # (batch, seq_len)
        output = model(x)
        print(f"Output shape: {output.shape}")
        return

    # Créer modèle
    model = BertEmotionClassifier(num_labels=28, freeze_bert=True)
    model.summary()

    # Test avec textes
    texts = [
        "I'm so happy today!",
        "This is really frustrating and annoying.",
        "I love spending time with my family.",
        "I'm nervous about the exam tomorrow."
    ]

    print(f"\nTest texts: {texts}")

    # Prédiction
    predictions, probabilities = model.predict(texts, threshold=0.3)

    print("\nPredictions:")
    for text, preds, probs in zip(texts, predictions, probabilities):
        print(f"  '{text[:40]}...'")
        print(f"    Emotions: {preds}")
        top_3 = np.argsort(probs)[-3:][::-1]
        emotions = GOEMOTIONS_CONFIG.get('emotions', [])
        print(f"    Top 3: {[(emotions[i], f'{probs[i]:.2f}') for i in top_3]}")

    print("\nTest réussi!")


if __name__ == "__main__":
    test_model()
