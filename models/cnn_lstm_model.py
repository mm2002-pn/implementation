"""
CNN-LSTM Model pour la classification des émotions à partir de signaux biométriques (WESAD)

Architecture:
    Input (batch, seq_len, features)
        ↓
    Conv1D layers (extraction de patterns locaux)
        ↓
    LSTM layer (capture des dépendances temporelles)
        ↓
    Fully Connected layers
        ↓
    Output (probabilities pour chaque classe)

Optimisé pour 8GB RAM avec batch_size=8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import CNN_LSTM_CONFIG, WESAD_CONFIG


class CNN_LSTM(nn.Module):
    """
    Modèle CNN-LSTM pour classification de séries temporelles.

    Le CNN extrait des features locales (patterns dans le signal),
    le LSTM capture les dépendances temporelles longues.

    Args:
        n_features: Nombre de features d'entrée (ex: 1 pour EDA seul)
        n_classes: Nombre de classes de sortie (ex: 3 pour baseline/stress/amusement)
        seq_len: Longueur de la séquence (ex: 40 pour 10s à 4Hz)
        config: Dictionnaire de configuration (optionnel)
    """

    def __init__(self, n_features=1, n_classes=3, seq_len=40, config=None):
        super(CNN_LSTM, self).__init__()

        # Configuration
        if config is None:
            config = CNN_LSTM_CONFIG

        self.n_features = n_features
        self.n_classes = n_classes
        self.seq_len = seq_len

        # Paramètres du modèle
        conv_filters = config.get('conv_filters', [64, 128])
        kernel_size = config.get('kernel_size', 3)
        lstm_hidden = config.get('lstm_hidden', 64)
        lstm_layers = config.get('lstm_layers', 1)
        dropout = config.get('dropout', 0.3)
        fc_hidden = config.get('fc_hidden', 32)

        # =====================================================================
        # Couches CNN (extraction de features locales)
        # =====================================================================
        # Conv1D attend (batch, channels, seq_len)
        # On transpose l'entrée de (batch, seq_len, features) vers (batch, features, seq_len)

        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=conv_filters[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2  # 'same' padding
        )
        self.bn1 = nn.BatchNorm1d(conv_filters[0])

        self.conv2 = nn.Conv1d(
            in_channels=conv_filters[0],
            out_channels=conv_filters[1],
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(conv_filters[1])

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout_cnn = nn.Dropout(dropout)

        # Calculer la taille après les convolutions
        # Après 2 couches de pooling: seq_len // 4
        self.seq_len_after_conv = seq_len // 4

        # =====================================================================
        # Couche LSTM (capture des dépendances temporelles)
        # =====================================================================
        self.lstm = nn.LSTM(
            input_size=conv_filters[1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )

        # =====================================================================
        # Couches Fully Connected (classification)
        # =====================================================================
        self.fc1 = nn.Linear(lstm_hidden, fc_hidden)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, n_classes)

        # Initialisation des poids
        self._init_weights()

    def _init_weights(self):
        """Initialisation Xavier/He pour une meilleure convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor de shape (batch, seq_len, n_features)
               Ex: (8, 40, 1) pour batch=8, 10 secondes à 4Hz, 1 feature (EDA)

        Returns:
            logits: Tensor de shape (batch, n_classes)
        """
        # x: (batch, seq_len, features) -> (batch, features, seq_len) pour Conv1D
        x = x.permute(0, 2, 1)

        # Bloc Conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Bloc Conv2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout_cnn(x)

        # x: (batch, channels, seq_len_reduced) -> (batch, seq_len_reduced, channels) pour LSTM
        x = x.permute(0, 2, 1)

        # LSTM
        # output: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Prendre le dernier hidden state
        x = h_n[-1]  # (batch, hidden_size)

        # Fully connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x

    def predict(self, x):
        """
        Prédit les classes (pas seulement les logits).

        Args:
            x: Tensor de shape (batch, seq_len, n_features)

        Returns:
            predictions: Tensor de shape (batch,) avec les classes prédites
            probabilities: Tensor de shape (batch, n_classes) avec les probabilités
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        return predictions, probabilities

    def count_parameters(self):
        """Compte le nombre de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Affiche un résumé du modèle."""
        print("="*60)
        print("CNN-LSTM Model Summary")
        print("="*60)
        print(f"Input shape: (batch, {self.seq_len}, {self.n_features})")
        print(f"Output shape: (batch, {self.n_classes})")
        print(f"Total parameters: {self.count_parameters():,}")
        print("="*60)
        print(self)
        print("="*60)


class CNN_LSTM_Lite(nn.Module):
    """
    Version allégée du CNN-LSTM pour ressources limitées.
    Moins de filtres, LSTM plus petit.
    """

    def __init__(self, n_features=1, n_classes=3, seq_len=40):
        super(CNN_LSTM_Lite, self).__init__()

        # CNN simplifié
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        # LSTM compact
        self.lstm = nn.LSTM(64, 32, batch_first=True)

        # Classification
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        # x: (batch, seq, features) -> (batch, features, seq)
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)

        # (batch, channels, seq) -> (batch, seq, channels)
        x = x.permute(0, 2, 1)

        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]

        return self.fc(x)


def create_model(n_features=1, n_classes=3, seq_len=40, lite=False):
    """
    Factory function pour créer le modèle.

    Args:
        n_features: Nombre de features d'entrée
        n_classes: Nombre de classes
        seq_len: Longueur de séquence
        lite: Si True, utilise la version allégée

    Returns:
        model: Instance du modèle
    """
    if lite:
        return CNN_LSTM_Lite(n_features, n_classes, seq_len)
    else:
        return CNN_LSTM(n_features, n_classes, seq_len)


def test_model():
    """Test rapide du modèle."""
    print("Test du modèle CNN-LSTM...")

    # Créer modèle
    model = CNN_LSTM(n_features=1, n_classes=3, seq_len=40)
    model.summary()

    # Données de test
    batch_size = 8
    x = torch.randn(batch_size, 40, 1)  # (batch, seq_len, features)

    # Forward pass
    print(f"\nInput shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")

    # Prédiction
    preds, probs = model.predict(x)
    print(f"Predictions: {preds}")
    print(f"Probabilities shape: {probs.shape}")

    print("\nTest réussi!")


if __name__ == "__main__":
    test_model()
