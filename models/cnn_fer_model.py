"""
CNN Model pour la classification des expressions faciales (FER2013)

Architecture:
    Input (batch, 1, 48, 48) - Image grayscale
        ↓
    Conv2D blocks avec BatchNorm et MaxPool
        ↓
    Flatten
        ↓
    Fully Connected layers avec Dropout
        ↓
    Output (7 classes d'émotions)

Optimisé pour 8GB RAM avec batch_size=8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import CNN_FER_CONFIG, FER2013_CONFIG


class CNN_FER(nn.Module):
    """
    CNN pour la reconnaissance d'expressions faciales.

    Architecture classique avec blocs Conv-BatchNorm-ReLU-Pool
    suivis de couches fully connected.

    Args:
        n_classes: Nombre de classes (défaut: 7 pour FER2013)
        config: Dictionnaire de configuration (optionnel)
    """

    def __init__(self, n_classes=7, config=None):
        super(CNN_FER, self).__init__()

        # Configuration
        if config is None:
            config = CNN_FER_CONFIG

        self.n_classes = n_classes

        # Paramètres
        conv_filters = config.get('conv_filters', [32, 64, 128])
        kernel_size = config.get('kernel_size', 3)
        dropout = config.get('dropout', 0.4)
        fc_hidden = config.get('fc_hidden', 512)

        # =====================================================================
        # Bloc Convolutionnel 1
        # Input: (batch, 1, 48, 48) -> Output: (batch, 32, 24, 24)
        # =====================================================================
        self.conv1 = nn.Conv2d(1, conv_filters[0], kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_filters[0])

        # =====================================================================
        # Bloc Convolutionnel 2
        # Input: (batch, 32, 24, 24) -> Output: (batch, 64, 12, 12)
        # =====================================================================
        self.conv2 = nn.Conv2d(conv_filters[0], conv_filters[1], kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_filters[1])

        # =====================================================================
        # Bloc Convolutionnel 3
        # Input: (batch, 64, 12, 12) -> Output: (batch, 128, 6, 6)
        # =====================================================================
        self.conv3 = nn.Conv2d(conv_filters[1], conv_filters[2], kernel_size=kernel_size, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_filters[2])

        # =====================================================================
        # Bloc Convolutionnel 4 (optionnel pour plus de profondeur)
        # Input: (batch, 128, 6, 6) -> Output: (batch, 128, 3, 3)
        # =====================================================================
        self.conv4 = nn.Conv2d(conv_filters[2], conv_filters[2], kernel_size=kernel_size, padding=1)
        self.bn4 = nn.BatchNorm2d(conv_filters[2])

        # Pooling et Dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_conv = nn.Dropout2d(dropout / 2)  # Dropout spatial plus léger

        # =====================================================================
        # Couches Fully Connected
        # Après 4 poolings: 48 -> 24 -> 12 -> 6 -> 3
        # Flatten: 128 * 3 * 3 = 1152
        # =====================================================================
        self.flatten_size = conv_filters[2] * 3 * 3  # 128 * 3 * 3 = 1152

        self.fc1 = nn.Linear(self.flatten_size, fc_hidden)
        self.bn_fc1 = nn.BatchNorm1d(fc_hidden)
        self.dropout_fc = nn.Dropout(dropout)

        self.fc2 = nn.Linear(fc_hidden, fc_hidden // 2)
        self.bn_fc2 = nn.BatchNorm1d(fc_hidden // 2)

        self.fc3 = nn.Linear(fc_hidden // 2, n_classes)

        # Initialisation des poids
        self._init_weights()

    def _init_weights(self):
        """Initialisation He pour les couches ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor de shape (batch, 48, 48) ou (batch, 1, 48, 48)
               Images grayscale normalisées [0, 1]

        Returns:
            logits: Tensor de shape (batch, n_classes)
        """
        # Ajouter dimension channel si nécessaire
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 48, 48) -> (batch, 1, 48, 48)

        # Bloc 1: (batch, 1, 48, 48) -> (batch, 32, 24, 24)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Bloc 2: (batch, 32, 24, 24) -> (batch, 64, 12, 12)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Bloc 3: (batch, 64, 12, 12) -> (batch, 128, 6, 6)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Bloc 4: (batch, 128, 6, 6) -> (batch, 128, 3, 3)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout_conv(x)

        # Flatten: (batch, 128, 3, 3) -> (batch, 1152)
        x = x.view(x.size(0), -1)

        # FC1
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)

        # FC2
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout_fc(x)

        # Output
        x = self.fc3(x)

        return x

    def predict(self, x):
        """
        Prédit les classes.

        Args:
            x: Tensor de shape (batch, 48, 48) ou (batch, 1, 48, 48)

        Returns:
            predictions: Tensor de shape (batch,)
            probabilities: Tensor de shape (batch, n_classes)
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
        print("CNN-FER Model Summary")
        print("="*60)
        print(f"Input shape: (batch, 1, 48, 48)")
        print(f"Output shape: (batch, {self.n_classes})")
        print(f"Total parameters: {self.count_parameters():,}")
        print("="*60)
        print(self)
        print("="*60)


class CNN_FER_Lite(nn.Module):
    """
    Version allégée du CNN pour ressources très limitées.
    Moins de filtres, moins de couches FC.
    """

    def __init__(self, n_classes=7):
        super(CNN_FER_Lite, self).__init__()

        # Convolutions simplifiées
        self.features = nn.Sequential(
            # Bloc 1: 48 -> 24
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Bloc 2: 24 -> 12
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Bloc 3: 12 -> 6
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNN_FER_Deep(nn.Module):
    """
    Version plus profonde du CNN avec skip connections (ResNet-like).
    Pour de meilleures performances si assez de mémoire.
    """

    def __init__(self, n_classes=7):
        super(CNN_FER_Deep, self).__init__()

        # Initial conv
        self.conv_init = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Residual blocks
        self.res1 = self._make_residual_block(32, 32)
        self.res2 = self._make_residual_block(32, 64, downsample=True)
        self.res3 = self._make_residual_block(64, 128, downsample=True)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.fc = nn.Linear(128, n_classes)

    def _make_residual_block(self, in_ch, out_ch, downsample=False):
        stride = 2 if downsample else 1

        layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )

        # Skip connection
        if downsample or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.skip = nn.Identity()

        return layers

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv_init(x)

        # Residual blocks (simplified without skip for now)
        x = F.relu(self.res1(x))
        x = F.relu(self.res2(x))
        x = F.relu(self.res3(x))

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def create_model(n_classes=7, variant='standard'):
    """
    Factory function pour créer le modèle.

    Args:
        n_classes: Nombre de classes
        variant: 'lite', 'standard', ou 'deep'

    Returns:
        model: Instance du modèle
    """
    if variant == 'lite':
        return CNN_FER_Lite(n_classes)
    elif variant == 'deep':
        return CNN_FER_Deep(n_classes)
    else:
        return CNN_FER(n_classes)


def test_model():
    """Test rapide du modèle."""
    print("Test du modèle CNN-FER...")

    # Créer modèle
    model = CNN_FER(n_classes=7)
    model.summary()

    # Données de test (images 48x48 normalisées)
    batch_size = 8
    x = torch.randn(batch_size, 48, 48)  # Sans dimension channel

    # Forward pass
    print(f"\nInput shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")

    # Prédiction
    preds, probs = model.predict(x)
    print(f"Predictions: {preds}")

    # Labels
    labels = FER2013_CONFIG['labels']
    print(f"\nPredicted emotions: {[labels[p.item()] for p in preds]}")

    print("\nTest réussi!")


if __name__ == "__main__":
    test_model()
