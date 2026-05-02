"""
Configuration centrale du projet de reconnaissance emotionnelle multimodale.
"""

import os
from pathlib import Path

# =============================================================================
# CHEMINS
# =============================================================================

# Racine du projet
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT.parent

# Datasets
WESAD_PATH = DATASET_ROOT / "WESAD" / "WESAD"
FER2013_PATH = DATASET_ROOT / "challenges-in-representation-learning-facial-expression-recognition-challenge" / "fer2013" / "fer2013" / "fer2013.csv"
GOEMOTIONS_PATH = DATASET_ROOT / "goemotions"

# Outputs
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"
PROCESSED_DIR = OUTPUT_DIR / "processed"

# Creer les dossiers si necessaire
for d in [MODELS_DIR, FIGURES_DIR, LOGS_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HYPERPARAMETRES GENERAUX
# =============================================================================

# Processing - OPTIMISE POUR 8GB RAM
CHUNK_SIZE = 500            # Petits chunks pour economiser RAM
RANDOM_SEED = 42

# Training - OPTIMISE POUR 8GB RAM
BATCH_SIZE = 8              # Petit batch pour 8GB RAM
LEARNING_RATE = 1e-4
EPOCHS = 20
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# =============================================================================
# CONFIGURATION WESAD (Biometrique)
# =============================================================================

WESAD_CONFIG = {
    "subjects": ["S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10",
                 "S11", "S13", "S14", "S15", "S16", "S17"],
    "labels": {
        0: "not_defined",
        1: "baseline",
        2: "stress",
        3: "amusement",
        4: "meditation"
    },
    "target_labels": [1, 2, 3],  # On garde baseline, stress, amusement
    "e4_signals": ["ACC", "BVP", "EDA", "HR", "TEMP"],
    "e4_sample_rates": {
        "ACC": 32,
        "BVP": 64,
        "EDA": 4,
        "HR": 1,
        "TEMP": 4
    },
    "window_size_sec": 60,      # Fenetre de 60 secondes
    "overlap": 0.5,             # 50% overlap entre fenetres
}

# =============================================================================
# CONFIGURATION FER2013 (Images faciales)
# =============================================================================

FER2013_CONFIG = {
    "image_size": 48,
    "num_classes": 7,
    "labels": {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "sad",
        5: "surprise",
        6: "neutral"
    },
    "augmentation": True,
}

# =============================================================================
# CONFIGURATION GOEMOTIONS (Texte)
# =============================================================================

GOEMOTIONS_CONFIG = {
    "model_name": "distilbert-base-uncased",  # Plus leger que bert-base
    "max_length": 128,
    "num_labels": 28,
    "emotions": [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness", "optimism",
        "pride", "realization", "relief", "remorse", "sadness", "surprise",
        "neutral"
    ],
}

# =============================================================================
# CONFIGURATION CNN-LSTM (Modele biometrique)
# =============================================================================

CNN_LSTM_CONFIG = {
    "conv_filters": [64, 128],
    "kernel_size": 3,
    "lstm_hidden": 64,
    "lstm_layers": 1,
    "dropout": 0.3,
    "fc_hidden": 32,
}

# =============================================================================
# CONFIGURATION CNN (Modele images)
# =============================================================================

CNN_FER_CONFIG = {
    "conv_filters": [32, 64, 128],
    "kernel_size": 3,
    "dropout": 0.4,
    "fc_hidden": 512,
}

# =============================================================================
# CONFIGURATION FUSION
# =============================================================================

FUSION_CONFIG = {
    "method": "weighted_average",  # Options: average, weighted_average, voting, mlp
    "weights": {
        "biometric": 0.4,
        "visual": 0.3,
        "text": 0.3
    }
}

# =============================================================================
# MAPPING EMOTIONS POUR FUSION
# =============================================================================

EMOTION_MAPPING = {
    # Mapping vers categories communes
    "stress": {
        "wesad": ["stress"],
        "fer2013": ["fear"],
        "goemotions": ["fear", "nervousness"]
    },
    "joy": {
        "wesad": ["amusement"],
        "fer2013": ["happy"],
        "goemotions": ["joy", "amusement", "excitement"]
    },
    "neutral": {
        "wesad": ["baseline"],
        "fer2013": ["neutral"],
        "goemotions": ["neutral"]
    },
    "anger": {
        "wesad": [],
        "fer2013": ["angry"],
        "goemotions": ["anger", "annoyance"]
    },
    "sadness": {
        "wesad": [],
        "fer2013": ["sad"],
        "goemotions": ["sadness", "disappointment", "grief"]
    }
}
