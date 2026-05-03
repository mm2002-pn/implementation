"""
Conversion des labels GoEmotions de 28 emotions multi-label
vers 7 emotions de base d'Ekman en single-label.

Reference academique:
    Demszky et al. (2020) - GoEmotions: A Dataset of Fine-Grained Emotions
    Mapping officiel propose par les auteurs vers la taxonomie d'Ekman (1992).

Strategie de conversion:
    1. Construire la matrice de mapping M (28, 7) depuis EMOTION_MAPPING
    2. Projeter les labels (n, 28) vers (n, 7) en sommant les contributions
    3. Single-label: prendre la categorie d'Ekman avec le plus de votes
    4. Tie-breaking: priorite a l'emotion la plus specifique (ordre EKMAN_EMOTIONS)
    5. Si aucune emotion presente: assigner a 'neutral' (categorie par defaut)

Ce module N'ENTRAINE RIEN. Il convertit uniquement des labels.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import EMOTION_MAPPING, EKMAN_EMOTIONS, GOEMOTIONS_CONFIG


# Liste ordonnee officielle des 28 emotions GoEmotions (ordre Demszky 2020)
GOEMOTIONS_28_LABELS = GOEMOTIONS_CONFIG['emotions']

# Index de la categorie 'neutral' dans EKMAN_EMOTIONS
NEUTRAL_INDEX = EKMAN_EMOTIONS.index('neutral')


def build_mapping_matrix():
    """
    Construit la matrice binaire de mapping M (28, 7).
    M[i, j] = 1 si l'emotion GoEmotions i appartient a la categorie Ekman j.

    Returns:
        M: numpy array (28, 7) - matrice de mapping
    """
    M = np.zeros((len(GOEMOTIONS_28_LABELS), len(EKMAN_EMOTIONS)), dtype=np.float32)

    for ekman_idx, ekman_name in enumerate(EKMAN_EMOTIONS):
        goe_emotions_in_category = EMOTION_MAPPING[ekman_name].get('goemotions', [])

        for goe_name in goe_emotions_in_category:
            if goe_name in GOEMOTIONS_28_LABELS:
                goe_idx = GOEMOTIONS_28_LABELS.index(goe_name)
                M[goe_idx, ekman_idx] = 1.0

    return M


# Matrice precalculee pour reutilisation
MAPPING_MATRIX = build_mapping_matrix()


def convert_28_to_7_ekman(labels_28, mode='single_label'):
    """
    Convertit les labels GoEmotions de 28 emotions vers 7 categories d'Ekman.

    Args:
        labels_28: numpy array (n, 28) - labels multi-label binaires
        mode: 'single_label' (retourne indices) ou 'multi_label' (retourne binaire)

    Returns:
        Si mode='single_label':
            labels_7: array (n,) - indice de classe Ekman dans [0, 6]
        Si mode='multi_label':
            labels_7: array (n, 7) - binaire multi-label

    Exemple:
        >>> labels_28 = np.zeros((1, 28))
        >>> labels_28[0, GOEMOTIONS_28_LABELS.index('joy')] = 1
        >>> labels_7 = convert_28_to_7_ekman(labels_28)
        >>> EKMAN_EMOTIONS[labels_7[0]]
        'happy'
    """
    if labels_28.ndim != 2 or labels_28.shape[1] != 28:
        raise ValueError(f"labels_28 doit etre (n, 28), recu {labels_28.shape}")

    # Projection: scores agregeés par categorie Ekman
    # scores[i, j] = nombre d'emotions GoEmotions de la categorie j presentes dans l'echantillon i
    scores = labels_28.astype(np.float32) @ MAPPING_MATRIX  # (n, 7)

    if mode == 'multi_label':
        # Binaire: 1 si au moins une emotion de la categorie est presente
        return (scores > 0).astype(np.float32)

    elif mode == 'single_label':
        # Indices de classe (single-label)
        labels_7 = np.argmax(scores, axis=1).astype(np.int64)

        # Cas particulier: aucun label present (toutes les colonnes a 0)
        # -> assigner a 'neutral' par defaut
        no_label_mask = (scores.sum(axis=1) == 0)
        labels_7[no_label_mask] = NEUTRAL_INDEX

        return labels_7

    else:
        raise ValueError(f"Mode inconnu: {mode}. Utiliser 'single_label' ou 'multi_label'.")


def convert_predictions_28_to_7_ekman(probs_28):
    """
    Convertit des PROBABILITES (n, 28) en probabilites (n, 7).

    Utile pour mapper les sorties d'un modele 28-classes vers 7 categories Ekman
    sans ré-entrainement.

    Args:
        probs_28: array (n, 28) - probabilites multi-label (sigmoid)

    Returns:
        probs_7: array (n, 7) - probabilites par categorie Ekman (normalisees)
    """
    # Pour chaque categorie Ekman, faire la moyenne des probabilites
    # des emotions GoEmotions correspondantes
    n_per_category = MAPPING_MATRIX.sum(axis=0, keepdims=True)  # (1, 7)
    n_per_category = np.maximum(n_per_category, 1.0)  # eviter division par zero

    # Somme des probabilites par categorie / nombre d'emotions par categorie
    probs_7 = (probs_28 @ MAPPING_MATRIX) / n_per_category  # (n, 7)

    # Renormaliser pour avoir une distribution valide
    sums = probs_7.sum(axis=1, keepdims=True)
    sums = np.maximum(sums, 1e-10)
    probs_7 = probs_7 / sums

    return probs_7


def get_class_distribution(labels_7):
    """Retourne la distribution des 7 classes Ekman."""
    counts = np.bincount(labels_7, minlength=7)
    distribution = {EKMAN_EMOTIONS[i]: int(counts[i]) for i in range(7)}
    return distribution


def print_mapping_summary():
    """Affiche un resume du mapping 28->7."""
    print("="*70)
    print("MAPPING GOEMOTIONS 28 -> 7 EMOTIONS D'EKMAN")
    print("="*70)
    print(f"Source : {len(GOEMOTIONS_28_LABELS)} emotions GoEmotions (Demszky 2020)")
    print(f"Cible  : {len(EKMAN_EMOTIONS)} categories d'Ekman (1992)")
    print()

    for ekman_idx, ekman_name in enumerate(EKMAN_EMOTIONS):
        goe_emotions = EMOTION_MAPPING[ekman_name].get('goemotions', [])
        print(f"  {ekman_name:10s} ({len(goe_emotions)} emotions)")
        for goe in goe_emotions:
            print(f"    <- {goe}")
        print()


# =============================================================================
# TEST
# =============================================================================

def test_conversion():
    """Test du module de conversion."""
    print_mapping_summary()

    print("="*70)
    print("TEST DE CONVERSION")
    print("="*70)

    # Cas 1: 'joy' uniquement -> doit donner 'happy'
    labels_28 = np.zeros((1, 28))
    labels_28[0, GOEMOTIONS_28_LABELS.index('joy')] = 1
    result = convert_28_to_7_ekman(labels_28)
    print(f"\nCas 1 - 'joy' seul:")
    print(f"  Resultat: classe {result[0]} = '{EKMAN_EMOTIONS[result[0]]}' (attendu: happy)")
    assert EKMAN_EMOTIONS[result[0]] == 'happy'

    # Cas 2: 'sadness' + 'grief' -> doit donner 'sad' (2 votes)
    labels_28 = np.zeros((1, 28))
    labels_28[0, GOEMOTIONS_28_LABELS.index('sadness')] = 1
    labels_28[0, GOEMOTIONS_28_LABELS.index('grief')] = 1
    result = convert_28_to_7_ekman(labels_28)
    print(f"\nCas 2 - 'sadness' + 'grief':")
    print(f"  Resultat: classe {result[0]} = '{EKMAN_EMOTIONS[result[0]]}' (attendu: sad)")
    assert EKMAN_EMOTIONS[result[0]] == 'sad'

    # Cas 3: aucun label -> doit donner 'neutral'
    labels_28 = np.zeros((1, 28))
    result = convert_28_to_7_ekman(labels_28)
    print(f"\nCas 3 - Aucun label:")
    print(f"  Resultat: classe {result[0]} = '{EKMAN_EMOTIONS[result[0]]}' (attendu: neutral)")
    assert EKMAN_EMOTIONS[result[0]] == 'neutral'

    # Cas 4: 'fear' + 'joy' -> joy a 11 emotions, fear a 2 -> selon scores: joy
    # Mais pour 1 emotion chacun, ils s'egalent -> argmax prend le premier
    # Verifions le comportement
    labels_28 = np.zeros((1, 28))
    labels_28[0, GOEMOTIONS_28_LABELS.index('fear')] = 1
    labels_28[0, GOEMOTIONS_28_LABELS.index('joy')] = 1
    result = convert_28_to_7_ekman(labels_28)
    print(f"\nCas 4 - 'fear' + 'joy' (1-1):")
    print(f"  Resultat: classe {result[0]} = '{EKMAN_EMOTIONS[result[0]]}'")
    print(f"  (argmax prend le premier en cas d'egalite: 'happy' car index 0)")

    # Cas 5: distribution sur batch
    np.random.seed(42)
    labels_batch = np.random.binomial(1, 0.05, size=(1000, 28)).astype(np.float32)
    result_batch = convert_28_to_7_ekman(labels_batch)
    print(f"\nCas 5 - Batch de 1000 echantillons aleatoires:")
    distribution = get_class_distribution(result_batch)
    for emotion, count in distribution.items():
        print(f"  {emotion:10s}: {count} ({count/1000*100:.1f}%)")

    # Test multi-label mode
    labels_multi = convert_28_to_7_ekman(labels_batch, mode='multi_label')
    print(f"\nCas 6 - Mode multi-label, shape: {labels_multi.shape}")
    print(f"  Moyenne labels par sample: {labels_multi.sum(axis=1).mean():.2f}")

    # Test mapping matrix
    print(f"\n=== Matrice de mapping M (28, 7) ===")
    print(f"  Forme: {MAPPING_MATRIX.shape}")
    print(f"  Somme par colonne (emotions par categorie):")
    for j, ekman in enumerate(EKMAN_EMOTIONS):
        print(f"    {ekman:10s}: {int(MAPPING_MATRIX[:, j].sum())} emotions")

    print("\n=== TOUS LES TESTS PASSES ===")


if __name__ == "__main__":
    test_conversion()
