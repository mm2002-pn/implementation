"""
Score de coherence multimodale.

Apport original du memoire pour la detection des congruences/incongruences
entre les predictions des trois modalites (biometrique, visuelle, textuelle).

Fondement theorique:
    Hypothese de la fuite emotionnelle (Ekman & Friesen, 1969).
    Les emotions s'expriment a travers plusieurs canaux que la personne
    ne peut tous controler simultanement. Une forte coherence inter-modale
    suggere une emotion authentique; une faible coherence suggere
    une dissimulation, une emotion masquee, ou un signal ambigu.

Definition:
    Coh(x) = 1 - (1 / C(M,2)) * sum_{i<j} D_KL(P_i || P_j)
    avec D_KL(P || Q) = sum_c P(c) * log(P(c) / Q(c))

Variantes implementees:
    1. Coherence par divergence de Kullback-Leibler
    2. Coherence par divergence de Jensen-Shannon (symetrique)
    3. Coherence par accord de prediction (top-1)

Reference: voir Chapitre V.5.2 du memoire pour la formulation complete.
"""

import numpy as np
from itertools import combinations


# =============================================================================
# DIVERGENCES
# =============================================================================

def _safe_log(x, eps=1e-10):
    """Logarithme avec stabilite numerique."""
    return np.log(np.clip(x, eps, 1.0))


def kl_divergence(P, Q, eps=1e-10):
    """
    Divergence de Kullback-Leibler entre deux distributions.

    D_KL(P || Q) = sum_c P(c) * log(P(c) / Q(c))

    Args:
        P, Q: arrays (batch, n_classes) ou (n_classes,) - distributions de probabilite
        eps: stabilite numerique

    Returns:
        D_KL: array (batch,) ou scalar
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    # Normaliser pour s'assurer que ce sont des distributions valides
    P = P / (P.sum(axis=-1, keepdims=True) + eps)
    Q = Q / (Q.sum(axis=-1, keepdims=True) + eps)

    P = np.clip(P, eps, 1.0)
    Q = np.clip(Q, eps, 1.0)

    return np.sum(P * (_safe_log(P) - _safe_log(Q)), axis=-1)


def js_divergence(P, Q, eps=1e-10):
    """
    Divergence de Jensen-Shannon (symetrique, bornee dans [0, log(2)]).

    JSD(P, Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
    avec M = 0.5 * (P + Q)

    Args:
        P, Q: arrays - distributions de probabilite
        eps: stabilite numerique

    Returns:
        JSD: array (batch,) ou scalar - divergence JS normalisee dans [0, 1]
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    M = 0.5 * (P + Q)

    jsd = 0.5 * kl_divergence(P, M, eps) + 0.5 * kl_divergence(Q, M, eps)

    # Normaliser dans [0, 1] (divergence JS bornee par log(2))
    return jsd / np.log(2.0)


# =============================================================================
# SCORES DE COHERENCE
# =============================================================================

def coherence_kl(predictions_list):
    """
    Score de coherence base sur la divergence KL.

    Coh(x) = 1 - (1 / C(M,2)) * sum_{i<j} JSD(P_i, P_j)

    Note: on utilise JSD au lieu de KL brut pour avoir un score borne et symetrique.

    Args:
        predictions_list: liste de M arrays (batch, n_classes) ALIGNES en classes

    Returns:
        coherence: array (batch,) - score dans [0, 1]
                   1 = parfaite coherence, 0 = incongruence totale
    """
    M = len(predictions_list)

    if M < 2:
        raise ValueError("Au moins 2 modalites necessaires pour calculer la coherence")

    # Verifier que toutes les modalites ont la meme dimension de classes
    n_classes_set = {p.shape[-1] for p in predictions_list}
    if len(n_classes_set) > 1:
        raise ValueError(
            f"Les modalites doivent etre alignees en classes. "
            f"Tailles trouvees: {n_classes_set}"
        )

    n_pairs = M * (M - 1) // 2
    batch_size = predictions_list[0].shape[0]

    total_jsd = np.zeros(batch_size)

    for i, j in combinations(range(M), 2):
        total_jsd += js_divergence(predictions_list[i], predictions_list[j])

    avg_jsd = total_jsd / n_pairs
    coherence = 1.0 - avg_jsd

    return np.clip(coherence, 0.0, 1.0)


def coherence_agreement(predictions_list):
    """
    Score de coherence base sur l'accord des predictions top-1.

    Coh_acc(x) = (2 / M(M-1)) * sum_{i<j} 1[argmax(P_i) == argmax(P_j)]

    Plus simple a interpreter: proportion de paires en accord.

    Args:
        predictions_list: liste de M arrays (batch, n_classes) ALIGNES en classes

    Returns:
        coherence: array (batch,) - score dans [0, 1]
    """
    M = len(predictions_list)

    if M < 2:
        raise ValueError("Au moins 2 modalites necessaires")

    # Predictions top-1
    top1 = [np.argmax(p, axis=-1) for p in predictions_list]

    n_pairs = M * (M - 1) // 2
    batch_size = predictions_list[0].shape[0]

    agreements = np.zeros(batch_size)

    for i, j in combinations(range(M), 2):
        agreements += (top1[i] == top1[j]).astype(np.float64)

    return agreements / n_pairs


# =============================================================================
# CLASSIFICATION DES NIVEAUX DE COHERENCE
# =============================================================================

def classify_coherence(coherence_scores, thresholds=(0.4, 0.7)):
    """
    Classifie les scores de coherence en niveaux qualitatifs.

    Args:
        coherence_scores: array (batch,) - scores dans [0, 1]
        thresholds: tuple (low, high) - seuils de classification

    Returns:
        labels: array (batch,) de strings
            - 'incongruent' : score < low
            - 'modere'      : low <= score < high
            - 'congruent'   : score >= high
    """
    low, high = thresholds
    labels = np.full(len(coherence_scores), 'modere', dtype=object)
    labels[coherence_scores < low] = 'incongruent'
    labels[coherence_scores >= high] = 'congruent'
    return labels


# =============================================================================
# SCORE COMBINE (FUSION + COHERENCE)
# =============================================================================

def fusion_with_coherence(predictions_list, fusion_fn, weights=None):
    """
    Calcule simultanement la fusion et le score de coherence.

    Args:
        predictions_list: liste de M arrays alignes (batch, n_classes)
        fusion_fn: fonction de fusion (ex: average_fusion)
        weights: poids optionnels pour la fusion

    Returns:
        fused_predictions: array (batch, n_classes)
        coherence_kl_score: array (batch,)
        coherence_agreement_score: array (batch,)
    """
    if weights is not None:
        fused = fusion_fn(predictions_list, weights=weights)
    else:
        fused = fusion_fn(predictions_list)

    coh_kl = coherence_kl(predictions_list)
    coh_acc = coherence_agreement(predictions_list)

    return fused, coh_kl, coh_acc


# =============================================================================
# UTILITAIRES POUR ANALYSE
# =============================================================================

def coherence_statistics(coherence_scores):
    """
    Statistiques descriptives du score de coherence.

    Args:
        coherence_scores: array (batch,)

    Returns:
        dict avec mean, std, min, max, quartiles, distribution par niveau
    """
    levels = classify_coherence(coherence_scores)
    unique, counts = np.unique(levels, return_counts=True)
    distribution = dict(zip(unique, counts.tolist()))

    return {
        'mean': float(np.mean(coherence_scores)),
        'std': float(np.std(coherence_scores)),
        'min': float(np.min(coherence_scores)),
        'max': float(np.max(coherence_scores)),
        'q25': float(np.percentile(coherence_scores, 25)),
        'q50': float(np.percentile(coherence_scores, 50)),
        'q75': float(np.percentile(coherence_scores, 75)),
        'n_total': len(coherence_scores),
        'distribution': distribution,
        'pct_congruent': float(np.mean(levels == 'congruent') * 100),
        'pct_modere': float(np.mean(levels == 'modere') * 100),
        'pct_incongruent': float(np.mean(levels == 'incongruent') * 100),
    }


# =============================================================================
# TEST
# =============================================================================

def test_coherence():
    """Test du module de coherence."""
    print("Test du module de coherence multimodale...\n")

    np.random.seed(42)
    batch_size = 100
    n_classes = 7  # 7 emotions d'Ekman

    # Cas 1: predictions tres coherentes (toutes pointent vers la meme classe)
    print("Cas 1: Predictions coherentes")
    base = np.zeros((batch_size, n_classes))
    base[:, 0] = 0.7  # classe 0 dominante
    base[:, 1:] = 0.05  # autres classes faibles

    P1 = base + np.random.randn(batch_size, n_classes) * 0.02
    P2 = base + np.random.randn(batch_size, n_classes) * 0.02
    P3 = base + np.random.randn(batch_size, n_classes) * 0.02
    P1 = np.abs(P1); P1 /= P1.sum(axis=1, keepdims=True)
    P2 = np.abs(P2); P2 /= P2.sum(axis=1, keepdims=True)
    P3 = np.abs(P3); P3 /= P3.sum(axis=1, keepdims=True)

    coh = coherence_kl([P1, P2, P3])
    acc = coherence_agreement([P1, P2, P3])
    print(f"  Coherence KL  : moy={coh.mean():.3f}, std={coh.std():.3f}")
    print(f"  Coherence Acc : moy={acc.mean():.3f}")

    # Cas 2: predictions divergentes (chaque modalite vote different)
    print("\nCas 2: Predictions incongruentes")
    P1 = np.zeros((batch_size, n_classes))
    P2 = np.zeros((batch_size, n_classes))
    P3 = np.zeros((batch_size, n_classes))
    for i in range(batch_size):
        P1[i, np.random.randint(0, 3)] = 0.8
        P2[i, np.random.randint(3, 5)] = 0.8
        P3[i, np.random.randint(5, 7)] = 0.8
    P1 += 0.02; P1 /= P1.sum(axis=1, keepdims=True)
    P2 += 0.02; P2 /= P2.sum(axis=1, keepdims=True)
    P3 += 0.02; P3 /= P3.sum(axis=1, keepdims=True)

    coh = coherence_kl([P1, P2, P3])
    acc = coherence_agreement([P1, P2, P3])
    print(f"  Coherence KL  : moy={coh.mean():.3f}, std={coh.std():.3f}")
    print(f"  Coherence Acc : moy={acc.mean():.3f}")

    # Cas 3: cas mixte (statistiques)
    print("\nCas 3: Statistiques sur cas mixte")
    P_mix = [
        np.random.dirichlet(np.ones(n_classes), size=batch_size),
        np.random.dirichlet(np.ones(n_classes), size=batch_size),
        np.random.dirichlet(np.ones(n_classes), size=batch_size),
    ]
    coh_mix = coherence_kl(P_mix)
    stats = coherence_statistics(coh_mix)
    print(f"  Statistiques: {stats}")

    print("\nTest reussi!")


if __name__ == "__main__":
    test_coherence()
