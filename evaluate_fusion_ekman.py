"""
Evaluation de la fusion multimodale avec mapping vers 7 emotions d'Ekman
et calcul du score de coherence multimodale.

Ce script complete evaluate.py (qui reste inchange comme preuve scientifique
des resultats unimodaux originaux) en ajoutant:
  1. Le mapping de chaque modalite vers les 7 emotions de base d'Ekman
  2. La fusion multimodale avec 4 strategies (avg, weighted, voting, max)
  3. Le calcul du score de coherence multimodale (apport original)
  4. L'analyse qualitative des cas congruents/incongruents

Usage:
    python evaluate_fusion_ekman.py
    python evaluate_fusion_ekman.py --weights 0.5,0.3,0.2
    python evaluate_fusion_ekman.py --device cuda

Sortie:
    outputs/figures/fusion_ekman_report.json
    outputs/figures/fusion_ekman_report.txt
    outputs/figures/coherence_distribution.png (si matplotlib disponible)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from configs.config import (
    MODELS_DIR, FIGURES_DIR,
    WESAD_CONFIG, FER2013_CONFIG, GOEMOTIONS_CONFIG,
    FUSION_CONFIG, EMOTION_MAPPING, EKMAN_EMOTIONS
)
from utils.data_loader import (
    get_wesad_loaders, get_fer2013_loaders, get_goemotions_loaders
)
from fusion.multimodal_fusion import (
    map_to_common_emotions,
    average_fusion, weighted_average_fusion, voting_fusion, max_fusion
)
from fusion.coherence_score import (
    coherence_kl, coherence_agreement, classify_coherence,
    coherence_statistics, fusion_with_coherence
)


# =============================================================================
# CHARGEMENT DES MODELES (delegue a evaluate.py)
# =============================================================================

def load_all_models(device='cpu'):
    """Charge les 3 modeles entraines (utilise les memes chemins qu'evaluate.py)."""
    from evaluate import load_model

    models = {}
    for name in ['wesad', 'fer2013', 'goemotions']:
        print(f"Chargement modele {name}...")
        model, config = load_model(name, device)
        if model is None:
            print(f"  ECHEC: modele {name} introuvable")
        else:
            models[name] = (model, config)
            print(f"  OK")
    return models


# =============================================================================
# GENERATION DES PREDICTIONS
# =============================================================================

def predict_wesad(model, device='cpu'):
    """Genere les predictions du modele WESAD sur le test set."""
    _, _, test_loader = get_wesad_loaders()
    if test_loader is None:
        return None, None

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="WESAD predictions", leave=False):
            X = X.to(device)
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(y.numpy())

    return np.vstack(all_probs), np.array(all_labels)


def predict_fer2013(model, device='cpu'):
    """Genere les predictions du modele FER2013 sur le test set."""
    _, _, test_loader = get_fer2013_loaders()
    if test_loader is None:
        return None, None

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="FER2013 predictions", leave=False):
            X = X.to(device)
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(y.numpy())

    return np.vstack(all_probs), np.array(all_labels)


def predict_goemotions(model, device='cpu'):
    """Genere les predictions du modele GoEmotions sur le test set."""
    try:
        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(GOEMOTIONS_CONFIG['model_name'])
    except ImportError:
        print("transformers non installe.")
        return None, None

    _, _, test_loader = get_goemotions_loaders(tokenizer=tokenizer)
    if test_loader is None:
        return None, None

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="GoEmotions predictions", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            outputs = model(input_ids, attention_mask)
            # Sigmoid pour multi-label
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.vstack(all_probs), np.vstack(all_labels)


# =============================================================================
# ANALYSE DE LA FUSION
# =============================================================================

def evaluate_fusion_strategies(P_wesad, P_fer, P_goe, weights=None):
    """
    Evalue les 4 strategies de fusion + score de coherence.

    Args:
        P_wesad, P_fer, P_goe: predictions ALIGNEES sur 7 emotions Ekman
        weights: poids pour la moyenne ponderee

    Returns:
        dict avec resultats de chaque strategie
    """
    if weights is None:
        weights = [0.4, 0.3, 0.3]

    predictions_list = [P_wesad, P_fer, P_goe]

    results = {}

    # Strategies de fusion
    results['average'] = {
        'predictions': average_fusion(predictions_list),
        'top1': None,
    }
    results['weighted_average'] = {
        'predictions': weighted_average_fusion(predictions_list, weights=weights),
        'weights': weights,
        'top1': None,
    }
    results['voting'] = {
        'predictions': voting_fusion(predictions_list),
        'top1': None,
    }
    results['max'] = {
        'predictions': max_fusion(predictions_list),
        'top1': None,
    }

    # Calculer top-1 et label final
    for strategy, data in results.items():
        data['top1'] = np.argmax(data['predictions'], axis=1)
        data['emotions'] = [EKMAN_EMOTIONS[i] for i in data['top1']]

    # Score de coherence
    results['coherence'] = {
        'kl': coherence_kl(predictions_list),
        'agreement': coherence_agreement(predictions_list),
    }
    results['coherence']['kl_levels'] = classify_coherence(results['coherence']['kl'])
    results['coherence']['stats_kl'] = coherence_statistics(results['coherence']['kl'])
    results['coherence']['stats_agreement'] = coherence_statistics(results['coherence']['agreement'])

    return results


# =============================================================================
# RAPPORT
# =============================================================================

def generate_report(results, output_dir):
    """Genere le rapport JSON + TXT."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()

    # Rapport JSON
    report = {
        'timestamp': timestamp,
        'method': 'fusion multimodale + coherence (mapping 7 emotions Ekman)',
        'fusion_strategies': {},
        'coherence': {
            'kl_statistics': results['coherence']['stats_kl'],
            'agreement_statistics': results['coherence']['stats_agreement'],
        }
    }

    for strategy in ['average', 'weighted_average', 'voting', 'max']:
        data = results[strategy]
        # Distribution des emotions predites
        unique, counts = np.unique(data['emotions'], return_counts=True)
        distribution = dict(zip(unique.tolist(), counts.tolist()))

        report['fusion_strategies'][strategy] = {
            'distribution_emotions': distribution,
            'n_samples': len(data['emotions']),
        }
        if 'weights' in data:
            report['fusion_strategies'][strategy]['weights'] = data['weights']

    json_path = output_dir / "fusion_ekman_report.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nRapport JSON: {json_path}")

    # Rapport TXT lisible
    txt_path = output_dir / "fusion_ekman_report.txt"
    with open(txt_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("RAPPORT FUSION MULTIMODALE + SCORE DE COHERENCE\n")
        f.write("Mapping: 7 emotions de base (Ekman, 1992)\n")
        f.write(f"Date: {timestamp}\n")
        f.write("="*70 + "\n\n")

        f.write("1. STRATEGIES DE FUSION\n")
        f.write("-"*70 + "\n")
        for strategy in ['average', 'weighted_average', 'voting', 'max']:
            data = results[strategy]
            unique, counts = np.unique(data['emotions'], return_counts=True)
            f.write(f"\n  {strategy.upper()}:\n")
            for em, cnt in zip(unique, counts):
                pct = cnt / len(data['emotions']) * 100
                f.write(f"    {em:10s}: {cnt:5d} ({pct:5.2f}%)\n")

        f.write("\n\n2. SCORE DE COHERENCE MULTIMODALE\n")
        f.write("-"*70 + "\n")

        f.write("\n  Coherence KL (basee sur Jensen-Shannon):\n")
        stats_kl = results['coherence']['stats_kl']
        f.write(f"    Moyenne   : {stats_kl['mean']:.4f}\n")
        f.write(f"    Ecart-type: {stats_kl['std']:.4f}\n")
        f.write(f"    Min/Max   : {stats_kl['min']:.4f} / {stats_kl['max']:.4f}\n")
        f.write(f"    Mediane   : {stats_kl['q50']:.4f}\n")
        f.write(f"    Distribution:\n")
        f.write(f"      Congruent  (>=0.70): {stats_kl['pct_congruent']:5.2f}%\n")
        f.write(f"      Modere   (0.40-0.69): {stats_kl['pct_modere']:5.2f}%\n")
        f.write(f"      Incongruent(<0.40) : {stats_kl['pct_incongruent']:5.2f}%\n")

        f.write("\n  Coherence par accord top-1:\n")
        stats_acc = results['coherence']['stats_agreement']
        f.write(f"    Moyenne: {stats_acc['mean']:.4f}\n")
        f.write(f"    Std    : {stats_acc['std']:.4f}\n")

    print(f"Rapport TXT : {txt_path}")
    return json_path, txt_path


def plot_coherence_distribution(coherence_scores, output_path):
    """Histogramme de la distribution du score de coherence."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib non disponible - figure non generee")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(coherence_scores, bins=30, color='steelblue', edgecolor='black', alpha=0.75)
    ax.axvline(0.4, color='red', linestyle='--', label='Seuil incongruence (0.40)')
    ax.axvline(0.7, color='green', linestyle='--', label='Seuil congruence (0.70)')
    ax.set_xlabel('Score de coherence multimodale')
    ax.set_ylabel("Nombre d'echantillons")
    ax.set_title("Distribution du score de coherence (KL/Jensen-Shannon)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure   : {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fusion multimodale + coherence (Ekman)")
    parser.add_argument("--weights", type=str, default="0.4,0.3,0.3",
                       help="Poids fusion (WESAD,FER,GoEmotions)")
    parser.add_argument("--device", type=str, default='cpu',
                       help="Device (cpu ou cuda)")
    args = parser.parse_args()

    weights = [float(w) for w in args.weights.split(",")]
    assert len(weights) == 3, "Trois poids requis"

    device = torch.device(args.device)
    print(f"Device: {device}\n")

    # 1. Charger modeles
    print("="*70)
    print("ETAPE 1/4 - CHARGEMENT DES MODELES")
    print("="*70)
    models = load_all_models(device)
    if len(models) < 3:
        print("\nERREUR: Les 3 modeles doivent etre disponibles.")
        return 1

    # 2. Generer predictions
    print("\n" + "="*70)
    print("ETAPE 2/4 - GENERATION DES PREDICTIONS")
    print("="*70)

    print("\nWESAD...")
    P_wesad_raw, y_wesad = predict_wesad(models['wesad'][0], device)
    print(f"  shape={P_wesad_raw.shape}")

    print("\nFER2013...")
    P_fer_raw, y_fer = predict_fer2013(models['fer2013'][0], device)
    print(f"  shape={P_fer_raw.shape}")

    print("\nGoEmotions...")
    P_goe_raw, y_goe = predict_goemotions(models['goemotions'][0], device)
    print(f"  shape={P_goe_raw.shape}")

    # 3. Mapping vers 7 emotions Ekman
    print("\n" + "="*70)
    print("ETAPE 3/4 - MAPPING VERS 7 EMOTIONS D'EKMAN")
    print("="*70)

    P_wesad = map_to_common_emotions(P_wesad_raw, 'wesad')
    P_fer = map_to_common_emotions(P_fer_raw, 'fer2013')
    P_goe = map_to_common_emotions(P_goe_raw, 'goemotions')

    print(f"\nApres mapping:")
    print(f"  WESAD     : {P_wesad_raw.shape} -> {P_wesad.shape}")
    print(f"  FER2013   : {P_fer_raw.shape} -> {P_fer.shape}")
    print(f"  GoEmotions: {P_goe_raw.shape} -> {P_goe.shape}")

    # Aligner sur le min des batchs (datasets non alignes)
    min_n = min(len(P_wesad), len(P_fer), len(P_goe))
    P_wesad = P_wesad[:min_n]
    P_fer = P_fer[:min_n]
    P_goe = P_goe[:min_n]
    print(f"\nAlignement: {min_n} echantillons")

    # 4. Fusion + coherence
    print("\n" + "="*70)
    print("ETAPE 4/4 - FUSION + COHERENCE")
    print("="*70)

    results = evaluate_fusion_strategies(P_wesad, P_fer, P_goe, weights=weights)

    # Affichage
    print("\nDistribution des emotions par strategie:")
    for strategy in ['average', 'weighted_average', 'voting', 'max']:
        unique, counts = np.unique(results[strategy]['emotions'], return_counts=True)
        print(f"\n  {strategy}:")
        for em, cnt in zip(unique, counts):
            print(f"    {em:10s}: {cnt} ({cnt/min_n*100:.1f}%)")

    print(f"\nScore de coherence (KL/JS):")
    stats = results['coherence']['stats_kl']
    print(f"  Moyenne : {stats['mean']:.4f}")
    print(f"  Mediane : {stats['q50']:.4f}")
    print(f"  Congruent  : {stats['pct_congruent']:.2f}%")
    print(f"  Modere     : {stats['pct_modere']:.2f}%")
    print(f"  Incongruent: {stats['pct_incongruent']:.2f}%")

    # Rapport
    print("\n" + "="*70)
    print("GENERATION DU RAPPORT")
    print("="*70)

    json_path, txt_path = generate_report(results, FIGURES_DIR)
    plot_coherence_distribution(
        results['coherence']['kl'],
        FIGURES_DIR / "coherence_distribution.png"
    )

    # Sauvegarder les predictions brutes pour analyse ulterieure
    np.savez_compressed(
        FIGURES_DIR / "predictions_ekman.npz",
        P_wesad=P_wesad, P_fer=P_fer, P_goe=P_goe,
        coherence_kl=results['coherence']['kl'],
        coherence_agreement=results['coherence']['agreement'],
        fusion_average=results['average']['predictions'],
        fusion_weighted=results['weighted_average']['predictions'],
        fusion_voting=results['voting']['predictions'],
        fusion_max=results['max']['predictions'],
    )
    print(f"Predictions sauvegardees: {FIGURES_DIR / 'predictions_ekman.npz'}")

    print("\n" + "="*70)
    print("TERMINE")
    print("="*70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
