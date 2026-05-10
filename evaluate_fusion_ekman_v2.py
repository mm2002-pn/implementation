"""
Evaluation de la fusion multimodale - VERSION 2 (avec GoEmotions Ekman7 re-entraine)

Difference cle vs evaluate_fusion_ekman.py (v1):
  - v1 : utilise l'ancien modele GoEmotions 28 classes + mapping post-hoc 28->7
  - v2 : utilise le NOUVEAU modele GoEmotions re-entraine directement sur 7 classes Ekman
         (single-label, CrossEntropy, F1=0.5807 vs 0.3738 sur 28 classes)

WESAD et FER2013 restent en mapping post-hoc (modeles natifs preserves comme preuve).

Sortie (suffixe _v2 pour ne pas ecraser les resultats v1) :
    outputs/figures/fusion_ekman_v2_report.json
    outputs/figures/fusion_ekman_v2_report.txt
    outputs/figures/coherence_distribution_v2.png
    outputs/figures/predictions_ekman_v2.npz

Usage:
    python evaluate_fusion_ekman_v2.py
    python evaluate_fusion_ekman_v2.py --device cuda
    python evaluate_fusion_ekman_v2.py --weights 0.4,0.3,0.3
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
    get_wesad_loaders, get_fer2013_loaders
)
from fusion.multimodal_fusion import (
    map_to_common_emotions,
    average_fusion, weighted_average_fusion, voting_fusion, max_fusion
)
from fusion.coherence_score import (
    coherence_kl, coherence_agreement, classify_coherence,
    coherence_statistics
)


# =============================================================================
# CHARGEMENT DES MODELES
# =============================================================================

def load_wesad_fer_models(device='cpu'):
    """Charge WESAD + FER2013 (modeles natifs preserves)."""
    from evaluate import load_model
    models = {}
    for name in ['wesad', 'fer2013']:
        print(f"Chargement modele {name}...")
        model, config = load_model(name, device)
        if model is None:
            print(f"  ECHEC: modele {name} introuvable")
        else:
            models[name] = (model, config)
            print(f"  OK")
    return models


def load_goemotions_ekman7(device='cpu'):
    """
    Charge le NOUVEAU modele GoEmotions re-entraine sur 7 classes Ekman.
    Cherche d'abord le checkpoint _best.pt, sinon _final.pt.
    """
    from models.bert_model import BertEmotionClassifier

    candidates = [
        MODELS_DIR / "bert_goemotions_ekman7_best.pt",
        MODELS_DIR / "bert_goemotions_ekman7_final.pt",
    ]
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        print(f"ECHEC: aucun checkpoint trouve dans {candidates}")
        return None, None

    print(f"Chargement modele GoEmotions Ekman7 depuis: {ckpt_path.name}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    model = BertEmotionClassifier(num_labels=7, freeze_bert=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  OK (num_labels=7, single-label)")
    return model, checkpoint.get('config', {})


# =============================================================================
# GENERATION DES PREDICTIONS
# =============================================================================

def predict_wesad(model, device='cpu'):
    _, _, test_loader = get_wesad_loaders()
    if test_loader is None:
        return None, None

    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="WESAD predictions", leave=False):
            X = X.to(device)
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(y.numpy())
    return np.vstack(all_probs), np.array(all_labels)


def predict_fer2013(model, device='cpu'):
    _, _, test_loader = get_fer2013_loaders()
    if test_loader is None:
        return None, None

    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="FER2013 predictions", leave=False):
            X = X.to(device)
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(y.numpy())
    return np.vstack(all_probs), np.array(all_labels)


def predict_goemotions_ekman7(model, device='cpu'):
    """
    Predictions du NOUVEAU modele 7 classes (single-label, softmax).
    Renvoie directement (N, 7) - pas besoin de mapping post-hoc.
    """
    from transformers import DistilBertTokenizer
    from train_goemotions_ekman7 import get_goemotions_ekman7_loaders

    tokenizer = DistilBertTokenizer.from_pretrained(GOEMOTIONS_CONFIG['model_name'])
    _, _, test_loader = get_goemotions_ekman7_loaders(
        batch_size=GOEMOTIONS_CONFIG.get('batch_size', 32),
        tokenizer=tokenizer,
        max_length=GOEMOTIONS_CONFIG['max_length']
    )

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="GoEmotions Ekman7 predictions", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)  # single-label -> softmax
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.vstack(all_probs), np.array(all_labels)


# =============================================================================
# ANALYSE DE LA FUSION
# =============================================================================

def evaluate_fusion_strategies(P_wesad, P_fer, P_goe, weights=None):
    if weights is None:
        weights = [0.4, 0.3, 0.3]

    predictions_list = [P_wesad, P_fer, P_goe]
    results = {}

    results['average'] = {'predictions': average_fusion(predictions_list)}
    results['weighted_average'] = {
        'predictions': weighted_average_fusion(predictions_list, weights=weights),
        'weights': weights,
    }
    results['voting'] = {'predictions': voting_fusion(predictions_list)}
    results['max'] = {'predictions': max_fusion(predictions_list)}

    for strategy, data in results.items():
        data['top1'] = np.argmax(data['predictions'], axis=1)
        data['emotions'] = [EKMAN_EMOTIONS[i] for i in data['top1']]

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

def generate_report(results, output_dir, suffix='_v2'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()

    report = {
        'timestamp': timestamp,
        'version': 'v2',
        'method': 'Fusion multimodale + coherence (GoEmotions RE-ENTRAINE 7 classes Ekman)',
        'note': 'WESAD/FER2013 mapping post-hoc, GoEmotions natif 7 classes (single-label)',
        'fusion_strategies': {},
        'coherence': {
            'kl_statistics': results['coherence']['stats_kl'],
            'agreement_statistics': results['coherence']['stats_agreement'],
        }
    }

    for strategy in ['average', 'weighted_average', 'voting', 'max']:
        data = results[strategy]
        unique, counts = np.unique(data['emotions'], return_counts=True)
        distribution = dict(zip(unique.tolist(), counts.tolist()))
        report['fusion_strategies'][strategy] = {
            'distribution_emotions': distribution,
            'n_samples': len(data['emotions']),
        }
        if 'weights' in data:
            report['fusion_strategies'][strategy]['weights'] = data['weights']

    json_path = output_dir / f"fusion_ekman{suffix}_report.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nRapport JSON: {json_path}")

    txt_path = output_dir / f"fusion_ekman{suffix}_report.txt"
    with open(txt_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("RAPPORT FUSION MULTIMODALE V2 + SCORE DE COHERENCE\n")
        f.write("GoEmotions: modele RE-ENTRAINE 7 classes (single-label)\n")
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
        f.write("\n  Coherence KL (Jensen-Shannon):\n")
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
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(coherence_scores, bins=30, color='seagreen', edgecolor='black', alpha=0.75)
    ax.axvline(0.4, color='red', linestyle='--', label='Seuil incongruence (0.40)')
    ax.axvline(0.7, color='green', linestyle='--', label='Seuil congruence (0.70)')
    ax.set_xlabel('Score de coherence multimodale')
    ax.set_ylabel("Nombre d'echantillons")
    ax.set_title("Distribution du score de coherence V2 (GoEmotions re-entraine 7 classes)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure   : {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fusion multimodale V2 (GoEmotions re-entraine)")
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
    print("ETAPE 1/4 - CHARGEMENT DES MODELES (V2)")
    print("="*70)

    models = load_wesad_fer_models(device)
    goe_model, goe_cfg = load_goemotions_ekman7(device)

    if len(models) < 2 or goe_model is None:
        print("\nERREUR: Modeles WESAD, FER2013 et GoEmotions Ekman7 requis.")
        print("Verifiez que bert_goemotions_ekman7_best.pt existe dans outputs/models/")
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

    print("\nGoEmotions Ekman7 (re-entraine)...")
    P_goe, y_goe = predict_goemotions_ekman7(goe_model, device)
    print(f"  shape={P_goe.shape}  <- DEJA en 7 classes, pas de mapping post-hoc")

    # 3. Mapping vers 7 emotions Ekman (UNIQUEMENT pour WESAD et FER2013)
    print("\n" + "="*70)
    print("ETAPE 3/4 - MAPPING VERS 7 EMOTIONS D'EKMAN (WESAD + FER2013)")
    print("="*70)

    P_wesad = map_to_common_emotions(P_wesad_raw, 'wesad')
    P_fer = map_to_common_emotions(P_fer_raw, 'fer2013')
    # P_goe : deja en 7 classes, on ne touche pas

    print(f"\nApres mapping:")
    print(f"  WESAD     : {P_wesad_raw.shape} -> {P_wesad.shape}  (post-hoc)")
    print(f"  FER2013   : {P_fer_raw.shape} -> {P_fer.shape}  (post-hoc)")
    print(f"  GoEmotions: {P_goe.shape}  (modele natif 7 classes, pas de mapping)")

    # Aligner sur le min des batchs
    min_n = min(len(P_wesad), len(P_fer), len(P_goe))
    P_wesad = P_wesad[:min_n]
    P_fer = P_fer[:min_n]
    P_goe = P_goe[:min_n]
    print(f"\nAlignement: {min_n} echantillons")

    # 4. Fusion + coherence
    print("\n" + "="*70)
    print("ETAPE 4/4 - FUSION + COHERENCE (V2)")
    print("="*70)

    results = evaluate_fusion_strategies(P_wesad, P_fer, P_goe, weights=weights)

    print("\nDistribution des emotions par strategie:")
    for strategy in ['average', 'weighted_average', 'voting', 'max']:
        unique, counts = np.unique(results[strategy]['emotions'], return_counts=True)
        print(f"\n  {strategy}:")
        for em, cnt in zip(unique, counts):
            print(f"    {em:10s}: {cnt} ({cnt/min_n*100:.1f}%)")

    print(f"\nScore de coherence V2 (KL/JS):")
    stats = results['coherence']['stats_kl']
    print(f"  Moyenne : {stats['mean']:.4f}")
    print(f"  Mediane : {stats['q50']:.4f}")
    print(f"  Congruent  : {stats['pct_congruent']:.2f}%")
    print(f"  Modere     : {stats['pct_modere']:.2f}%")
    print(f"  Incongruent: {stats['pct_incongruent']:.2f}%")

    # Rapport
    print("\n" + "="*70)
    print("GENERATION DU RAPPORT V2")
    print("="*70)

    json_path, txt_path = generate_report(results, FIGURES_DIR, suffix='_v2')
    plot_coherence_distribution(
        results['coherence']['kl'],
        FIGURES_DIR / "coherence_distribution_v2.png"
    )

    np.savez_compressed(
        FIGURES_DIR / "predictions_ekman_v2.npz",
        P_wesad=P_wesad, P_fer=P_fer, P_goe=P_goe,
        coherence_kl=results['coherence']['kl'],
        coherence_agreement=results['coherence']['agreement'],
        fusion_average=results['average']['predictions'],
        fusion_weighted=results['weighted_average']['predictions'],
        fusion_voting=results['voting']['predictions'],
        fusion_max=results['max']['predictions'],
    )
    print(f"Predictions sauvegardees: {FIGURES_DIR / 'predictions_ekman_v2.npz'}")

    print("\n" + "="*70)
    print("TERMINE - V2 (avec GoEmotions Ekman7 re-entraine)")
    print("="*70)
    print("\nPour comparer V1 vs V2 :")
    print("  V1: outputs/figures/fusion_ekman_report.{json,txt}")
    print("  V2: outputs/figures/fusion_ekman_v2_report.{json,txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
