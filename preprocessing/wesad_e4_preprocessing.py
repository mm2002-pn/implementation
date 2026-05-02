"""
Preprocessing WESAD utilisant les donnees E4 (leger, compatible 8GB RAM)
Les labels sont extraits des timestamps dans quest.csv

Signaux E4: EDA (4Hz), HR (1Hz), TEMP (4Hz), ACC (32Hz)
Labels: 0=baseline, 1=stress, 2=amusement
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Ajouter le path parent pour imports
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import WESAD_PATH, WESAD_CONFIG, PROCESSED_DIR, RANDOM_SEED


def parse_quest_csv(quest_path):
    """
    Parse le fichier quest.csv pour extraire les timestamps des phases.

    Returns:
        dict avec les phases et leurs timestamps (start, end) en secondes
    """
    phases = {}

    with open(quest_path, 'r') as f:
        lines = f.readlines()

    # Trouver ORDER, START, END
    order_line = None
    start_line = None
    end_line = None

    for line in lines:
        if line.startswith('# ORDER'):
            order_line = line.strip().split(';')
        elif line.startswith('# START'):
            start_line = line.strip().split(';')
        elif line.startswith('# END'):
            end_line = line.strip().split(';')

    if not all([order_line, start_line, end_line]):
        return None

    # Parser les phases
    # ORDER: Base, TSST, Medi 1, Fun, Medi 2, ...
    phase_names = order_line[1:]  # Skip "# ORDER"
    start_times = start_line[1:]  # Skip "# START"
    end_times = end_line[1:]      # Skip "# END"

    for i, name in enumerate(phase_names):
        name = name.strip()
        if not name:
            continue

        try:
            start = float(start_times[i]) * 60  # Convertir minutes -> secondes
            end = float(end_times[i]) * 60

            # Mapper vers nos labels
            if 'Base' in name:
                label = 0  # baseline
            elif 'TSST' in name:
                label = 1  # stress
            elif 'Fun' in name:
                label = 2  # amusement
            else:
                label = -1  # ignore (meditation, reading, etc.)

            if label >= 0:
                phases[name] = {
                    'start': start,
                    'end': end,
                    'label': label
                }
        except (ValueError, IndexError):
            continue

    return phases


def load_e4_signal(csv_path):
    """
    Charge un signal E4 depuis un CSV.

    Returns:
        data: numpy array des valeurs
        start_time: timestamp unix de debut
        sample_rate: frequence d'echantillonnage
    """
    try:
        # Lire tout le fichier (petits fichiers E4)
        with open(csv_path, 'r') as f:
            lines = f.readlines()

        if len(lines) < 3:
            return None, None, None

        # Premiere ligne: timestamp unix
        start_time = float(lines[0].strip().split(',')[0])

        # Deuxieme ligne: sample rate
        sample_rate = float(lines[1].strip().split(',')[0])

        # Reste: donnees
        data = []
        for line in lines[2:]:
            values = line.strip().split(',')
            if len(values) == 1:
                data.append(float(values[0]))
            else:
                data.append([float(v) for v in values])

        data = np.array(data, dtype=np.float32)

        return data, start_time, sample_rate

    except Exception as e:
        print(f"Erreur lecture {csv_path}: {e}")
        return None, None, None


def create_labeled_windows(signal, sample_rate, phases, window_size=10, overlap=0.5):
    """
    Cree des fenetres labelisees a partir du signal et des phases.

    Args:
        signal: numpy array (n_samples,) ou (n_samples, n_features)
        sample_rate: frequence echantillonnage
        phases: dict des phases avec start, end, label
        window_size: taille fenetre en secondes
        overlap: pourcentage overlap

    Returns:
        X: (n_windows, window_samples, n_features)
        y: (n_windows,)
    """
    window_samples = int(window_size * sample_rate)
    step = int(window_samples * (1 - overlap))

    # S'assurer que signal est 2D
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    n_samples = len(signal)

    X = []
    y = []

    for phase_name, phase_info in phases.items():
        start_sample = int(phase_info['start'] * sample_rate)
        end_sample = int(phase_info['end'] * sample_rate)
        label = phase_info['label']

        # Verifier les bornes
        start_sample = max(0, start_sample)
        end_sample = min(n_samples, end_sample)

        if end_sample - start_sample < window_samples:
            continue

        # Creer fenetres pour cette phase
        pos = start_sample
        while pos + window_samples <= end_sample:
            window = signal[pos:pos + window_samples]
            X.append(window)
            y.append(label)
            pos += step

    if len(X) == 0:
        return None, None

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def normalize_signal(signal):
    """Normalisation Z-score."""
    mean = np.mean(signal, axis=0, keepdims=True)
    std = np.std(signal, axis=0, keepdims=True) + 1e-8
    return (signal - mean) / std


def process_subject_e4(subject_id, window_size=10, overlap=0.5, signals_to_use=['EDA', 'HR']):
    """
    Traite un sujet en utilisant les donnees E4.

    Args:
        subject_id: ex "S2"
        window_size: taille fenetre en secondes
        overlap: overlap entre fenetres
        signals_to_use: liste des signaux a utiliser

    Returns:
        X, y ou None, None si erreur
    """
    subject_path = WESAD_PATH / subject_id
    e4_path = subject_path / f"{subject_id}_E4_Data"
    quest_path = subject_path / f"{subject_id}_quest.csv"

    # Verifier existence
    if not e4_path.exists():
        print(f"  E4 non trouve: {e4_path}")
        return None, None

    if not quest_path.exists():
        print(f"  Quest non trouve: {quest_path}")
        return None, None

    # Parser les phases/labels
    phases = parse_quest_csv(quest_path)
    if not phases:
        print(f"  Impossible de parser les phases")
        return None, None

    print(f"  Phases trouvees: {list(phases.keys())}")

    # Charger et traiter chaque signal
    all_X = []
    all_y = None

    for signal_name in signals_to_use:
        csv_path = e4_path / f"{signal_name}.csv"
        if not csv_path.exists():
            print(f"  Signal {signal_name} non trouve")
            continue

        data, start_time, sample_rate = load_e4_signal(csv_path)
        if data is None:
            continue

        print(f"  {signal_name}: {len(data)} samples, {sample_rate}Hz")

        # Normaliser
        data = normalize_signal(data)

        # Creer fenetres
        X, y = create_labeled_windows(
            data, sample_rate, phases,
            window_size=window_size, overlap=overlap
        )

        if X is not None:
            all_X.append((signal_name, X, sample_rate))
            if all_y is None:
                all_y = y

    if len(all_X) == 0:
        return None, None

    # Pour simplifier, on utilise le signal avec le plus de features
    # Ou on peut resampler tous au meme rate

    # Utiliser EDA comme signal principal (4Hz, stable)
    for name, X, sr in all_X:
        if name == 'EDA':
            print(f"  Utilisation {name}: {X.shape}")
            return X, all_y

    # Fallback: premier signal disponible
    name, X, sr = all_X[0]
    print(f"  Utilisation {name}: {X.shape}")
    return X, all_y


def preprocess_wesad_e4(window_size=10, overlap=0.5, signals=['EDA', 'HR'], save=True):
    """
    Preprocessing complet WESAD avec donnees E4.
    Compatible 8GB RAM.

    Args:
        window_size: taille fenetre en secondes
        overlap: overlap entre fenetres
        signals: signaux a utiliser
        save: sauvegarder les resultats
    """
    print("="*60)
    print("PREPROCESSING WESAD (E4 Data)")
    print(f"Window: {window_size}s, Overlap: {overlap*100}%")
    print(f"Signaux: {signals}")
    print("="*60)

    subjects = WESAD_CONFIG["subjects"]

    X_all = []
    y_all = []
    subject_ids = []

    for i, subject_id in enumerate(tqdm(subjects, desc="Sujets")):
        print(f"\n[{i+1}/{len(subjects)}] {subject_id}")

        X, y = process_subject_e4(
            subject_id,
            window_size=window_size,
            overlap=overlap,
            signals_to_use=signals
        )

        if X is not None and y is not None:
            X_all.append(X)
            y_all.append(y)
            subject_ids.extend([subject_id] * len(X))
            print(f"  -> {len(X)} fenetres")
        else:
            print(f"  -> Pas de donnees")

        gc.collect()

    if len(X_all) == 0:
        print("\nAucune donnee extraite!")
        return None, None

    # Combiner
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    print("\n" + "="*60)
    print("RESULTATS")
    print("="*60)
    print(f"X shape: {X_all.shape}")
    print(f"y shape: {y_all.shape}")
    print(f"Distribution:")
    print(f"  0 (baseline):  {np.sum(y_all == 0)}")
    print(f"  1 (stress):    {np.sum(y_all == 1)}")
    print(f"  2 (amusement): {np.sum(y_all == 2)}")

    if save:
        output_path = PROCESSED_DIR / "wesad_e4_processed.npz"
        np.savez_compressed(
            output_path,
            X=X_all,
            y=y_all,
            subjects=np.array(subject_ids)
        )
        print(f"\nSauvegarde: {output_path}")
        print(f"Taille: {output_path.stat().st_size / 1e6:.2f} MB")

    return X_all, y_all


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocessing WESAD E4")
    parser.add_argument("--window", type=int, default=10, help="Taille fenetre (secondes)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap (0-1)")
    parser.add_argument("--signals", nargs='+', default=['EDA'], help="Signaux a utiliser")

    args = parser.parse_args()

    preprocess_wesad_e4(
        window_size=args.window,
        overlap=args.overlap,
        signals=args.signals
    )
