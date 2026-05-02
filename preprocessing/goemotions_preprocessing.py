"""
Preprocessing GoEmotions (Text Emotion Recognition)
Optimise pour 8GB RAM - Traitement par chunks avec tokenization

Texte: Commentaires Reddit en anglais
Labels: 28 emotions (multi-label)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

# Ajouter le path parent pour imports
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import (
    GOEMOTIONS_PATH, GOEMOTIONS_CONFIG, PROCESSED_DIR, CHUNK_SIZE, RANDOM_SEED
)


# =============================================================================
# TOKENIZATION SIMPLE (sans Hugging Face pour preprocessing)
# =============================================================================

class SimpleTokenizer:
    """
    Tokenizer simple pour preprocessing initial.
    Le vrai tokenizer BERT sera utilise a l'entrainement.
    """

    def __init__(self, max_length=128):
        self.max_length = max_length
        self.vocab = {}
        self.word_counts = {}

    def fit(self, texts):
        """Construit le vocabulaire a partir des textes."""
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                self.word_counts[word] = self.word_counts.get(word, 0) + 1

        # Trier par frequence et assigner des IDs
        sorted_words = sorted(self.word_counts.items(), key=lambda x: -x[1])

        self.vocab = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3
        }

        for i, (word, _) in enumerate(sorted_words):
            self.vocab[word] = i + 4

    def _tokenize(self, text):
        """Tokenization basique."""
        if not isinstance(text, str):
            return []
        # Lowercase et split simple
        text = text.lower()
        # Garder seulement alphanumerique et espaces
        text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
        return text.split()

    def encode(self, text):
        """Encode un texte en sequence d'IDs."""
        words = self._tokenize(text)

        # Ajouter [CLS] au debut et [SEP] a la fin
        ids = [self.vocab['[CLS]']]

        for word in words[:self.max_length - 2]:
            ids.append(self.vocab.get(word, self.vocab['[UNK]']))

        ids.append(self.vocab['[SEP]'])

        # Padding
        while len(ids) < self.max_length:
            ids.append(self.vocab['[PAD]'])

        return ids[:self.max_length]

    def save(self, path):
        """Sauvegarde le vocabulaire."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'max_length': self.max_length
            }, f)

    @classmethod
    def load(cls, path):
        """Charge le vocabulaire."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tokenizer = cls(max_length=data['max_length'])
        tokenizer.vocab = data['vocab']
        return tokenizer


# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def get_emotion_columns():
    """Retourne la liste des colonnes d'emotions."""
    return GOEMOTIONS_CONFIG["emotions"]


def process_chunk(chunk, emotion_cols):
    """
    Traite un chunk du CSV.

    Args:
        chunk: pandas DataFrame
        emotion_cols: liste des colonnes d'emotions

    Returns:
        texts: liste des textes
        labels: numpy array (n, 28) multi-label
    """
    texts = chunk['text'].fillna('').tolist()

    # Extraire labels multi-label
    labels = chunk[emotion_cols].values.astype(np.float32)

    return texts, labels


def preprocess_goemotions(chunk_size=None, save=True, save_chunks=False,
                          build_vocab=True):
    """
    Preprocessing complet GoEmotions par chunks.

    Args:
        chunk_size: taille des chunks
        save: sauvegarder le resultat
        save_chunks: sauvegarder par chunks (economie RAM)
        build_vocab: construire un vocabulaire simple

    Returns:
        texts, labels si save_chunks=False
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE

    goemotions_path = Path(GOEMOTIONS_PATH)

    # Trouver tous les CSV
    csv_files = sorted(goemotions_path.glob("goemotions_*.csv"))

    if not csv_files:
        print(f"Aucun fichier CSV trouve dans {goemotions_path}")
        return None, None

    print("="*60)
    print("PREPROCESSING GOEMOTIONS")
    print(f"Fichiers: {[f.name for f in csv_files]}")
    print(f"Chunk size: {chunk_size}")
    print("="*60)

    emotion_cols = get_emotion_columns()

    # Verifier que les colonnes existent
    sample_df = pd.read_csv(csv_files[0], nrows=5)
    available_cols = [c for c in emotion_cols if c in sample_df.columns]

    if len(available_cols) != len(emotion_cols):
        print(f"Attention: {len(emotion_cols) - len(available_cols)} colonnes manquantes")
        emotion_cols = available_cols

    print(f"Emotions: {len(emotion_cols)} colonnes")

    if save_chunks:
        # Mode economie RAM
        chunk_dir = PROCESSED_DIR / "goemotions_chunks"
        chunk_dir.mkdir(exist_ok=True)

        chunk_idx = 0
        all_texts_for_vocab = []

        for csv_file in csv_files:
            print(f"\nTraitement {csv_file.name}...")

            for chunk in tqdm(pd.read_csv(csv_file, chunksize=chunk_size)):
                texts, labels = process_chunk(chunk, emotion_cols)

                # Sauvegarder
                chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}.npz"

                # Pour les textes, on sauvegarde en JSON
                text_path = chunk_dir / f"chunk_{chunk_idx:04d}_texts.json"

                np.savez_compressed(chunk_path, labels=labels)
                with open(text_path, 'w', encoding='utf-8') as f:
                    json.dump(texts, f)

                if build_vocab:
                    all_texts_for_vocab.extend(texts[:100])  # Echantillon pour vocab

                chunk_idx += 1
                del texts, labels, chunk

        # Construire vocabulaire
        if build_vocab and all_texts_for_vocab:
            print("\nConstruction vocabulaire...")
            tokenizer = SimpleTokenizer(max_length=GOEMOTIONS_CONFIG["max_length"])
            tokenizer.fit(all_texts_for_vocab)
            tokenizer.save(PROCESSED_DIR / "goemotions_vocab.json")
            print(f"Vocabulaire: {len(tokenizer.vocab)} mots")

        print(f"\n{chunk_idx} chunks sauvegardes dans {chunk_dir}")
        return None, None

    else:
        # Mode standard
        all_texts = []
        all_labels = []

        for csv_file in csv_files:
            print(f"\nTraitement {csv_file.name}...")

            for chunk in tqdm(pd.read_csv(csv_file, chunksize=chunk_size)):
                texts, labels = process_chunk(chunk, emotion_cols)
                all_texts.extend(texts)
                all_labels.append(labels)

                del chunk
                import gc
                gc.collect()

        all_labels = np.vstack(all_labels)

        print(f"\nResultat final:")
        print(f"  Textes: {len(all_texts)}")
        print(f"  Labels shape: {all_labels.shape}")

        # Distribution des emotions
        print("\nDistribution des emotions (top 10):")
        emotion_counts = all_labels.sum(axis=0)
        sorted_idx = np.argsort(emotion_counts)[::-1]

        for i in sorted_idx[:10]:
            print(f"  {emotion_cols[i]:15s}: {int(emotion_counts[i]):6d}")

        # Multi-label stats
        labels_per_sample = all_labels.sum(axis=1)
        print(f"\nLabels par exemple:")
        print(f"  Moyenne: {labels_per_sample.mean():.2f}")
        print(f"  Max: {labels_per_sample.max():.0f}")
        print(f"  Exemples sans label: {np.sum(labels_per_sample == 0)}")

        if save:
            # Sauvegarder
            output_path = PROCESSED_DIR / "goemotions_processed.npz"
            text_path = PROCESSED_DIR / "goemotions_texts.json"

            np.savez_compressed(output_path, labels=all_labels)
            with open(text_path, 'w', encoding='utf-8') as f:
                json.dump(all_texts, f)

            print(f"\nSauvegarde:")
            print(f"  Labels: {output_path}")
            print(f"  Textes: {text_path}")

            # Construire vocabulaire
            if build_vocab:
                print("\nConstruction vocabulaire...")
                tokenizer = SimpleTokenizer(max_length=GOEMOTIONS_CONFIG["max_length"])
                tokenizer.fit(all_texts)
                tokenizer.save(PROCESSED_DIR / "goemotions_vocab.json")
                print(f"Vocabulaire: {len(tokenizer.vocab)} mots")

        return all_texts, all_labels


def load_goemotions_chunks(chunk_dir=None):
    """
    Charge les donnees GoEmotions depuis les chunks.
    Generateur pour economiser RAM.

    Yields:
        texts, labels pour chaque chunk
    """
    if chunk_dir is None:
        chunk_dir = PROCESSED_DIR / "goemotions_chunks"

    chunk_dir = Path(chunk_dir)

    if not chunk_dir.exists():
        print(f"Dossier non trouve: {chunk_dir}")
        return

    label_files = sorted(chunk_dir.glob("chunk_*.npz"))

    for label_path in label_files:
        # Charger labels
        data = np.load(label_path)
        labels = data['labels']

        # Charger textes
        text_path = label_path.with_suffix('').with_suffix('_texts.json')
        with open(text_path, 'r', encoding='utf-8') as f:
            texts = json.load(f)

        yield texts, labels

        del data, labels, texts


def create_train_val_test_split(texts, labels, train_ratio=0.7, val_ratio=0.15):
    """
    Split les donnees en train/val/test.
    """
    np.random.seed(RANDOM_SEED)

    n = len(texts)
    indices = np.random.permutation(n)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return {
        'texts_train': [texts[i] for i in train_idx],
        'labels_train': labels[train_idx],
        'texts_val': [texts[i] for i in val_idx],
        'labels_val': labels[val_idx],
        'texts_test': [texts[i] for i in test_idx],
        'labels_test': labels[test_idx]
    }


def get_data_info():
    """Affiche les informations sur GoEmotions."""
    print("\n" + "="*60)
    print("INFORMATIONS GOEMOTIONS")
    print("="*60)

    goemotions_path = Path(GOEMOTIONS_PATH)

    if not goemotions_path.exists():
        print(f"Dossier non trouve: {goemotions_path}")
        return

    csv_files = sorted(goemotions_path.glob("goemotions_*.csv"))

    total_rows = 0
    for csv_file in csv_files:
        size_mb = csv_file.stat().st_size / 1e6
        n_rows = sum(1 for _ in open(csv_file, 'r', encoding='utf-8')) - 1
        total_rows += n_rows
        print(f"{csv_file.name}: {n_rows:,} lignes, {size_mb:.1f} MB")

    print(f"\nTotal: {total_rows:,} exemples")

    # Colonnes
    sample_df = pd.read_csv(csv_files[0], nrows=5)
    print(f"\nColonnes ({len(sample_df.columns)}):")

    emotion_cols = [c for c in sample_df.columns if c in GOEMOTIONS_CONFIG["emotions"]]
    other_cols = [c for c in sample_df.columns if c not in emotion_cols]

    print(f"  Metadata: {other_cols}")
    print(f"  Emotions: {len(emotion_cols)} colonnes")

    # Exemple
    print("\nExemple de texte:")
    print(f"  '{sample_df['text'].iloc[0][:100]}...'")


def visualize_distribution(labels, emotion_cols, save_path=None):
    """Visualise la distribution des emotions."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib non disponible")
        return

    counts = labels.sum(axis=0)
    sorted_idx = np.argsort(counts)[::-1]

    plt.figure(figsize=(14, 6))
    plt.bar(range(len(emotion_cols)),
            [counts[i] for i in sorted_idx],
            color='steelblue')
    plt.xticks(range(len(emotion_cols)),
               [emotion_cols[i] for i in sorted_idx],
               rotation=45, ha='right')
    plt.xlabel("Emotion")
    plt.ylabel("Nombre d'exemples")
    plt.title("Distribution des emotions dans GoEmotions")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Figure sauvegardee: {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocessing GoEmotions")
    parser.add_argument("--info", action="store_true", help="Afficher infos dataset")
    parser.add_argument("--chunks", action="store_true", help="Sauvegarder par chunks")
    parser.add_argument("--chunk-size", type=int, default=500, help="Taille chunks")
    parser.add_argument("--no-vocab", action="store_true", help="Ne pas construire vocabulaire")

    args = parser.parse_args()

    if args.info:
        get_data_info()
    else:
        preprocess_goemotions(
            chunk_size=args.chunk_size,
            save_chunks=args.chunks,
            build_vocab=not args.no_vocab
        )
