"""
DataLoaders PyTorch pour les 3 datasets.
Optimises pour 8GB RAM avec chargement par chunks.
"""

import os
import sys
import gc
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent.parent))
from configs.config import (
    PROCESSED_DIR, BATCH_SIZE, RANDOM_SEED,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
)


# =============================================================================
# DATASET WESAD (Signaux biometriques)
# =============================================================================

class WESADDataset(Dataset):
    """Dataset pour WESAD (signaux biometriques)."""

    def __init__(self, X, y, transform=None):
        """
        Args:
            X: numpy array (n_samples, seq_len, n_features)
            y: numpy array (n_samples,)
            transform: transformations optionnelles
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


def load_wesad_data(processed_path=None):
    """
    Charge les donnees WESAD preprocessees.

    Returns:
        X, y ou None si fichier non trouve
    """
    if processed_path is None:
        processed_path = PROCESSED_DIR / "wesad_processed.npz"

    if not processed_path.exists():
        print(f"Fichier non trouve: {processed_path}")
        return None, None

    data = np.load(processed_path)
    return data['X'], data['y']


def get_wesad_loaders(batch_size=None, train_ratio=0.7, val_ratio=0.15):
    """
    Cree les DataLoaders pour WESAD.

    Returns:
        train_loader, val_loader, test_loader
    """
    if batch_size is None:
        batch_size = BATCH_SIZE

    X, y = load_wesad_data()
    if X is None:
        return None, None, None

    # Split
    n = len(X)
    indices = np.random.RandomState(RANDOM_SEED).permutation(n)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # Datasets
    dataset = WESADDataset(X, y)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"WESAD - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    return train_loader, val_loader, test_loader


# =============================================================================
# DATASET FER2013 (Images faciales)
# =============================================================================

class FER2013Dataset(Dataset):
    """Dataset pour FER2013 (images faciales)."""

    def __init__(self, X, y, transform=None):
        """
        Args:
            X: numpy array (n_samples, 48, 48) normalise [0,1]
            y: numpy array (n_samples,)
            transform: transformations optionnelles
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


class FER2013ChunkedDataset(Dataset):
    """
    Dataset FER2013 charge par chunks (economie RAM).
    Charge UN sample a la fois pour minimiser la RAM.
    """

    def __init__(self, chunk_dir=None, indices=None):
        """
        Args:
            chunk_dir: dossier contenant les chunks .npz
            indices: indices globaux a utiliser (pour train/val/test split)
        """
        if chunk_dir is None:
            chunk_dir = PROCESSED_DIR / "fer2013_chunks"

        self.chunk_dir = Path(chunk_dir)
        self.chunk_files = sorted(self.chunk_dir.glob("chunk_*.npz"))

        if not self.chunk_files:
            raise FileNotFoundError(f"Aucun chunk trouve dans {chunk_dir}")

        # Calculer la taille de chaque chunk et les offsets
        self.chunk_sizes = []
        self.chunk_offsets = [0]

        for chunk_path in self.chunk_files:
            with np.load(chunk_path) as data:
                size = len(data['y'])
            self.chunk_sizes.append(size)
            self.chunk_offsets.append(self.chunk_offsets[-1] + size)

        self.total_size = self.chunk_offsets[-1]

        # Indices a utiliser (tous par defaut)
        if indices is not None:
            self.indices = indices
        else:
            self.indices = list(range(self.total_size))

    def __len__(self):
        return len(self.indices)

    def _get_chunk_and_local_idx(self, global_idx):
        """Trouve le chunk et l'index local pour un index global."""
        for i, (start, end) in enumerate(zip(self.chunk_offsets[:-1], self.chunk_offsets[1:])):
            if start <= global_idx < end:
                return i, global_idx - start
        raise IndexError(f"Index {global_idx} hors limites")

    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        chunk_idx, local_idx = self._get_chunk_and_local_idx(global_idx)

        # Charger seulement le sample necessaire
        with np.load(self.chunk_files[chunk_idx]) as data:
            x = torch.FloatTensor(data['X'][local_idx].copy())
            y = int(data['y'][local_idx])

        return x, torch.tensor(y, dtype=torch.long)


def load_fer2013_data(processed_path=None):
    """
    Charge les donnees FER2013 preprocessees (version complete).

    Returns:
        X, y ou None si fichier non trouve
    """
    if processed_path is None:
        processed_path = PROCESSED_DIR / "fer2013_processed.npz"

    if not processed_path.exists():
        print(f"Fichier non trouve: {processed_path}")
        return None, None

    data = np.load(processed_path)
    return data['X'], data['y']


def get_fer2013_loaders(batch_size=None, train_ratio=0.7, val_ratio=0.15, use_chunks=True):
    """
    Cree les DataLoaders pour FER2013.

    Args:
        batch_size: taille du batch
        train_ratio, val_ratio: ratios de split
        use_chunks: True pour charger par chunks (economie RAM)

    Returns:
        train_loader, val_loader, test_loader
    """
    if batch_size is None:
        batch_size = BATCH_SIZE

    if use_chunks:
        chunk_dir = PROCESSED_DIR / "fer2013_chunks"
        if not chunk_dir.exists():
            print("Chunks non trouves, tentative chargement fichier complet...")
            use_chunks = False

    if use_chunks:
        # Mode chunks
        dataset = FER2013ChunkedDataset(chunk_dir)
        n = len(dataset)

        indices = np.random.RandomState(RANDOM_SEED).permutation(n)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_dataset = FER2013ChunkedDataset(chunk_dir, indices[:train_end].tolist())
        val_dataset = FER2013ChunkedDataset(chunk_dir, indices[train_end:val_end].tolist())
        test_dataset = FER2013ChunkedDataset(chunk_dir, indices[val_end:].tolist())

    else:
        # Mode fichier complet
        X, y = load_fer2013_data()
        if X is None:
            return None, None, None

        n = len(X)
        indices = np.random.RandomState(RANDOM_SEED).permutation(n)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        dataset = FER2013Dataset(X, y)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"FER2013 - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


# =============================================================================
# DATASET GOEMOTIONS (Texte)
# =============================================================================

class GoEmotionsDataset(Dataset):
    """Dataset pour GoEmotions (texte multi-label)."""

    def __init__(self, texts, labels, tokenizer=None, max_length=128):
        """
        Args:
            texts: liste de strings
            labels: numpy array (n_samples, 28)
            tokenizer: tokenizer BERT (optionnel, utilise a l'entrainement)
            max_length: longueur max des sequences
        """
        self.texts = texts
        self.labels = torch.FloatTensor(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        if self.tokenizer is not None:
            # Tokenization BERT
            encoded = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'labels': labels
            }
        else:
            # Retourne juste le texte brut
            return {'text': text, 'labels': labels}


class GoEmotionsChunkedDataset(Dataset):
    """Dataset GoEmotions charge par chunks."""

    def __init__(self, chunk_dir=None, indices=None, tokenizer=None, max_length=128):
        if chunk_dir is None:
            chunk_dir = PROCESSED_DIR / "goemotions_chunks"

        self.chunk_dir = Path(chunk_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Trouver tous les chunks
        self.label_files = sorted(self.chunk_dir.glob("chunk_*.npz"))

        if not self.label_files:
            raise FileNotFoundError(f"Aucun chunk trouve dans {chunk_dir}")

        # Calculer tailles et offsets
        self.chunk_sizes = []
        self.chunk_offsets = [0]

        for label_path in self.label_files:
            data = np.load(label_path)
            size = len(data['labels'])
            self.chunk_sizes.append(size)
            self.chunk_offsets.append(self.chunk_offsets[-1] + size)
            del data

        self.total_size = self.chunk_offsets[-1]
        self.indices = indices if indices is not None else list(range(self.total_size))

        # Cache
        self._current_chunk_idx = -1
        self._current_texts = None
        self._current_labels = None

    def __len__(self):
        return len(self.indices)

    def _get_chunk_and_local_idx(self, global_idx):
        for i, (start, end) in enumerate(zip(self.chunk_offsets[:-1], self.chunk_offsets[1:])):
            if start <= global_idx < end:
                return i, global_idx - start
        raise IndexError(f"Index {global_idx} hors limites")

    def _load_chunk(self, chunk_idx):
        if chunk_idx != self._current_chunk_idx:
            # Liberer les anciens chunks
            if self._current_labels is not None:
                del self._current_labels
                del self._current_texts
                self._current_labels = None
                self._current_texts = None
                gc.collect()

            label_path = self.label_files[chunk_idx]
            text_path = label_path.with_suffix('').with_suffix('_texts.json')

            # Utiliser mmap pour economiser la memoire
            data = np.load(label_path, mmap_mode='r')
            self._current_labels = data['labels']

            with open(text_path, 'r', encoding='utf-8') as f:
                self._current_texts = json.load(f)

            self._current_chunk_idx = chunk_idx

    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        chunk_idx, local_idx = self._get_chunk_and_local_idx(global_idx)

        self._load_chunk(chunk_idx)

        text = self._current_texts[local_idx]
        labels = torch.FloatTensor(self._current_labels[local_idx])

        if self.tokenizer is not None:
            encoded = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'labels': labels
            }

        return {'text': text, 'labels': labels}


def get_goemotions_loaders(batch_size=None, train_ratio=0.7, val_ratio=0.15,
                           tokenizer=None, max_length=128, use_chunks=True):
    """
    Cree les DataLoaders pour GoEmotions.

    Args:
        batch_size: taille du batch
        train_ratio, val_ratio: ratios de split
        tokenizer: tokenizer BERT
        max_length: longueur max des sequences
        use_chunks: True pour charger par chunks

    Returns:
        train_loader, val_loader, test_loader
    """
    if batch_size is None:
        batch_size = BATCH_SIZE

    chunk_dir = PROCESSED_DIR / "goemotions_chunks"

    if use_chunks and chunk_dir.exists():
        # Mode chunks
        dataset = GoEmotionsChunkedDataset(chunk_dir)
        n = len(dataset)

        indices = np.random.RandomState(RANDOM_SEED).permutation(n)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_dataset = GoEmotionsChunkedDataset(
            chunk_dir, indices[:train_end].tolist(), tokenizer, max_length
        )
        val_dataset = GoEmotionsChunkedDataset(
            chunk_dir, indices[train_end:val_end].tolist(), tokenizer, max_length
        )
        test_dataset = GoEmotionsChunkedDataset(
            chunk_dir, indices[val_end:].tolist(), tokenizer, max_length
        )

    else:
        # Mode fichier complet
        label_path = PROCESSED_DIR / "goemotions_processed.npz"
        text_path = PROCESSED_DIR / "goemotions_texts.json"

        if not label_path.exists() or not text_path.exists():
            print("Donnees GoEmotions non trouvees. Executez le preprocessing.")
            return None, None, None

        labels = np.load(label_path)['labels']
        with open(text_path, 'r', encoding='utf-8') as f:
            texts = json.load(f)

        n = len(texts)
        indices = np.random.RandomState(RANDOM_SEED).permutation(n)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_texts = [texts[i] for i in indices[:train_end]]
        val_texts = [texts[i] for i in indices[train_end:val_end]]
        test_texts = [texts[i] for i in indices[val_end:]]

        train_labels = labels[indices[:train_end]]
        val_labels = labels[indices[train_end:val_end]]
        test_labels = labels[indices[val_end:]]

        train_dataset = GoEmotionsDataset(train_texts, train_labels, tokenizer, max_length)
        val_dataset = GoEmotionsDataset(val_texts, val_labels, tokenizer, max_length)
        test_dataset = GoEmotionsDataset(test_texts, test_labels, tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"GoEmotions - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Test des DataLoaders...\n")

    # Test FER2013
    print("="*50)
    print("Test FER2013 DataLoader")
    print("="*50)

    try:
        train_loader, val_loader, test_loader = get_fer2013_loaders(batch_size=8)

        if train_loader:
            batch = next(iter(train_loader))
            X, y = batch
            print(f"Batch X shape: {X.shape}")
            print(f"Batch y shape: {y.shape}")
            print(f"Labels: {y[:5]}")
    except Exception as e:
        print(f"Erreur: {e}")

    print()
