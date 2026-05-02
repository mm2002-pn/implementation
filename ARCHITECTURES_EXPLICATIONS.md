# Architectures Deep Learning pour la Reconnaissance Emotionnelle

## Vue d'ensemble du projet

```
+------------------+     +------------------+     +------------------+
|    WESAD         |     |    FER2013       |     |   GoEmotions     |
|  (Biometrique)   |     |    (Images)      |     |    (Texte)       |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         v                        v                        v
+--------+---------+     +--------+---------+     +--------+---------+
|    CNN-LSTM      |     |   CNN / ViT      |     |      BERT        |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         v                        v                        v
    [Prediction]            [Prediction]            [Prediction]
         |                        |                        |
         +------------------------+------------------------+
                                  |
                                  v
                    +-------------+-------------+
                    |     FUSION MULTIMODALE    |
                    |  (Agregation predictions) |
                    +---------------------------+
                                  |
                                  v
                         [Emotion Finale]
```

---

## 1. CNN-LSTM (pour donnees biometriques - WESAD)

### Qu'est-ce que c'est ?

**CNN (Convolutional Neural Network)** + **LSTM (Long Short-Term Memory)** combines pour traiter des series temporelles.

### Pourquoi cette architecture ?

Les donnees WESAD sont des **series temporelles** (HR, EDA, ACC echantillonnes dans le temps):
- **CNN** : Extrait des patterns locaux dans les signaux (pics, variations)
- **LSTM** : Capture les dependances temporelles longues (evolution du stress)

### Architecture detaillee

```
Entree: Signal biometrique (ex: EDA sur 60 secondes)
        Shape: (batch_size, sequence_length, n_features)
        Exemple: (32, 256, 5) -> 32 echantillons, 256 timestamps, 5 capteurs

        +---------------------------+
        |  Conv1D (filters=64)      |  <- Detecte patterns locaux
        |  kernel_size=3            |
        +------------+--------------+
                     |
        +------------v--------------+
        |  Conv1D (filters=128)     |  <- Patterns plus complexes
        |  + MaxPooling1D           |
        +------------+--------------+
                     |
        +------------v--------------+
        |  LSTM (units=64)          |  <- Capture temporalite
        |  return_sequences=False   |
        +------------+--------------+
                     |
        +------------v--------------+
        |  Dense (units=32, ReLU)   |
        +------------+--------------+
                     |
        +------------v--------------+
        |  Dense (units=n_classes)  |  <- Sortie: probas emotions
        |  Softmax                  |
        +---------------------------+

Sortie: Probabilites [baseline, stress, amusement, meditation]
```

### Code simplifie (PyTorch)

```python
class CNN_LSTM(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        # CNN pour extraction de features
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

        # LSTM pour temporalite
        self.lstm = nn.LSTM(128, 64, batch_first=True)

        # Classification
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len) pour Conv1d
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)  # Retour format LSTM

        _, (h_n, _) = self.lstm(x)
        x = F.relu(self.fc1(h_n[-1]))
        return self.fc2(x)
```

---

## 2. BERT (pour donnees textuelles - GoEmotions)

### Qu'est-ce que c'est ?

**BERT** = Bidirectional Encoder Representations from Transformers (Google, 2018)

C'est un modele de langage **pre-entraine** sur des milliards de mots qui comprend le contexte des phrases.

### Pourquoi BERT ?

- **Pre-entraine** : Deja appris la langue anglaise (GoEmotions est en anglais)
- **Bidirectionnel** : Comprend le contexte avant ET apres chaque mot
- **Fine-tuning** : On adapte juste la derniere couche pour notre tache

### Architecture detaillee

```
Entree: Texte "That game hurt."

        +---------------------------+
        |      TOKENIZATION         |
        |  [CLS] That game hurt [SEP]|
        +------------+--------------+
                     |
        +------------v--------------+
        |    BERT ENCODER           |
        |  (12 couches Transformer) |
        |  - Self-Attention         |
        |  - Feed-Forward           |
        +------------+--------------+
                     |
        +------------v--------------+
        |  [CLS] Token Embedding    |  <- Representation de la phrase
        |  (768 dimensions)         |
        +------------+--------------+
                     |
        +------------v--------------+
        |  Classification Head      |
        |  Linear(768 -> 28)        |  <- 28 emotions GoEmotions
        |  Sigmoid (multi-label)    |
        +---------------------------+

Sortie: Probabilites pour chaque emotion [0.1, 0.8, 0.05, ...]
```

### Hugging Face pour BERT

```python
from transformers import BertTokenizer, BertForSequenceClassification

# Charger modele pre-entraine
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=28,  # 28 emotions
    problem_type="multi_label_classification"
)

# Tokenization
inputs = tokenizer("That game hurt.", return_tensors="pt", padding=True)

# Prediction
outputs = model(**inputs)
probas = torch.sigmoid(outputs.logits)  # Multi-label
```

---

## 3. ViT - Vision Transformer (pour images - FER2013)

### Qu'est-ce que c'est ?

**ViT** = Vision Transformer (Google, 2020)

Applique l'architecture Transformer (comme BERT) aux images en les decoupant en "patches".

### Alternative: CNN classique (ResNet, VGG)

Pour FER2013 (images 48x48), on peut aussi utiliser un **CNN classique** qui est plus leger:

```
Option 1: CNN Simple (recommande pour RAM limitee)
Option 2: ViT (plus puissant mais plus lourd)
```

### Architecture CNN pour FER2013

```
Entree: Image 48x48 grayscale
        Shape: (batch_size, 1, 48, 48)

        +---------------------------+
        |  Conv2D(1->32, 3x3)       |
        |  BatchNorm + ReLU         |
        |  MaxPool(2x2)             |  -> 24x24
        +------------+--------------+
                     |
        +------------v--------------+
        |  Conv2D(32->64, 3x3)      |
        |  BatchNorm + ReLU         |
        |  MaxPool(2x2)             |  -> 12x12
        +------------+--------------+
                     |
        +------------v--------------+
        |  Conv2D(64->128, 3x3)     |
        |  BatchNorm + ReLU         |
        |  MaxPool(2x2)             |  -> 6x6
        +------------+--------------+
                     |
        +------------v--------------+
        |  Flatten                  |
        |  128 * 6 * 6 = 4608       |
        +------------+--------------+
                     |
        +------------v--------------+
        |  Dense(512) + Dropout     |
        +------------+--------------+
                     |
        +------------v--------------+
        |  Dense(7) + Softmax       |  <- 7 emotions FER2013
        +---------------------------+

Sortie: [angry, disgust, fear, happy, sad, surprise, neutral]
```

### Architecture ViT (si assez de RAM)

```
Entree: Image 48x48

        +---------------------------+
        |  Decoupage en patches     |
        |  16 patches de 12x12      |
        +------------+--------------+
                     |
        +------------v--------------+
        |  Linear Embedding         |
        |  + Position Embedding     |
        +------------+--------------+
                     |
        +------------v--------------+
        |  Transformer Encoder      |
        |  (6-12 couches)           |
        |  - Multi-Head Attention   |
        |  - MLP                    |
        +------------+--------------+
                     |
        +------------v--------------+
        |  [CLS] Token -> Dense(7)  |
        +---------------------------+
```

---

## 4. Hugging Face - Role dans le projet

### Qu'est-ce que Hugging Face ?

**Hugging Face** est une plateforme/bibliotheque qui fournit:
- Des **modeles pre-entraines** (BERT, ViT, etc.)
- Des **tokenizers** pour le texte
- Des **pipelines** simples d'utilisation
- La bibliotheque `transformers`

### Utilisation dans notre projet

| Composant | Hugging Face ? | Justification |
|-----------|----------------|---------------|
| **BERT (texte)** | OUI | Indispensable pour charger BERT pre-entraine |
| **ViT (images)** | OPTIONNEL | Possible mais CNN simple suffit pour 48x48 |
| **CNN-LSTM (bio)** | NON | Architecture custom, pas de pre-training utile |

### Installation

```bash
pip install transformers torch datasets
```

### Ce que Hugging Face apporte

```python
# SANS Hugging Face (BERT from scratch):
# - Implementer 12 couches Transformer
# - Entrainer sur des milliards de mots
# - Plusieurs semaines de calcul GPU
# -> IMPOSSIBLE avec ressources limitees

# AVEC Hugging Face:
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
# -> Modele pret en 2 lignes, pre-entraine par Google
```

### Modeles recommandes pour le projet

```python
# Pour GoEmotions (texte anglais)
"bert-base-uncased"          # 110M params, bon compromis
"distilbert-base-uncased"    # 66M params, plus leger (recommande si RAM limitee)

# Pour FER2013 (images) - si on utilise ViT
"google/vit-base-patch16-224"  # Necessite resize 48->224
```

---

## 5. Strategie de Fusion Multimodale

### Late Fusion (Agregation de predictions)

Comme les datasets ne sont pas synchronises, on utilise la **late fusion**:

```
           +-------------+
           |  CNN-LSTM   | -> P_bio = [0.1, 0.7, 0.2]  (stress haut)
           +-------------+

           +-------------+
           |    CNN      | -> P_img = [0.0, 0.1, 0.8, 0.05, 0.05]  (happy)
           +-------------+

           +-------------+
           |    BERT     | -> P_txt = [0.2, 0.6, ...]  (28 emotions)
           +-------------+
                  |
                  v
        +---------+---------+
        |  Mapping Labels   |  <- Aligner les emotions
        +---------+---------+
                  |
                  v
        +---------+---------+
        |  Fusion           |
        |  - Moyenne        |
        |  - Vote majoritaire|
        |  - Weighted avg   |
        |  - MLP fusion     |
        +---------+---------+
                  |
                  v
           [Emotion Finale]
```

---

## 6. Resume: Technologies utilisees

| Technologie | Usage | Installation |
|-------------|-------|--------------|
| **PyTorch** | Framework deep learning | `pip install torch` |
| **Hugging Face Transformers** | BERT pre-entraine | `pip install transformers` |
| **NumPy/Pandas** | Manipulation donnees | `pip install numpy pandas` |
| **Scikit-learn** | Metrics, split | `pip install scikit-learn` |
| **Matplotlib/Seaborn** | Visualisation | `pip install matplotlib seaborn` |

### Commande d'installation complete

```bash
pip install torch transformers numpy pandas scikit-learn matplotlib seaborn tqdm
```

---

## 7. Considerations RAM limitee

### Strategies pour economiser la RAM

1. **Chunked processing** : Traiter les donnees par petits lots
2. **Gradient checkpointing** : Reduire memoire GPU
3. **DistilBERT** au lieu de BERT : 40% plus leger
4. **Mixed precision** : float16 au lieu de float32
5. **DataLoader avec workers** : Charger donnees progressivement

```python
# Exemple chunked processing
for chunk in pd.read_csv('data.csv', chunksize=1000):
    process(chunk)

# Exemple DataLoader
DataLoader(dataset, batch_size=16, num_workers=2, pin_memory=True)
```
