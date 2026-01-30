# ML3 - Sentiment Analysis con RNN

Sistema di sentiment analysis basato su RNN (LSTM/GRU) per il dataset Twitter US Airline Sentiment, implementato in PyTorch.

## 📋 Descrizione

Questo progetto implementa un sistema completo di sentiment analysis per classificare tweet relativi a compagnie aeree in tre categorie:
- **Negative** (negativo)
- **Neutral** (neutrale)  
- **Positive** (positivo)

Il modello utilizza una architettura RNN con layer LSTM o GRU bidirezionali, supporto per GPU, validation set per early stopping, e metriche di valutazione dettagliate.

## 🏗️ Architettura

### Struttura del Progetto
```
ML3/
├── data_loader.py      # Caricamento e preprocessing dei dati
├── model.py            # Architettura del modello RNN
├── train.py            # Loop di training con validation
├── test.py             # Evaluation e visualizzazioni
├── utils.py            # Funzioni di utilità
├── main.py             # Script principale
├── requirements.txt    # Dipendenze
└── README.md          # Questa documentazione
```

### Architettura del Modello
1. **Embedding Layer**: converte token in embeddings (default: 100 dimensioni)
2. **RNN Layer**: LSTM o GRU bidirezionale (default: 2 layer, 128 hidden units)
3. **Dropout Layer**: regolarizzazione (default: 0.3)
4. **Fully Connected Layer**: classificazione finale (3 classi)

## 🚀 Installazione

### Prerequisiti
- Python 3.8+
- PyTorch 2.0+

### Installare le dipendenze
```bash
cd ML3
pip install -r requirements.txt
```

## 📊 Dataset

### Download
Il dataset deve essere scaricato manualmente da una di queste fonti:
- **Kaggle**: [Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)
- **OpenML**: [Dataset 43397](https://www.openml.org/d/43397)

### Posizionamento
Dopo il download, posizionare il file CSV come:
```
ML3/data/airline_tweets.csv
```

### Caratteristiche
- ~14,640 tweet
- 3 classi di sentiment
- Split: 60% train, 20% validation, 20% test

## 💻 Utilizzo

### Training del Modello

**Training base con parametri di default:**
```bash
python main.py --mode train
```

**Training con iperparametri personalizzati:**
```bash
python main.py --mode train \
    --num_epochs 30 \
    --batch_size 128 \
    --learning_rate 0.0005 \
    --hidden_dim 256 \
    --rnn_type GRU
```

### Testing del Modello

```bash
python main.py --mode test
```

### Parametri Disponibili

#### Parametri dei Dati
- `--data_dir`: Directory dei dati (default: `./data`)
- `--batch_size`: Dimensione del batch (default: `64`)
- `--max_len`: Lunghezza massima sequenza (default: `50`)
- `--vocab_size`: Dimensione massima vocabulary (default: `10000`)

#### Parametri del Modello
- `--embedding_dim`: Dimensione embeddings (default: `100`)
- `--hidden_dim`: Dimensione hidden state (default: `128`)
- `--num_layers`: Numero di layer RNN (default: `2`)
- `--dropout`: Dropout rate (default: `0.3`)
- `--bidirectional`: RNN bidirezionale (default: `True`)
- `--rnn_type`: Tipo di RNN - `LSTM` o `GRU` (default: `LSTM`)

#### Parametri di Training
- `--num_epochs`: Numero di epoch (default: `20`)
- `--learning_rate`: Learning rate (default: `0.001`)
- `--patience`: Patience per early stopping (default: `5`)

#### Altri Parametri
- `--mode`: Modalità - `train` o `test` (default: `train`)
- `--save_dir`: Directory per salvare i modelli (default: `./models`)
- `--no_cuda`: Disabilita GPU
- `--plot`: Genera grafici (default: `True`)
- `--examples`: Valuta esempi di tweet (default: `True`)

## 📈 Output

### Durante il Training
- Loss e accuracy per ogni epoch (train e validation)
- Early stopping quando validation non migliora
- Salvataggio del modello migliore

### Dopo il Testing
- Accuracy, Precision, Recall, F1-score globali
- Metriche per classe
- Confusion matrix (visualizzata e salvata come immagine)
- Predizioni su tweet di esempio

### File Generati
```
models/
├── best_model.pth              # Modello migliore salvato
├── training_history.png         # Grafico loss e accuracy
└── confusion_matrix.png         # Matrice di confusione

data/
└── vocabulary.pkl              # Vocabulary salvato
```

## 🔍 Esempio di Utilizzo

### Training Completo
```bash
# 1. Scaricare il dataset e posizionarlo in ML3/data/airline_tweets.csv

# 2. Eseguire il training
python main.py --mode train --num_epochs 20 --batch_size 64

# 3. Il modello migliore verrà salvato in ./models/best_model.pth
```

### Solo Testing
```bash
python main.py --mode test
```

### Predizione su Testo Personalizzato
```python
from train import load_model
from test import predict_sentiment
from model import create_model
from utils import load_vocabulary
import torch

# Carica vocabulary
word2idx, idx2word = load_vocabulary('./data/vocabulary.pkl')

# Crea e carica modello
model = create_model(len(word2idx))
model = load_model(model, './models/best_model.pth')

# Predici sentiment
text = "This airline is amazing! Great service!"
predict_sentiment(model, text, word2idx)
```

## 📊 Risultati Attesi

Con i parametri di default, ci si aspetta:
- **Test Accuracy**: 75-85%
- **Training Time**: 5-15 minuti (con GPU)
- **Early Stopping**: tipicamente dopo 10-15 epoch

Le prestazioni possono variare in base a:
- Iperparametri scelti
- Split casuale dei dati
- Hardware disponibile

## 🛠️ Componenti Principali

### `data_loader.py`
- Download e caricamento dataset
- Preprocessing del testo (cleaning, tokenization)
- Split train/val/test (60/20/20)
- Creazione del vocabulary
- PyTorch DataLoader

### `model.py`
- Classe `SentimentRNN`
- Supporto LSTM e GRU
- Configurazione bidirezionale
- Inizializzazione pesi

### `train.py`
- Loop di training
- Validation ad ogni epoch
- Early stopping
- Model checkpointing
- Supporto GPU

### `test.py`
- Valutazione su test set
- Calcolo metriche dettagliate
- Visualizzazioni (confusion matrix, training curves)
- Predizioni su esempi

### `utils.py`
- Text preprocessing
- Vocabulary building
- Sequence padding
- Metrics calculation

## 🎯 Caratteristiche Principali

✅ Architettura RNN con LSTM/GRU  
✅ Split train/validation/test  
✅ Early stopping basato su validation  
✅ Supporto GPU automatico  
✅ Metriche di valutazione complete  
✅ Visualizzazioni (confusion matrix, training history)  
✅ Hyperparameter tuning via CLI  
✅ Salvataggio del modello migliore  
✅ Predizione su nuovi testi  

## 📝 Note

- Il modello usa **padding** per uniformare le lunghezze delle sequenze
- Il **vocabulary** viene costruito solo sul training set
- Il **best model** viene selezionato in base alla validation accuracy
- **Early stopping** previene overfitting
- Le parole fuori vocabulary vengono mappate a `<UNK>`

## 🐛 Troubleshooting

**Errore: Dataset not found**
- Scaricare il dataset e posizionarlo in `ML3/data/airline_tweets.csv`

**CUDA out of memory**
- Ridurre `--batch_size` o `--hidden_dim`
- Usare `--no_cuda` per CPU

**Accuracy bassa**
- Aumentare `--num_epochs`
- Provare diversi iperparametri
- Usare `--rnn_type GRU` invece di LSTM

## 👨‍💻 Autore

Progetto sviluppato per il corso di Fondamenti di Intelligenza Artificiale.

## 📄 Licenza

Questo progetto è rilasciato sotto licenza MIT.
