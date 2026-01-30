from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from model import SentimentRNN
import pandas as pd
import torch
import torch.nn as nn

def load_dataset():
    dataset = fetch_openml(
        data_id=43397,
        as_frame=True
    )
    df = dataset.frame
    return df

def prepare_data(df):
    texts = df["tweet_text"].astype(str)
    raw_labels = df["tweet_sentiment_value"].astype(int)

    # Mappatura esplicita
    label_map = {
        -1: 0,  # negative
         0: 1,  # neutral
         1: 2   # positive
    }

    labels = raw_labels.map(label_map)

    # rimuoviamo eventuali righe con label non mappate
    mask = labels.notna()
    texts = texts[mask]
    labels = labels[mask].astype(int)

    return texts.reset_index(drop=True), labels.reset_index(drop=True)


def tokenize(text):
    text = text.lower()
    return text.split()

def build_vocab(texts, max_vocab_size=10000):
    counter = Counter()

    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, (word, _) in enumerate(counter.most_common(max_vocab_size - 2), start=2):
        vocab[word] = i

    return vocab

def text_to_sequence(text, vocab):
    tokens = tokenize(text)
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

def pad_sequence(seq, max_len):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [0] * (max_len - len(seq))

class TweetDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]

        seq = text_to_sequence(text, self.vocab)
        seq = pad_sequence(seq, self.max_len)

        return torch.tensor(seq), torch.tensor(label)





if __name__ == "__main__":

    # 1. Carico dataset
    df = load_dataset()
    texts, labels = prepare_data(df)

    # 2. Split train / test
    texts_train, texts_test, labels_train, labels_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # 2.1 Split train / validation (10% del train)
    texts_train, texts_val, labels_train, labels_val = train_test_split(
        texts_train,
        labels_train,
        test_size=0.1,
        random_state=42,
        stratify=labels_train
    )

    # 3. Vocabolario (SOLO dai testi di training)
    vocab = build_vocab(texts_train)

    # 4. Dataset e DataLoader
    BATCH_SIZE = 32
    MAX_LEN = 20

    train_dataset = TweetDataset(texts_train, labels_train, vocab, max_len=MAX_LEN)
    val_dataset = TweetDataset(texts_val, labels_val, vocab, max_len=MAX_LEN)
    test_dataset = TweetDataset(texts_test, labels_test, vocab, max_len=MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Modello
    VOCAB_SIZE = len(vocab)
    EMBED_DIM = 3000
    HIDDEN_DIM = 32
    NUM_LAYERS = 2
    NUM_CLASSES = 3

    model = SentimentRNN(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        num_layers=NUM_LAYERS
    )

    # 6. Loss e optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 7. Training
    EPOCHS = 50
    PATIENCE = 3
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        total_loss = 0.0
        model.train()

        for inputs, targets in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_outputs = model(val_inputs)
                v_loss = criterion(val_outputs, val_targets)
                val_loss += v_loss.item()
                
                val_predictions = val_outputs.argmax(dim=1)
                val_correct += (val_predictions == val_targets).sum().item()
                val_total += val_targets.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        
        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            # print("  Model saved!")
        else:
            patience_counter += 1
            # print(f"  Patience {patience_counter}/{PATIENCE}")
            
        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 8. TEST
    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
