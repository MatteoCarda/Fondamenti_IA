import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()

        # Trasforma parola (indice) → vettore
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        # RNN con memoria (LSTM)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Classificatore finale
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        """

        # x → embedding
        x = self.embedding(x)
        # x: [batch_size, seq_len, embed_dim]

        # LSTM
        _, (hidden, _) = self.lstm(x)
        # hidden: [1, batch_size, hidden_dim]

        # Prendiamo l’ultimo hidden state
        out = hidden[-1]
        # out: [batch_size, hidden_dim]

        # Classificazione
        out = self.fc(out)
        # out: [batch_size, num_classes]

        return out
