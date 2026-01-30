import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)
        
        # rnn_out shape: (batch_size, seq_len, hidden_dim)
        # hidden shape: (num_layers, batch_size, hidden_dim)
        _, hidden = self.rnn(x)
        
        # Take the last hidden state
        # hidden shape is (1, batch_size, hidden_dim) since num_layers=1
        out = hidden[-1, :, :]
        
        out = self.fc(out)
        return out