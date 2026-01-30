"""
RNN model architecture for sentiment analysis.
"""

import torch
import torch.nn as nn


class SentimentRNN(nn.Module):
    """
    RNN-based sentiment analysis model.
    
    Supports both LSTM and GRU architectures with optional bidirectionality.
    """
    
    def __init__(
        self,
        vocab_size,
        num_classes,
        embedding_dim=100,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        rnn_type='LSTM',
        padding_idx=0
    ):
        """
        Initialize the RNN model.
        
        Args:
            vocab_size: Size of the vocabulary
            num_classes: Number of output classes
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden state
            num_layers: Number of RNN layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional RNN
            rnn_type: Type of RNN ('LSTM' or 'GRU')
            padding_idx: Index used for padding
        """
        super(SentimentRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=padding_idx
        )
        
        # RNN layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"Invalid rnn_type: {rnn_type}. Must be 'LSTM' or 'GRU'")
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        # If bidirectional, hidden_dim is doubled
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate values."""
        # Initialize embedding weights
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Initialize RNN weights
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        # Initialize FC layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Embedding: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # RNN: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, hidden_dim * num_directions)
        if self.rnn_type == 'LSTM':
            rnn_out, (hidden, cell) = self.rnn(embedded)
        else:  # GRU
            rnn_out, hidden = self.rnn(embedded)
        
        # Use the last hidden state
        # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)
        
        if self.bidirectional:
            # Concatenate forward and backward hidden states from the last layer
            # Forward: hidden[-2, :, :]
            # Backward: hidden[-1, :, :]
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            # Just use the last layer
            hidden = hidden[-1, :, :]
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        # Fully connected layer: (batch_size, hidden_dim * num_directions) -> (batch_size, num_classes)
        logits = self.fc(hidden)
        
        return logits


def create_model(
    vocab_size,
    num_classes,
    embedding_dim=100,
    hidden_dim=128,
    num_layers=2,
    dropout=0.3,
    bidirectional=True,
    rnn_type='LSTM',
    padding_idx=0
):
    """
    Create and return a SentimentRNN model.
    
    Args:
        vocab_size: Size of the vocabulary
        num_classes: Number of output classes
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of hidden state
        num_layers: Number of RNN layers
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional RNN
        rnn_type: Type of RNN ('LSTM' or 'GRU')
        padding_idx: Index used for padding
        
    Returns:
        SentimentRNN model
    """
    model = SentimentRNN(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        rnn_type=rnn_type,
        padding_idx=padding_idx
    )
    
    # Print model summary
    print(f"Created {rnn_type} model:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Bidirectional: {bidirectional}")
    print(f"  Dropout: {dropout}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


if __name__ == "__main__":
    # Test the model creation
    print("Testing model creation...")
    
    # Test with default parameters
    model = create_model(vocab_size=10000, num_classes=3)
    
    # Test forward pass
    batch_size = 4
    seq_len = 50
    dummy_input = torch.randint(0, 10000, (batch_size, seq_len))
    
    output = model(dummy_input)
    print(f"\nTest forward pass:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output shape: ({batch_size}, 3)")
    
    print("\n✓ Model test successful!")
