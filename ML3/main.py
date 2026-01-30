"""
Main script to run sentiment analysis training and testing.
"""

import torch
from data_loader import prepare_data
from model import create_model
from train import train_model, load_model
from test import test_model

def main():
    """
    Main function to run the complete pipeline.
    """
    # Configuration
    DATA_DIR = './data'
    BATCH_SIZE = 64
    MAX_LEN = 50
    VOCAB_SIZE = 10000
    
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    RNN_TYPE = 'GRU'
    
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    PATIENCE = 5
    
    SAVE_DIR = './models'
    
    # Device - automatic detection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # Step 1: Prepare Data
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION")
    print("="*70)
    dataloaders, word2idx, idx2word, num_classes = prepare_data(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
        vocab_size=VOCAB_SIZE
    )
    
    vocab_size = len(word2idx)
    
    # Step 2: Create Model
    print("\n" + "="*70)
    print("STEP 2: MODEL CREATION")
    print("="*70)
    model = create_model(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=True,
        rnn_type=RNN_TYPE
    )
    
    # Step 3: Training
    print("\n" + "="*70)
    print("STEP 3: TRAINING")
    print("="*70)
    history, best_model_path = train_model(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
        save_dir=SAVE_DIR,
        patience=PATIENCE
    )
    
    # Step 4: Load Best Model and Test
    print("\n" + "="*70)
    print("STEP 4: TESTING")
    print("="*70)
    model = load_model(model, best_model_path, device=device)
    metrics = test_model(model, dataloaders['test'], device=device, num_classes=num_classes)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()