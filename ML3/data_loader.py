"""
Data loading and preprocessing for Airlines Tweets Sentiment dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from utils import clean_text, text_to_indices, pad_sequences, build_vocabulary, save_vocabulary


class AirlineTweetsDataset(Dataset):
    """PyTorch Dataset for Airlines Tweets Sentiment."""
    
    def __init__(self, texts, labels, word2idx, max_len=50):
        """
        Initialize dataset.
        
        Args:
            texts: List of tweet texts
            labels: List of sentiment labels
            word2idx: Vocabulary mapping
            max_len: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len
        
        # Convert texts to indices
        sequences = [text_to_indices(text, word2idx) for text in texts]
        self.padded_sequences = pad_sequences(sequences, max_len=max_len)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.padded_sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


def download_and_load_data(data_dir='./data'):
    """
    Load the Airlines Tweets Sentiment dataset from OpenML (ID 43397).
    This is the dataset specified by the professor.
    
    Args:
        data_dir: Directory for caching (optional)
        
    Returns:
        DataFrame with tweets and sentiment labels  
    """
    os.makedirs(data_dir, exist_ok=True)
    
    print("Loading dataset from OpenML (ID: 43397)...")
    print("This is the dataset specified for the assignment.")
    
    try:
        from sklearn.datasets import fetch_openml
        
        # Fetch the dataset from OpenML
        # Dataset ID 43397 is the Twitter US Airline Sentiment dataset
        print("Fetching from OpenML...")
        data = fetch_openml(data_id=43397, as_frame=True, parser='auto')
        
        # Get the dataframe
        df = data.frame
        
        print(f"[OK] Successfully loaded {len(df)} rows from OpenML")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check what columns we have and standardize them
        print(f"\nDataset info:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # The OpenML dataset might have different column names
        # Let's check and rename appropriately
        if 'text' in df.columns and 'airline_sentiment' in df.columns:
            return df
        else:
            # Try to find the right columns
            # Common possibilities for OpenML format
            column_mapping = {}
            
            # Find text column
            for col in df.columns:
                col_lower = col.lower()
                if 'text' in col_lower or 'tweet' in col_lower or 'message' in col_lower:
                    column_mapping[col] = 'text'
                    break
            
            # Find sentiment column
            for col in df.columns:
                col_lower = col.lower()
                if 'sentiment' in col_lower or 'label' in col_lower or 'class' in col_lower or 'target' in col_lower:
                    column_mapping[col] = 'airline_sentiment'
                    break
            
            if column_mapping:
                df = df.rename(columns=column_mapping)
                print(f"Renamed columns: {column_mapping}")
            
            # If still missing columns, use the target from OpenML
            if 'airline_sentiment' not in df.columns and hasattr(data, 'target'):
                df['airline_sentiment'] = data.target
                print("Added 'airline_sentiment' from OpenML target")
            
            return df
        
    except ImportError:
        print("[ERROR] sklearn not installed properly")
        print("Please install: pip install scikit-learn")
        raise
        
    except Exception as e:
        print(f"[ERROR] OpenML loading failed: {e}")
        print("\nCould not load dataset from OpenML (ID 43397)")
        print("\nPlease check:")
        print("- Internet connection")
        print("- sklearn installation: pip install scikit-learn")
        print("\nOpenML dataset page:")
        print("https://www.openml.org/search?type=data&status=active&id=43397")
        raise FileNotFoundError(f"Could not load dataset from OpenML: {e}")


def preprocess_data(df, text_column='text', label_column='airline_sentiment'):
    """
    Preprocess the dataset.
    
    Args:
        df: Input DataFrame
        text_column: Name of text column
        label_column: Name of label column
        
    Returns:
        texts: List of cleaned texts
        labels: Numeric labels (0=negative, 1=neutral, 2=positive)
        label_mapping: Dictionary mapping label names to numbers
    """
    # Remove any rows with NaN in critical columns
    print(f"Initial dataset size: {len(df)} rows")
    
    # Check if columns exist
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found. Available columns: {df.columns.tolist()}")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found. Available columns: {df.columns.tolist()}")
    
    df = df.dropna(subset=[text_column, label_column])
    print(f"After removing NaN: {len(df)} rows")
    
    # Map sentiment to numeric labels
    label_mapping = {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    }
    
    texts = df[text_column].astype(str).tolist()
    
    # Check if labels are already numeric or need mapping
    if df[label_column].dtype in [int, float, 'int64', 'float64']:
        # Already numeric (OpenML format)
        labels = df[label_column].astype(int).tolist()
        print("Labels are already numeric (OpenML format)")
    else:
        # String labels need mapping
        labels = df[label_column].map(label_mapping).tolist()
        print("Mapped string labels to numeric values")
    
    # Remove any invalid entries
    valid_indices = []
    for i, (t, l) in enumerate(zip(texts, labels)):
        # Check if text is valid
        text_valid = (t and str(t).strip() != '' and 
                     str(t).lower() not in ['nan', 'none', 'null'])
        # Check if label is valid
        label_valid = (l is not None and not pd.isna(l) and 
                      isinstance(l, (int, np.integer)) and l in [0, 1, 2])
        
        if text_valid and label_valid:
            valid_indices.append(i)
    
    texts = [texts[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]
    
    print(f"Total valid samples: {len(texts)}")
    print(f"Label distribution: {pd.Series(labels).value_counts().sort_index().to_dict()}")
    
    if len(texts) == 0:
        raise ValueError("No valid samples found after preprocessing!")
    
    # Final check: ensure no NaN in labels
    if any(pd.isna(l) for l in labels):
        raise ValueError("Found NaN values in labels after preprocessing!")
    
    return texts, labels, label_mapping


def create_data_splits(texts, labels, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    """
    Split data into train, validation, and test sets.
    
    Args:
        texts: List of texts
        labels: List of labels
        train_size: Fraction for training
        val_size: Fraction for validation
        test_size: Fraction for test
        random_state: Random seed
        
    Returns:
        Dictionary with train, val, and test splits
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1.0"
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels, 
        test_size=(val_size + test_size),
        random_state=random_state,
        stratify=labels
    )
    
    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio),
        random_state=random_state,
        stratify=y_temp
    )
    
    print(f"\nData splits:")
    print(f"  Train:      {len(X_train)} samples ({train_size*100:.0f}%)")
    print(f"  Validation: {len(X_val)} samples ({val_size*100:.0f}%)")
    print(f"  Test:       {len(X_test)} samples ({test_size*100:.0f}%)")
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


def create_dataloaders(splits, word2idx, batch_size=64, max_len=50, num_workers=0):
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        splits: Dictionary with train, val, test splits
        word2idx: Vocabulary mapping
        batch_size: Batch size
        max_len: Maximum sequence length
        num_workers: Number of workers for data loading
        
    Returns:
        Dictionary with train, val, test DataLoaders
    """
    train_dataset = AirlineTweetsDataset(
        splits['train'][0], splits['train'][1], word2idx, max_len
    )
    val_dataset = AirlineTweetsDataset(
        splits['val'][0], splits['val'][1], word2idx, max_len
    )
    test_dataset = AirlineTweetsDataset(
        splits['test'][0], splits['test'][1], word2idx, max_len
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def prepare_data(data_dir='./data', batch_size=64, max_len=50, vocab_size=10000):
    """
    Complete data preparation pipeline.
    
    Args:
        data_dir: Directory containing data
        batch_size: Batch size for DataLoaders
        max_len: Maximum sequence length
        vocab_size: Maximum vocabulary size
        
    Returns:
        dataloaders: Dictionary with train, val, test DataLoaders
        word2idx: Vocabulary mapping
        idx2word: Reverse vocabulary mapping
        num_classes: Number of classes
    """
    print("="*50)
    print("DATA PREPARATION PIPELINE")
    print("="*50)
    
    # Load data
    print("\n1. Loading dataset...")
    df = download_and_load_data(data_dir)
    
    # Preprocess
    print("\n2. Preprocessing...")
    texts, labels, label_mapping = preprocess_data(df)
    
    # Create splits
    print("\n3. Creating data splits...")
    splits = create_data_splits(texts, labels)
    
    # Build vocabulary (only on training data)
    print("\n4. Building vocabulary...")
    train_texts = splits['train'][0]
    word2idx, idx2word = build_vocabulary(train_texts, min_freq=2, max_vocab_size=vocab_size)
    
    # Save vocabulary
    vocab_path = os.path.join(data_dir, 'vocabulary.pkl')
    save_vocabulary(word2idx, idx2word, vocab_path)
    
    # Create dataloaders
    print("\n5. Creating DataLoaders...")
    dataloaders = create_dataloaders(splits, word2idx, batch_size, max_len)
    
    num_classes = len(label_mapping)
    
    print("\n" + "="*50)
    print("DATA PREPARATION COMPLETE")
    print("="*50)
    print(f"Vocabulary size: {len(word2idx)}")
    print(f"Max sequence length: {max_len}")
    print(f"Batch size: {batch_size}")
    print(f"Number of classes: {num_classes}")
    print("="*50 + "\n")
    
    return dataloaders, word2idx, idx2word, num_classes


if __name__ == "__main__":
    # Test data loading
    dataloaders, word2idx, idx2word, num_classes = prepare_data()
    
    # Print sample batch
    for batch_texts, batch_labels in dataloaders['train']:
        print(f"Batch shape: {batch_texts.shape}")
        print(f"Labels shape: {batch_labels.shape}")
        print(f"Sample label: {batch_labels[0].item()}")
        break
