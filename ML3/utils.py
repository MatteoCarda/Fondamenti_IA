"""
Utility functions for text preprocessing and metrics calculation.
"""

import re
import pickle
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
import torch


def clean_text(text: str) -> str:
    """
    Clean and normalize tweet text.
    
    Args:
        text: Raw tweet text
        
    Returns:
        Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtag symbol but keep text
    text = re.sub(r'#', '', text)
    
    # Remove special characters and digits, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Cleaned text
        
    Returns:
        List of tokens
    """
    return text.split()


def build_vocabulary(texts: List[str], min_freq: int = 2, max_vocab_size: int = 10000) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build vocabulary from list of texts.
    
    Args:
        texts: List of text strings
        min_freq: Minimum frequency for a word to be included
        max_vocab_size: Maximum vocabulary size
        
    Returns:
        word2idx: Dictionary mapping words to indices
        idx2word: Dictionary mapping indices to words
    """
    # Count word frequencies
    word_counts = Counter()
    for text in texts:
        tokens = tokenize(clean_text(text))
        word_counts.update(tokens)
    
    # Get most common words
    most_common = word_counts.most_common(max_vocab_size)
    
    # Filter by minimum frequency
    vocab_words = [word for word, count in most_common if count >= min_freq]
    
    # Create mappings (reserve 0 for padding, 1 for unknown)
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for idx, word in enumerate(vocab_words, start=2):
        word2idx[word] = idx
    
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    print(f"Vocabulary size: {len(word2idx)}")
    
    return word2idx, idx2word


def text_to_indices(text: str, word2idx: Dict[str, int]) -> List[int]:
    """
    Convert text to list of word indices.
    
    Args:
        text: Input text
        word2idx: Vocabulary mapping
        
    Returns:
        List of word indices
    """
    tokens = tokenize(clean_text(text))
    return [word2idx.get(token, word2idx['<UNK>']) for token in tokens]


def pad_sequences(sequences: List[List[int]], max_len: int = 50, padding_value: int = 0) -> np.ndarray:
    """
    Pad sequences to the same length.
    
    Args:
        sequences: List of sequences
        max_len: Maximum sequence length
        padding_value: Value to use for padding
        
    Returns:
        Padded sequences as numpy array
    """
    padded = np.full((len(sequences), max_len), padding_value, dtype=np.int64)
    
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]
    
    return padded


def save_vocabulary(word2idx: Dict[str, int], idx2word: Dict[int, str], filepath: str):
    """
    Save vocabulary to file.
    
    Args:
        word2idx: Word to index mapping
        idx2word: Index to word mapping
        filepath: Path to save file
    """
    with open(filepath, 'wb') as f:
        pickle.dump({'word2idx': word2idx, 'idx2word': idx2word}, f)
    print(f"Vocabulary saved to {filepath}")


def load_vocabulary(filepath: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Load vocabulary from file.
    
    Args:
        filepath: Path to vocabulary file
        
    Returns:
        word2idx and idx2word dictionaries
    """
    with open(filepath, 'rb') as f:
        vocab = pickle.load(f)
    print(f"Vocabulary loaded from {filepath}")
    return vocab['word2idx'], vocab['idx2word']


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    for i in range(num_classes):
        metrics[f'precision_class_{i}'] = precision_per_class[i] if i < len(precision_per_class) else 0
        metrics[f'recall_class_{i}'] = recall_per_class[i] if i < len(recall_per_class) else 0
        metrics[f'f1_class_{i}'] = f1_per_class[i] if i < len(f1_per_class) else 0
    
    return metrics


def print_metrics(metrics: Dict[str, float], class_names: List[str] = None):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        class_names: Optional names for classes
    """
    if class_names is None:
        class_names = ['negative', 'neutral', 'positive']
    
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    print("\nPer-class Metrics:")
    print("-"*50)
    for i, class_name in enumerate(class_names):
        if f'precision_class_{i}' in metrics:
            print(f"{class_name.capitalize()}:")
            print(f"  Precision: {metrics[f'precision_class_{i}']:.4f}")
            print(f"  Recall:    {metrics[f'recall_class_{i}']:.4f}")
            print(f"  F1-Score:  {metrics[f'f1_class_{i}']:.4f}")
    
    print("="*50)
