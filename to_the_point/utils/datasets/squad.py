"""SQuAD dataset loader for to_the_point.

Provides tokenized SQuAD data for question-answering tasks.
Simplified version for testing analytical models.
"""
import torch


def load_squad_data(max_samples=1000, seq_length=100, num_classes=2):
    """Load SQuAD training data or synthetic replacement.
    
    For simplicity, we treat SQuAD as a binary classification task:
    predicting start/end positions or answer existence.
    
    Args:
        max_samples: Maximum number of samples
        seq_length: Sequence length for context+question
        num_classes: Number of output classes (2 for binary, or vocab_size for span prediction)
        
    Returns:
        X: Input sequences of shape (max_samples, seq_length)
        Y: Classification targets of shape (max_samples, num_classes)
    """
    # Generate synthetic QA data for testing
    print(f"Using synthetic SQuAD-like data ({max_samples} samples, seq_len={seq_length})")
    
    # Create synthetic token sequences (context + question)
    vocab_size = 10000
    X = torch.randint(0, vocab_size, (max_samples, seq_length))
    
    # Binary classification: does answer exist in context?
    Y = torch.zeros(max_samples, num_classes)
    for i in range(max_samples):
        # Simple pattern: classify based on first token
        class_idx = X[i, 0].item() % num_classes
        Y[i, class_idx] = 1
    
    return X.float(), Y


def load_squad_test_data(max_samples=200, seq_length=100, num_classes=2):
    """Load SQuAD test data or synthetic replacement.
    
    Args:
        max_samples: Maximum number of test samples
        seq_length: Sequence length
        num_classes: Number of output classes
        
    Returns:
        X: Input sequences of shape (max_samples, seq_length)
        Y: Classification targets of shape (max_samples, num_classes)
    """
    print(f"Using synthetic SQuAD-like test data ({max_samples} samples)")
    
    vocab_size = 10000
    X = torch.randint(0, vocab_size, (max_samples, seq_length))
    Y = torch.zeros(max_samples, num_classes)
    for i in range(max_samples):
        class_idx = X[i, 0].item() % num_classes
        Y[i, class_idx] = 1
    
    return X.float(), Y
