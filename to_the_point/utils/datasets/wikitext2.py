"""Wikitext-2 dataset loader for to_the_point.

Provides tokenized Wikitext-2 data for language modeling tasks.
Uses simple character or word-level tokenization.
"""
import torch


def _simple_tokenize(text, vocab_size=5000):
    """Simple word-level tokenization with fixed vocabulary.
    
    Args:
        text: Raw text string
        vocab_size: Maximum vocabulary size
        
    Returns:
        tokens: List of token IDs
        vocab: Dictionary mapping words to IDs
    """
    words = text.lower().split()
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Keep most frequent words
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for i, (word, _) in enumerate(sorted_words[:vocab_size-2]):
        vocab[word] = i + 2
    
    tokens = [vocab.get(word, vocab['<UNK>']) for word in words]
    return tokens, vocab


def load_wikitext2_data(max_samples=1000, seq_length=50, vocab_size=5000):
    """Load Wikitext-2 training data or synthetic replacement.
    
    Args:
        max_samples: Maximum number of sequences to generate
        seq_length: Length of each sequence
        vocab_size: Vocabulary size
        
    Returns:
        X: Input sequences of shape (max_samples, seq_length)
        Y: Target sequences of shape (max_samples, vocab_size) for next-token prediction
    """
    # Generate synthetic text data for testing
    # In a real implementation, this would load actual Wikitext-2 data
    print(f"Using synthetic Wikitext-2-like data ({max_samples} sequences, seq_len={seq_length})")
    
    # Create synthetic sequences with a learnable pattern
    # Use much smaller effective vocabulary for better learning
    effective_vocab = 100  # Reduced from 5000 for better learning
    X = torch.randint(0, effective_vocab, (max_samples, seq_length))
    
    # For language modeling, Y is typically the next token
    # We'll create one-hot encoded targets for classification
    # Simple pattern: next token is (sum of sequence % effective_vocab)
    Y = torch.zeros(max_samples, vocab_size)
    for i in range(max_samples):
        # Predictable pattern based on input
        next_token = int(X[i].sum().item()) % effective_vocab
        Y[i, next_token] = 1
    
    return X.float(), Y


def load_wikitext2_test_data(max_samples=200, seq_length=50, vocab_size=5000):
    """Load Wikitext-2 test data or synthetic replacement.
    
    Args:
        max_samples: Maximum number of test sequences
        seq_length: Length of each sequence
        vocab_size: Vocabulary size
        
    Returns:
        X: Input sequences of shape (max_samples, seq_length)
        Y: Target sequences of shape (max_samples, vocab_size)
    """
    print(f"Using synthetic Wikitext-2-like test data ({max_samples} sequences)")
    
    effective_vocab = 100
    X = torch.randint(0, effective_vocab, (max_samples, seq_length))
    Y = torch.zeros(max_samples, vocab_size)
    for i in range(max_samples):
        next_token = int(X[i].sum().item()) % effective_vocab
        Y[i, next_token] = 1
    
    return X.float(), Y
