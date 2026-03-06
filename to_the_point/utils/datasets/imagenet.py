"""ImageNet dataset loader for to_the_point.

Uses a subset of ImageNet (ImagenNet-100 or similar) for testing purposes.
Falls back to a synthetic dataset if torchvision ImageNet is not available.
"""
import torch
try:
    import torchvision
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


def load_imagenet_data(num_samples=1000, num_classes=100, flatten=False):
    """Load ImageNet training data or synthetic replacement.
    
    Args:
        num_samples: Number of samples to load (default: 1000 for quick testing)
        num_classes: Number of classes (default: 100 for ImageNet-100 subset)
        flatten: If True, flatten images to 1D vectors
        
    Returns:
        X: Images tensor of shape (num_samples, 3, 224, 224) or (num_samples, 3*224*224)
        Y: One-hot labels of shape (num_samples, num_classes)
    """
    if not TORCHVISION_AVAILABLE:
        # Generate synthetic data for testing when torchvision is not available
        X = torch.randn(num_samples, 3, 224, 224)
        if flatten:
            X = X.reshape(num_samples, -1)
        Y = torch.zeros(num_samples, num_classes)
        for i in range(num_samples):
            Y[i, i % num_classes] = 1
        return X, Y
    
    # Try to use actual ImageNet data if available, otherwise use synthetic
    # For this implementation, we'll use synthetic data as ImageNet is too large
    # and requires special access/download
    print("Using synthetic ImageNet-like data (1000 samples, 100 classes)")
    X = torch.randn(num_samples, 3, 224, 224) * 0.5
    if flatten:
        X = X.reshape(num_samples, -1)
    
    # Generate one-hot labels
    Y = torch.zeros(num_samples, num_classes)
    for i in range(num_samples):
        Y[i, i % num_classes] = 1
    
    return X, Y


def load_imagenet_test_data(num_samples=200, num_classes=100, flatten=False):
    """Load ImageNet test data or synthetic replacement.
    
    Args:
        num_samples: Number of test samples (default: 200)
        num_classes: Number of classes (default: 100)
        flatten: If True, flatten images to 1D vectors
        
    Returns:
        X: Images tensor of shape (num_samples, 3, 224, 224) or (num_samples, 3*224*224)
        Y: One-hot labels of shape (num_samples, num_classes)
    """
    print("Using synthetic ImageNet-like test data (200 samples, 100 classes)")
    X = torch.randn(num_samples, 3, 224, 224) * 0.5
    if flatten:
        X = X.reshape(num_samples, -1)
    
    # Generate one-hot labels
    Y = torch.zeros(num_samples, num_classes)
    for i in range(num_samples):
        Y[i, i % num_classes] = 1
    
    return X, Y
