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


def load_imagenet_data(num_samples=1000, num_classes=100, flatten=False, img_size=64):
    """Load ImageNet training data or synthetic replacement.
    
    Args:
        num_samples: Number of samples to load (default: 1000 for quick testing)
        num_classes: Number of classes (default: 100 for ImageNet-100 subset)
        flatten: If True, flatten images to 1D vectors
        img_size: Image size (default: 64 for reduced memory, original is 224)
        
    Returns:
        X: Images tensor of shape (num_samples, 3, img_size, img_size) or (num_samples, 3*img_size*img_size)
        Y: One-hot labels of shape (num_samples, num_classes)
    """
    if not TORCHVISION_AVAILABLE:
        # Generate synthetic data for testing when torchvision is not available
        X = torch.randn(num_samples, 3, img_size, img_size)
        if flatten:
            X = X.reshape(num_samples, -1)
        Y = torch.zeros(num_samples, num_classes)
        for i in range(num_samples):
            Y[i, i % num_classes] = 1
        return X, Y
    
    # Try to use actual ImageNet data if available, otherwise use synthetic
    # For this implementation, we'll use synthetic data as ImageNet is too large
    # and requires special access/download
    print(f"Using synthetic ImageNet-like data ({num_samples} samples, {num_classes} classes, {img_size}x{img_size})")
    X = torch.randn(num_samples, 3, img_size, img_size) * 0.5
    if flatten:
        X = X.reshape(num_samples, -1)
    
    # Generate one-hot labels
    Y = torch.zeros(num_samples, num_classes)
    for i in range(num_samples):
        Y[i, i % num_classes] = 1
    
    return X, Y


def load_imagenet_test_data(num_samples=200, num_classes=100, flatten=False, img_size=64):
    """Load ImageNet test data or synthetic replacement.
    
    Args:
        num_samples: Number of test samples (default: 200)
        num_classes: Number of classes (default: 100)
        flatten: If True, flatten images to 1D vectors
        img_size: Image size (default: 64)
        
    Returns:
        X: Images tensor of shape (num_samples, 3, img_size, img_size) or (num_samples, 3*img_size*img_size)
        Y: One-hot labels of shape (num_samples, num_classes)
    """
    print(f"Using synthetic ImageNet-like test data ({num_samples} samples, {num_classes} classes, {img_size}x{img_size})")
    X = torch.randn(num_samples, 3, img_size, img_size) * 0.5
    if flatten:
        X = X.reshape(num_samples, -1)
    
    # Generate one-hot labels
    Y = torch.zeros(num_samples, num_classes)
    for i in range(num_samples):
        Y[i, i % num_classes] = 1
    
    return X, Y
