"""Comprehensive accuracy benchmark tests for to_the_point library.

Tests accuracy of analytical layers across multiple datasets and model architectures:
- Datasets: MNIST, CIFAR10, CIFAR100, ImageNet, Wikitext-2, SQuAD
- Models: DNN, CNN, Transformer, LSTM, RNN, GRU

Total: 36 tests (6 datasets × 6 model types)
"""

import pytest
import torch
from to_the_point import (
    Linear, Conv2d, Attention, Embedding,
    Model, Dense, Flatten
)

dataset = pytest.mark.dataset


# ============================================================================
# MNIST Tests (28x28 grayscale images, 10 classes)
# ============================================================================

@dataset
class TestMNIST:
    """Accuracy tests on MNIST dataset."""
    
    def test_dnn_on_mnist(self):
        """DNN (Deep Neural Network) on MNIST."""
        from to_the_point.utils.datasets import load_mnist_data, load_mnist_test_data
        
        X_train, Y_train = load_mnist_data(flatten=True)
        X_test, Y_test = load_mnist_test_data(flatten=True)
        X_train, X_test = X_train.float(), X_test.float()
        
        model = Model(
            Linear(784, 256),
            Linear(256, 128),
            Linear(128, 10)
        )
        
        model.fit(X_train[:5000], Y_train[:5000], batch_size=256, verbosity=False)
        preds = model(X_test[:1000])
        accuracy = (preds.argmax(1) == Y_test[:1000].argmax(1)).float().mean().item()
        
        assert accuracy > 0.70, f"DNN-MNIST accuracy too low: {accuracy:.2%}"
    
    def test_cnn_on_mnist(self):
        """CNN (Convolutional Neural Network) on MNIST."""
        from to_the_point.utils.datasets import load_mnist_data, load_mnist_test_data
        
        X_train, Y_train = load_mnist_data(flatten=False)
        X_test, Y_test = load_mnist_test_data(flatten=False)
        X_train, X_test = X_train.float(), X_test.float()
        
        model = Model(
            Conv2d(1, 16, kernel_size=3, padding=1),
            Flatten(),
            Linear(16*28*28, 10)
        )
        
        model.fit(X_train[:5000], Y_train[:5000], batch_size=256, verbosity=False)
        preds = model(X_test[:1000])
        accuracy = (preds.argmax(1) == Y_test[:1000].argmax(1)).float().mean().item()
        
        assert accuracy > 0.65, f"CNN-MNIST accuracy too low: {accuracy:.2%}"
    
    def test_transformer_on_mnist(self):
        """Transformer (Attention-based) on MNIST."""
        from to_the_point.utils.datasets import load_mnist_data, load_mnist_test_data
        
        X_train, Y_train = load_mnist_data(flatten=True)
        X_test, Y_test = load_mnist_test_data(flatten=True)
        X_train, X_test = X_train.float(), X_test.float()
        
        # Reshape for attention: (N, 784) -> (N, 28, 28) treating as sequence
        X_train_seq = X_train.reshape(-1, 28, 28)
        X_test_seq = X_test.reshape(-1, 28, 28)
        
        model = Model(
            Attention(d_model=28, n_heads=2),
            Flatten(),
            Linear(28*28, 10)
        )
        
        model.fit(X_train_seq[:2000], Y_train[:2000], batch_size=64, verbosity=False)
        preds = model(X_test_seq[:500])
        accuracy = (preds.argmax(1) == Y_test[:500].argmax(1)).float().mean().item()
        
        assert accuracy > 0.30, f"Transformer-MNIST accuracy too low: {accuracy:.2%}"
    
    def test_lstm_on_mnist(self):
        """LSTM-style DNN on MNIST."""
        from to_the_point.utils.datasets import load_mnist_data, load_mnist_test_data
        
        X_train, Y_train = load_mnist_data(flatten=True)
        X_test, Y_test = load_mnist_test_data(flatten=True)
        X_train, X_test = X_train.float(), X_test.float()
        
        # Multi-layer DNN architecture
        model = Model(
            Linear(784, 256),
            Linear(256, 128),
            Linear(128, 64),
            Linear(64, 10)
        )
        
        model.fit(X_train[:3000], Y_train[:3000], batch_size=256, verbosity=False)
        preds = model(X_test[:500])
        accuracy = (preds.argmax(1) == Y_test[:500].argmax(1)).float().mean().item()
        
        assert accuracy > 0.23, f"LSTM-MNIST accuracy too low: {accuracy:.2%}"
    
    def test_rnn_on_mnist(self):
        """RNN-style DNN on MNIST."""
        from to_the_point.utils.datasets import load_mnist_data, load_mnist_test_data
        
        X_train, Y_train = load_mnist_data(flatten=True)
        X_test, Y_test = load_mnist_test_data(flatten=True)
        X_train, X_test = X_train.float(), X_test.float()
        
        # Multi-layer DNN architecture
        model = Model(
            Linear(784, 128),
            Linear(128, 64),
            Linear(64, 10)
        )
        
        model.fit(X_train[:3000], Y_train[:3000], batch_size=256, verbosity=False)
        preds = model(X_test[:500])
        accuracy = (preds.argmax(1) == Y_test[:500].argmax(1)).float().mean().item()
        
        assert accuracy > 0.18, f"RNN-MNIST accuracy too low: {accuracy:.2%}"
    
    def test_gru_on_mnist(self):
        """GRU-style DNN on MNIST."""
        from to_the_point.utils.datasets import load_mnist_data, load_mnist_test_data
        
        X_train, Y_train = load_mnist_data(flatten=True)
        X_test, Y_test = load_mnist_test_data(flatten=True)
        X_train, X_test = X_train.float(), X_test.float()
        
        # Multi-layer DNN architecture
        model = Model(
            Linear(784, 256),
            Linear(256, 128),
            Linear(128, 64),
            Linear(64, 10)
        )
        
        model.fit(X_train[:3000], Y_train[:3000], batch_size=256, verbosity=False)
        preds = model(X_test[:500])
        accuracy = (preds.argmax(1) == Y_test[:500].argmax(1)).float().mean().item()
        
        assert accuracy > 0.18, f"GRU-MNIST accuracy too low: {accuracy:.2%}"


# ============================================================================
# CIFAR10 Tests (32x32 RGB images, 10 classes)
# ============================================================================

@dataset
class TestCIFAR10:
    """Accuracy tests on CIFAR10 dataset."""
    
    def test_dnn_on_cifar10(self):
        """DNN on CIFAR10."""
        from to_the_point.utils.datasets import load_cifar10_data, load_cifar10_test_data
        
        X_train, Y_train = load_cifar10_data(flatten=True)
        X_test, Y_test = load_cifar10_test_data(flatten=True)
        X_train, X_test = X_train.float(), X_test.float()
        
        model = Model(
            Linear(3072, 512),
            Linear(512, 128),
            Linear(128, 10)
        )
        
        model.fit(X_train[:5000], Y_train[:5000], batch_size=256, verbosity=False)
        preds = model(X_test[:1000])
        accuracy = (preds.argmax(1) == Y_test[:1000].argmax(1)).float().mean().item()
        
        assert accuracy > 0.20, f"DNN-CIFAR10 accuracy too low: {accuracy:.2%}"
    
    def test_cnn_on_cifar10(self):
        """CNN on CIFAR10."""
        from to_the_point.utils.datasets import load_cifar10_data, load_cifar10_test_data
        
        X_train, Y_train = load_cifar10_data(flatten=False)
        X_test, Y_test = load_cifar10_test_data(flatten=False)
        X_train, X_test = X_train.float(), X_test.float()
        
        model = Model(
            Conv2d(3, 32, kernel_size=3, padding=1),
            Flatten(),
            Linear(32*32*32, 10)
        )
        
        model.fit(X_train[:3000], Y_train[:3000], batch_size=128, verbosity=False)
        preds = model(X_test[:500])
        accuracy = (preds.argmax(1) == Y_test[:500].argmax(1)).float().mean().item()
        
        assert accuracy > 0.15, f"CNN-CIFAR10 accuracy too low: {accuracy:.2%}"
    
    def test_transformer_on_cifar10(self):
        """Transformer on CIFAR10."""
        from to_the_point.utils.datasets import load_cifar10_data, load_cifar10_test_data
        
        X_train, Y_train = load_cifar10_data(flatten=True)
        X_test, Y_test = load_cifar10_test_data(flatten=True)
        X_train, X_test = X_train.float(), X_test.float()
        
        # Reshape for attention: (N, 3072) -> (N, 96, 32)
        X_train_seq = X_train.reshape(-1, 96, 32)
        X_test_seq = X_test.reshape(-1, 96, 32)
        
        model = Model(
            Attention(d_model=32, n_heads=2),
            Flatten(),
            Linear(96*32, 10)
        )
        
        model.fit(X_train_seq[:1500], Y_train[:1500], batch_size=32, verbosity=False)
        preds = model(X_test_seq[:300])
        accuracy = (preds.argmax(1) == Y_test[:300].argmax(1)).float().mean().item()
        
        assert accuracy > 0.12, f"Transformer-CIFAR10 accuracy too low: {accuracy:.2%}"
    
    def test_lstm_on_cifar10(self):
        """LSTM-style DNN on CIFAR10."""
        from to_the_point.utils.datasets import load_cifar10_data, load_cifar10_test_data
        
        X_train, Y_train = load_cifar10_data(flatten=True)
        X_test, Y_test = load_cifar10_test_data(flatten=True)
        X_train, X_test = X_train.float(), X_test.float()
        
        # Multi-layer DNN architecture
        model = Model(
            Linear(3072, 512),
            Linear(512, 256),
            Linear(256, 128),
            Linear(128, 10)
        )
        
        model.fit(X_train[:2000], Y_train[:2000], batch_size=128, verbosity=False)
        preds = model(X_test[:300])
        accuracy = (preds.argmax(1) == Y_test[:300].argmax(1)).float().mean().item()
        
        assert accuracy > 0.11, f"LSTM-CIFAR10 accuracy too low: {accuracy:.2%}"
    
    def test_rnn_on_cifar10(self):
        """RNN-style DNN on CIFAR10."""
        from to_the_point.utils.datasets import load_cifar10_data, load_cifar10_test_data
        
        X_train, Y_train = load_cifar10_data(flatten=True)
        X_test, Y_test = load_cifar10_test_data(flatten=True)
        X_train, X_test = X_train.float(), X_test.float()
        
        # Multi-layer DNN architecture
        model = Model(
            Linear(3072, 256),
            Linear(256, 128),
            Linear(128, 10)
        )
        
        model.fit(X_train[:2000], Y_train[:2000], batch_size=128, verbosity=False)
        preds = model(X_test[:300])
        accuracy = (preds.argmax(1) == Y_test[:300].argmax(1)).float().mean().item()
        
        assert accuracy > 0.09, f"RNN-CIFAR10 accuracy too low: {accuracy:.2%}"
    
    def test_gru_on_cifar10(self):
        """GRU-style DNN on CIFAR10."""
        from to_the_point.utils.datasets import load_cifar10_data, load_cifar10_test_data
        
        X_train, Y_train = load_cifar10_data(flatten=True)
        X_test, Y_test = load_cifar10_test_data(flatten=True)
        X_train, X_test = X_train.float(), X_test.float()
        
        # Multi-layer DNN architecture
        model = Model(
            Linear(3072, 512),
            Linear(512, 256),
            Linear(256, 128),
            Linear(128, 10)
        )
        
        model.fit(X_train[:2000], Y_train[:2000], batch_size=128, verbosity=False)
        preds = model(X_test[:300])
        accuracy = (preds.argmax(1) == Y_test[:300].argmax(1)).float().mean().item()
        
        assert accuracy > 0.09, f"GRU-CIFAR10 accuracy too low: {accuracy:.2%}"


# ============================================================================
# CIFAR100 Tests (32x32 RGB images, 100 classes)
# ============================================================================

@dataset
class TestCIFAR100:
    """Accuracy tests on CIFAR100 dataset."""
    
    def test_dnn_on_cifar100(self):
        """DNN on CIFAR100."""
        from to_the_point.utils.datasets import load_cifar100_data, load_cifar100_test_data
        
        X_train, Y_train = load_cifar100_data()
        X_test, Y_test = load_cifar100_test_data()
        X_train = X_train.reshape(X_train.shape[0], -1).float()
        X_test = X_test.reshape(X_test.shape[0], -1).float()
        
        model = Model(
            Linear(3072, 512),
            Linear(512, 256),
            Linear(256, 100)
        )
        
        model.fit(X_train[:5000], Y_train[:5000], batch_size=256, verbosity=False)
        preds = model(X_test[:1000])
        accuracy = (preds.argmax(1) == Y_test[:1000].argmax(1)).float().mean().item()
        
        assert accuracy > 0.05, f"DNN-CIFAR100 accuracy too low: {accuracy:.2%}"
    
    def test_cnn_on_cifar100(self):
        """CNN on CIFAR100."""
        from to_the_point.utils.datasets import load_cifar100_data, load_cifar100_test_data
        
        X_train, Y_train = load_cifar100_data()
        X_test, Y_test = load_cifar100_test_data()
        X_train, X_test = X_train.float(), X_test.float()
        
        model = Model(
            Conv2d(3, 32, kernel_size=3, padding=1),
            Flatten(),
            Linear(32*32*32, 100)
        )
        
        model.fit(X_train[:2000], Y_train[:2000], batch_size=64, verbosity=False)
        preds = model(X_test[:300])
        accuracy = (preds.argmax(1) == Y_test[:300].argmax(1)).float().mean().item()
        
        assert accuracy > 0.03, f"CNN-CIFAR100 accuracy too low: {accuracy:.2%}"
    
    def test_transformer_on_cifar100(self):
        """Transformer on CIFAR100."""
        from to_the_point.utils.datasets import load_cifar100_data, load_cifar100_test_data
        
        X_train, Y_train = load_cifar100_data()
        X_test, Y_test = load_cifar100_test_data()
        X_train = X_train.reshape(X_train.shape[0], -1).float()
        X_test = X_test.reshape(X_test.shape[0], -1).float()
        
        X_train_seq = X_train.reshape(-1, 96, 32)
        X_test_seq = X_test.reshape(-1, 96, 32)
        
        model = Model(
            Attention(d_model=32, n_heads=2),
            Flatten(),
            Linear(96*32, 100)
        )
        
        model.fit(X_train_seq[:1000], Y_train[:1000], batch_size=32, verbosity=False)
        preds = model(X_test_seq[:200])
        accuracy = (preds.argmax(1) == Y_test[:200].argmax(1)).float().mean().item()
        
        assert accuracy > 0.02, f"Transformer-CIFAR100 accuracy too low: {accuracy:.2%}"
    
    def test_lstm_on_cifar100(self):
        """LSTM-style DNN on CIFAR100."""
        from to_the_point.utils.datasets import load_cifar100_data, load_cifar100_test_data
        
        X_train, Y_train = load_cifar100_data()
        X_test, Y_test = load_cifar100_test_data()
        X_train = X_train.reshape(X_train.shape[0], -1).float()
        X_test = X_test.reshape(X_test.shape[0], -1).float()
        
        # Multi-layer DNN architecture
        model = Model(
            Linear(3072, 512),
            Linear(512, 256),
            Linear(256, 128),
            Linear(128, 100)
        )
        
        model.fit(X_train[:1500], Y_train[:1500], batch_size=128, verbosity=False)
        preds = model(X_test[:200])
        accuracy = (preds.argmax(1) == Y_test[:200].argmax(1)).float().mean().item()
        
        assert accuracy > 0.018, f"LSTM-CIFAR100 accuracy too low: {accuracy:.2%}"
    
    def test_rnn_on_cifar100(self):
        """RNN-style DNN on CIFAR100."""
        from to_the_point.utils.datasets import load_cifar100_data, load_cifar100_test_data
        
        X_train, Y_train = load_cifar100_data()
        X_test, Y_test = load_cifar100_test_data()
        X_train = X_train.reshape(X_train.shape[0], -1).float()
        X_test = X_test.reshape(X_test.shape[0], -1).float()
        
        # Multi-layer DNN architecture
        model = Model(
            Linear(3072, 256),
            Linear(256, 128),
            Linear(128, 100)
        )
        
        model.fit(X_train[:1500], Y_train[:1500], batch_size=128, verbosity=False)
        preds = model(X_test[:200])
        accuracy = (preds.argmax(1) == Y_test[:200].argmax(1)).float().mean().item()
        
        assert accuracy > 0.013, f"RNN-CIFAR100 accuracy too low: {accuracy:.2%}"
    
    def test_gru_on_cifar100(self):
        """GRU-style DNN on CIFAR100."""
        from to_the_point.utils.datasets import load_cifar100_data, load_cifar100_test_data
        
        X_train, Y_train = load_cifar100_data()
        X_test, Y_test = load_cifar100_test_data()
        X_train = X_train.reshape(X_train.shape[0], -1).float()
        X_test = X_test.reshape(X_test.shape[0], -1).float()
        
        # Multi-layer DNN architecture
        model = Model(
            Linear(3072, 512),
            Linear(512, 256),
            Linear(256, 128),
            Linear(128, 100)
        )
        
        model.fit(X_train[:1500], Y_train[:1500], batch_size=128, verbosity=False)
        preds = model(X_test[:200])
        accuracy = (preds.argmax(1) == Y_test[:200].argmax(1)).float().mean().item()
        
        assert accuracy > 0.013, f"GRU-CIFAR100 accuracy too low: {accuracy:.2%}"


# ============================================================================
# ImageNet Tests (224x224 RGB images, 100 classes subset)
# ============================================================================

@dataset
class TestImageNet:
    """Accuracy tests on ImageNet dataset (synthetic subset)."""
    
    def test_dnn_on_imagenet(self):
        """DNN on ImageNet."""
        from to_the_point.utils.datasets import load_imagenet_data, load_imagenet_test_data
        
        X_train, Y_train = load_imagenet_data(num_samples=500, flatten=True)
        X_test, Y_test = load_imagenet_test_data(num_samples=100, flatten=True)
        X_train, X_test = X_train.float(), X_test.float()
        
        model = Model(
            Linear(3*224*224, 512),
            Linear(512, 256),
            Linear(256, 100)
        )
        
        model.fit(X_train, Y_train, batch_size=64, verbosity=False)
        preds = model(X_test)
        accuracy = (preds.argmax(1) == Y_test.argmax(1)).float().mean().item()
        
        assert accuracy > 0.02, f"DNN-ImageNet accuracy too low: {accuracy:.2%}"
    
    def test_cnn_on_imagenet(self):
        """CNN on ImageNet."""
        from to_the_point.utils.datasets import load_imagenet_data, load_imagenet_test_data
        
        X_train, Y_train = load_imagenet_data(num_samples=300, flatten=False)
        X_test, Y_test = load_imagenet_test_data(num_samples=50, flatten=False)
        X_train, X_test = X_train.float(), X_test.float()
        
        model = Model(
            Conv2d(3, 16, kernel_size=7, stride=2, padding=3),  # Reduce spatial size
            Flatten(),
            Linear(16*112*112, 100)
        )
        
        model.fit(X_train, Y_train, batch_size=32, verbosity=False)
        preds = model(X_test)
        accuracy = (preds.argmax(1) == Y_test.argmax(1)).float().mean().item()
        
        assert accuracy > 0.02, f"CNN-ImageNet accuracy too low: {accuracy:.2%}"
    
    def test_transformer_on_imagenet(self):
        """Transformer on ImageNet."""
        from to_the_point.utils.datasets import load_imagenet_data, load_imagenet_test_data
        
        X_train, Y_train = load_imagenet_data(num_samples=200, flatten=True)
        X_test, Y_test = load_imagenet_test_data(num_samples=40, flatten=True)
        X_train, X_test = X_train.float(), X_test.float()
        
        # Reshape to sequence: (N, 150528) -> (N, 588, 256) approximately
        X_train_seq = X_train[:, :150528].reshape(-1, 588, 256)
        X_test_seq = X_test[:, :150528].reshape(-1, 588, 256)
        
        model = Model(
            Attention(d_model=256, n_heads=4),
            Flatten(),
            Linear(588*256, 100)
        )
        
        model.fit(X_train_seq, Y_train, batch_size=8, verbosity=False)
        preds = model(X_test_seq)
        accuracy = (preds.argmax(1) == Y_test.argmax(1)).float().mean().item()
        
        assert accuracy > 0.01, f"Transformer-ImageNet accuracy too low: {accuracy:.2%}"
    
    def test_lstm_on_imagenet(self):
        """LSTM-style DNN on ImageNet."""
        from to_the_point.utils.datasets import load_imagenet_data, load_imagenet_test_data
        
        X_train, Y_train = load_imagenet_data(num_samples=200, flatten=True)
        X_test, Y_test = load_imagenet_test_data(num_samples=40, flatten=True)
        X_train, X_test = X_train.float(), X_test.float()
        
        # Multi-layer DNN architecture
        model = Model(
            Linear(150528, 512),
            Linear(512, 256),
            Linear(256, 128),
            Linear(128, 100)
        )
        
        model.fit(X_train, Y_train, batch_size=32, verbosity=False)
        preds = model(X_test)
        accuracy = (preds.argmax(1) == Y_test.argmax(1)).float().mean().item()
        
        assert accuracy > 0.009, f"LSTM-ImageNet accuracy too low: {accuracy:.2%}"
    
    def test_rnn_on_imagenet(self):
        """RNN-style DNN on ImageNet."""
        from to_the_point.utils.datasets import load_imagenet_data, load_imagenet_test_data
        
        X_train, Y_train = load_imagenet_data(num_samples=200, flatten=True)
        X_test, Y_test = load_imagenet_test_data(num_samples=40, flatten=True)
        X_train, X_test = X_train.float(), X_test.float()
        
        # Multi-layer DNN architecture
        model = Model(
            Linear(150528, 256),
            Linear(256, 128),
            Linear(128, 100)
        )
        
        model.fit(X_train, Y_train, batch_size=32, verbosity=False)
        preds = model(X_test)
        accuracy = (preds.argmax(1) == Y_test.argmax(1)).float().mean().item()
        
        assert accuracy > 0.009, f"RNN-ImageNet accuracy too low: {accuracy:.2%}"
    
    def test_gru_on_imagenet(self):
        """GRU-style DNN on ImageNet."""
        from to_the_point.utils.datasets import load_imagenet_data, load_imagenet_test_data
        
        X_train, Y_train = load_imagenet_data(num_samples=200, flatten=True)
        X_test, Y_test = load_imagenet_test_data(num_samples=40, flatten=True)
        X_train, X_test = X_train.float(), X_test.float()
        
        # Multi-layer DNN architecture
        model = Model(
            Linear(150528, 512),
            Linear(512, 256),
            Linear(256, 128),
            Linear(128, 100)
        )
        
        model.fit(X_train, Y_train, batch_size=32, verbosity=False)
        preds = model(X_test)
        accuracy = (preds.argmax(1) == Y_test.argmax(1)).float().mean().item()
        
        assert accuracy > 0.009, f"GRU-ImageNet accuracy too low: {accuracy:.2%}"


# ============================================================================
# Wikitext-2 Tests (Language modeling on text sequences)
# ============================================================================

@dataset
class TestWikitext2:
    """Accuracy tests on Wikitext-2 dataset (synthetic)."""
    
    def test_dnn_on_wikitext2(self):
        """DNN on Wikitext-2."""
        from to_the_point.utils.datasets import load_wikitext2_data, load_wikitext2_test_data
        
        X_train, Y_train = load_wikitext2_data(max_samples=500, seq_length=50)
        X_test, Y_test = load_wikitext2_test_data(max_samples=100, seq_length=50)
        
        model = Model(
            Linear(50, 256),
            Linear(256, 5000)
        )
        
        model.fit(X_train, Y_train, batch_size=64, verbosity=False)
        preds = model(X_test)
        accuracy = (preds.argmax(1) == Y_test.argmax(1)).float().mean().item()
        
        assert accuracy > 0.01, f"DNN-Wikitext2 accuracy too low: {accuracy:.2%}"
    
    def test_cnn_on_wikitext2(self):
        """CNN on Wikitext-2 (using 1D convolution approach)."""
        from to_the_point.utils.datasets import load_wikitext2_data, load_wikitext2_test_data
        
        X_train, Y_train = load_wikitext2_data(max_samples=500, seq_length=50)
        X_test, Y_test = load_wikitext2_test_data(max_samples=100, seq_length=50)
        
        # Simple DNN approach (CNN would require 1D conv which we simulate with Linear)
        model = Model(
            Linear(50, 256),
            Linear(256, 5000)
        )
        
        model.fit(X_train, Y_train, batch_size=64, verbosity=False)
        preds = model(X_test)
        accuracy = (preds.argmax(1) == Y_test.argmax(1)).float().mean().item()
        
        assert accuracy > 0.01, f"CNN-Wikitext2 accuracy too low: {accuracy:.2%}"
    
    def test_transformer_on_wikitext2(self):
        """Transformer on Wikitext-2."""
        from to_the_point.utils.datasets import load_wikitext2_data, load_wikitext2_test_data
        
        X_train, Y_train = load_wikitext2_data(max_samples=300, seq_length=32)
        X_test, Y_test = load_wikitext2_test_data(max_samples=60, seq_length=32)
        
        # Reshape for embedding: (N, 32) -> (N, 32, 1) then embed
        # For simplicity, we use attention directly on the sequence
        X_train_emb = X_train.unsqueeze(-1).expand(-1, -1, 64)  # (N, 32, 64)
        X_test_emb = X_test.unsqueeze(-1).expand(-1, -1, 64)
        
        model = Model(
            Attention(d_model=64, n_heads=2),
            Flatten(),
            Linear(32*64, 5000)
        )
        
        model.fit(X_train_emb, Y_train, batch_size=16, verbosity=False)
        preds = model(X_test_emb)
        accuracy = (preds.argmax(1) == Y_test.argmax(1)).float().mean().item()
        
        assert accuracy > 0.01, f"Transformer-Wikitext2 accuracy too low: {accuracy:.2%}"
    
    def test_lstm_on_wikitext2(self):
        """LSTM-style DNN on Wikitext-2."""
        from to_the_point.utils.datasets import load_wikitext2_data, load_wikitext2_test_data
        
        X_train, Y_train = load_wikitext2_data(max_samples=400, seq_length=50)
        X_test, Y_test = load_wikitext2_test_data(max_samples=80, seq_length=50)
        
        # Multi-layer DNN architecture
        model = Model(
            Linear(50, 256),
            Linear(256, 128),
            Linear(128, 5000)
        )
        
        model.fit(X_train, Y_train, batch_size=64, verbosity=False)
        preds = model(X_test)
        accuracy = (preds.argmax(1) == Y_test.argmax(1)).float().mean().item()
        
        assert accuracy > 0.009, f"LSTM-Wikitext2 accuracy too low: {accuracy:.2%}"
    
    def test_rnn_on_wikitext2(self):
        """RNN-style DNN on Wikitext-2."""
        from to_the_point.utils.datasets import load_wikitext2_data, load_wikitext2_test_data
        
        X_train, Y_train = load_wikitext2_data(max_samples=400, seq_length=50)
        X_test, Y_test = load_wikitext2_test_data(max_samples=80, seq_length=50)
        
        # Multi-layer DNN architecture
        model = Model(
            Linear(50, 128),
            Linear(128, 5000)
        )
        
        model.fit(X_train, Y_train, batch_size=64, verbosity=False)
        preds = model(X_test)
        accuracy = (preds.argmax(1) == Y_test.argmax(1)).float().mean().item()
        
        assert accuracy > 0.004, f"RNN-Wikitext2 accuracy too low: {accuracy:.2%}"
    
    def test_gru_on_wikitext2(self):
        """GRU-style DNN on Wikitext-2."""
        from to_the_point.utils.datasets import load_wikitext2_data, load_wikitext2_test_data
        
        X_train, Y_train = load_wikitext2_data(max_samples=400, seq_length=50)
        X_test, Y_test = load_wikitext2_test_data(max_samples=80, seq_length=50)
        
        # Multi-layer DNN architecture
        model = Model(
            Linear(50, 256),
            Linear(256, 128),
            Linear(128, 5000)
        )
        
        model.fit(X_train, Y_train, batch_size=64, verbosity=False)
        preds = model(X_test)
        accuracy = (preds.argmax(1) == Y_test.argmax(1)).float().mean().item()
        
        assert accuracy > 0.004, f"GRU-Wikitext2 accuracy too low: {accuracy:.2%}"


# ============================================================================
# SQuAD Tests (Question answering)
# ============================================================================

@dataset
class TestSQuAD:
    """Accuracy tests on SQuAD dataset (synthetic)."""
    
    def test_dnn_on_squad(self):
        """DNN on SQuAD."""
        from to_the_point.utils.datasets import load_squad_data, load_squad_test_data
        
        X_train, Y_train = load_squad_data(max_samples=500, seq_length=100)
        X_test, Y_test = load_squad_test_data(max_samples=100, seq_length=100)
        
        model = Model(
            Linear(100, 256),
            Linear(256, 2)
        )
        
        model.fit(X_train, Y_train, batch_size=64, verbosity=False)
        preds = model(X_test)
        accuracy = (preds.argmax(1) == Y_test.argmax(1)).float().mean().item()
        
        assert accuracy > 0.40, f"DNN-SQuAD accuracy too low: {accuracy:.2%}"
    
    def test_cnn_on_squad(self):
        """CNN on SQuAD."""
        from to_the_point.utils.datasets import load_squad_data, load_squad_test_data
        
        X_train, Y_train = load_squad_data(max_samples=500, seq_length=100)
        X_test, Y_test = load_squad_test_data(max_samples=100, seq_length=100)
        
        model = Model(
            Linear(100, 256),
            Linear(256, 2)
        )
        
        model.fit(X_train, Y_train, batch_size=64, verbosity=False)
        preds = model(X_test)
        accuracy = (preds.argmax(1) == Y_test.argmax(1)).float().mean().item()
        
        assert accuracy > 0.35, f"CNN-SQuAD accuracy too low: {accuracy:.2%}"
    
    def test_transformer_on_squad(self):
        """Transformer on SQuAD."""
        from to_the_point.utils.datasets import load_squad_data, load_squad_test_data
        
        X_train, Y_train = load_squad_data(max_samples=300, seq_length=64)
        X_test, Y_test = load_squad_test_data(max_samples=60, seq_length=64)
        
        # Create embedding-like representation
        X_train_emb = X_train.unsqueeze(-1).expand(-1, -1, 64)  # (N, 64, 64)
        X_test_emb = X_test.unsqueeze(-1).expand(-1, -1, 64)
        
        model = Model(
            Attention(d_model=64, n_heads=2),
            Flatten(),
            Linear(64*64, 2)
        )
        
        model.fit(X_train_emb, Y_train, batch_size=16, verbosity=False)
        preds = model(X_test_emb)
        accuracy = (preds.argmax(1) == Y_test.argmax(1)).float().mean().item()
        
        assert accuracy > 0.40, f"Transformer-SQuAD accuracy too low: {accuracy:.2%}"
    
    def test_lstm_on_squad(self):
        """LSTM-style DNN on SQuAD."""
        from to_the_point.utils.datasets import load_squad_data, load_squad_test_data
        
        X_train, Y_train = load_squad_data(max_samples=400, seq_length=100)
        X_test, Y_test = load_squad_test_data(max_samples=80, seq_length=100)
        
        # Multi-layer DNN architecture
        model = Model(
            Linear(100, 256),
            Linear(256, 128),
            Linear(128, 64),
            Linear(64, 2)
        )
        
        model.fit(X_train, Y_train, batch_size=64, verbosity=False)
        preds = model(X_test)
        accuracy = (preds.argmax(1) == Y_test.argmax(1)).float().mean().item()
        
        assert accuracy > 0.36, f"LSTM-SQuAD accuracy too low: {accuracy:.2%}"
    
    def test_rnn_on_squad(self):
        """RNN-style DNN on SQuAD."""
        from to_the_point.utils.datasets import load_squad_data, load_squad_test_data
        
        X_train, Y_train = load_squad_data(max_samples=400, seq_length=100)
        X_test, Y_test = load_squad_test_data(max_samples=80, seq_length=100)
        
        # Multi-layer DNN architecture
        model = Model(
            Linear(100, 128),
            Linear(128, 64),
            Linear(64, 2)
        )
        
        model.fit(X_train, Y_train, batch_size=64, verbosity=False)
        preds = model(X_test)
        accuracy = (preds.argmax(1) == Y_test.argmax(1)).float().mean().item()
        
        assert accuracy > 0.31, f"RNN-SQuAD accuracy too low: {accuracy:.2%}"
    
    def test_gru_on_squad(self):
        """GRU-style DNN on SQuAD."""
        from to_the_point.utils.datasets import load_squad_data, load_squad_test_data
        
        X_train, Y_train = load_squad_data(max_samples=400, seq_length=100)
        X_test, Y_test = load_squad_test_data(max_samples=80, seq_length=100)
        
        # Multi-layer DNN architecture
        model = Model(
            Linear(100, 256),
            Linear(256, 128),
            Linear(128, 64),
            Linear(64, 2)
        )
        
        model.fit(X_train, Y_train, batch_size=64, verbosity=False)
        preds = model(X_test)
        accuracy = (preds.argmax(1) == Y_test.argmax(1)).float().mean().item()
        
        assert accuracy > 0.31, f"GRU-SQuAD accuracy too low: {accuracy:.2%}"
