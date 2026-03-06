# to_the_point

**Analytical replacements for autograd layers.**

`to_the_point` provides drop-in analytical (closed-form) alternatives for common neural-network building blocks. Instead of learning weights through gradient-based backpropagation, every layer solves for optimal parameters using ridge regression, covariance accumulation, or other direct methods — making training a single-pass operation.

## Layers

| Layer | Description |
|---|---|
| `Linear` | Ridge-regression linear layer (`W = Σ_xx⁻¹ Σ_xy`) |
| `Conv2d` | Analytical convolution via unfolded patch regression |
| `Attention` | Multi-head self-attention fitted through covariance analysis |
| `Embedding` | Token (+ optional positional) embeddings fitted analytically |
| `Polynomial` | Polynomial feature expansion with ridge solve |
| `Model` | Sequential container with per-layer analytical fitting |
| `Residual` | Model with a residual (skip) connection |
| `Dense` | Linear + ReLU |
| `Flatten` | Reshape to `(batch, -1)` |
| `Recursive` | Linear layer with an internal memory stack |
| `UnEmbed` | Placeholder inverse-embedding layer |

## Quick start

```bash
pip install -e .
```
or
```bash
pip install git+https://github.com/infiplexity-pixel/to_the_point
```

```python
import torch
from to_the_point import Linear, Model, Flatten

# Create an analytical model
model = Model(
    Flatten(),
    Linear(784, 128),
    Linear(128, 10),
)

# One-shot analytical fit (no epochs, no optimizer)
X_train = torch.randn(1000, 1, 28, 28)
Y_train = torch.zeros(1000, 10)

model.fit(X_train, Y_train, batch_size=256)

# Inference works like any nn.Module
predictions = model(X_train)
```

## Utilities

| Utility | Description |
|---|---|
| `torch_to_analytical(fn)` | Wrap any pointwise function (e.g. `torch.relu`) as an analytical layer |
| `load_mnist_data` / `load_mnist_test_data` | Load MNIST (requires `torchvision`) |
| `load_cifar10_data` / `load_cifar10_test_data` | Load CIFAR-10 |
| `load_cifar100_data` / `load_cifar100_test_data` | Load CIFAR-100 |

## Testing

```bash
# Unit tests (synthetic data, no downloads)
pytest tests/ --ignore=tests/test_datasets.py -v

# Dataset integration tests (downloads MNIST, CIFAR-10, CIFAR-100)
pip install torchvision
pytest tests/test_datasets.py -v -m dataset
```

### Cosine-similarity test

`tests/test_cosine_similarity.py` trains the same attention configuration both analytically and via gradient descent on identical synthetic data, then asserts that the resulting attention matrices have positive cosine similarity — verifying that the analytical path learns a comparable representation.

## CI

The GitHub Actions workflow (`.github/workflows/tests.yml`) runs:

- **Unit tests** on every push and PR (Python 3.10 / 3.11 / 3.12).
- **Dataset benchmarks** on every release, training on MNIST, CIFAR-10, and CIFAR-100 and asserting above-chance accuracy.

## License

MIT
