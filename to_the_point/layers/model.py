import torch
import math
from tqdm import tqdm

from .template import AnalyticalBase
from .linear import Linear


class Model(torch.nn.Module):
    """Sequential container for analytical layers.

    Works like ``torch.nn.Sequential`` but supports analytical fitting via
    ``fit`` / ``fit_batch`` / ``finalize_fit``.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x, *args):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def fit(
        self,
        X: torch.Tensor,
        Y,
        P=None,
        batch_size=None,
        momentum=0,
        verbosity=True,
        finalize=True,
        *args,
        **kwargs,
    ):
        num_samples = X.shape[0]
        if batch_size is None:
            batch_size = num_samples

        for layer_idx, layer in enumerate(
            tqdm(self.layers, desc="Layers") if verbosity else self.layers
        ):
            if hasattr(layer, "fit_batch"):
                self._fit_layer_batched(
                    layer,
                    layer_idx,
                    X,
                    Y,
                    P,
                    batch_size,
                    momentum,
                    verbosity,
                    finalize,
                    *args,
                    **kwargs,
                )

            if hasattr(layer, "fit"):
                layer.fit(
                    X,
                    Y,
                    P=P,
                    batch_size=batch_size,
                    momentum=momentum,
                    verbosity=verbosity,
                    finalize=finalize,
                    *args,
                    **kwargs,
                )

            X = self._forward_layer_batched(
                layer, layer_idx, X, batch_size, verbosity
            )

    def _fit_layer_batched(
        self,
        layer,
        layer_idx,
        X,
        Y,
        P,
        batch_size,
        momentum,
        verbosity,
        finalize,
        *args,
        **kwargs,
    ):
        num_samples = X.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        for batch_idx in (
            tqdm(range(num_batches), desc=f"Layer {layer_idx} Fit")
            if verbosity
            else range(num_batches)
        ):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)

            Y_batch = Y[start_idx:end_idx]
            P_batch = P[start_idx:end_idx] if P is not None else None
            X_batch = X[start_idx:end_idx]

            layer.fit_batch(
                X_batch, Y_batch, P_batch, momentum, verbosity=verbosity, *args, **kwargs
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if hasattr(layer, "finalize_fit") and finalize:
            layer.finalize_fit(*args, **kwargs)

    def _forward_layer_batched(self, layer, layer_idx, X, batch_size, verbosity):
        num_samples = X.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size

        with torch.no_grad():
            test_output = layer.forward(X[:1])
            output_shape = test_output.shape[1:]
            output_dtype = test_output.dtype
            output_device = test_output.device
            del test_output

        new_X = torch.empty(
            (num_samples, *output_shape), dtype=output_dtype, device=output_device
        )

        for batch_idx in (
            tqdm(range(num_batches), desc=f"Layer {layer_idx} Forward")
            if verbosity
            else range(num_batches)
        ):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_samples)

            batch_output = layer.forward(X[start_idx:end_idx])
            new_X[start_idx:end_idx] = batch_output

            del batch_output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return new_X

    def finalize_all_layers(self, *args, **kwargs):
        for layer in self.layers:
            if hasattr(layer, "finalize_fit"):
                layer.finalize_fit(*args, **kwargs)

    def batched_forward(self, x, batch_size=64):
        if batch_size is None:
            batch_size = min(64, x.shape[0])

        num_batches = math.ceil(x.shape[0] / batch_size)
        outputs = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, x.shape[0])

            batch_output = x[start_idx:end_idx]
            for layer in self.layers:
                batch_output = layer.forward(batch_output)

            outputs.append(batch_output.cpu())

            del batch_output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result = torch.cat(outputs, dim=0).to(x.device)
        return result

    def iterate(self, X, Y):
        current_X = X
        for layer in self.layers:
            if hasattr(layer, "iterate"):
                layer.iterate(current_X, Y)
            current_X = layer.forward(current_X)

            if torch.cuda.is_available() and current_X.device.type == "cuda":
                torch.cuda.empty_cache()

    def __repr__(self):
        return f"{self.layers}"

    def __del__(self):
        try:
            for layer in self.layers:
                del layer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

class Residual(Model):
    """Model with a residual (skip) connection."""

    def forward(self, x, *args, **kwargs):
        out = super().forward(x)
        out += x @ torch.eye(x.shape[-1], out.shape[-1], device=x.device)
        return out


class Dense(Linear):
    """Linear layer followed by ReLU activation."""

    def forward(self, x, *args, **kwargs):
        return torch.nn.functional.relu(super().forward(x))


class Flatten(AnalyticalBase):
    """Flatten all dimensions except the batch dimension."""

    def forward(self, x, *args, **kwargs):
        return x.reshape(x.shape[0], -1)
