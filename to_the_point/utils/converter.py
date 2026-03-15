import torch
from typing import Callable


def torch_to_analytical(func: Callable):
    """Wrap an arbitrary callable (e.g. ``torch.relu``) as an analytical layer.

    The returned module has no-op ``fit_batch`` and ``finalize_fit`` methods so
    it can be composed inside a :class:`to_the_point.Model`.
    """

    class NonLinear(torch.nn.Module):
        def forward(self, x, *args, **kwargs):
            return func(x)

        def fit_batch(self, *args, **kwargs):
            pass

        def finalize_fit(self, *args, **kwargs):
            pass

    return NonLinear()
