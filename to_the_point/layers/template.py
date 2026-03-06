import torch


class AnalyticalBase(torch.nn.Module):
    """Base template for all analytical layers in to_the_point.

    Every layer inherits from this class and may override ``forward``,
    ``fit_batch`` and ``finalize_fit``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x

    def fit_batch(self, *args, **kwargs):
        pass

    def finalize_fit(self, *args, **kwargs):
        pass
