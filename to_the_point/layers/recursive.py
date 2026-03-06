import torch
from .linear import Linear


class Recursive(Linear):
    """Recursive layer that maintains a stack of previous inputs.

    At each forward call the current input is appended to an internal
    stack and concatenated with previous inputs before being passed
    through the underlying ``Linear`` layer.
    """

    def __init__(self, in_features, out_features, stack_size, device="cpu"):
        super().__init__(
            in_features + (in_features * stack_size), out_features, device
        )
        self.max_size = stack_size
        self.stack: list = []

    def forward(self, x: torch.Tensor):
        orig_x = x.clone()
        if len(self.stack) != self.max_size:
            x = torch.hstack(
                [
                    x,
                    *self.stack,
                    *[torch.zeros_like(x) for _ in range(self.max_size - len(self.stack))],
                ]
            )
        else:
            x = torch.hstack([x, *self.stack])
        self.stack.append(orig_x)
        if len(self.stack) > self.max_size:
            self.stack.pop(0)
        return super().forward(x)

    def fit_batch(self, X_batch, Y_batch, P_batch=None, momentum=0, *args, **kwargs):
        if len(self.stack) != self.max_size:
            X_batch = torch.hstack(
                [
                    X_batch,
                    *self.stack,
                    *[
                        torch.zeros_like(X_batch)
                        for _ in range(self.max_size - len(self.stack))
                    ],
                ]
            )
        else:
            X_batch = torch.hstack([X_batch, *self.stack])
        return super().fit_batch(X_batch, Y_batch, P_batch, momentum)

    def finalize_fit(self, N=None, dampening=1e-8):
        super().finalize_fit(N, dampening)
        self.stack = []
