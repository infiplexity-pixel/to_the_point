import torch
from .linear import Linear

class Recursive(Linear):
    """Analytical recursive layer with stacked input history.

    Maintains a stack of previous inputs and concatenates them with the
    current input before passing through the underlying analytical ``Linear``
    layer. At each forward pass, the current input is appended to the stack,
    and the stack is managed to not exceed ``stack_size``.

    The stack is cleared when ``finalize_fit`` is called, ensuring each
    training epoch starts fresh.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        stack_size (int): Maximum number of previous inputs to maintain in the stack.
        device (str, optional): Device to allocate tensors on. Default: "cpu".
        param_batch_size (int, optional): Batch size for parameter updates. Default: 256.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stack_size: int,
        device: str = "cpu",
        param_batch_size: int = 256,
    ):
        # Recursive layer expands input by concatenating stack_size previous inputs
        expanded_in_features = in_features * (1 + stack_size)

        super().__init__(
            in_features=expanded_in_features,
            out_features=out_features,
            device=device,
            param_batch_size=param_batch_size,
        )

        self.input_dim = in_features
        self.stack_size = stack_size
        self.stack: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Stack Management
    # ------------------------------------------------------------------
    def _build_augmented_input(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate current input with stacked history.

        If the stack is not yet full, pads with zeros to maintain consistent
        input dimensionality.

        Args:
            x (torch.Tensor): Current input, shape (..., in_features).

        Returns:
            torch.Tensor: Augmented input with history, shape (..., in_features * (1 + stack_size)).
        """
        padding_needed = self.stack_size - len(self.stack)

        if padding_needed > 0:
            # Stack not yet full: pad with zeros
            padding = [torch.zeros_like(x) for _ in range(padding_needed)]
            try:
                augmented = torch.hstack([x, *self.stack, *padding])
            except RuntimeError:
                augmented = torch.hstack([x, *[torch.zeros_like(x) for _ in range(self.stack_size)]])
        else:
            # Stack is full: concatenate current input with full stack
            augmented = torch.hstack([x, *self.stack])

        return augmented

    def _push_to_stack(self, x: torch.Tensor) -> None:
        """Add input to the front of the stack and maintain max size.

        Args:
            x (torch.Tensor): Input to add, shape (..., in_features).
        """
        self.stack.append(x)
        if len(self.stack) > self.stack_size:
            self.stack.pop(0)

    def _clear_stack(self) -> None:
        """Clear the input stack."""
        self.stack = []

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, no_stack=False) -> torch.Tensor:
        """Forward pass with recursive history concatenation.

        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor, shape (batch_size, out_features).
        """
        # Clone to avoid modifying input and ensure stack isolation
        x_clone = x.clone()

        # Augment input with stacked history
        x_augmented = self._build_augmented_input(x_clone)

        # Add current input to stack for next forward pass
        if not no_stack:
            self._push_to_stack(x_clone)

        # Forward through analytical linear layer
        return super().forward(x_augmented)

    # ------------------------------------------------------------------
    # Analytical Fitting
    # ------------------------------------------------------------------
    def fit_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        P_batch: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Fit on a batch with recursive history concatenation.

        Augments the batch with stacked history before passing to the
        analytical ridge regression accumulator.

        Args:
            X_batch (torch.Tensor): Input batch, shape (batch_size, in_features).
            Y_batch (torch.Tensor): Target batch, shape (batch_size, out_features).
            P_batch (torch.Tensor, optional): Sample weights. Default: None.
        """
        if X_batch.shape[0] == 0:
            return

        # Augment the entire batch with stacked history
        X_batch_augmented = self._build_augmented_input(X_batch)

        # Add last sample(s) to stack for next batch
        # (typically the last sample in the batch becomes the oldest history)
        if X_batch.shape[0] > 0:
            self._push_to_stack(X_batch[-1:])

        # Delegate to parent's analytical fitting
        super().fit_batch(X_batch_augmented, Y_batch, P_batch, *args, **kwargs)

    # ------------------------------------------------------------------
    # Training Lifecycle
    # ------------------------------------------------------------------
    def finalize_fit(self, N: int | None = None, dampening: float = 1e-8) -> None:
        """Finalize analytical fitting and reset stack for next epoch.

        Solves the accumulated ridge regression system and clears the stack
        to prepare for the next training epoch.

        Args:
            N (int, optional): Total number of samples. If None, uses ``sample_count``.
            dampening (float, optional): Regularization strength. Default: 1e-8.
        """
        super().finalize_fit(N=N, dampening=dampening)
        self._clear_stack()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"Recursive("
            f"input_dim={self.input_dim}, "
            f"out_features={self.out_features}, "
            f"stack_size={self.stack_size}, "
            f"stack_len={len(self.stack)})"
        )