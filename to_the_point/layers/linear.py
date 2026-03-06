import torch


class Linear(torch.nn.Module):
    """Analytical linear layer.

    Accumulates sufficient statistics (X^T X and X^T Y) across batches and
    solves for the optimal weight matrix via ridge regression in
    ``finalize_fit``.
    """

    def __init__(self, in_features, out_features, device="cpu", param_batch_size=256):
        super().__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.param_batch_size = param_batch_size

        self.weight = torch.nn.Parameter(
            torch.randn((in_features, out_features), device=device)
        )
        self.bias = torch.nn.Parameter(
            torch.randn((1, out_features), device=device)
        )
        self.is_fitted = False

        self.S = None
        self.S2 = None
        self.S3 = None

        self.cross_var = torch.zeros((in_features, in_features), device=device)
        self.co_cross_var = torch.zeros((in_features, out_features), device=device)

        self.sample_count = 0

        self.sum_X = torch.zeros((1, in_features), device=device)
        self.sum_Y = torch.zeros((1, out_features), device=device)

        self._should_clear_data = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x):
        if x.shape[-1] != self.in_features and self.S is not None:
            x = x @ self.S

        if self.out_features > self.param_batch_size:
            outputs = []
            for start_idx in range(0, self.out_features, self.param_batch_size):
                end_idx = min(start_idx + self.param_batch_size, self.out_features)
                weight_batch = self.weight[:, start_idx:end_idx]
                bias_batch = self.bias[:, start_idx:end_idx]
                output_batch = (x @ weight_batch) + bias_batch
                outputs.append(output_batch)
            return torch.cat(outputs, dim=-1)
        else:
            return (x @ self.weight) + self.bias

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit_batch(self, X_batch, Y_batch, P_batch=None, *args, **kwargs):
        if X_batch.shape[0] == 0:
            return

        X_batch = X_batch.reshape(-1, X_batch.shape[-1])

        if P_batch is None:
            weights = None
            batch_size = X_batch.shape[0]
        else:
            weights = P_batch
            batch_size = torch.sum(P_batch)

        if X_batch.shape[-1] != self.in_features and self.S is not None:
            X_batch = X_batch @ self.S
        elif X_batch.shape[-1] != self.in_features and self.S is None:
            self.S = torch.randn(
                X_batch.shape[-1], self.in_features, device=self.device
            )
            X_batch = X_batch @ self.S

        if Y_batch is not None and Y_batch.shape[-1] != self.out_features:
            if self.S2 is None:
                self.S2 = torch.randn(
                    Y_batch.shape[-1],
                    self.out_features,
                    dtype=Y_batch.dtype,
                    device=self.device,
                )
            Y_batch = Y_batch @ self.S2

        if weights is not None:
            Xw = weights * X_batch
            Yw = weights * Y_batch
        else:
            Xw = X_batch
            Yw = Y_batch

        self.cross_var += torch.einsum("bi,bj->ij", Xw, Xw)
        self.co_cross_var += torch.einsum("bi,bj->ij", Xw, Yw)

        self.sum_X += Xw.sum(dim=0, keepdim=True)
        self.sum_Y += Yw.sum(dim=0, keepdim=True)
        self.sample_count += batch_size

    def finalize_fit(self, N=None, dampening=0):
        if self.sample_count == 0:
            return

        if N is None:
            N = self.sample_count

        cross_var_normalized = self.cross_var
        co_cross_var_normalized = self.co_cross_var

        weight_matrix = N * cross_var_normalized - self.sum_X.T @ self.sum_X
        rhs = N * co_cross_var_normalized - self.sum_X.T @ self.sum_Y

        weight_matrix += dampening * torch.eye(self.in_features, device=self.device)

        W = self._cholesky_solve(weight_matrix) @ rhs
        b = (self.sum_Y - self.sum_X @ W) / N

        self.weight.data = W
        self.bias.data = b
        self.is_fitted = True

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------
    def _cholesky_solve(self, A):
        try:
            L = torch.linalg.cholesky(A)
            I = torch.eye(A.size(0), device=A.device)
            return torch.cholesky_solve(I, L)
        except RuntimeError:
            return self._lu_solve(A)

    def _lu_solve(self, A):
        try:
            I = torch.eye(A.size(0), device=A.device)
            return torch.linalg.solve(A, I)
        except RuntimeError:
            return torch.linalg.pinv(A)

    def __repr__(self):
        return f"Linear({self.in_features}, {self.out_features})"
