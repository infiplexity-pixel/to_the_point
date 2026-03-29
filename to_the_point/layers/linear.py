import torch


class Linear(torch.nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        device="cpu",
        param_batch_size=256,
        lsmr_iters=300,
        lsmr_tol=1e-8,
        dtype=torch.float64,
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.in_features = in_features
        self.out_features = out_features
        self.param_batch_size = param_batch_size

        self.lsmr_iters = lsmr_iters
        self.lsmr_tol = lsmr_tol

        self.weight = torch.nn.Parameter(
            torch.randn(
                in_features,
                out_features,
                device=device,
                dtype=dtype,
            )
        )

        self.bias = torch.nn.Parameter(
            torch.randn(
                1,
                out_features,
                device=device,
                dtype=dtype,
            )
        )

        self.is_fitted = False

        # Random projections
        self.S = None
        self.S2 = None

        # Stored batches
        self.X_batches = []
        self.Y_batches = []
        self.batch_sizes = []

        self.sample_count = 0

        self.sum_X = torch.zeros(
            1,
            in_features,
            device=device,
            dtype=dtype,
        )

        self.sum_Y = torch.zeros(
            1,
            out_features,
            device=device,
            dtype=dtype,
        )

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    def forward(self, x):

        if x.shape[-1] != self.in_features and self.S is not None:
            x = x @ self.S.to(device=x.device, dtype=x.dtype)

        if self.out_features > self.param_batch_size:

            outputs = []

            for start in range(
                0,
                self.out_features,
                self.param_batch_size,
            ):

                end = min(
                    start + self.param_batch_size,
                    self.out_features,
                )

                w = self.weight[:, start:end]
                b = self.bias[:, start:end]

                outputs.append((x @ w) + b)

            return torch.cat(outputs, dim=-1)

        else:

            return (x @ self.weight.to(device=x.device, dtype=x.dtype)) + self.bias.to(device=x.device, dtype=x.dtype)

    # ---------------------------------------------------------
    # Fit batch
    # ---------------------------------------------------------

    def fit_batch(self, X_batch, Y_batch, P_batch=None, *args, **kwargs):

        if X_batch.shape[0] == 0:
            return

        X_batch = X_batch.reshape(-1, X_batch.shape[-1])

        X_batch = X_batch.to(self.dtype)
        Y_batch = Y_batch.to(self.dtype)

        # Input projection
        if X_batch.shape[-1] != self.in_features:

            if self.S is None:

                self.S = torch.randn(
                    X_batch.shape[-1],
                    self.in_features,
                    device=self.device,
                    dtype=self.dtype,
                )

            X_batch = X_batch @ self.S

        # Output projection
        if Y_batch.shape[-1] != self.out_features:

            if self.S2 is None:

                self.S2 = torch.randn(
                    Y_batch.shape[-1],
                    self.out_features,
                    device=self.device,
                    dtype=self.dtype,
                )

            Y_batch = Y_batch @ self.S2

        # Optional weighting
        if P_batch is not None:

            X_batch = P_batch * X_batch
            Y_batch = P_batch * Y_batch

        self.X_batches.append(X_batch)
        self.Y_batches.append(Y_batch)

        self.batch_sizes.append(X_batch.shape[0])

        self.sum_X += X_batch.sum(dim=0, keepdim=True)
        self.sum_Y += Y_batch.sum(dim=0, keepdim=True)

        self.sample_count += X_batch.shape[0]

    # ---------------------------------------------------------
    # Matrix-vector products
    # ---------------------------------------------------------

    def _Xv(self, v):

        outputs = []

        for Xb in self.X_batches:
            outputs.append(Xb @ v)

        return torch.cat(outputs, dim=0)

    def _XT_u(self, u):

        result = torch.zeros(
            self.in_features,
            u.shape[1],
            device=self.device,
            dtype=self.dtype,
        )

        offset = 0

        for Xb, n in zip(
            self.X_batches,
            self.batch_sizes,
        ):

            ub = u[offset : offset + n]

            result += Xb.T @ ub

            offset += n

        return result

    # ---------------------------------------------------------
    # LSMR Solver
    # ---------------------------------------------------------

    def _lsmr_single(self, y, damp):

        x = torch.zeros(
            self.in_features,
            1,
            device=self.device,
            dtype=self.dtype,
        )

        u = y.clone()

        beta = torch.norm(u)

        if beta > 0:
            u /= beta

        v = self._XT_u(u)

        alpha = torch.norm(v)

        if alpha > 0:
            v /= alpha

        w = v.clone()

        phi = beta
        rho = alpha

        for _ in range(self.lsmr_iters):

            u = self._Xv(v) - alpha * u

            beta = torch.norm(u)

            if beta > 0:
                u /= beta

            v = self._XT_u(u) - beta * v

            alpha = torch.norm(v)

            if alpha > 0:
                v /= alpha

            rho_bar = torch.sqrt(
                rho**2 + damp**2 + beta**2
            )

            c = rho / rho_bar
            s = beta / rho_bar

            theta = s * alpha
            rho = -c * alpha

            phi_bar = c * phi
            phi = s * phi

            x += (phi_bar / rho_bar) * w

            w = v - (theta / rho_bar) * w
            
            if torch.abs(phi) < self.lsmr_tol:
                break

        return x

    # ---------------------------------------------------------
    # Finalize
    # ---------------------------------------------------------

    def finalize_fit(self, dampening=1e-6, *args, **kwargs):

        if self.sample_count == 0:
            return

        Y = torch.cat(self.Y_batches, dim=0)

        W = torch.zeros(
            self.in_features,
            self.out_features,
            device=self.device,
            dtype=self.dtype,
        )

        for col in range(self.out_features):

            y = Y[:, col : col + 1]

            W[:, col : col + 1] = self._lsmr_single(
                y,
                dampening,
            )

        N = self.sample_count

        b = (
            self.sum_Y
            - self.sum_X @ W
        ) / N

        self.weight.data = W.float()
        self.bias.data = b

        self.is_fitted = True

    # ---------------------------------------------------------

    def clear_batches(self):

        self.X_batches.clear()
        self.Y_batches.clear()
        self.batch_sizes.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self):

        return f"Linear({self.in_features}, {self.out_features})"