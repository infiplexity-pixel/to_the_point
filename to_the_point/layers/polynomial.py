import torch
import numpy as np
import math
from typing import Optional


class Polynomial(torch.nn.Module):
    """Polynomial feature regressor with ridge regression.

    Projects inputs through random projections, constructs polynomial
    features (powers + cross-terms), and solves for weights analytically.
    Supports chunked / batched accumulation for large datasets.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_degree: int = 3,
        device: str = "cpu",
        n_components: int = 128,
        max_cross_terms: int = 512,
        use_cross_terms: bool = True,
        alpha: float = 1e-4,
        max_chunk_size: int = 1000,
        max_samples_per_chunk: int = 10000,
    ):
        super().__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.n_degree = n_degree
        self.n_components = n_components
        self.use_cross_terms = use_cross_terms
        self.alpha = alpha
        self.max_chunk_size = max_chunk_size
        self.max_samples_per_chunk = max_samples_per_chunk

        if in_features >= n_components:
            self.projection = torch.nn.init.orthogonal_(
                torch.empty(in_features, n_components, device=device)
            )
        else:
            self.projection = torch.randn(
                in_features, n_components, device=device
            ) / np.sqrt(n_components)

        self.feature_config = {
            "bias": True,
            "min_power": 1,
            "max_power": n_degree,
            "cross_terms": use_cross_terms,
            "interaction_depth": min(
                4, max(2, int(np.log2(max_cross_terms // max(n_components, 1))))
            ),
            "max_cross_terms": max_cross_terms,
            "use_sqrt": True,
        }

        self.S = None
        self._precalculate_feature_dim_and_chunks()

        self.weight = None
        self.bias = None
        self.is_fitted = False

        self.phiT_phi = None
        self.phiT_Y = None
        self.phi_sum = None
        self.Y_sum = None
        self.sample_count = 0

    # ------------------------------------------------------------------
    def _precalculate_feature_dim_and_chunks(self):
        n_features = self.n_components
        dim = 0
        self.feature_chunks = []
        self.cross_term_chunks = []
        current_offset = 0

        if self.feature_config["bias"]:
            self.feature_chunks.append(("bias", 0, 1))
            dim += 1
            current_offset += 1

        min_p = self.feature_config["min_power"]
        max_p = min(self.n_degree, self.feature_config["max_power"])
        for deg in range(min_p, max_p + 1):
            chunk_size = n_features
            self.feature_chunks.append((f"power_{deg}", current_offset, chunk_size))
            dim += chunk_size
            current_offset += chunk_size

        cross_dim = 0
        if self.feature_config["cross_terms"]:
            depth = self.feature_config["interaction_depth"]
            total_cross_terms = self.feature_config["max_cross_terms"]

            n_pairwise = 0
            n_triples = 0
            n_quads = 0

            if depth >= 2:
                n_pairwise = min(
                    (n_features * (n_features - 1)) // 2, total_cross_terms
                )

            if depth >= 3 and n_features >= 3:
                remaining = total_cross_terms - n_pairwise
                if remaining > 0:
                    n_triples = min(
                        (n_features * (n_features - 1) * (n_features - 2)) // 6,
                        remaining,
                    )

            if depth >= 4 and n_features >= 4:
                remaining = total_cross_terms - n_pairwise - n_triples
                if remaining > 0:
                    n_quads = min(
                        (
                            n_features
                            * (n_features - 1)
                            * (n_features - 2)
                            * (n_features - 3)
                        )
                        // 24,
                        remaining,
                    )

            cross_start = current_offset

            if n_pairwise > 0:
                n_pair_chunks = math.ceil(n_pairwise / self.max_chunk_size)
                for i in range(n_pair_chunks):
                    start_idx = i * self.max_chunk_size
                    end_idx = min((i + 1) * self.max_chunk_size, n_pairwise)
                    chunk_size = end_idx - start_idx
                    self.cross_term_chunks.append(
                        ("pairwise", cross_start + start_idx, chunk_size, start_idx, end_idx)
                    )

            current_offset += n_pairwise

            if n_triples > 0:
                n_triple_chunks = math.ceil(n_triples / self.max_chunk_size)
                for i in range(n_triple_chunks):
                    start_idx = i * self.max_chunk_size
                    end_idx = min((i + 1) * self.max_chunk_size, n_triples)
                    chunk_size = end_idx - start_idx
                    self.cross_term_chunks.append(
                        (
                            "triples",
                            cross_start + n_pairwise + start_idx,
                            chunk_size,
                            start_idx,
                            end_idx,
                        )
                    )

            current_offset += n_triples

            if n_quads > 0:
                n_quad_chunks = math.ceil(n_quads / self.max_chunk_size)
                for i in range(n_quad_chunks):
                    start_idx = i * self.max_chunk_size
                    end_idx = min((i + 1) * self.max_chunk_size, n_quads)
                    chunk_size = end_idx - start_idx
                    self.cross_term_chunks.append(
                        (
                            "quads",
                            cross_start + n_pairwise + n_triples + start_idx,
                            chunk_size,
                            start_idx,
                            end_idx,
                        )
                    )

            current_offset += n_quads
            cross_dim = n_pairwise + n_triples + n_quads
            dim += cross_dim

        if self.feature_config.get("use_sqrt", False):
            self.feature_chunks.append(("sqrt", current_offset, n_features))
            dim += n_features
            current_offset += n_features

        self.feature_dim = dim
        self.cross_dim = cross_dim

    # ------------------------------------------------------------------
    def _project(self, X: torch.Tensor) -> torch.Tensor:
        result = []
        for i in range(0, X.shape[0], self.max_samples_per_chunk):
            chunk = X[i : i + self.max_samples_per_chunk]
            result.append(
                chunk @ self.projection.to(device=chunk.device, dtype=chunk.dtype)
            )
        return torch.cat(result, dim=0)

    def create_polynomial_features_chunked(self, X: torch.Tensor) -> torch.Tensor:
        X_proj = self._project(X)
        n_samples = X_proj.shape[0]

        phi = torch.zeros(
            (n_samples, self.feature_dim), device=X.device, dtype=X.dtype
        )

        for chunk_type, offset, size in self.feature_chunks:
            if chunk_type == "bias":
                phi[:, offset] = 1.0
            elif chunk_type.startswith("power_"):
                deg = int(chunk_type.split("_")[1])
                phi[:, offset : offset + size] = X_proj**deg
            elif chunk_type == "sqrt":
                phi[:, offset : offset + size] = torch.sqrt(
                    torch.abs(X_proj) + 1e-8
                )

        if self.feature_config["cross_terms"] and self.cross_dim > 0:
            for (
                chunk_type,
                offset,
                chunk_size,
                local_start,
                local_end,
            ) in self.cross_term_chunks:
                phi_chunk = self._create_cross_terms_chunk(
                    X_proj, chunk_type, local_start, local_end
                )
                phi[:, offset : offset + chunk_size] = phi_chunk

        return phi

    def _create_cross_terms_chunk(
        self,
        X_proj: torch.Tensor,
        chunk_type: str,
        start_idx: int,
        end_idx: int,
    ) -> torch.Tensor:
        n_samples, n_features = X_proj.shape
        chunk_size = end_idx - start_idx
        chunk = torch.zeros(
            (n_samples, chunk_size), device=X_proj.device, dtype=X_proj.dtype
        )

        if chunk_type == "pairwise":
            idx = start_idx
            col_idx = 0
            for i in range(n_features):
                if idx >= start_idx + chunk_size:
                    break
                pairs_with_i = n_features - i - 1
                if idx + pairs_with_i > start_idx:
                    first_j = max(i + 1, i + 1 + (start_idx - idx))
                    last_j = min(n_features, i + 1 + (start_idx + chunk_size - idx))
                    n_cols = last_j - first_j
                    if n_cols > 0:
                        chunk[:, col_idx : col_idx + n_cols] = (
                            X_proj[:, i : i + 1] * X_proj[:, first_j:last_j]
                        )
                        col_idx += n_cols
                idx += pairs_with_i

        elif chunk_type == "triples":
            idx = start_idx
            col_idx = 0
            for i in range(n_features):
                if idx >= start_idx + chunk_size:
                    break
                for j in range(i + 1, n_features):
                    if idx >= start_idx + chunk_size:
                        break
                    triples_with_ij = n_features - j - 1
                    if idx + triples_with_ij > start_idx:
                        first_k = max(j + 1, j + 1 + (start_idx - idx))
                        last_k = min(
                            n_features, j + 1 + (start_idx + chunk_size - idx)
                        )
                        n_cols = last_k - first_k
                        if n_cols > 0:
                            chunk[:, col_idx : col_idx + n_cols] = (
                                X_proj[:, i : i + 1]
                                * X_proj[:, j : j + 1]
                                * X_proj[:, first_k:last_k]
                            )
                            col_idx += n_cols
                    idx += triples_with_ij

        elif chunk_type == "quads":
            idx = start_idx
            col_idx = 0
            for i in range(n_features):
                if idx >= start_idx + chunk_size:
                    break
                for j in range(i + 1, n_features):
                    if idx >= start_idx + chunk_size:
                        break
                    for k in range(j + 1, n_features):
                        if idx >= start_idx + chunk_size:
                            break
                        quads_with_ijk = n_features - k - 1
                        if idx + quads_with_ijk > start_idx:
                            first_l = max(k + 1, k + 1 + (start_idx - idx))
                            last_l = min(
                                n_features,
                                k + 1 + (start_idx + chunk_size - idx),
                            )
                            n_cols = last_l - first_l
                            if n_cols > 0:
                                chunk[:, col_idx : col_idx + n_cols] = (
                                    X_proj[:, i : i + 1]
                                    * X_proj[:, j : j + 1]
                                    * X_proj[:, k : k + 1]
                                    * X_proj[:, first_l:last_l]
                                )
                                col_idx += n_cols
                        idx += quads_with_ijk
        return chunk

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit_batch_chunked(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        batch_chunk_size: Optional[int] = None,
    ) -> None:
        if len(X.shape) == 3:
            X = X.reshape(-1, X.shape[-1])
        if X.shape[0] == 0:
            return

        if Y.shape[-1] != self.out_features:
            if self.S is None:
                self.S = torch.randn(Y.shape[-1], self.out_features)
            Y = Y @ self.S

        if batch_chunk_size is None:
            batch_chunk_size = self.max_samples_per_chunk

        if self.phiT_phi is None:
            self.phiT_phi = torch.zeros(
                (self.feature_dim, self.feature_dim),
                device=self.device,
                dtype=X.dtype,
            )
            self.phiT_Y = torch.zeros(
                (self.feature_dim, self.out_features),
                device=self.device,
                dtype=Y.dtype,
            )
            self.phi_sum = torch.zeros(
                self.feature_dim, device=self.device, dtype=X.dtype
            )
            self.Y_sum = torch.zeros(
                self.out_features, device=self.device, dtype=Y.dtype
            )

        for i in range(0, X.shape[0], batch_chunk_size):
            X_chunk = X[i : i + batch_chunk_size]
            Y_chunk = Y[i : i + batch_chunk_size]

            phi_chunk = self.create_polynomial_features_chunked(X_chunk)

            self.phiT_phi += phi_chunk.T @ phi_chunk
            self.phiT_Y += phi_chunk.T @ Y_chunk
            self.phi_sum += phi_chunk.sum(dim=0)
            self.Y_sum += Y_chunk.sum(dim=0)
            self.sample_count += X_chunk.shape[0]

    def fit_batch(self, X: torch.Tensor, Y: torch.Tensor, *args, **kwargs) -> None:
        self.fit_batch_chunked(X, Y)

    def finalize_fit(self, *args, **kwargs) -> None:
        if self.sample_count == 0:
            raise RuntimeError("No data accumulated")

        N = self.sample_count
        phi_mean = self.phi_sum / N
        Y_mean = self.Y_sum / N

        phiT_phi_c = self.phiT_phi - N * torch.outer(phi_mean, phi_mean)
        phiT_Y_c = self.phiT_Y - N * torch.outer(phi_mean, Y_mean)

        A = phiT_phi_c + self.alpha * torch.eye(
            self.feature_dim, device=self.device, dtype=phiT_phi_c.dtype
        )
        self.weight = torch.nn.Parameter(torch.linalg.pinv(A) @ phiT_Y_c)
        self.bias = torch.nn.Parameter(Y_mean - phi_mean @ self.weight)
        self.is_fitted = True

        self.phiT_phi = None
        self.phiT_Y = None
        self.phi_sum = None
        self.Y_sum = None
        self.sample_count = 0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def forward_chunked(
        self, X: torch.Tensor, output_chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        if output_chunk_size is None:
            output_chunk_size = self.max_samples_per_chunk

        if X.shape[0] > output_chunk_size:
            results = []
            for i in range(0, X.shape[0], output_chunk_size):
                X_chunk = X[i : i + output_chunk_size]
                phi_chunk = self.create_polynomial_features_chunked(X_chunk)
                result_chunk = phi_chunk @ self.weight.to(
                    device=phi_chunk.device, dtype=phi_chunk.dtype
                )
                result_chunk += self.bias.to(
                    device=phi_chunk.device, dtype=phi_chunk.dtype
                )
                results.append(result_chunk)
            return torch.cat(results, dim=0)
        else:
            phi = self.create_polynomial_features_chunked(X)
            return phi @ self.weight.to(
                device=phi.device, dtype=phi.dtype
            ) + self.bias.to(device=phi.device, dtype=phi.dtype)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.forward_chunked(X)

    create_polynomial_features = create_polynomial_features_chunked
