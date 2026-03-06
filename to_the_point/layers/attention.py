import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict


class Attention(nn.Module):
    """Analytical Attention layer with batch-by-batch fitting.

    Learns attention patterns incrementally by accumulating covariance
    statistics and solving for optimal Q/K/V projections via ridge
    regression.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        fit_bias: bool = True,
        init_sharpness: float = 0.3,
        learn_temperature: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)

        self.learn_temperature = learn_temperature
        if learn_temperature:
            self.temperature = nn.Parameter(torch.tensor(init_sharpness))
        else:
            self.temperature = torch.tensor(init_sharpness)

        self.output_proj = nn.Linear(d_model, d_model)

        self.register_buffer("covariance_qq", torch.zeros(d_model, d_model))
        self.register_buffer("covariance_kk", torch.zeros(d_model, d_model))
        self.register_buffer("covariance_vv", torch.zeros(d_model, d_model))
        self.register_buffer("covariance_xy", torch.zeros(d_model, d_model))
        self.register_buffer("batch_count", torch.tensor(0))

        self.learning_rates = {
            "q": 0.1,
            "k": 0.1,
            "v": 0.1,
            "temperature": 0.01,
        }

        self.attention_stats: Dict[str, list] = {
            "max_attention": [],
            "mean_attention": [],
            "sharpness": [],
            "fitting_loss": [],
        }

        self._init_sharp_weights()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _init_sharp_weights(self):
        nn.init.normal_(self.w_q.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.w_k.weight, mean=0.0, std=0.08)
        nn.init.normal_(self.w_v.weight, mean=0.0, std=0.02)

        nn.init.uniform_(self.w_q.bias, -0.05, 0.05)
        nn.init.uniform_(self.w_k.bias, -0.05, 0.05)
        nn.init.constant_(self.w_v.bias, 0.0)

        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    # ------------------------------------------------------------------
    # Core attention computation
    # ------------------------------------------------------------------
    def compute_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = Q.size(0)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.learn_temperature:
            scores = scores / (self.temperature.abs() + 1e-8)
        else:
            scores = scores / (self.temperature + 1e-8)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        if return_weights:
            return output, attn_weights
        return output, None

    # ------------------------------------------------------------------
    # Analytical fitting
    # ------------------------------------------------------------------
    def analytical_fit(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict:
        batch_size, seq_len, _ = X_batch.shape
        device = X_batch.device

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            mask_matrix = mask_expanded.reshape(
                batch_size, seq_len, mask_expanded.shape[-1]
            ) @ mask_expanded.reshape(
                batch_size, mask_expanded.shape[-2], mask_expanded.shape[-1]
            ).transpose(2, 1)
        else:
            mask_matrix = torch.ones(batch_size, seq_len, seq_len, device=device)

        with torch.no_grad():
            X_flat = X_batch.view(-1, self.d_model)
            Y_flat = Y_batch.view(-1, self.d_model)

            Q_current = self.w_q(X_batch).view(-1, self.d_model)
            K_current = self.w_k(X_batch).view(-1, self.d_model)
            V_current = self.w_v(X_batch).view(-1, self.d_model)

            n = batch_size * seq_len
            cov_qq_batch = Q_current.T @ Q_current / n
            cov_kk_batch = K_current.T @ K_current / n
            cov_vv_batch = V_current.T @ V_current / n
            cov_xy_batch = X_flat.T @ Y_flat / n

        alpha = 0.1
        if self.batch_count == 0:
            self.covariance_qq.copy_(cov_qq_batch)
            self.covariance_kk.copy_(cov_kk_batch)
            self.covariance_vv.copy_(cov_vv_batch)
            self.covariance_xy.copy_(cov_xy_batch)
        else:
            self.covariance_qq = (1 - alpha) * self.covariance_qq + alpha * cov_qq_batch
            self.covariance_kk = (1 - alpha) * self.covariance_kk + alpha * cov_kk_batch
            self.covariance_vv = (1 - alpha) * self.covariance_vv + alpha * cov_vv_batch
            self.covariance_xy = (1 - alpha) * self.covariance_xy + alpha * cov_xy_batch

        self.batch_count += 1

        with torch.no_grad():
            reg = 1e-6
            XTX_reg = X_flat.T @ X_flat + reg * torch.eye(self.d_model, device=device)
            XTY = X_flat.T @ Y_flat

            W_q_opt = torch.linalg.pinv(XTX_reg) @ XTY
            lr_q = self.learning_rates["q"]
            self.w_q.weight.data = (1 - lr_q) * self.w_q.weight.data + lr_q * W_q_opt

            noise = torch.randn_like(W_q_opt) * 0.01
            W_k_opt = W_q_opt * 0.9 + noise
            lr_k = self.learning_rates["k"]
            self.w_k.weight.data = (1 - lr_k) * self.w_k.weight.data + lr_k * W_k_opt

            if self.batch_count > 1:
                W_v_opt = torch.linalg.pinv(XTX_reg) @ XTY
                lr_v = self.learning_rates["v"]
                self.w_v.weight.data = (1 - lr_v) * self.w_v.weight.data + lr_v * W_v_opt

        Q = self.w_q(X_batch)
        K = self.w_k(X_batch)
        V = self.w_v(X_batch)

        output, attn_weights = self.compute_attention(Q, K, V, mask)
        output = self.output_proj(output)

        fitting_loss = F.mse_loss(output, Y_batch)

        if attn_weights is not None:
            attn_max = attn_weights.max().item()
            attn_mean = attn_weights.mean().item()
            sharpness = attn_weights.std().item() / (attn_mean + 1e-8)

            self.attention_stats["max_attention"].append(attn_max)
            self.attention_stats["mean_attention"].append(attn_mean)
            self.attention_stats["sharpness"].append(sharpness)
            self.attention_stats["fitting_loss"].append(fitting_loss.item())

        stats = {
            "fitting_loss": fitting_loss.item(),
            "batch_count": self.batch_count.item(),
            "covariance_norm": self.covariance_qq.norm().item(),
            "temperature": (
                self.temperature.item()
                if self.learn_temperature
                else self.temperature
            ),
        }
        return stats

    def fit_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        update_temperature: bool = True,
        projection_std: float = 0.02,
        *args,
        **kwargs,
    ) -> Dict:
        device = X_batch.device
        batch_size, seq_len, d_model = X_batch.shape

        y_original_shape = Y_batch.shape

        if Y_batch.shape != X_batch.shape:
            Y_flat = Y_batch.view(batch_size, -1)
            y_features = Y_flat.shape[1]

            if (
                not hasattr(self, "y_projection")
                or self.y_projection.shape != (y_features, d_model * seq_len)
            ):
                self.y_projection = nn.Parameter(
                    torch.randn(y_features, d_model * seq_len, device=device)
                    * projection_std
                )

            Y_projected = torch.matmul(Y_flat, self.y_projection)
            Y_batch = Y_projected.view(batch_size, seq_len, d_model)

        stats = self.analytical_fit(X_batch, Y_batch, mask)

        stats["y_original_shape"] = list(y_original_shape)
        stats["y_projected_shape"] = list(Y_batch.shape)
        stats["projection_used"] = y_original_shape != Y_batch.shape

        if (
            update_temperature
            and self.learn_temperature
            and len(self.attention_stats["sharpness"]) > 0
        ):
            recent_sharpness = (
                np.mean(self.attention_stats["sharpness"][-10:])
                if len(self.attention_stats["sharpness"]) >= 10
                else self.attention_stats["sharpness"][-1]
            )

            target_sharpness = 1.0
            sharpness_error = target_sharpness - recent_sharpness

            lr_temp = self.learning_rates["temperature"]
            temp_update = lr_temp * sharpness_error * 0.1

            with torch.no_grad():
                new_temp = self.temperature + temp_update
                new_temp = torch.clamp(new_temp, 0.1, 5.0)
                self.temperature.copy_(new_temp)

            stats["temperature_update"] = temp_update
            stats["sharpness"] = recent_sharpness

        if self.batch_count > 2:
            self._update_output_projection(X_batch, Y_batch)

        return stats

    def _update_output_projection(
        self, X_batch: torch.Tensor, Y_batch: torch.Tensor
    ):
        device = X_batch.device

        with torch.no_grad():
            Q = self.w_q(X_batch)
            K = self.w_k(X_batch)
            V = self.w_v(X_batch)
            attn_output, _ = self.compute_attention(Q, K, V, return_weights=False)

            X_flat = attn_output.view(-1, self.d_model)
            Y_flat = Y_batch.view(-1, self.d_model)

            reg = 1e-6
            XTX = X_flat.T @ X_flat + reg * torch.eye(self.d_model, device=device)
            XTY = X_flat.T @ Y_flat

            try:
                W_opt = torch.linalg.pinv(XTX) @ XTY
                lr = 0.05
                self.output_proj.weight.data = (
                    (1 - lr) * self.output_proj.weight.data + lr * W_opt
                )
            except RuntimeError:
                try:
                    W_opt = (torch.linalg.pinv(XTX) @ XTY).T
                    lr = 0.05
                    self.output_proj.weight.data = (
                        (1 - lr) * self.output_proj.weight.data + lr * W_opt
                    )
                except RuntimeError:
                    pass

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key = query
        value = query
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        output, attn_weights = self.compute_attention(Q, K, V, mask, return_weights=True)
        output = self.output_proj(output)
        return output

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def get_attention_weights(
        self,
        query: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return attention weights for ``query`` (self-attention)."""
        Q = self.w_q(query)
        K = self.w_k(query)
        V = self.w_v(query)
        _, attn_weights = self.compute_attention(Q, K, V, mask, return_weights=True)
        return attn_weights

    def get_fitting_summary(self) -> Dict:
        if self.batch_count == 0:
            return {"batch_count": 0, "message": "No fitting performed yet"}

        summary: Dict = {
            "total_batches": self.batch_count.item(),
            "covariance_qq_norm": self.covariance_qq.norm().item(),
            "covariance_kk_norm": self.covariance_kk.norm().item(),
            "covariance_vv_norm": self.covariance_vv.norm().item(),
            "temperature": (
                self.temperature.item()
                if self.learn_temperature
                else self.temperature
            ),
            "learning_rates": self.learning_rates.copy(),
        }

        if self.attention_stats["fitting_loss"]:
            recent_losses = self.attention_stats["fitting_loss"][-10:]
            summary["avg_recent_loss"] = np.mean(recent_losses)
            summary["loss_std"] = np.std(recent_losses)

        if self.attention_stats["sharpness"]:
            recent_sharpness = self.attention_stats["sharpness"][-10:]
            summary["avg_sharpness"] = np.mean(recent_sharpness)
            summary["sharpness_std"] = np.std(recent_sharpness)

        return summary

    def reset_fitting(self):
        self.covariance_qq.zero_()
        self.covariance_kk.zero_()
        self.covariance_vv.zero_()
        self.covariance_xy.zero_()
        self.batch_count.zero_()
        for key in self.attention_stats:
            self.attention_stats[key] = []

    def set_learning_rates(
        self,
        q: float = None,
        k: float = None,
        v: float = None,
        temperature: float = None,
    ):
        if q is not None:
            self.learning_rates["q"] = q
        if k is not None:
            self.learning_rates["k"] = k
        if v is not None:
            self.learning_rates["v"] = v
        if temperature is not None:
            self.learning_rates["temperature"] = temperature
