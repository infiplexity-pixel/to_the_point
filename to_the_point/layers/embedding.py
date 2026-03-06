import torch
import math
from typing import Optional


class Embedding(torch.nn.Module):
    """Analytical embedding layer with optional positional encoding.

    Supports sinusoidal (fixed) and trainable positional embeddings.
    Token embeddings can be fitted analytically via ``fit_batch``.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        max_seq_len: int = 512,
        use_positional: bool = False,
        positional_trainable: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.use_positional = use_positional
        self.positional_trainable = positional_trainable
        self.dtype = dtype
        self.device = device if device is not None else torch.device("cpu")

        self.W = torch.nn.Parameter(
            torch.randn(
                self.vocab_size,
                self.embed_dim,
                device=self.device,
                dtype=self.dtype,
            )
            * 0.02
        )

        if self.use_positional:
            if self.positional_trainable:
                self.positional_W = torch.nn.Parameter(
                    torch.randn(
                        max_seq_len,
                        embed_dim,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    * 0.02
                )
            else:
                self.positional_W = torch.nn.Parameter(
                    self.create_sinusoidal_positions(max_seq_len, embed_dim)
                )

        self.projection = None

    def create_sinusoidal_positions(
        self, max_seq_len: int, embed_dim: int
    ) -> torch.Tensor:
        position = torch.arange(
            max_seq_len, device=self.device, dtype=self.dtype
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, device=self.device, dtype=self.dtype)
            * (-math.log(10000.0) / embed_dim)
        )

        pos_emb = torch.zeros(
            max_seq_len, embed_dim, device=self.device, dtype=self.dtype
        )
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        return pos_emb

    def forward(
        self, x: torch.Tensor, positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x.to(device=self.device)
        token_embeds = self.W.data[x.long()]

        if self.use_positional:
            batch_size, seq_len = (
                x.shape if len(x.shape) == 2 else (1, len(x))
            )

            if positions is None:
                positions = (
                    torch.arange(seq_len, device=self.device)
                    .unsqueeze(0)
                    .expand(batch_size, seq_len)
                )
            else:
                positions = positions.long().to(device=self.device)

            if self.positional_trainable:
                pos_embeds = self.positional_W.data[positions]
            else:
                pos_indices = positions % self.max_seq_len
                pos_embeds = self.positional_W.data[pos_indices]

            if len(token_embeds.shape) == 2 and len(pos_embeds.shape) == 3:
                pos_embeds = pos_embeds.reshape(-1, self.embed_dim)

            token_embeds = token_embeds + pos_embeds

        return token_embeds

    def fit_batch(
        self,
        Xbatch: torch.Tensor,
        Ybatch: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> None:
        Xbatch = Xbatch.to(device=self.device)
        Ybatch = Ybatch.to(device=self.device, dtype=self.dtype)

        N, seq_len = Xbatch.shape

        if (len(Ybatch.shape) != 3) or Ybatch.shape[-1] != self.embed_dim:
            Ybatch = Ybatch.reshape(N, -1)
            if self.projection is None:
                self.projection = torch.eye(
                    Ybatch.shape[-1],
                    seq_len * self.embed_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
            Ybatch = Ybatch @ self.projection
            Ybatch = Ybatch.reshape(N, seq_len, self.embed_dim)

        if (
            self.use_positional
            and positions is not None
            and self.positional_trainable
        ):
            positions = positions.to(device=self.device)
            pos_embeds = self.positional_W.data[positions.long()]
            Ybatch = Ybatch - pos_embeds

        embed_dim = self.embed_dim

        X_flat = Xbatch.reshape(-1).long()
        Y_flat = Ybatch.reshape(-1, embed_dim)

        counts = torch.bincount(X_flat, minlength=self.vocab_size)

        self.W.data.scatter_add_(
            0, X_flat.unsqueeze(1).expand(-1, embed_dim), Y_flat
        )

        mask = counts > 0
        if mask.any():
            self.W[mask] /= counts[mask].unsqueeze(1)

        if (
            self.use_positional
            and self.positional_trainable
            and positions is not None
        ):
            self.update_positional_embeddings(positions, Ybatch, Xbatch)

    def update_positional_embeddings(
        self,
        positions: torch.Tensor,
        Ybatch: torch.Tensor,
        Xbatch: torch.Tensor,
    ) -> None:
        N, seq_len = Xbatch.shape
        embed_dim = self.embed_dim

        pos_flat = positions.reshape(-1).long()
        Y_flat = Ybatch.reshape(-1, embed_dim)
        X_flat = Xbatch.reshape(-1).long()

        token_embeds = self.W[X_flat]
        pos_targets = Y_flat - token_embeds

        pos_counts = torch.bincount(pos_flat, minlength=self.max_seq_len)

        pos_accum = torch.zeros(
            self.max_seq_len, embed_dim, dtype=self.dtype, device=self.device
        )
        pos_accum.scatter_add_(
            0, pos_flat.unsqueeze(1).expand(-1, embed_dim), pos_targets
        )

        mask = pos_counts > 0
        if mask.any():
            pos_updates = pos_accum[mask] / pos_counts[mask].unsqueeze(1)
            self.positional_W.data[mask] = (
                0.9 * self.positional_W.data[mask] + 0.1 * pos_updates
            )

    def finalize_fit(self, *args, **kwargs) -> None:
        pass

    def get_positional_embeddings(
        self, seq_len: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        if not self.use_positional:
            return None
        if seq_len is None:
            seq_len = self.max_seq_len
        return self.positional_W[:seq_len]
