import torch
import torch.nn.functional as F
import torch.linalg as LA


class Conv2d(torch.nn.Module):
    """Analytical Conv2D layer.

    Solves ``W = (X^T X + γI)^{-1} X^T Y`` for optimal convolutional
    filters using ridge regression on unfolded image patches.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        device="cpu",
        gamma=1e2,
        use_cholesky=True,
        bias=True,
    ):
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.gamma = gamma
        self.use_cholesky = use_cholesky
        self.has_bias = bias

        kh, kw = self.kernel_size
        self.kernel_elements = in_channels * kh * kw
        self.total_params = self.kernel_elements + int(bias)

        self.weight = torch.nn.Parameter(
            torch.randn(out_channels, in_channels, kh, kw, device=device)
        )
        self.bias = (
            torch.nn.Parameter(torch.randn(out_channels, device=device))
            if bias
            else None
        )

        self.R = torch.zeros(
            self.total_params, self.total_params, device=self.device
        )
        self.QTY = torch.zeros(
            self.total_params, self.out_channels, device=self.device
        )
        self.sample_count = 0

    def _reset_stats(self):
        self.R.zero_()
        self.QTY.zero_()
        self.sample_count = 0

    def output_shape(self, input_shape):
        H, W = input_shape[-2:]
        kh, kw = self.kernel_size
        H_out = (H + 2 * self.padding - self.dilation * (kh - 1) - 1) // self.stride + 1
        W_out = (W + 2 * self.padding - self.dilation * (kw - 1) - 1) // self.stride + 1
        return H_out, W_out

    def forward(self, x):
        if x.numel() == 0:
            return x
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

    def _extract_patches(self, x):
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
        )
        patches = patches.transpose(1, 2).reshape(-1, self.kernel_elements)
        return patches

    def _label_encoding_projection(self, y_batch, H_out, W_out):
        B, num_classes = y_batch.shape
        target_dim = self.out_channels * H_out * W_out

        Q_l = (
            torch.randn(num_classes, target_dim, device=self.device, dtype=y_batch.dtype)
            * 0.1
        )
        Y_projected = y_batch @ Q_l
        Y_feature_maps = Y_projected.reshape(B, self.out_channels, H_out, W_out)
        return Y_feature_maps

    def fit_batch(self, X_batch, y_batch, *args, **kwargs):
        if X_batch.shape[0] == 0:
            return

        B = X_batch.shape[0]
        H_out, W_out = self.output_shape(X_batch.shape)
        num_patches = B * H_out * W_out

        X_patches = self._extract_patches(X_batch)
        ones = torch.ones(num_patches, 1, device=self.device, dtype=X_patches.dtype)
        X_aug = torch.cat([X_patches, ones], dim=1)

        Y_batch_proj = self._label_encoding_projection(y_batch, H_out, W_out)
        Y_flat = Y_batch_proj.permute(0, 2, 3, 1).reshape(
            num_patches, self.out_channels
        )

        self.R += X_aug.T @ X_aug
        self.QTY += X_aug.T @ Y_flat
        self.sample_count += num_patches

    def finalize_fit(self, *args, **kwargs):
        if self.sample_count == 0:
            return

        R_reg = self.R + self.gamma * torch.eye(
            self.total_params, device=self.device
        )

        try:
            if self.use_cholesky:
                L = torch.linalg.cholesky(R_reg)
                weights_bias = torch.cholesky_solve(self.QTY, L)
            else:
                weights_bias = LA.pinv(R_reg) @ self.QTY

            if self.bias is not None:
                weights_flat = weights_bias[:-1]
                bias = weights_bias[-1]
            else:
                weights_flat = weights_bias

            kh, kw = self.kernel_size
            self.weight.data = weights_flat.reshape(
                self.in_channels, kh, kw, self.out_channels
            ).permute(3, 0, 1, 2)

            if self.bias is not None:
                self.bias.data = bias

        except RuntimeError:
            weights_bias = LA.pinv(R_reg) @ self.QTY

    def train_analytical(self, train_loader):
        self.train()
        self._reset_stats()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            if y_batch.dim() == 1:
                y_batch = F.one_hot(y_batch, num_classes=10).float()
            y_batch = y_batch.to(self.device)
            self.fit_batch(X_batch, y_batch)

        self.finalize_fit()
