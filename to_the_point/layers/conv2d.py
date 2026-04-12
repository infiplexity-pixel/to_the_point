import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        device="cpu",
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.device = device
        self.S = None

        k_h, k_w = kernel_size

        self.weight = nn.Parameter(
            torch.zeros(
                out_channels,
                in_channels,
                k_h,
                k_w,
                device=device
            )
        )

        self.bias = nn.Parameter(
            torch.zeros(out_channels, device=device)
        )

        self.reset_accumulators()

    ########################################

    def reset_accumulators(self):

        k_h, k_w = self.kernel_size

        self.size = self.in_channels * k_h * k_w

        self.R_xx = torch.zeros(
            self.size,
            self.size,
            device=self.weight.device
        )

        self.R_xy = torch.zeros(
            self.out_channels,
            self.size,
            device=self.weight.device
        )

    ########################################
    def _compute_output_size(self, input_size):
        """
        Compute output size given input size (H or W).
        
        Args:
            input_size: Height or width of input
            
        Returns:
            output_size: Height or width of output
        """
        return ((input_size + 2 * self.padding - self.kernel_size[0]) // 1 + 1)
    def fit_batch(self, x, y, *args, **kwargs):

        # Handle target projection if needed
        target_h = self._compute_output_size(x.shape[-2])
        target_w = self._compute_output_size(x.shape[-1])
        
        if y.shape != (x.shape[0], self.out_channels, target_h, target_w):
            y = y[:, :, None, None]
            if self.S is None:
                self.S = torch.randn(
                    y.shape[1], self.out_channels, target_h, target_w,
                    device=y.device, dtype=y.dtype
                )
            y = F.conv_transpose2d(y, self.S, stride=(target_h, target_w), padding=0)
        k_h, k_w = self.kernel_size
        x_pad = F.pad(
            x,
            (self.padding,
             self.padding,
             self.padding,
             self.padding)
        )

        ################################
        # accumulate statistics
        ################################
        from tqdm import tqdm
        for b in (tqdm(range(x.shape[0]), desc="Accumulating Conv2d statistics, this may take a while...")) if x.shape[0] > 1_000 else range(x.shape[0]):

            xb = x_pad[b]
            yb = y[b]

            for h in range(target_h):
                for w in range(target_w):

                    patch = xb[
                        :,
                        h:h+k_h,
                        w:w+k_w
                    ].reshape(-1)

                    y_vec = yb[
                        :,
                        h,
                        w
                    ]

                    self.R_xx += torch.outer(
                        patch,
                        patch
                    )

                    self.R_xy += torch.outer(
                        y_vec,
                        patch
                    )


    ########################################

    def finalize_fit(
        self,
        regularization=1e-6,
        *args,
        **kwargs
    ):

        k_h, k_w = self.kernel_size

        A = self.R_xx.clone()

        # regularization (critical)
        A += regularization * torch.eye(
            self.size,
            device=A.device
        )

        A_inv = torch.linalg.pinv(A)

        W = torch.zeros(
            self.out_channels,
            self.in_channels,
            k_h,
            k_w,
            device=A.device
        )

        for oc in range(self.out_channels):

            w_vec = A_inv @ self.R_xy[oc]

            W[oc] = w_vec.reshape(
                self.in_channels,
                k_h,
                k_w
            )

        self.weight.data = W

        return self.weight.data

    ########################################

    def forward(self, x):

        return F.conv2d(
            x,
            self.weight,
            self.bias,
            padding=self.padding
        )