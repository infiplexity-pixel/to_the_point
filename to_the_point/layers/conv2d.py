import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        k_h, k_w = kernel_size

        self.weight = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels,
                k_h,
                k_w
            )
        )

        self.bias = nn.Parameter(
            torch.zeros(out_channels)
        )

        self.reset_accumulators()
        self.S = None

    ########################################

    def reset_accumulators(self):

        k_h, k_w = self.kernel_size

        R_size_h = 2*k_h - 1
        R_size_w = 2*k_w - 1

        self.R_xx = torch.zeros(
            self.in_channels,
            self.in_channels,
            R_size_h,
            R_size_w
        )

        self.R_xy = torch.zeros(
            self.out_channels,
            self.in_channels,
            k_h,
            k_w
        )
    def _compute_output_size(self, input_size):
        """
        Compute output size given input size (H or W).
        
        Args:
            input_size: Height or width of input
            
        Returns:
            output_size: Height or width of output
        """
        return ((input_size + 2 * self.padding * (self.kernel_size[0] - 1) - 1)
                // 1 + 1)

    ########################################

    def fit_batch(self, x, y, *args, **kwargs):

        B = x.shape[0]
        if y.shape != (x.shape[0], self.out_channels, self._compute_output_size(x.shape[-2]), self._compute_output_size(x.shape[-1])):
            if self.S is None:
                self.S = torch.randn(y.shape[-1], self.out_channels*self._compute_output_size(x.shape[-2])*self._compute_output_size(x.shape[-1]), device=y.device, dtype=y.dtype)
            y = y @ self.S
            y = y.reshape(x.shape[0], self.out_channels, self._compute_output_size(x.shape[-2]), self._compute_output_size(x.shape[-1]))
        k_h, k_w = self.kernel_size

        pad = (k_h - 1, k_w - 1)

        for b in range(B):

            xb = x[b]
            yb = y[b]

            ################################
            # R_xx accumulation
            ################################

            for ic1 in range(self.in_channels):
                for ic2 in range(self.in_channels):

                    auto = F.conv2d(
                        xb[ic1:ic1+1]
                        .unsqueeze(0),

                        xb[ic2:ic2+1]
                        .flip(-1,-2)
                        .unsqueeze(0),

                        padding=pad
                    )[0,0]

                    self.R_xx[
                        ic1,
                        ic2
                    ] += auto

            ################################
            # R_xy accumulation
            ################################

            for oc in range(self.out_channels):
                for ic in range(self.in_channels):

                    cross = F.conv2d(
                        xb[ic:ic+1]
                        .unsqueeze(0),

                        yb[oc:oc+1]
                        .flip(-1,-2)
                        .unsqueeze(0),

                        padding=self.kernel_size[0]-1
                    )[0,0]

                    self.R_xy[
                        oc,
                        ic
                    ] += cross

    ########################################

    def finalize_fit(self, regularization=1e-6, *args, **kwargs):

        C_in = self.in_channels
        C_out = self.out_channels

        k_h, k_w = self.kernel_size

        center_h = self.R_xx.shape[2] // 2
        center_w = self.R_xx.shape[3] // 2

        size = C_in * k_h * k_w

        ################################
        # Build convolution matrix
        ################################

        A = torch.zeros(size, size)

        row = 0

        for ic1 in range(C_in):
            for i in range(k_h):
                for j in range(k_w):

                    col = 0

                    for ic2 in range(C_in):
                        for u in range(k_h):
                            for v in range(k_w):

                                di = i - u
                                dj = j - v

                                A[row, col] = self.R_xx[
                                    ic1,
                                    ic2,
                                    center_h + di,
                                    center_w + dj
                                ]

                                col += 1

                    row += 1

        ################################
        # Solve per output channel
        ################################

        A += regularization * torch.eye(size)

        A_pinv = torch.linalg.pinv(A)

        W_est = torch.zeros(
            C_out,
            C_in,
            k_h,
            k_w
        )

        for oc in range(C_out):

            b = self.R_xy[oc].reshape(-1)

            w_vec = A_pinv @ b

            W_est[oc] = w_vec.reshape(
                C_in,
                k_h,
                k_w
            )

        self.weight.data = W_est

        self.reset_accumulators()

        return self.weight.data

    ########################################

    def forward(self, x):

        return F.conv2d(
            x,
            self.weight,
            self.bias,
            padding=self.padding
        )