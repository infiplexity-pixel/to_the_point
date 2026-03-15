import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any
import math


class Conv2d(nn.Module):
    """
    Pure analytical Conv2D layer with fit_batch and finalize_fit methods.
    No gradients, no autograd, just pure torch tensor math.
    Subclass of nn.Module for compatibility.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        init_method: str = 'analytical'  # 'analytical', 'xavier', 'he'
    ):
        """
        Initialize the convolutional layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel
            stride: Stride for convolution
            padding: Padding to apply
            bias: Whether to use bias
            init_method: Initialization method
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.has_bias = bias
        self.init_method = init_method
        
        # Training state
        self.training = True
        self._batch_count = 0
        self._grad_accumulator = None
        self._bias_grad_accumulator = None
        
        # Register parameters as buffers (no gradients)
        self.register_buffer('kernel', torch.zeros(self.out_channels, self.in_channels, 
                                                   self.kernel_size[0], self.kernel_size[1]))
        if self.has_bias:
            self.register_buffer('bias', torch.zeros(self.out_channels))
        else:
            self.bias = None
        
        # Initialize parameters
        self._initialize_parameters()
        
        # Store original patterns for reference
        self.register_buffer('_original_kernel', self.kernel.clone())
        self._original_bias = self.bias.clone() if self.has_bias else None
        
    def _initialize_parameters(self):
        """Initialize weights and biases with chosen method."""
        k_h, k_w = self.kernel_size
        
        if self.init_method == 'analytical':
            # Pure analytical pattern for verification
            kernel_data = torch.zeros((self.out_channels, self.in_channels, k_h, k_w))
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    for i in range(k_h):
                        for j in range(k_w):
                            # Pattern: channel-based values with spatial variation
                            kernel_data[oc, ic, i, j] = (
                                (oc * 0.1) + 
                                (ic * 0.01) + 
                                ((i * k_h + j + 1) * 0.001)
                            )
            self.kernel.data = kernel_data
            
            if self.has_bias:
                bias_data = torch.zeros(self.out_channels)
                for oc in range(self.out_channels):
                    bias_data[oc] = oc * 0.01  # Small bias values
                self.bias.data = bias_data
                    
        elif self.init_method == 'xavier':
            # Xavier/Glorot initialization
            limit = math.sqrt(6 / (self.in_channels * k_h * k_w + self.out_channels * k_h * k_w))
            self.kernel.data.uniform_(-limit, limit)
            
            if self.has_bias:
                self.bias.data.zero_()
                
        elif self.init_method == 'he':
            # He initialization (for ReLU)
            std = math.sqrt(2 / (self.in_channels * k_h * k_w))
            self.kernel.data.normal_(0, std)
            
            if self.has_bias:
                self.bias.data.zero_()
        
        # Add small noise to break symmetry
        if self.init_method == 'analytical':
            self.kernel.data += torch.randn_like(self.kernel) * 0.001
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using pure torch tensors.
        
        Args:
            x: Input of shape (batch, in_channels, height, width)
            
        Returns:
            Output of shape (batch, out_channels, out_height, out_width)
        """
        batch_size, in_channels, in_height, in_width = x.shape
        
        # Calculate output dimensions
        out_height = (in_height + 2 * self.padding - self.kernel_size[0]) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size[1]) // self.stride + 1
        
        # Apply padding
        if self.padding > 0:
            x_padded = torch.nn.functional.pad(
                x,
                (self.padding, self.padding, self.padding, self.padding),
                mode='constant'
            )
        else:
            x_padded = x
        
        # Initialize output
        output = torch.zeros((batch_size, self.out_channels, out_height, out_width), device=x.device)
        
        # Perform convolution (im2col method for efficiency)
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * self.stride
                        h_end = h_start + self.kernel_size[0]
                        w_start = ow * self.stride
                        w_end = w_start + self.kernel_size[1]
                        
                        # Extract window
                        window = x_padded[b, :, h_start:h_end, w_start:w_end]
                        
                        # Compute convolution
                        conv_sum = torch.sum(self.kernel[oc] * window)
                        
                        # Add bias
                        if self.has_bias:
                            conv_sum += self.bias[oc]
                            
                        output[b, oc, oh, ow] = conv_sum
        
        return output
    
    def _compute_gradients(self, x: torch.Tensor, error: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute gradients analytically.
        
        Args:
            x: Input batch
            error: Error signal (prediction - target)
            
        Returns:
            kernel_grad, bias_grad
        """
        batch_size, _, in_height, in_width = x.shape
        _, _, out_height, out_width = error.shape
        
        # Apply padding to input for gradient computation
        if self.padding > 0:
            x_padded = torch.nn.functional.pad(
                x,
                (self.padding, self.padding, self.padding, self.padding),
                mode='constant'
            )
        else:
            x_padded = x
        
        # Initialize gradients
        kernel_grad = torch.zeros_like(self.kernel)
        bias_grad = torch.zeros(self.out_channels, device=x.device) if self.has_bias else None
        
        # Compute kernel gradients
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    for i in range(self.kernel_size[0]):
                        for j in range(self.kernel_size[1]):
                            grad_sum = 0.0
                            
                            for oh in range(out_height):
                                for ow in range(out_width):
                                    h_pos = oh * self.stride + i
                                    w_pos = ow * self.stride + j
                                    
                                    grad_sum += (
                                        error[b, oc, oh, ow] * 
                                        x_padded[b, ic, h_pos, w_pos]
                                    )
                            
                            kernel_grad[oc, ic, i, j] += grad_sum / batch_size
        
        # Compute bias gradients
        if self.has_bias:
            for oc in range(self.out_channels):
                bias_grad[oc] = torch.mean(error[:, oc])
        
        return kernel_grad, bias_grad
    
    def fit_batch(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        learning_rate: float = 0.01,
        loss_fn: str = 'mse',
        accumulate_gradients: bool = False, *args, **kwargs
    ) -> Dict[str, float]:
        """
        Train on a single batch using analytical gradient computation.
        
        Args:
            X_batch: Input batch
            Y_batch: Target batch
            learning_rate: Learning rate for updates
            loss_fn: Loss function ('mse' or 'cross_entropy')
            accumulate_gradients: Whether to accumulate gradients for later
            
        Returns:
            Dictionary with loss and other metrics
        """
        # Ensure tensors are on same device
        device = X_batch.device
        if self.kernel.device != device:
            self.kernel = self.kernel.to(device)
            if self.has_bias and self.bias is not None:
                self.bias = self.bias.to(device)
        
        # Forward pass
        Y_pred = self.forward(X_batch)
        
        # Compute loss and error
        if loss_fn == 'mse':
            # Mean squared error
            error = Y_pred - Y_batch
            loss = torch.mean(error ** 2).item()
            
        elif loss_fn == 'cross_entropy':
            # Cross-entropy with softmax
            # Apply softmax
            exp_pred = torch.exp(Y_pred - torch.max(Y_pred, dim=1, keepdim=True)[0])
            softmax_pred = exp_pred / torch.sum(exp_pred, dim=1, keepdim=True)
            
            # Convert targets to one-hot if needed
            if len(Y_batch.shape) == 1 or Y_batch.shape[1] == 1:
                Y_onehot = torch.zeros_like(softmax_pred)
                Y_onehot[torch.arange(len(Y_batch)), Y_batch.flatten().long()] = 1
            else:
                Y_onehot = Y_batch
            
            # Cross-entropy loss
            loss = -torch.mean(torch.sum(Y_onehot * torch.log(softmax_pred + 1e-8), dim=1)).item()
            
            # Error for gradient (softmax derivative)
            error = softmax_pred - Y_onehot
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")
        
        # Compute gradients
        kernel_grad, bias_grad = self._compute_gradients(X_batch, error)
        
        # Accumulate gradients if requested
        if accumulate_gradients:
            if self._grad_accumulator is None:
                self._grad_accumulator = kernel_grad
                self._bias_grad_accumulator = bias_grad
            else:
                self._grad_accumulator += kernel_grad
                if bias_grad is not None:
                    self._bias_grad_accumulator += bias_grad
            self._batch_count += 1
        else:
            # Update parameters immediately
            self.kernel.data -= learning_rate * kernel_grad
            if self.has_bias and bias_grad is not None:
                self.bias.data -= learning_rate * bias_grad
        
        return {
            'loss': loss,
            'batch_size': len(X_batch),
            'learning_rate': learning_rate
        }
    
    def fit_batch_pattern_preserving(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        learning_rate: float = 0.01,
        pattern_strength: float = 0.1,
        loss_fn: str = 'mse'
    ) -> Dict[str, float]:
        """
        Train while preserving analytical patterns.
        
        Args:
            X_batch: Input batch
            Y_batch: Target batch
            learning_rate: Learning rate
            pattern_strength: How strongly to preserve patterns (0-1)
            loss_fn: Loss function
            
        Returns:
            Dictionary with loss and metrics
        """
        # Ensure tensors are on same device
        device = X_batch.device
        if self.kernel.device != device:
            self.kernel = self.kernel.to(device)
            if self.has_bias and self.bias is not None:
                self.bias = self.bias.to(device)
        
        # Forward pass
        Y_pred = self.forward(X_batch)
        
        # Compute loss and error
        if loss_fn == 'mse':
            error = Y_pred - Y_batch
            loss = torch.mean(error ** 2).item()
        else:
            # Cross-entropy with softmax
            exp_pred = torch.exp(Y_pred - torch.max(Y_pred, dim=1, keepdim=True)[0])
            softmax_pred = exp_pred / torch.sum(exp_pred, dim=1, keepdim=True)
            
            if len(Y_batch.shape) == 1 or Y_batch.shape[1] == 1:
                Y_onehot = torch.zeros_like(softmax_pred)
                Y_onehot[torch.arange(len(Y_batch)), Y_batch.flatten().long()] = 1
            else:
                Y_onehot = Y_batch
            
            loss = -torch.mean(torch.sum(Y_onehot * torch.log(softmax_pred + 1e-8), dim=1)).item()
            error = softmax_pred - Y_onehot
        
        # Compute gradients
        kernel_grad, bias_grad = self._compute_gradients(X_batch, error)
        
        # Pattern-preserving update
        # Gradient descent part
        kernel_update = learning_rate * kernel_grad
        
        # Pattern restoration part (pull back toward original pattern)
        if hasattr(self, '_original_kernel') and self._original_kernel is not None:
            pattern_restore = pattern_strength * (self._original_kernel - self.kernel)
        else:
            pattern_restore = 0
        
        # Combined update
        self.kernel.data += pattern_restore - kernel_update
        
        if self.has_bias and bias_grad is not None:
            bias_update = learning_rate * bias_grad
            if hasattr(self, '_original_bias') and self._original_bias is not None:
                bias_restore = pattern_strength * (self._original_bias - self.bias)
            else:
                bias_restore = 0
            
            self.bias.data += bias_restore - bias_update
        
        return {
            'loss': loss,
            'pattern_strength': pattern_strength,
            'learning_rate': learning_rate
        }
    
    def finalize_fit(self, method: str = 'average', *args, **kwargs) -> Dict[str, Any]:
        """
        Finalize training by applying accumulated gradients.
        
        Args:
            method: How to apply accumulated gradients ('average' or 'sum')
            
        Returns:
            Dictionary with finalization stats
        """
        if self._grad_accumulator is None or self._batch_count == 0:
            return {'status': 'no_gradients_accumulated'}
        
        if method == 'average':
            # Average the accumulated gradients
            self.kernel.data -= self._grad_accumulator / self._batch_count
            if self.has_bias and self._bias_grad_accumulator is not None:
                self.bias.data -= self._bias_grad_accumulator / self._batch_count
            update_type = 'averaged'
            
        elif method == 'sum':
            # Sum the accumulated gradients
            self.kernel.data -= self._grad_accumulator
            if self.has_bias and self._bias_grad_accumulator is not None:
                self.bias.data -= self._bias_grad_accumulator
            update_type = 'summed'
        else:
            raise ValueError(f"Unknown finalization method: {method}")
        
        # Clear accumulators
        stats = {
            'status': 'success',
            'method': update_type,
            'batches_accumulated': self._batch_count,
            'kernel_update_norm': torch.norm(self._grad_accumulator / self._batch_count).item()
        }
        
        self._grad_accumulator = None
        self._bias_grad_accumulator = None
        self._batch_count = 0
        
        return stats
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of the layer."""
        return {
            'kernel_mean': self.kernel.mean().item(),
            'kernel_std': self.kernel.std().item(),
            'kernel_min': self.kernel.min().item(),
            'kernel_max': self.kernel.max().item(),
            'bias_mean': self.bias.mean().item() if self.has_bias else None,
            'bias_std': self.bias.std().item() if self.has_bias else None,
            'training': self.training,
            'batches_accumulated': self._batch_count
        }
    
    def reset_accumulators(self):
        """Reset gradient accumulators."""
        self._grad_accumulator = None
        self._bias_grad_accumulator = None
        self._batch_count = 0
    
    def verify_pattern(self, tolerance: float = 0.1) -> Tuple[bool, bool]:
        """
        Verify if kernel still follows analytical pattern.
        
        Returns:
            (kernel_ok, bias_ok)
        """
        if self.init_method != 'analytical':
            return True, True  # Non-analytical init doesn't need to follow pattern
        
        kernel_ok = True
        bias_ok = True
        
        k_h, k_w = self.kernel_size
        
        # Check kernel pattern
        for oc in range(self.out_channels):
            for ic in range(self.in_channels):
                for i in range(k_h):
                    for j in range(k_w):
                        expected = (
                            (oc * 0.1) + 
                            (ic * 0.01) + 
                            ((i * k_h + j + 1) * 0.001)
                        )
                        actual = self.kernel[oc, ic, i, j].item()
                        if abs(actual - expected) > tolerance:
                            kernel_ok = False
        
        # Check bias pattern
        if self.has_bias:
            for oc in range(self.out_channels):
                expected = oc * 0.01
                actual = self.bias[oc].item()
                if abs(actual - expected) > tolerance:
                    bias_ok = False
        
        return kernel_ok, bias_ok
    
    def reset_to_pattern(self):
        """Reset parameters to original analytical pattern."""
        if hasattr(self, '_original_kernel') and self._original_kernel is not None:
            self.kernel.data = self._original_kernel.clone()
        if hasattr(self, '_original_bias') and self._original_bias is not None:
            self.bias.data = self._original_bias.clone()