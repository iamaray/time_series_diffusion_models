import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import Tuple

class BaseConstructor(nn.Module):
    def __init__(self):
        super(BaseConstructor, self).__init__()
        
    def forward(self, x: torch.Tensor, yt: torch.Tensor, t: int):
        return yt
    
class ConditionalConstructor(nn.Module):
    def __init__(self, in_seq: int, in_feats: int, out_seq:int):
        super(BaseConstructor, self).__init__()
        self.proj = nn.Linear(in_features=in_seq * in_feats, out_features=out_seq)
        
    def forward(self, x: torch.Tensor, yt: torch.Tensor, t: int):
        x = x.view(-1)
        proj = self.proj(x)
        
        return torch.cat([x.unsqueeze(-1), yt.unsqueeze(-1)], dim=0)

def compute_posterior(forward_process, t):
    beta_t = forward_process.betas[t]
    alpha_t = forward_process.alphas[t]
    alpha_bar = sqrt_bar**2
    alpha_bar_prev = forward_process.raw_coeffs[t - 1]**2

    # posterior q(x_{t-1} | x_t, x0)
    var_post = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar)
    coeff_x0 = torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar)
    coeff_xt = torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar)

    mean_post = (coeff_x0 * predicted_x0) + coeff_xt * yt
    noise = torch.randn_like(yt)
    yt = mean_post + torch.sqrt(var_post) * noise
    
class DiffusionModel(nn.Module):
    def __init__(
        self, 
        model: nn.Module, 
        data_pred: bool, 
        input_constructor: nn.Module, 
        output_shape: Tuple[int, int]):
        
        super(DiffusionModel, self).__init__()

        self.model = model
        self.data_pred = data_pred
        self.input_constructor = input_constructor
        self.output_shape = output_shape

    def forward(self, x: torch.Tensor, yt: torch.Tensor, t: int):
        return self.model(self.input_constructor(x, yt, t))

    def predict(self, x: torch.Tensor, yt: torch.Tensor, t: int):
        return self.forward(x, yt, t)

    def inference(
        self, 
        x: torch.Tensor,
        forward_process,
        num_inference_steps: int = None,
        train_norm = None):
        
        """
        Full inference loop for time series diffusion model.

        Args:
            x: Input features/conditioning information
            forward_process: ForwardProcess object containing noise schedule
            num_inference_steps: Number of denoising steps (defaults to forward_process.T)

        Returns:
            Generated time series data
        """
        device = x.device
        batch_size = x.shape[0]

        if num_inference_steps is None:
            num_inference_steps = forward_process.T

        yt = torch.randn(batch_size, *self.output_shape, device=device)

        for t in reversed(range(num_inference_steps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            model_out = self.predict(x, yt, t_tensor)

            if self.data_pred:
                predicted_x0 = model_out
                sqrt_bar = forward_process.raw_coeffs[t]
                predicted_noise = (yt - sqrt_bar * predicted_x0) / forward_process.noise_coeffs[t]
            else:
                predicted_noise = model_out
                sqrt_bar = forward_process.raw_coeffs[t]
                predicted_x0 = (yt - forward_process.noise_coeffs[t] * predicted_noise) / sqrt_bar

            if t > 0:
                compute_posterior(forward_process, t)
            else:
                yt = predicted_x0

        
        
            return yt
        
