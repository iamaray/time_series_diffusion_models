import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class ForwardProcess:
    def __init__(self, variance_schedule: torch.Tensor):
        self.betas = variance_schedule
        self.T = variance_schedule.shape[0]
        
        self.alphas = 1 - self.betas
        bar_alphas = torch.cumprod(self.alphas, dim=0)
        self.SNR = bar_alphas / (1 - bar_alphas)
        
        self.raw_coeffs = torch.sqrt(bar_alphas)
        self.noise_coeffs = torch.sqrt(1 - bar_alphas)
    
    def __call__(self, y0):
        return [ 
                self.raw_coeffs[t] * y0 + self.noise_coeffs[t] * torch.randn_like(y0) 
                for t in range(self.T) ]

class DiffusionLoader:
    def __init__(self, base_loader: DataLoader, forward_process: ForwardProcess):
        self.base_loader = base_loader
        self.forward_process = forward_process
        
    def __len__(self):
        return len(self.base_loader)
    
    def __iter__(self):
        for x, y0 in self.base_loader:
            y_seq = self.forward_process(y0)
            yield x, y_seq